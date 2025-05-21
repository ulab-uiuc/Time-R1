# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import time
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from torch.distributed import barrier
import torch.distributed as dist

import datetime  
import json

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_data_metrics(batch, use_critic=True):
    """计算训练指标，只保留time_prediction任务的必要统计"""
    sequence_score = batch.batch['token_level_scores'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    # 只保留必要的Critic指标
    metrics = {
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
    }
    
    # 添加Values相关指标(如果使用critic)
    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        
        metrics.update({
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        })
    
    # 只保留time_prediction任务的统计
    task_type = "time_prediction"
    
    # 将rewards数据直接记录到time_prediction类别下
    metrics.update({
        f'rewards/{task_type}/overall_reward_mean': torch.mean(sequence_score).detach().item(),
        f'rewards/{task_type}/overall_reward_max': torch.max(sequence_score).detach().item(),
        f'rewards/{task_type}/overall_reward_min': torch.min(sequence_score).detach().item(),
    })

    # 添加pred_reward指标
    if hasattr(batch, 'meta_info') and 'pred_rewards' in batch.meta_info:
        all_pred_rewards = torch.tensor(batch.meta_info['pred_rewards'])
        metrics.update({
            f'rewards/{task_type}/pred_reward_mean': torch.mean(all_pred_rewards).item(),
            f'rewards/{task_type}/pred_reward_max': torch.max(all_pred_rewards).item(),
            f'rewards/{task_type}/pred_reward_min': torch.min(all_pred_rewards).item(),
        })
    
    # 长度统计也只针对time_prediction任务
    metrics.update({
        f'response_length/{task_type}/mean': torch.mean(response_length).detach().item(),
        f'response_length/{task_type}/max': torch.max(response_length).detach().item(),
        f'response_length/{task_type}/min': torch.min(response_length).detach().item(),
        f'response_length/{task_type}/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        
        f'prompt_length/{task_type}/mean': torch.mean(prompt_length).detach().item(),
        f'prompt_length/{task_type}/max': torch.max(prompt_length).detach().item(),
        f'prompt_length/{task_type}/min': torch.min(prompt_length).detach().item(),
        f'prompt_length/{task_type}/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    })
    
    return metrics

def compute_data_metrics_time_reasoning(batch, use_critic=True):
    """计算训练指标，包括每种任务类型的独立统计"""
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    # Critic部分指标 - 保留关键评估指标，但移除rewards相关指标
    metrics = {
        # 保留Critic核心指标
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
    }
    
    # 将score相关指标移动到rewards类别
    metrics.update({
        'rewards/overall/mean': torch.mean(sequence_score).detach().item(),
        'rewards/overall/max': torch.max(sequence_score).detach().item(),
        'rewards/overall/min': torch.min(sequence_score).detach().item(),
    })

    # 新增 overall 的 pred_reward 和 consistency_penalty 指标
    if hasattr(batch, 'meta_info'):
        # 为 overall 添加 pred_reward 指标
        if 'pred_rewards' in batch.meta_info:
            all_pred_rewards = torch.tensor(batch.meta_info['pred_rewards'])
            metrics.update({
                'rewards/overall/pred_reward_mean': torch.mean(all_pred_rewards).item(),
                'rewards/overall/pred_reward_max': torch.max(all_pred_rewards).item(),
                'rewards/overall/pred_reward_min': torch.min(all_pred_rewards).item(),
            })
        
        # 为 overall 添加 consistency_penalty 指标
        if 'consistency_penalties' in batch.meta_info:
            all_consistency_penalties = torch.tensor(batch.meta_info['consistency_penalties'])
            metrics.update({
                'rewards/overall/consistency_penalty_mean': torch.mean(all_consistency_penalties).item(),
                'rewards/overall/consistency_penalty_max': torch.max(all_consistency_penalties).item(),
                'rewards/overall/consistency_penalty_min': torch.min(all_consistency_penalties).item(),
            })
    
    # 添加Values相关指标(如果使用critic)
    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        
        metrics.update({
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        })

    # 总体长度统计 
    metrics.update({
        'response_length/overall/mean': torch.mean(response_length).detach().item(),
        'response_length/overall/max': torch.max(response_length).detach().item(),
        'response_length/overall/min': torch.min(response_length).detach().item(),
        'response_length/overall/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        
        'prompt_length/overall/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/overall/max': torch.max(prompt_length).detach().item(),
        'prompt_length/overall/min': torch.min(prompt_length).detach().item(),
        'prompt_length/overall/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    })
    
    # 按任务类型分离指标统计
    if hasattr(batch, 'meta_info') and 'task_types' in batch.meta_info:
        task_types = batch.meta_info['task_types']
        unique_tasks = set(task_types)
        
        # 预计算一些可能需要的字典
        reward_components = {}
        format_components = {}
        consistency_penalties = {}
        
        # 如果这些信息在meta_info中可用
        if 'pred_rewards' in batch.meta_info:
            pred_rewards = batch.meta_info['pred_rewards']
            for i, task in enumerate(task_types):
                if task not in reward_components:
                    reward_components[task] = []
                reward_components[task].append(pred_rewards[i])
                
        # if 'format_bonuses' in batch.meta_info:
        #     format_bonuses = batch.meta_info['format_bonuses']
        #     for i, task in enumerate(task_types):
        #         if task not in format_components:
        #             format_components[task] = []
        #         format_components[task].append(format_bonuses[i])
                
        if 'consistency_penalties' in batch.meta_info:
            consistency_penalties_list = batch.meta_info['consistency_penalties']
            for i, task in enumerate(task_types):
                if task not in consistency_penalties:
                    consistency_penalties[task] = []
                consistency_penalties[task].append(consistency_penalties_list[i])
        
        # 统计每种任务类型的指标
        for task in unique_tasks:
            task_indices = [i for i, t in enumerate(task_types) if t == task]
            if not task_indices:
                continue
                
            task_indices_tensor = torch.tensor(task_indices)
            
            # 该任务的奖励分数
            task_scores = sequence_score[task_indices_tensor]
            
            # 该任务的响应长度
            task_resp_len = response_length[task_indices_tensor]
            
            # 该任务的提示长度
            task_prompt_len = prompt_length[task_indices_tensor]
            
            # 添加分类任务的奖励指标
            metrics.update({
                f'rewards/{task}/mean': torch.mean(task_scores).detach().item(),
                f'rewards/{task}/max': torch.max(task_scores).detach().item(),
                f'rewards/{task}/min': torch.min(task_scores).detach().item(),
            })
            
            # 添加分类任务的长度指标
            metrics.update({
                f'response_length/{task}/mean': torch.mean(task_resp_len).detach().item(),
                f'response_length/{task}/max': torch.max(task_resp_len).detach().item(),
                f'response_length/{task}/min': torch.min(task_resp_len).detach().item(),
                
                f'prompt_length/{task}/mean': torch.mean(task_prompt_len).detach().item(),
                f'prompt_length/{task}/max': torch.max(task_prompt_len).detach().item(),
                f'prompt_length/{task}/min': torch.min(task_prompt_len).detach().item(),
            })
            
            # 添加预测奖励组件（如果可用）
            if task in reward_components and reward_components[task]:
                task_pred_rewards = torch.tensor(reward_components[task])
                metrics.update({
                    f'rewards/{task}/pred_reward_mean': torch.mean(task_pred_rewards).item(),
                    f'rewards/{task}/pred_reward_max': torch.max(task_pred_rewards).item(),
                    f'rewards/{task}/pred_reward_min': torch.min(task_pred_rewards).item(),
                })
                
            # # 添加格式奖励组件（如果可用）
            # if task in format_components and format_components[task]:
            #     task_format_bonuses = torch.tensor(format_components[task])
            #     metrics.update({
            #         f'rewards/{task}/format_bonus_mean': torch.mean(task_format_bonuses).item(),
            #         f'rewards/{task}/format_bonus_max': torch.max(task_format_bonuses).item(),
            #         f'rewards/{task}/format_bonus_min': torch.min(task_format_bonuses).item(),
            #     })
                
            # 添加一致性惩罚组件（如果可用）
            if task in consistency_penalties and consistency_penalties[task]:
                task_consistency = torch.tensor(consistency_penalties[task])
                metrics.update({
                    f'rewards/{task}/consistency_penalty_mean': torch.mean(task_consistency).item(),
                    f'rewards/{task}/consistency_penalty_max': torch.max(task_consistency).item(),
                    f'rewards/{task}/consistency_penalty_min': torch.min(task_consistency).item(),
                })
    
    return metrics

# def compute_data_metrics(batch, use_critic=True):
#     # TODO: add response length
#     sequence_score = batch.batch['token_level_scores'].sum(-1)
#     sequence_reward = batch.batch['token_level_rewards'].sum(-1)

#     advantages = batch.batch['advantages']
#     returns = batch.batch['returns']

#     max_response_length = batch.batch['responses'].shape[-1]

#     prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
#     response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

#     max_prompt_length = prompt_mask.size(-1)

#     response_info = _compute_response_info(batch)
#     prompt_length = response_info['prompt_length']
#     response_length = response_info['response_length']

#     valid_adv = torch.masked_select(advantages, response_mask)
#     valid_returns = torch.masked_select(returns, response_mask)

#     if use_critic:
#         values = batch.batch['values']
#         valid_values = torch.masked_select(values, response_mask)
#         return_diff_var = torch.var(valid_returns - valid_values)
#         return_var = torch.var(valid_returns)

#     metrics = {
#         # score
#         'critic/score/mean':
#             torch.mean(sequence_score).detach().item(),
#         'critic/score/max':
#             torch.max(sequence_score).detach().item(),
#         'critic/score/min':
#             torch.min(sequence_score).detach().item(),
#         # reward
#         'critic/rewards/mean':
#             torch.mean(sequence_reward).detach().item(),
#         'critic/rewards/max':
#             torch.max(sequence_reward).detach().item(),
#         'critic/rewards/min':
#             torch.min(sequence_reward).detach().item(),
#         # adv
#         'critic/advantages/mean':
#             torch.mean(valid_adv).detach().item(),
#         'critic/advantages/max':
#             torch.max(valid_adv).detach().item(),
#         'critic/advantages/min':
#             torch.min(valid_adv).detach().item(),
#         # returns
#         'critic/returns/mean':
#             torch.mean(valid_returns).detach().item(),
#         'critic/returns/max':
#             torch.max(valid_returns).detach().item(),
#         'critic/returns/min':
#             torch.min(valid_returns).detach().item(),
#         **({
#             # values
#             'critic/values/mean': torch.mean(valid_values).detach().item(),
#             'critic/values/max': torch.max(valid_values).detach().item(),
#             'critic/values/min': torch.min(valid_values).detach().item(),
#             # vf explained var
#             'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
#         } if use_critic else {}),

#         # response length
#         'response_length/mean':
#             torch.mean(response_length).detach().item(),
#         'response_length/max':
#             torch.max(response_length).detach().item(),
#         'response_length/min':
#             torch.min(response_length).detach().item(),
#         'response_length/clip_ratio':
#             torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
#         # prompt length
#         'prompt_length/mean':
#             torch.mean(prompt_length).detach().item(),
#         'prompt_length/max':
#             torch.max(prompt_length).detach().item(),
#         'prompt_length/min':
#             torch.min(prompt_length).detach().item(),
#         'prompt_length/clip_ratio':
#             torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
#     }
#     return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # # 安全获取 Rank
        # if dist.is_initialized():
        #     self.is_main_process = (dist.get_rank() == 0)
        # else:
        #     self.is_main_process = True  # 单进程模式
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):

        # import torch
        from torch.utils.data import DataLoader
        # # 创建一个固定种子的随机生成器
        # shuffle_generator = torch.Generator()
        # shuffle_generator.manual_seed(1024)

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                        #    generator=shuffle_generator,  # 指定固定种子的生成器
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        
        # from torch.utils.data import Subset

        # # 设置需要抽取的样本数
        # num_val_samples = 1024

        # # 获取所有样本的索引，并用固定种子（可选）保证可复现性
        # indices = torch.randperm(len(self.val_dataset), generator=shuffle_generator)[:num_val_samples].tolist()

        # # 构造子集
        # subset_val_dataset = Subset(self.val_dataset, indices)


        self.val_dataloader = DataLoader(dataset=self.val_dataset, #self.val_dataset, subset_val_dataset
                                         batch_size=len(self.val_dataset), #len(self.val_dataset),
                                         shuffle=True,
                                        #  generator=shuffle_generator,  # 指定固定种子的生成器
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """增强的验证评估方法，添加按月份统计结果和响应长度统计"""
        reward_tensor_lst = []
        pred_reward_lst = []
        year_month_lst = []
        response_length_lst = []  # 添加响应长度收集
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # 验证生成和评估过程
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            
            # 计算响应长度
            response_info = _compute_response_info(test_batch)
            response_length_lst.append(response_info['response_length'])
            
            # 获取详细的评估结果
            test_batch.meta_info['global_step'] = self.global_steps
            reward_details = self.val_reward_fn.get_detailed_metrics(test_batch)
            reward_tensor = reward_details['reward_tensor']
            pred_rewards = reward_details.get('pred_rewards', [0.0] * reward_tensor.shape[0])
            year_month_info = reward_details.get('year_month_info', ['unknown'] * reward_tensor.shape[0])
            
            reward_tensor_lst.append(reward_tensor)
            pred_reward_lst.append(pred_rewards)
            year_month_lst.append(year_month_info)

        # 合并所有批次的结果
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        pred_rewards = np.concatenate(pred_reward_lst, axis=0)
        year_month_info = np.concatenate(year_month_lst, axis=0)
        response_length = torch.cat(response_length_lst, dim=0).cpu()  # 合并所有响应长度
        
        # 总体结果统计
        metric_dict = {
            'val/time_prediction/overall_reward': np.mean(reward_tensor.numpy()),
            'val/time_prediction/pred_reward': np.mean(pred_rewards),
            # 添加响应长度统计
            'response_length/time_prediction/val_mean': torch.mean(response_length).item(),
            'response_length/time_prediction/val_max': torch.max(response_length).item(),
            'response_length/time_prediction/val_min': torch.min(response_length).item(),
            # 可以添加更多响应长度统计
            'response_length/time_prediction/val_median': torch.median(response_length).item(),
        }
        
        # 按月份分组统计
        month_rewards = {}
        month_pred_rewards = {}
        month_response_lengths = {}  # 添加按月份的响应长度统计
        
        for i in range(len(reward_tensor)):
            month = year_month_info[i]
            if month not in month_rewards:
                month_rewards[month] = []
                month_pred_rewards[month] = []
                month_response_lengths[month] = []
            
            month_rewards[month].append(reward_tensor[i].item())
            month_pred_rewards[month].append(pred_rewards[i])
            month_response_lengths[month].append(response_length[i].item())
        
        # 添加按月份统计的指标
        for month in sorted(month_rewards.keys()):
            if month != "unknown":
                metric_dict[f'val/time_prediction/overall_reward_{month}'] = np.mean(month_rewards[month])
                metric_dict[f'val/time_prediction/pred_reward_{month}'] = np.mean(month_pred_rewards[month])
                # 添加每个月的响应长度统计
                metric_dict[f'response_length/time_prediction/val_mean_{month}'] = np.mean(month_response_lengths[month])
        
        # 打印统计结果
        print(f"\n===== VALIDATION RESULTS (Step {self.global_steps}) =====")
        print(f"Overall: score={metric_dict['val/time_prediction/overall_reward']:.4f}, pred_reward={metric_dict['val/time_prediction/pred_reward']:.4f}")
        
        for month in sorted(month_rewards.keys()):
            if month != "unknown":
                # print(f"Month {month}: score={np.mean(month_rewards[month]):.4f}, pred_reward={np.mean(month_pred_rewards[month]):.4f}, samples={len(month_rewards[month])}")
                print(f"Month {month}: score={np.mean(month_rewards[month]):.4f}, pred_reward={np.mean(month_pred_rewards[month]):.4f}")        
        # 将结果保存到单独的文件中
        validation_summary = {
            "global_step": self.global_steps,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_metrics": {
                "overall_reward": float(metric_dict['val/time_prediction/overall_reward']),
                "pred_reward": float(metric_dict['val/time_prediction/pred_reward']),
                "sample_count": len(reward_tensor)
            },
            "monthly_metrics": {}
        }
        
        for month in sorted(month_rewards.keys()):
            if month != "unknown":
                validation_summary["monthly_metrics"][month] = {
                    "overall_reward": float(np.mean(month_rewards[month])),
                    "pred_reward": float(np.mean(month_pred_rewards[month])),
                    "sample_count": len(month_rewards[month])
                }
        
        # 保存验证统计结果
        with open("/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/output_log/validation_summary_prediction_directly_from_base.jsonl", "a", encoding="utf-8") as f:
        # with open("/data/zliu331/temporal_reasoning/TinyZero/output_log/validation_summary_time_prediction_zero.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(validation_summary, ensure_ascii=False) + "\n")
        
        return metric_dict
    

    def _validate_time_reasoning(self):
        """增强的验证评估方法，添加任务类型分析和更多指标"""
        reward_tensor_lst = []
        data_source_lst = []
        pred_reward_lst = []
        task_type_lst = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # 验证生成和评估过程 - 原有代码保持不变
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            
            # 获取详细的评估结果，包括分项指标
            test_batch.meta_info['global_step'] = self.global_steps
            reward_details = self.val_reward_fn.get_detailed_metrics(test_batch)
            reward_tensor = reward_details['reward_tensor']
            pred_rewards = reward_details.get('pred_rewards', [0.0] * reward_tensor.shape[0])
            task_types = reward_details.get('task_types', ['unknown'] * reward_tensor.shape[0])
            
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('ability', ['unknown'] * reward_tensor.shape[0]))
            pred_reward_lst.append(pred_rewards)
            task_type_lst.append(task_types)

        # 合并所有批次的结果
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        pred_rewards = np.concatenate(pred_reward_lst, axis=0)
        task_types = np.concatenate(task_type_lst, axis=0)
        
        # 按数据源统计指标
        data_source_reward = {}
        data_source_pred_reward = {}
        
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_pred_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_pred_reward[data_source].append(pred_rewards[i])

        # 生成指标字典
        metric_dict = {}
        
        # 按数据源添加指标
        for data_source, rewards in data_source_reward.items():
            # metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            # metric_dict[f'val/pred_reward/{data_source}'] = np.mean(data_source_pred_reward[data_source])
            metric_dict[f'val/test_score_overall'] = np.mean(rewards)
            metric_dict[f'val/pred_reward_overall'] = np.mean(data_source_pred_reward[data_source])
        
        # 按任务类型添加指标
        task_rewards = {}
        task_pred_rewards = {}
        
        for i in range(reward_tensor.shape[0]):
            task = task_types[i]
            if task not in task_rewards:
                task_rewards[task] = []
                task_pred_rewards[task] = []
            task_rewards[task].append(reward_tensor[i].item())
            task_pred_rewards[task].append(pred_rewards[i])
        
        for task, rewards in task_rewards.items():
            metric_dict[f'val/test_score_{task}'] = np.mean(rewards)
            metric_dict[f'val/pred_reward_{task}'] = np.mean(task_pred_rewards[task])

        return metric_dict

    # def _validate(self):
    #     reward_tensor_lst = []
    #     data_source_lst = []
    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)
    #         # test_batch = test_batch.to('cuda')

    #         # we only do validation on rule-based rm
    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
    #             return {}

    #         test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
    #         test_gen_batch.meta_info = {
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': False,
    #             'validate': True,
    #         }

    #         # pad to be divisible by dp_size
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    #         # unpad
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    #         print('validation generation end')

    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # evaluate using reward_function
    #         # for certain reward function (e.g. sandbox), the generation can overlap with reward
    #         reward_tensor = self.val_reward_fn(test_batch)

    #         reward_tensor_lst.append(reward_tensor)
    #         data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

    #     reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
    #     data_sources = np.concatenate(data_source_lst, axis=0)
    #     # evaluate test_score based on data source
    #     data_source_reward = {}
    #     for i in range(reward_tensor.shape[0]):
    #         data_source = data_sources[i]
    #         if data_source not in data_source_reward:
    #             data_source_reward[data_source] = []
    #         data_source_reward[data_source].append(reward_tensor[i].item())

    #     metric_dict = {}
    #     for data_source, rewards in data_source_reward.items():
    #         metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

    #     return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # # 使用时间戳和全局步数构造唯一目录名称
        # unique_actor_dir = f"actor_step_{self.global_steps}_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        # actor_local_path = os.path.join(self.config.trainer.default_local_dir, unique_actor_dir)
        
        # actor_remote_path = None
        # if self.config.trainer.default_hdfs_dir is not None:
        #     # 如果需要，也对远程目录做类似的唯一命名
        #     unique_remote_dir = f"actor_step_{self.global_steps}_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        #     actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, unique_remote_dir)
        # 同步所有进程
        # barrier()
        # if self.is_main_process:
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)
        # else:
        #     print(f'Not main process, skip saving checkpoint for actor. {self.is_main_process=}')
        # # 再次同步
        # barrier()

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # val_metrics = self._validate()
            val_metrics = self._validate_time_reasoning()      
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 在调用reward_fn前将全局步骤添加到batch - 添加在这里
                        batch.meta_info['global_step'] = self.global_steps
                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            # val_metrics: dict = self._validate()
                            val_metrics: dict = self._validate_time_reasoning()
                            # logger.log(data=val_metrics, step=self.global_steps)   # print already in _validate
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics  
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_time_reasoning(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        # val_metrics = self._validate()
                        val_metrics = self._validate_time_reasoning()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
