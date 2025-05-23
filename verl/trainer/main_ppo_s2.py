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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
from verl import DataProto
import torch
import json
from verl.utils.reward_score import gsm8k, math, multiply, countdown, news, time_prediction
from verl.trainer.ppo.ray_trainer_s2 import RayPPOTrainer
import torch.distributed as dist

# Add in the import section at the top of the file
import numpy as np

# Add before RewardManager class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source == 'new_york_times':
        return time_prediction.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # Added: Tracking statistics for prediction tasks
        self.task_stats = {
            "time_prediction": {"count": 0, "total_score": 0.0}
        }
        self.last_global_step = -1  # Record the global steps of the last processing

    def get_detailed_metrics(self, data: DataProto):
        """Get detailed reward metrics for verification stage
        Return reward_tensor and related pred_rewards and task_types"""
        global_step = data.meta_info.get('global_step', 0)

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pred_rewards = []
        task_types = []
        
        # Add month information collection
        year_month_info = []
        
        # Add this line to track the printed data source
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode the complete sequence for printing
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            response_str = "<think>" + self.tokenizer.decode(valid_response_ids)
            
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            
            # Extract task type and month information
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            task_type = "time_prediction"
            
            # Get Month Information
            year = extra_info.get('year', None)
            month = extra_info.get('month', None)
            if year is not None and month is not None:
                year_month_info.append(f"{year}-{month:02d}")
            else:
                year_month_info.append("unknown")
            
            # Merge task types to ground_truth
            if task_type and isinstance(ground_truth, dict):
                ground_truth = ground_truth.copy()
                ground_truth['task'] = task_type
                ground_truth['global_step'] = global_step
            
            # Get complete rating information
            score, pred_reward, format_bonus, tag_format_score, tag_count_score, _, detected_task = compute_score_fn(
                solution_str=response_str, 
                ground_truth=ground_truth
            )

            # Add to test set log
            log_data = {
                "sequences_str": sequences_str,
                "ground_truth": ground_truth,
                "total_score": score,
                "pred_reward": pred_reward,
                "format_bonus": format_bonus,
                "tag_format_score": tag_format_score,
                "tag_count_score": tag_count_score,
                "task_type": detected_task,
                "is_validation": True,
                "global_step": global_step,
                "year_month": year_month_info[-1]
            }
            
            # Write to a special verification log file
            with open("Time-R1/output_log/validation_prediction_output.jsonl", "a", encoding="utf-8") as f:  # path to your log file
                f.write(json.dumps(log_data, ensure_ascii=False, cls=NumpyEncoder) + "\n")
            
            # Add verification sample printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
                
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n===== VALIDATION EXAMPLE =====")
                print(sequences_str, f'\nground_truth: {ground_truth}', 
                    f'total_score: {score}', f'VAL task: {task_type}', 
                    f'pred: {pred_reward}', f'format: {format_bonus}', 
                    f'tag_format: {tag_format_score}', f'tag_count: {tag_count_score}')
            
            reward_tensor[i, valid_response_length - 1] = score
            pred_rewards.append(pred_reward)
            task_types.append(detected_task)
        
        return {
            'reward_tensor': reward_tensor,
            'pred_rewards': pred_rewards,
            'task_types': task_types,
            'year_month_info': year_month_info
        }

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # Print step information only when steps change
        global_step = data.meta_info.get('global_step', 0)
        if global_step != self.last_global_step:
            self.last_global_step = global_step
            if global_step % 5 == 0 or global_step == 1:
                print(f"Global Steps {global_step}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Collect task type information and predict rewards
        task_types = []
        pred_rewards = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            response_str = "<think>" + self.tokenizer.decode(valid_response_ids) 

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Extract task type information
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            task_type = "time_prediction"
            
            # Key modification: Merge extra_info into ground_truth
            if task_type and isinstance(ground_truth, dict):
                ground_truth = ground_truth.copy()
                ground_truth['task'] = task_type
                ground_truth['global_step'] = global_step

            # Select the appropriate scoring function
            score, pred_reward, format_bonus, tag_format_score, tag_count_score, _, detected_task = compute_score_fn(
                solution_str=response_str, 
                ground_truth=ground_truth
            )
            
            # Record task type
            task_types.append(detected_task)
            pred_rewards.append(pred_reward)
            
            # Update task statistics
            if detected_task in self.task_stats:
                self.task_stats[detected_task]["count"] += 1
                self.task_stats[detected_task]["total_score"] += score
            
            reward_tensor[i, valid_response_length - 1] = score

            # Save print information as JSONL
            log_data = {
                "sequences_str": sequences_str,
                "ground_truth": ground_truth,
                "total_score": score,
                "pred_reward": pred_reward,
                "format_bonus": format_bonus,
                "tag_format_score": tag_format_score,
                "tag_count_score": tag_count_score,
                "task_type": detected_task,
                "global_step": global_step
            }

            with open("Time-R1/output_log/prediction_output.jsonl", "a", encoding="utf-8") as f:  # path to your log file
                f.write(json.dumps(log_data, ensure_ascii=False, cls=NumpyEncoder) + "\n")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str, f'\nground_truth: {ground_truth}', 
                      f'total_score: {score}', f'task: {task_type}', 
                      f'pred: {pred_reward}', f'format: {format_bonus}', 
                      f'tag_format: {tag_format_score}', f'tag_count: {tag_count_score}')

        # Add task type to the meta_info of data
        data.meta_info['task_types'] = task_types
        data.meta_info['pred_rewards'] = pred_rewards

        return reward_tensor



import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    print(config.actor_rollout_ref.model.path)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    print(  local_path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=5)  #0

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=3)  #1 

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # # Initialize distributed process groups
    # if not dist.is_initialized():
    #     dist.init_process_group(
    #         backend="nccl",
    #         init_method="env://",
    #         rank=int(os.environ.get("LOCAL_RANK", 0)),
    #         world_size=4
    #     )

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
