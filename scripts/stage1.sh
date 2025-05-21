export CUDA_VISIBLE_DEVICES=6,7,8,9 ###
export WANDB_ENTITY=zijialiu-university-of-illinois-urbana-champaign ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero:$PYTHONPATH
set -e

model_name=Qwen2.5-3B-Instruct ###
BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_conprehension_ablation_no_dynamic_reward/conprehension/ablation_no_dynamic_reward/actor/global_step_180
# BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_comprehension_theta1/comprehension/combined_tasks_dynamic_alpha/actor/global_step_60
# BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_conprehension_phase2_1/conprehension_phase2/combined_tasks_from_inference_easy/actor/global_step_360
# BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_conprehension_phase1_1/conprehension_phase1/inference_easy_from_base/actor/global_step_30
# BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_conprehension_phase1/conprehension_phase1/inference_easy_from_base/actor/global_step_40
# BASE_MODEL=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/Qwen/Qwen2.5-3B-Instruct #/data/models/${model_name} ###
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/Qwen/Qwen2.5-3B-Instruct #/data/models/${model_name} ###
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_time_reasoning_combined_tasks_from_dynamic_alpha_increasing/time_reasoning/combined_tasks_from_dynamic_alpha_increasing/actor/global_step_780
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_time_reasoning_inferring_easy_alpha_0.1/time_reasoning/inferring_easy_alpha_0.1/actor/global_step_60
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_time_reasoning_combined_tasks_from_inferring_easy_dynamic_alpha/time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha/actor/global_step_380
# /data/zliu331/temporal_reasoning/TinyZero/check_points_zero1/news_inference/Qwen2.5-3B-Instruct/actor/global_step_30
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_zero1/news_inference/Qwen2.5-3B-Instruct/actor/global_step_180 #/data/models/${model_name} ###
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_train_from_easy/news_inference/Qwen2.5-3B-Instruct/actor/global_step_360 #/data/models/${model_name} ###
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_train_easy_kl_5e-3/news_inference/Qwen2.5-3B-Instruct/actor/global_step_200 #/data/models/${model_name} ###
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_train_easy_kl_5e-3/news_inference/train_easy_kl_5e-3/actor/global_step_200
# BASE_MODEL=/data/zliu331/temporal_reasoning/TinyZero/check_points_train_easy/news_inference/Qwen2.5-3B-Instruct/actor/global_step_90
# task=time_reasoning #countdown ###
task=comprehension
# task=conprehension_phase2
# task=prediction
OUTPUT_BASE_DIR=/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero ###
# OUTPUT_BASE_DIR=/data/zliu331/temporal_reasoning/TinyZero ###
# EXPERIMENT_NAME=${task}/combined_tasks_dynamic_alpha
EXPERIMENT_NAME=${task}/ablation_no_dynamic_reward
# EXPERIMENT_NAME=${task}/combined_tasks_from_inference_easy
# EXPERIMENT_NAME=${task}/inference_easy_from_base #with_generated_1
# EXPERIMENT_NAME=${task}/directly_from_base #with_generated_1
# EXPERIMENT_NAME=${task}/combined_tasks_from_inferring_easy_alpha_0.1 # finetune_mixed_hard_2024JanToDec #Qwen2.5-3B-Instruct ###
# EXPERIMENT_NAME=${task}/combined_tasks_from_inferring_easy_dynamic_alpha

DATA_DIR=${OUTPUT_BASE_DIR}/datasets
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_comprehension_theta1_1
OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_conprehension_ablation_no_dynamic_reward_1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_conprehension_phase2_1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_conprehension_phase1_1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_prediction_from_base #with_generated_1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_time_prediction_from_base #with_generated_1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_time_reasoning_combined_tasks_from_inferring_easy_alpha_0.1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_time_reasoning_combined_tasks_from_dynamic_alpha_increasing  #train_from_easy_temperature_1.2_rollout_11
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_time_reasoning_inferring_easy_alpha_0.1
# OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_time_reasoning_combined_tasks_from_inferring_easy_dynamic_alpha
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
ROLLOUT_TP_SIZE=$N_GPUS


# python ./data_gen/${task}.py --local_dir $DATA_DIR    train_time_prediction_with_generated_1  small_test_time_prediction   train_time_inferring_easy  train_time_reasoning_dynamic_alpha.parquet   test_time_reasoning_combined


python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$DATA_DIR/train_time_reasoning_dynamic_alpha.parquet \
data.val_files=$DATA_DIR/test_time_reasoning_combined.parquet \
data.train_batch_size=128 \
data.val_batch_size=512 \
data.prompt_key=prompt \
data.max_prompt_length=1024 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=2e-6 \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
actor_rollout_ref.actor.optim.total_training_steps=1000 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size=16 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.grad_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.critic_warmup=0 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.project_name=Temporal_Reasoning \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=${OUTPUT_DIR}/${EXPERIMENT_NAME} \
trainer.n_gpus_per_node=${N_GPUS} \
trainer.nnodes=1 \
trainer.save_freq=20 \
trainer.test_freq=10 \
trainer.total_training_steps=1000 \
trainer.total_epochs=15 2>&1 | tee verl_demo.log