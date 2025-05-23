export CUDA_VISIBLE_DEVICES=0,1,2,3   ###  GPU numbers
export WANDB_ENTITY=     ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e

model_name=Qwen2.5-3B-Instruct ###
BASE_MODEL=     ### stage1_phase2 checkpoint path
task=comprehension
OUTPUT_BASE_DIR=Time-R1     
EXPERIMENT_NAME=${task}/theta1   #all tasks, dynamic alpha

DATA_DIR=${OUTPUT_BASE_DIR}/datasets
OUTPUT_DIR=${OUTPUT_BASE_DIR}/check_points_theta1
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
ROLLOUT_TP_SIZE=$N_GPUS


python3 -m verl.trainer.main_ppo_s1_p3 \
algorithm.adv_estimator=grpo \
data.train_files=$DATA_DIR/train_comprehension_combined.parquet \
data.val_files=$DATA_DIR/test_comprehension_combined.parquet \
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