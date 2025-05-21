export CUDA_VISIBLE_DEVICES=7 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_LAUNCH_BLOCKING=1
set -e
set -o pipefail

step=30

# python news_generator.py \
#     --model_path /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/check_points_time_prediction_with_generated_1/time_prediction/with_generated/actor/global_step_${step} \
#     --output_dir /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_with_generated_1 \
#     --start_month "2024-08" \
#     --end_month "2025-02"

# python analyze_generation_diversity_monthly.py \
#     --input_dir /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_with_generated_1 \
#     --output_dir /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_with_generated_1/outputs_filtered \
#     --count_per_topic 5

# python analyze_monthly_similarity_baseline.py \
#     --diverse_news_file /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_with_generated_1/outputs_filtered/all_diverse_news.jsonl \
#     --real_news_2024_file /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl \
#     --real_news_2025_file /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl \
#     --output_dir /mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_with_generated_1/analysis


python news_generator.py \
    --model_path /data/zliu331/temporal_reasoning/TinyZero/check_points_time_prediction_from_base/time_prediction/from_base/actor/global_step_${step} \
    --output_dir /data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_from_base/step_${step} \
    --start_month "2024-08" \
    --end_month "2025-02"

python analyze_generation_diversity_monthly.py \
    --input_dir /data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_from_base/step_${step} \
    --output_dir /data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_from_base/step_${step}/outputs_filtered \
    --count_per_topic 5

python analyze_monthly_similarity_baseline.py \
    --diverse_news_file /data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_from_base/step_${step}/outputs_filtered/all_diverse_news.jsonl \
    --real_news_2024_file /data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl \
    --real_news_2025_file /data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl \
    --output_dir /data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results_from_base/step_${step}/analysis
