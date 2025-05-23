export CUDA_VISIBLE_DEVICES=0 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_LAUNCH_BLOCKING=1
set -e
set -o pipefail


python news_generator.py \
    --model_path check_points_theta2 \
    --output_dir result \
    --start_month "2024-08" \
    --end_month "2025-02"

python analyze_generation_diversity_monthly.py \
    --input_dir result \
    --output_dir result/outputs_filtered \
    --count_per_topic 5

python analyze_monthly_similarity.py \
    --diverse_news_file result/outputs_filtered/all_diverse_news.jsonl \
    --real_news_2024_file /Time-R1/datasets/nyt_years/2024.jsonl \
    --real_news_2025_file /Time-R1/datasets/nyt_years/2025.jsonl \
    --output_dir result/analysis
