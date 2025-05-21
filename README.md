# Time-R1: Towards Comprehensive Temporal Reasoning in LLMs

**Time-R1 is a framework designed to endow Language Models (LLMs) with comprehensive temporal reasoning capabilities, enabling them to understand past events, predict future occurrences, and creatively generate plausible future scenarios.**

This repository contains the official code, the Time-Bench dataset, and pre-trained model checkpoints for our paper:
> **Time-R1: Towards Comprehensive Temporal Reasoning in LLMs** > Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You  
> *Conference/Journal (e.g., NeurIPS 2025) - Please update with publication details* > [Link to Paper - e.g., arXiv link or official publication link]

---

## 🌟 Overview

Large Language Models (LLMs) demonstrate impressive capabilities but often lack robust temporal intelligence. Time-R1 addresses this by introducing a novel **three-stage reinforcement learning (RL) curriculum** driven by a meticulously designed **dynamic rule-based reward system**. Our approach progressively builds:
1.  **(Stage 1: Comprehension)** Foundational temporal understanding and logical event-time mappings from historical data.
2.  **(Stage 2: Prediction)** Skills to predict future event times, especially for events beyond the model's knowledge cutoff, using synthetic data to ensure rigorous training and prevent information leakage.
3.  **(Stage 3: Generation)** Strong generalization to creatively generate plausible future scenarios *without direct fine-tuning* for this task, leveraging capabilities from the first two stages.

Our experiments show that a 3B-parameter Time-R1 model significantly outperforms models over 200 times its size on challenging future event prediction and creative scenario generation benchmarks.

---

## 🚀 Key Features

* **Comprehensive Temporal Reasoning:** Unified capabilities for understanding, prediction, and creative generation related to time.
* **Novel 3-Stage RL Curriculum:** Progressively builds temporal skills from foundational understanding to advanced future-oriented reasoning.
* **Dynamic Reward System:** Meticulously designed rewards guide the LLM effectively.
* **State-of-the-Art Performance:** Our 3B Time-R1 model surpasses significantly larger models on key temporal tasks.
* **Time-Bench Dataset:** A new large-scale, multi-task temporal reasoning dataset derived from ten years of news data.
* **Pre-trained Models:** Release of Time-R1 model checkpoints.

---

## 📚 Released Resources

* **Time-Bench Dataset:** [Link to Dataset - e.g., Hugging Face Datasets, GitHub LFS, or download script]
    * Contains over 200,000 examples with explicit temporal annotations.
    * Covers diverse tasks: timestamp inference, time-gap estimation, event ordering, and masked time entity completion.
    * Further details on dataset construction can be found in Appendix [X] of our paper and [link to your dataset appendix/documentation if separate].
* **Time-R1 Model Checkpoints:** [Link to Model Checkpoints - e.g., Hugging Face Hub, Google Drive]
    * Includes checkpoints for $\theta_1$ (after Stage 1) and $\theta_2$ (after Stage 2).
* **Source Code:** For training Time-R1 and evaluating on Time-Bench.

---

## 🛠️ Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ulab-uiuc/Time-R1.git](https://github.com/ulab-uiuc/Time-R1.git) # Or your actual repo URL
    cd Time-R1
    ```
2.  **Create Conda Environment (Recommended):**
    ```bash
    conda create -n timer1 python=3.9 # Or your preferred Python version
    conda activate timer1
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or if you use poetry/other package managers, provide instructions here.
    # Mention any specific CUDA versions or other system-level dependencies if crucial.
    ```
    *(Please provide a `requirements.txt` file in your repository.)*

---

## 📊 Dataset (Time-Bench)

The Time-Bench dataset is central to training and evaluating Time-R1.
* **Download:** [Instructions or link to download/access Time-Bench]
* **Structure:** Briefly describe the data format (e.g., JSONL, Parquet) and key fields.
    ```
    Example structure:
    {
        "task": "time_inferring",
        "event": {"headline": "...", "abstract": "..."},
        "ground_truth": {"event_pub_date": "YYYY-MM"},
        // ... other fields ...
    }
    ```
* **Preprocessing:** If any preprocessing steps are required for the dataset, list them here or point to a script (e.g., `scripts/preprocess_data.py`).

---

## ⚙️ Training Time-R1

Detailed training configurations, including hyperparameters, are provided in Appendix [Y] of our paper (e.g., `Appendix~\ref{app:config_details}`).

**Example Training Command (Stage 1 - Phase 1):**
*(This is an illustrative example based on your `work.sh`. Please adapt with actual runnable commands and necessary arguments.)*
```bash
# Ensure datasets are in the correct DATA_DIR
# Modify BASE_MODEL and OUTPUT_DIR as needed

export CUDA_VISIBLE_DEVICES=0,1,2,3 # Or your GPU setup
export DATA_DIR=/path/to/your/datasets
export BASE_MODEL=/path/to/Qwen2.5-3B-Instruct 
export OUTPUT_DIR=/path/to/your/output_checkpoints
export EXPERIMENT_NAME=stage1/phase1_infer_easy

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_stage1_phase1.parquet \
    data.val_files=$DATA_DIR/val_stage1.parquet \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.total_training_steps=100 \
    trainer.experiment_name=<span class="math-inline">EXPERIMENT\_NAME \\
    trainer\.default\_local\_dir\=</span>{OUTPUT_DIR}/${EXPERIMENT_NAME} 