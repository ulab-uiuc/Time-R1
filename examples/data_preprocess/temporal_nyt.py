import os
import json
import re
from datetime import datetime
import pandas as pd

#############################
# 1. 一些辅助函数
#############################

def parse_year_month_from_true(date_str: str):
    """
    解析真实发布时间字符串，期望格式为 "YYYY-MM"。
    返回 (year, month)，若解析失败则返回 (None, None)。
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except:
        return None, None

def parse_prediction(pred_str: str):
    """
    从模型预测字段中去掉 "Prediction:" 并返回剩余内容。
    例如 "Prediction: 2025-03" -> "2025-03"
    """
    return pred_str.replace("Prediction:", "").strip()

def month_difference(pred_year: int, pred_month: int, true_year: int, true_month: int) -> int:
    """
    计算预测日期与真实日期之间的月份差 diff。
    diff < 0 表示预测日期早于真实日期（提前），
    diff > 0 表示预测日期晚于真实日期（滞后）。
    """
    return (pred_year - true_year) * 12 + (pred_month - true_month)

#############################
# 2. 生成 Prompt 的函数
#############################

def make_prefix_inference(headline: str, abstract: str):
    """
    生成适用于新闻事件发生时间推理任务的对话式前缀。
    """
    prefix = (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Please carefully read the following news article information:\n"
        f"Headline: {headline}\n"
        f"Abstract: {abstract}\n"
        "For the purpose of this inference, assume that the event described in the article definitely occurs (or will occur). "
        "Based on the information provided and your general knowledge, determine the specific occurrence date of the event.\n"
        "- Output the event's occurrence date in the format 'YYYY-MM'.\n"
        "- Do not output 'No event' under any circumstances. Always provide your best inferred date, even if the information is ambiguous.\n"
        "- Show your work in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>2025-03</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )
    return prefix

#############################
# 3.1 主逻辑：读取、筛选、构造数据、输出 Parquet
#############################

def build_train_easy_parquet(
    input_dir: str,
    years: range,
    output_file: str = "train_easy_nyt.parquet"
):
    """
    读取从 2016-0.jsonl 到 2023-0.jsonl 的文件，
    筛选预测误差小于等于 3 个月的数据，将其转换成与 Countdown 类似的格式保存为 Parquet 文件。
    """
    
    # 准备一个列表，用来存储所有满足条件的样本
    all_samples = []
    index_counter = 0
    
    for year in years:
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                headline = data.get("headline", "")
                abstract = data.get("abstract", "")
                true_pub_date = data.get("true_pub_date", "")  # 例如 "2017-06"
                model_prediction = data.get("model_prediction", "")  # 例如 "Prediction: 2023-02"

                # 解析真实日期
                true_year, true_month = parse_year_month_from_true(true_pub_date)
                if true_year is None or true_month is None:
                    continue  # 无法解析真实日期，跳过
                
                # 解析预测日期
                pred_content = parse_prediction(model_prediction)  # 例如 "2023-02"
                # 这里的实验数据中不应出现 "No event"，如果出现也可直接跳过
                try:
                    pred_year, pred_month = map(int, pred_content.split("-"))
                except:
                    continue  # 无法解析预测日期，跳过
                
                # 计算月份差，判断是否在 +/-3 个月范围内
                diff = month_difference(pred_year, pred_month, true_year, true_month)
                if abs(diff) <= 3:
                    # 符合条件的样本才收集
                    prompt_text = make_prefix_inference(headline, abstract)
                    
                    # ground_truth 可以放真实日期，方便后续做评估
                    ground_truth = {
                        "true_pub_date": true_pub_date
                    }
                    
                    # 构造一个与 Countdown 类似的数据结构
                    # prompt -> list of dict with "role" and "content"
                    sample = {
                        "data_source": "new_york_times",  
                        "prompt": [{
                            "role": "user",
                            "content": prompt_text
                        }],
                        "ability": "news_inference",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": ground_truth
                        },
                        "extra_info": {
                            "split": "train_easy",
                            "index": index_counter
                        }
                    }
                    all_samples.append(sample)
                    index_counter += 1

    # 转换为 Pandas DataFrame 并写入 Parquet
    df = pd.DataFrame(all_samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(all_samples)} samples saved to {output_file}.")

#############################
# 3.2 构建 train_nyt.parquet 的主逻辑
#############################

def build_train_nyt_parquet(
    input_dir: str,
    years: range,
    output_file: str = "train_nyt.parquet"
):
    """
    读取从 2016-0.jsonl 到 2023-0.jsonl 的文件，
    筛选预测误差大于 3 个月或者无法解析预测日期的数据，
    将其转换成与 Countdown 类似的格式保存为 Parquet 文件。
    """
    all_samples = []
    index_counter = 0

    for year in years:
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                headline = data.get("headline", "")
                abstract = data.get("abstract", "")
                true_pub_date = data.get("true_pub_date", "")  # 例如 "2017-06"
                model_prediction = data.get("model_prediction", "")  # 例如 "Prediction: 2023-02"

                # 解析真实日期
                true_year, true_month = parse_year_month_from_true(true_pub_date)
                if true_year is None or true_month is None:
                    continue  # 无法解析真实日期则跳过

                # 尝试解析预测日期
                try:
                    pred_content = parse_prediction(model_prediction)  # 例如 "2023-02"
                    pred_year, pred_month = map(int, pred_content.split("-"))
                    diff = month_difference(pred_year, pred_month, true_year, true_month)
                except Exception as e:
                    # 如果解析预测日期失败，则认为满足条件（无法解析）
                    diff = None

                # 当预测日期解析失败，或预测误差大于 3 个月时，收集该样本
                if diff is None or abs(diff) > 3:
                    prompt_text = make_prefix_inference(headline, abstract)
                    ground_truth = {"true_pub_date": true_pub_date}
                    sample = {
                        "data_source": "new_york_times",
                        "prompt": [{
                            "role": "user",
                            "content": prompt_text
                        }],
                        "ability": "news_inference",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": ground_truth
                        },
                        "extra_info": {
                            "split": "train",
                            "index": index_counter
                        }
                    }
                    all_samples.append(sample)
                    index_counter += 1

    df = pd.DataFrame(all_samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(all_samples)} samples saved to {output_file}.")

#############################
# 3.3 构建 test_nyt.parquet 的主逻辑
#############################

def build_test_nyt_parquet(input_dir: str, years: range, output_file: str = "test_nyt.parquet"):
    """
    读取 2024-0.jsonl 和 2025-0.jsonl 文件，
    将其中所有数据（有效解析真实发布时间的）转换成与 Countdown 类似的格式保存为 Parquet 文件，
    用于构建验证集 test_nyt.parquet。
    """
    all_samples = []
    index_counter = 0

    for year in years:
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                headline = data.get("headline", "")
                abstract = data.get("abstract", "")
                true_pub_date = data.get("true_pub_date", "")  # 例如 "2017-06"
                # 对于验证集，我们不对 model_prediction 做过滤，直接全部包含
                model_prediction = data.get("model_prediction", "")

                # 解析真实发布时间
                true_year, true_month = parse_year_month_from_true(true_pub_date)
                if true_year is None or true_month is None:
                    continue  # 如果无法解析真实发布时间，则跳过

                prompt_text = make_prefix_inference(headline, abstract)
                ground_truth = {"true_pub_date": true_pub_date}

                sample = {
                    "data_source": "new_york_times",
                    "prompt": [{
                        "role": "user",
                        "content": prompt_text
                    }],
                    "ability": "news_inference",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth
                    },
                    "extra_info": {
                        "split": "test",
                        "index": index_counter
                    }
                }
                all_samples.append(sample)
                index_counter += 1

    df = pd.DataFrame(all_samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(all_samples)} samples saved to {output_file}.")

#############################
# 4. 实际运行
#############################
if __name__ == "__main__":
    # 假设你的文件都放在 /data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result/
    input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"
    years = range(2016, 2024)  # 2016 ~ 2023, 2024, 2026
    
    # 生成 train_easy.parquet
    build_train_easy_parquet(input_dir, years, output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_easy_nyt.parquet")

    # # 生成 train_nyt.parquet 文件
    # build_train_nyt_parquet(input_dir, years, output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_nyt.parquet")

    # # 生成 test_nyt.parquet 文件
    # build_test_nyt_parquet(input_dir, years, output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_nyt.parquet")