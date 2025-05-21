import os
import json
import pandas as pd
from datetime import datetime

def parse_year_month_from_true(date_str: str):
    """
    解析真实发布时间字符串，期望格式为 "YYYY-MM"，
    返回 (year, month)；若解析失败则返回 (None, None)。
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def parse_prediction(pred_str: str):
    """
    去掉 "Prediction:" 前缀，返回剩余内容。
    """
    return pred_str.replace("Prediction:", "").strip()

def month_difference(pred_year, pred_month, true_year, true_month):
    """
    计算预测日期与真实日期之间的月份差。
    """
    return (pred_year - true_year) * 12 + (pred_month - true_month)

# 预设基础数据所在目录（存放 f"{year}-0.jsonl" 文件）
input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"

# 验证集 Parquet 文件路径
val_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_nyt.parquet"

# 加载验证集数据
df_val = pd.read_parquet(val_parquet)

errors = []            # 用于存放能解析预测日期的新闻的误差（单位：月）
cannot_parse_count = 0 # 统计无法解析预测日期的新闻数
not_found_count = 0    # 统计在基础数据中未匹配到新闻的数目

# 遍历验证集中的每一条记录
for idx, row in df_val.iterrows():
    # 提取 prompt 中的内容（假设 prompt 为 list，取第一个元素的 content）
    prompt_content = row['prompt'][0]['content']
    
    # 从 prompt 中提取出包含 headline 的那一行，假设格式为 "Headline: {headline}"
    headline = None
    for line in prompt_content.splitlines():
        if line.startswith("Headline:"):
            headline = line.replace("Headline:", "").strip()
            break
    if not headline:
        print(f"Row {idx}: No headline found in prompt.")
        cannot_parse_count += 1
        continue

    # 获取真实发布日期，格式例如 "2024-08"，从 reward_model 中提取
    true_pub_date = row['reward_model']['ground_truth']['true_pub_date']
    true_year, true_month = parse_year_month_from_true(true_pub_date)
    if true_year is None:
        print(f"Row {idx}: Cannot parse true_pub_date: {true_pub_date}")
        cannot_parse_count += 1
        continue

    # 构造对应年份的基础数据文件路径
    jsonl_file = os.path.join(input_dir, f"{true_year}-0.jsonl")
    if not os.path.isfile(jsonl_file):
        print(f"Row {idx}: JSONL file not found: {jsonl_file}")
        not_found_count += 1
        continue

    # 在对应的 jsonl 文件中查找 headline 匹配的记录
    matched_record = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            # 假设 headline 完全匹配即可（可根据需要调整匹配策略）
            if data.get("headline", "").strip() == headline:
                matched_record = data
                break
    if not matched_record:
        print(f"Row {idx}: No matching record found for headline: {headline} in {jsonl_file}")
        not_found_count += 1
        continue

    # 从匹配记录中获取预测日期
    model_prediction = matched_record.get("model_prediction", "")
    pred_str = parse_prediction(model_prediction)
    try:
        pred_year, pred_month = map(int, pred_str.split("-"))
    except Exception:
        print(f"Row {idx}: Cannot parse prediction: {model_prediction}")
        cannot_parse_count += 1
        continue

    # 计算预测误差（单位：月），取绝对值
    error = abs(month_difference(pred_year, pred_month, true_year, true_month))
    errors.append(error)

# 计算平均预测误差（仅针对能解析的新闻）
if errors:
    average_error = sum(errors) / len(errors)
else:
    average_error = None

print("Average original prediction error (in months) for parsed news:", average_error)
print("Number of news with unparseable prediction date:", cannot_parse_count)
print("Number of news not found in JSONL files:", not_found_count)
print("Total news processed with parsed predictions:", len(errors))




import os
import json
import pandas as pd
from datetime import datetime
from verl.utils.reward_score.news import date_prediction_reward

def parse_year_month_from_true(date_str: str):
    """
    解析真实发布时间字符串，期望格式为 "YYYY-MM"，
    返回 (year, month)；若解析失败则返回 (None, None)。
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def parse_prediction(pred_str: str):
    """
    去掉 "Prediction:" 前缀，返回剩余内容。
    """
    return pred_str.replace("Prediction:", "").strip()

def month_difference(pred_year, pred_month, true_year, true_month):
    """
    计算预测日期与真实日期之间的月份差。
    """
    return (pred_year - true_year) * 12 + (pred_month - true_month)

# 预设基础数据所在目录（存放 f"{year}-0.jsonl" 文件）
input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"

# 验证集 Parquet 文件路径
val_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_nyt.parquet"

# 加载验证集数据
df_val = pd.read_parquet(val_parquet)

rewards = []            # 用于存放能解析预测日期的新闻的 date_prediction_reward 分数
cannot_parse_count = 0 # 统计无法解析预测日期的新闻数
not_found_count = 0    # 统计在基础数据中未匹配到新闻的数目

# 遍历验证集中的每一条记录
for idx, row in df_val.iterrows():
    # 提取 prompt 中的内容（假设 prompt 为 list，取第一个元素的 content）
    prompt_content = row['prompt'][0]['content']
    
    # 从 prompt 中提取出包含 headline 的那一行，假设格式为 "Headline: {headline}"
    headline = None
    for line in prompt_content.splitlines():
        if line.startswith("Headline:"):
            headline = line.replace("Headline:", "").strip()
            break
    if not headline:
        print(f"Row {idx}: No headline found in prompt.")
        cannot_parse_count += 1
        continue

    # 获取真实发布日期，格式例如 "2024-08"，从 reward_model 中提取
    true_pub_date = row['reward_model']['ground_truth']['true_pub_date']
    true_year, true_month = parse_year_month_from_true(true_pub_date)
    if true_year is None:
        print(f"Row {idx}: Cannot parse true_pub_date: {true_pub_date}")
        cannot_parse_count += 1
        continue

    # 构造对应年份的基础数据文件路径
    jsonl_file = os.path.join(input_dir, f"{true_year}-0.jsonl")
    if not os.path.isfile(jsonl_file):
        print(f"Row {idx}: JSONL file not found: {jsonl_file}")
        not_found_count += 1
        continue

    # 在对应的 jsonl 文件中查找 headline 匹配的记录
    matched_record = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            # 假设 headline 完全匹配即可（可根据需要调整匹配策略）
            if data.get("headline", "").strip() == headline:
                matched_record = data
                break
    if not matched_record:
        print(f"Row {idx}: No matching record found for headline: {headline} in {jsonl_file}")
        not_found_count += 1
        continue

    # 从匹配记录中获取预测日期
    model_prediction = matched_record.get("model_prediction", "")
    pred_str_value = parse_prediction(model_prediction)
    try:
        pred_year, pred_month = map(int, pred_str_value.split("-"))
    except Exception:
        print(f"Row {idx}: Cannot parse prediction: {model_prediction}")
        cannot_parse_count += 1
        continue

    # 格式化预测日期为字符串，例如 "2025-03"
    pred_date_str = f"{pred_year}-{pred_month:02d}"

    # 使用 news.py 中的 date_prediction_reward 计算奖励分数
    reward_score = date_prediction_reward(pred_date_str, true_pub_date, alpha=0.05)
    rewards.append(reward_score)

# 计算平均 date_prediction_reward（仅针对能解析的新闻）
if rewards:
    average_reward = sum(rewards) / len(rewards)
else:
    average_reward = None

print("Average original date_prediction_reward for parsed news:", average_reward)
print("Number of news with unparseable prediction date:", cannot_parse_count)
print("Number of news not found in JSONL files:", not_found_count)
print("Total news processed with parsed predictions:", len(rewards))