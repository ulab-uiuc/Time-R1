import os
import json
import random
from datetime import datetime
import pandas as pd
import pickle

def construct_prefix_time_diff(event1, event2):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Please carefully read the following two news article information:\n"
        "News article 1:\n"
        f"Headline: {event1['headline']}\n"
        f"Abstract: {event1['abstract']}\n"
        "News article 2:\n"
        f"Headline: {event2['headline']}\n"
        f"Abstract: {event2['abstract']}\n"
        "For the purpose of this inference, assume that the two events described in the articles definitely occur. "
        "Based on the information provided and your general knowledge, determine the specific occurrence date for each event and then calculate the month difference between these two dates.\n"
        "- You can recall the events related to these two and their occurrence dates to help you infer.\n"
        "- Provide your answer in the following format:\n"
        "  'Event 1: YYYY-MM, Event 2: YYYY-MM. Month difference: XX.'\n"
        "- Do not output 'No event' under any circumstances. Always provide your best inferred dates, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>Event 1: 2023-01, Event 2: 2021-11. Month difference: 14.</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def construct_prefix_time_order(event1, event2, event3):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Please carefully read the following three news article information:\n"
        "News article 1:\n"
        f"Headline: {event1['headline']}\n"
        f"Abstract: {event1['abstract']}\n"
        "News article 2:\n"
        f"Headline: {event2['headline']}\n"
        f"Abstract: {event2['abstract']}\n"
        "News article 3:\n"
        f"Headline: {event3['headline']}\n"
        f"Abstract: {event3['abstract']}\n"
        "For the purpose of this inference, assume that the three events described in the articles definitely occur. "
        "Based on the information provided and your general knowledge, determine the specific occurrence date for each event and then arrange the three events in ascending chronological order.\n"
        "- You can recall the events related to these three and their occurrence dates to help you infer.\n"
        "- Provide your answer in the following format:\n"
        "  'Event 1: YYYY-MM, Event 2: YYYY-MM, Event 3: YYYY-MM. Event order: X-X-X.'\n"
        "- Do not output 'No event' under any circumstances. Always provide your best inferred dates, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>Event 1: 2023-03, Event 2: 2020-11, Event 3: 2023-08. Event order: 2-1-3.</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def construct_prefix_time_completion(event):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Please carefully read the following news article information:\n"
        f"Headline: {event['headline']}\n"
        f"Abstract: {event['abstract']}\n"
        "For the purpose of this inference, assume that the event described in the article definitely occurs. "
        "In the article, one time expression has been masked using either <YEAR> or <MONTH>. "
        "Based on the information provided and your general knowledge, determine the specific occurrence date for the event and fill in the missing time entity by replacing the mask with the appropriate value.\n"
        "- For the occurrence date, use a complete 4-digit year and a 2-digit month (e.g., 2023-01).\n"
        "- For a missing year, provide a complete 4-digit year (e.g., 2020).\n"
        "- For a missing month, provide the full month name with correct capitalization (e.g., June).\n"
        "- You can recall the events related to this article and their occurrence dates to help you infer.\n"
        "- Provide your answer in the following format:\n"
        "  'Event: YYYY-MM. Missing entity: XXXXX.'\n"
        "- Do not output 'No event' under any circumstances. Always provide your best inferred dates, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>Event: 2021-10. Missing entity: December.</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def construct_prefix_time_inferring(event):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Please carefully read the following news article information:\n"
        f"Headline: {event['headline']}\n"
        f"Abstract: {event['abstract']}\n"
        "For the purpose of this inference, assume that the event described in the article definitely occurs. "
        "Based on the information provided and your general knowledge, determine the specific occurrence date of the event.\n"
        "- You can recall the events related to this article and their occurrence dates to help you infer.\n"
        "- Output the event's occurrence date in the format 'YYYY-MM'.\n"
        "- Do not output 'No event' under any circumstances. Always provide your best inferred date, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>2023-12</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def parse_year_month_from_true(date_str: str):
    """
    Parse a date string in the format 'YYYY-MM'. Returns (year, month) or (None, None) if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def month_difference(year1, month1, year2, month2):
    """
    Compute the month difference between two events.
    For this task, the difference is computed as the absolute difference in months.
    """
    return abs((year2 - year1) * 12 + (month2 - month1))

def event_date_to_numeric(event):
    """Convert event date to a single numeric value for easier comparison"""
    return event["year"] * 12 + event["month"]

def get_event_order(events):
    """
    Determine the chronological order of three events.
    Returns a string like "1-3-2" indicating the ordering from earliest to latest.
    """
    # Create a list of (event_index, date_value) tuples
    event_dates = [(i+1, event_date_to_numeric(event)) for i, event in enumerate(events)]
    
    # Sort by date (second element in the tuple)
    sorted_events = sorted(event_dates, key=lambda x: x[1])
    
    # Extract just the event indices from the sorted list
    order = "-".join(str(idx) for idx, _ in sorted_events)
    
    return order

def load_events(input_dir, years, split_ratio=0.9, random_seed=1024):
    """
    Load events from files and split them into train and test sets.
    Returns a tuple of (train_events, test_events)
    """
    all_events = []
    
    # Iterate over each year's file
    for year in years:
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue
                
            for line in lines:
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue

                true_pub_date = data.get("true_pub_date", "")
                year_val, month_val = parse_year_month_from_true(true_pub_date)
                if year_val is None or month_val is None:
                    continue

                event = {
                    "headline": data.get("headline", ""),
                    "abstract": data.get("abstract", ""),
                    "true_pub_date": true_pub_date,
                    "year": year_val,
                    "month": month_val
                }
                all_events.append(event)
    
    # Split events into train and test sets - 与event_order_nyt.py完全相同
    random.seed(random_seed)
    random.shuffle(all_events)
    split_index = int(split_ratio * len(all_events))
    
    return all_events[:split_index], all_events[split_index:]

def build_time_difference_dataset(events, output_file, num_samples, split_name):
    """
    Build a dataset for time difference between two events.
    """
    if len(events) < 2:
        print(f"Not enough events to generate {split_name} set. Need at least 2 events.")
        return
    
    samples = []
    index_counter = 0
    
    for _ in range(num_samples):
        # Randomly select 2 different events
        event1, event2 = random.sample(events, 2)
        
        # Construct the prefix with the two selected events
        prefix = construct_prefix_time_diff(event1, event2)
        
        # Calculate the month difference
        diff = month_difference(event1["year"], event1["month"], event2["year"], event2["month"])
        
        ground_truth = {
            "event1_pub_date": event1["true_pub_date"],
            "event2_pub_date": event2["true_pub_date"],
            "month_difference": diff
        }
        
        sample = {
            "data_source": "new_york_times",
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "news_time_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split_name,
                "index": index_counter,
                "task": "time_difference"
            }
        }
        samples.append(sample)
        index_counter += 1
    
    df = pd.DataFrame(samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(samples)} {split_name} samples saved to {output_file}.")

def build_time_ordering_dataset(events, output_file, num_samples, split_name):
    """
    Build a dataset for time ordering of three events.
    """
    if len(events) < 3:
        print(f"Not enough events to generate {split_name} set. Need at least 3 events.")
        return
    
    samples = []
    index_counter = 0
    
    for _ in range(num_samples):
        # Randomly select 3 different events
        selected_events = random.sample(events, 3)
        
        # Construct the prefix with the three selected events
        prefix = construct_prefix_time_order(selected_events[0], selected_events[1], selected_events[2])
        
        # Determine the correct chronological order
        event_order = get_event_order(selected_events)
        
        ground_truth = {
            "event1_pub_date": selected_events[0]["true_pub_date"],
            "event2_pub_date": selected_events[1]["true_pub_date"],
            "event3_pub_date": selected_events[2]["true_pub_date"],
            "event_order": event_order
        }
        
        sample = {
            "data_source": "new_york_times",
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "news_time_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split_name,
                "index": index_counter,
                "task": "time_ordering"
            }
        }
        samples.append(sample)
        index_counter += 1
    
    df = pd.DataFrame(samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(samples)} {split_name} samples saved to {output_file}.")

def build_time_completion_dataset(masked_events, output_file, num_samples, split_name):
    """
    Build a dataset for time entity completion task.
    This function samples from the masked events dataset and builds prompts for
    the time entity completion task.
    """
    if len(masked_events) < num_samples:
        print(f"Warning: Requested {num_samples} samples but only {len(masked_events)} masked events available.")
        num_samples = len(masked_events)
    
    # Randomly sample from the masked events
    selected_events = random.sample(masked_events, num_samples)
    
    samples = []
    index_counter = 0
    
    for event in selected_events:
        # Construct the prefix with the selected masked event
        prefix = construct_prefix_time_completion(event)
        
        ground_truth = {
            "event_pub_date": event["true_pub_date"],
            "mask_type": event["mask_type"],
            "masked_entity": event["masked_info"]
        }
        
        sample = {
            "data_source": "new_york_times",
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "news_time_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split_name,
                "index": index_counter,
                "task": "time_completion"
            }
        }
        samples.append(sample)
        index_counter += 1
    
    df = pd.DataFrame(samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(samples)} {split_name} samples saved to {output_file}.")

def build_time_inferring_dataset(events, output_file, num_samples, split_name):
    """
    Build a dataset for directly inferring the occurrence date of a single event.
    This function samples random events and builds prompts for the time inferring task.
    """
    if len(events) < num_samples:
        print(f"Warning: Requested {num_samples} samples but only {len(events)} events available.")
        num_samples = len(events)
    
    # Randomly sample from the events
    selected_events = random.sample(events, num_samples)
    
    samples = []
    index_counter = 0
    
    for event in selected_events:
        # Construct the prefix with the selected event
        prefix = construct_prefix_time_inferring(event)
        
        ground_truth = {
            "event_pub_date": event["true_pub_date"]
        }
        
        sample = {
            "data_source": "new_york_times",
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "news_time_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": split_name,
                "index": index_counter,
                "task": "time_inferring"
            }
        }
        samples.append(sample)
        index_counter += 1
    
    df = pd.DataFrame(samples)
    df.to_parquet(output_file, index=False)
    print(f"Finished! {len(samples)} {split_name} samples saved to {output_file}.")

if __name__ == "__main__":
    # 使用与event_order_nyt.py相同的数据加载逻辑
    input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"
    years = range(2016, 2024)  # Using data from 2016 to 2023
    
    # 检查是否存在已保存的训练/测试数据划分
    split_cache_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_train_test_split.pkl"
    
    if os.path.exists(split_cache_file):
        # 如果存在缓存文件，直接加载已分割的数据
        print(f"Loading train/test split from cache: {split_cache_file}")
        with open(split_cache_file, 'rb') as f:
            train_events, test_events = pickle.load(f)
    else:
        # 如果不存在缓存文件，创建并保存分割
        print(f"Creating new train/test split and saving to: {split_cache_file}")
        train_events, test_events = load_events(input_dir, years)
        
        # 保存分割结果供未来使用
        with open(split_cache_file, 'wb') as f:
            pickle.dump((train_events, test_events), f)
    
    # 构建训练数据集 (50,000 samples)
    build_time_difference_dataset(
        train_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_difference.parquet",
        num_samples=50000,
        split_name="train"
    )
    
    # 构建测试数据集 (5,000 samples)
    build_time_difference_dataset(
        test_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_difference.parquet",
        num_samples=5000,
        split_name="test"
    )

    # Build training dataset (40,000 samples)
    build_time_ordering_dataset(
        train_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_ordering.parquet",
        num_samples=50000,
        split_name="train"
    )
    
    # Build test dataset (4,000 samples)
    build_time_ordering_dataset(
        test_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_ordering.parquet",
        num_samples=5000,
        split_name="test"
    )

    # 构建直接时间推断任务数据集
    build_time_inferring_dataset(
        train_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_inferring.parquet",
        num_samples=50000,  # 如果数据足够，创建50000个样本 
        split_name="train"
    )
    
    build_time_inferring_dataset(
        test_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_inferring.parquet",
        num_samples=5000,   # 如果数据足够，创建5000个测试样本
        split_name="test"
    )

    # 加载掩码处理过的数据集
    train_masked_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/masked_time_entity/train_masked.jsonl"
    test_masked_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/masked_time_entity/test_masked.jsonl"
    
    train_masked_events = []
    test_masked_events = []
    
    # 加载训练集
    with open(train_masked_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_masked_events.append(json.loads(line.strip()))
    
    # 加载测试集
    with open(test_masked_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_masked_events.append(json.loads(line.strip()))
    
    print(f"Loaded {len(train_masked_events)} masked training events and {len(test_masked_events)} masked test events.")
    
    # 构建时间实体补全任务数据集
    build_time_completion_dataset(
        train_masked_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_completion.parquet",
        num_samples=50000,  # 如果数据足够，创建50000个样本
        split_name="train"
    )
    
    build_time_completion_dataset(
        test_masked_events,
        output_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_completion.parquet",
        num_samples=5000,   # 如果数据足够，创建5000个测试样本
        split_name="test"
    )








