import os
import json
import random
from datetime import datetime
import pandas as pd
import pickle

def construct_prefix(event1, event2, event3):
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
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>Event 1: 2022-01, Event 2: 2023-12, Event 3: 2023-06. Event order: 1-3-2.</answer>.\n"
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
    
    # Split events into train and test sets
    random.seed(random_seed)
    random.shuffle(all_events)
    split_index = int(split_ratio * len(all_events))
    
    return all_events[:split_index], all_events[split_index:]

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
        prefix = construct_prefix(selected_events[0], selected_events[1], selected_events[2])
        
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
            "ability": "news_inference",
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

if __name__ == "__main__":
    # The input directory where the original JSONL files are stored
    input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"
    years = range(2016, 2024)  # Using data from 2016 to 2023
    
    # Load and split events
    train_events, test_events = load_events(input_dir, years)
    
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