import os
import json
import pandas as pd
from datetime import datetime

def parse_year_month_from_true(date_str: str):
    """Parses the real release time string, the expected format is "YYYY-MM",
    Returns (year, month); returns (None, None) if parsing fails."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def parse_prediction(pred_str: str):
    """Remove the "Prediction:" prefix and return the remaining content."""
    return pred_str.replace("Prediction:", "").strip()

def month_difference(pred_year, pred_month, true_year, true_month):
    """Calculate the month difference between the predicted date and the real date."""
    return (pred_year - true_year) * 12 + (pred_month - true_month)

# Preset the directory where the basic data is located (store the f"{year}-0.jsonl" file)
input_dir = "Time-R1/preliminary/original_ability_result"

# Verification Set Parquet File Path
val_parquet = "Time-R1/datasets/small_test_nyt.parquet"

# Load the verification set data
df_val = pd.read_parquet(val_parquet)

errors = []            # Error used to store news that can analyze predicted dates (unit: month)
cannot_parse_count = 0 # Statistics the number of news that cannot be parsed for predicted dates
not_found_count = 0    # Statistics the number of news that has not been matched in the basic data

# traverse every record in the verification set
for idx, row in df_val.iterrows():
    # Extract the content in prompt (assuming prompt is list, take the content of the first element)
    prompt_content = row['prompt'][0]['content']
    
    # Extract the line containing the headline from prompt, assuming the format is "Headline: {headline}"
    headline = None
    for line in prompt_content.splitlines():
        if line.startswith("Headline:"):
            headline = line.replace("Headline:", "").strip()
            break
    if not headline:
        print(f"Row {idx}: No headline found in prompt.")
        cannot_parse_count += 1
        continue

    # Get the real release date, format such as "2024-08", extracted from reward_model
    true_pub_date = row['reward_model']['ground_truth']['true_pub_date']
    true_year, true_month = parse_year_month_from_true(true_pub_date)
    if true_year is None:
        print(f"Row {idx}: Cannot parse true_pub_date: {true_pub_date}")
        cannot_parse_count += 1
        continue

    # Construct the path of the basic data file for the corresponding year
    jsonl_file = os.path.join(input_dir, f"{true_year}-0.jsonl")
    if not os.path.isfile(jsonl_file):
        print(f"Row {idx}: JSONL file not found: {jsonl_file}")
        not_found_count += 1
        continue

    # Find headline matching records in the corresponding jsonl file
    matched_record = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            # Assume that the headline matches exactly (the matching strategy can be adjusted as needed)
            if data.get("headline", "").strip() == headline:
                matched_record = data
                break
    if not matched_record:
        print(f"Row {idx}: No matching record found for headline: {headline} in {jsonl_file}")
        not_found_count += 1
        continue

    # Get the predicted date from the matched record
    model_prediction = matched_record.get("model_prediction", "")
    pred_str = parse_prediction(model_prediction)
    try:
        pred_year, pred_month = map(int, pred_str.split("-"))
    except Exception:
        print(f"Row {idx}: Cannot parse prediction: {model_prediction}")
        cannot_parse_count += 1
        continue

    # Calculate the prediction error (unit: month), take the absolute value
    error = abs(month_difference(pred_year, pred_month, true_year, true_month))
    errors.append(error)

# Calculate the average prediction error (only for analytical news)
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
from verl.utils.reward_score.time_reasoning_fixed_alpha import date_prediction_reward

def parse_year_month_from_true(date_str: str):
    """Parses the real release time string, the expected format is "YYYY-MM",
    Returns (year, month); returns (None, None) if parsing fails."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def parse_prediction(pred_str: str):
    """Remove the "Prediction:" prefix and return the remaining content."""
    return pred_str.replace("Prediction:", "").strip()

def month_difference(pred_year, pred_month, true_year, true_month):
    """Calculate the month difference between the predicted date and the real date."""
    return (pred_year - true_year) * 12 + (pred_month - true_month)

# Preset the directory where the basic data is located (store the f"{year}-0.jsonl" file)
input_dir = "Time-R1/preliminary/original_ability_result"

# Verification Set Parquet File Path
val_parquet = "Time-R1/datasets/small_test_nyt.parquet"

# Load the verification set data
df_val = pd.read_parquet(val_parquet)

rewards = []            # Used to store date_prediction_reward scores for news that can parse predicted dates
cannot_parse_count = 0 # Statistics the number of news that cannot be parsed for predicted dates
not_found_count = 0    # Statistics the number of news that has not been matched in the basic data

# traverse every record in the verification set
for idx, row in df_val.iterrows():
    # Extract the content in prompt (assuming prompt is list, take the content of the first element)
    prompt_content = row['prompt'][0]['content']
    
    # Extract the line containing the headline from prompt, assuming the format is "Headline: {headline}"
    headline = None
    for line in prompt_content.splitlines():
        if line.startswith("Headline:"):
            headline = line.replace("Headline:", "").strip()
            break
    if not headline:
        print(f"Row {idx}: No headline found in prompt.")
        cannot_parse_count += 1
        continue

    # Get the real release date, format such as "2024-08", extracted from reward_model
    true_pub_date = row['reward_model']['ground_truth']['true_pub_date']
    true_year, true_month = parse_year_month_from_true(true_pub_date)
    if true_year is None:
        print(f"Row {idx}: Cannot parse true_pub_date: {true_pub_date}")
        cannot_parse_count += 1
        continue

    # Construct the path of the basic data file for the corresponding year
    jsonl_file = os.path.join(input_dir, f"{true_year}-0.jsonl")
    if not os.path.isfile(jsonl_file):
        print(f"Row {idx}: JSONL file not found: {jsonl_file}")
        not_found_count += 1
        continue

    # Find headline matching records in the corresponding jsonl file
    matched_record = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            # Assume that the headline matches exactly (the matching strategy can be adjusted as needed)
            if data.get("headline", "").strip() == headline:
                matched_record = data
                break
    if not matched_record:
        print(f"Row {idx}: No matching record found for headline: {headline} in {jsonl_file}")
        not_found_count += 1
        continue

    # Get the predicted date from the matched record
    model_prediction = matched_record.get("model_prediction", "")
    pred_str_value = parse_prediction(model_prediction)
    try:
        pred_year, pred_month = map(int, pred_str_value.split("-"))
    except Exception:
        print(f"Row {idx}: Cannot parse prediction: {model_prediction}")
        cannot_parse_count += 1
        continue

    # Format the prediction date as a string, for example "2025-03"
    pred_date_str = f"{pred_year}-{pred_month:02d}"

    # Use date_prediction_reward in news.py to calculate reward scores
    reward_score = date_prediction_reward(pred_date_str, true_pub_date, alpha=0.05)
    rewards.append(reward_score)

# Calculate the average date_prediction_reward (only for parsable news)
if rewards:
    average_reward = sum(rewards) / len(rewards)
else:
    average_reward = None

print("Average original date_prediction_reward for parsed news:", average_reward)
print("Number of news with unparseable prediction date:", cannot_parse_count)
print("Number of news not found in JSONL files:", not_found_count)
print("Total news processed with parsed predictions:", len(rewards))