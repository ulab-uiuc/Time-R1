# import json
# from datetime import datetime

# # File name can be modified according to actual situation
# filename = "Time-R1/preliminary/original_ability_result/2023-0.jsonl"

# # Initialize the counter for each indicator
# metrics = {
# "Prediction: No event": 0, # Prediction will not happen
# "Same month": 0, # The prediction is consistent with the actual occurrence month
# "Early less than 3 months": 0, # Prediction is less than 3 months ahead of the real
# "Late less than 3 months": 0, # The prediction lags less than 3 months compared to the real
# "Early between 3 and 6 months": 0, # The prediction is less than 6 months ahead of the real one, but more than 3 months
# "Late between 3 and 6 months": 0, # The prediction lags less than 6 months compared to the real, but exceeds 3 months
# "Early between 6 and 12 months": 0, # The prediction is less than 12 months ahead of the real one, but more than 6 months
# "Late between 6 and 12 months": 0, # The prediction lags less than 12 months compared to the real, but exceeds 6 months
# "Early between 12 and 24 months": 0, # The prediction is less than 24 months ahead of the real one, but more than 12 months
# "Late between 12 and 24 months": 0, # The prediction lags less than 24 months compared to the real, but exceeds 12 months
# "Early more than 24 months": 0, # Prediction is more than 24 months ahead of the real
# "Late more than 24 months": 0 # The prediction lags more than 24 months compared to the real
# }

# total = 0

# def parse_year_month_from_true(date_str):
#     """
# parse the real release time string, the expected format is "YYYY-MM".
# Returns a tuple (year, month), and returns (None, None) if parsing fails.
#     """
#     try:
#         dt = datetime.strptime(date_str, "%Y-%m")
#     except Exception as e:
#         dt = None
#     return (dt.year, dt.month) if dt is not None else (None, None)

# def parse_prediction(pred_str):
#     """
# parse the model prediction field, remove the "Prediction:" prefix, and return the following content
#     """
#     return pred_str.replace("Prediction:", "").strip()

# with open(filename, "r", encoding="utf-8") as f:
#     for line in f:
#         total += 1
#         data = json.loads(line)
#         true_pub_date = data.get("true_pub_date", "")
#         model_prediction = data.get("model_prediction", "")
        
#         pred_content = parse_prediction(model_prediction)
#         if pred_content == "No event":
#             metrics["Prediction: No event"] += 1
#             continue
        
# # Try to parse the predicted year and month (the format should be "YYYY-MM")
#         try:
#             pred_year, pred_month = map(int, pred_content.split("-"))
#         except Exception as e:
#             print("Error parsing prediction date:", model_prediction)
#             continue
        
# # parse the year and month of the real release time
#         true_year, true_month = parse_year_month_from_true(true_pub_date)
#         if true_year is None or true_month is None:
#             print("Error parsing true_pub_date:", true_pub_date)
#             continue
        
# # Calculate the difference in the month between the forecast and the real date
# # diff < 0 means prediction is advanced, diff > 0 means prediction lag
#         diff = (pred_year - true_year) * 12 + (pred_month - true_month)
        
#         if diff == 0:
#             metrics["Same month"] += 1
#         elif diff < 0:
#             diff_abs = abs(diff)
#             if diff_abs < 3:
#                 metrics["Early less than 3 months"] += 1
#             elif 3 <= diff_abs < 6:
#                 metrics["Early between 3 and 6 months"] += 1
#             elif 6 <= diff_abs < 12:
#                 metrics["Early between 6 and 12 months"] += 1
#             elif 12 <= diff_abs < 24:
#                 metrics["Early between 12 and 24 months"] += 1
#             else:
#                 metrics["Early more than 24 months"] += 1
#         else:  # diff > 0
#             if diff < 3:
#                 metrics["Late less than 3 months"] += 1
#             elif 3 <= diff < 6:
#                 metrics["Late between 3 and 6 months"] += 1
#             elif 6 <= diff < 12:
#                 metrics["Late between 6 and 12 months"] += 1
#             elif 12 <= diff < 24:
#                 metrics["Late between 12 and 24 months"] += 1
#             else:
#                 metrics["Late more than 24 months"] += 1

# # Output statistics
# print(f"Total results: {total}\n")
# for key, count in metrics.items():
#     percent = (count / total * 100) if total > 0 else 0
#     print(f"{key}: {count} ({percent:.2f}%)")


import json
from datetime import datetime

def parse_year_month_from_true(date_str):
    """Parses the real release time string, with the expected format as "YYYY-MM".
    Returns a tuple (year, month), and returns (None, None) if parsing fails."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m")
    except Exception as e:
        dt = None
    return (dt.year, dt.month) if dt is not None else (None, None)

def parse_prediction(pred_str):
    """Parses the model prediction field, removes the "Prediction:" prefix, and returns the following content"""
    return pred_str.replace("Prediction:", "").strip()

def process_file(filename):
    """Processes data files for a single year, returning the total number of digits of the file and the count of each metric."""
    metrics = {
        "Prediction: No event": 0,                             
        "Same month": 0,                                       
        "Early less than 3 months": 0,                         
        "Late less than 3 months": 0,                          
        "Early between 3 and 6 months": 0,                     
        "Late between 3 and 6 months": 0,                      
        "Early between 6 and 12 months": 0,                    
        "Late between 6 and 12 months": 0,                     
        "Early between 12 and 24 months": 0,                   
        "Late between 12 and 24 months": 0,                    
        "Early more than 24 months": 0,                        
        "Late more than 24 months": 0                          
    }
    total = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            true_pub_date = data.get("true_pub_date", "")
            model_prediction = data.get("model_prediction", "")
            
            pred_content = parse_prediction(model_prediction)
            if pred_content == "No event":
                metrics["Prediction: No event"] += 1
                continue
            
            # Try to parse the predicted date, the format should be "YYYY-MM"
            try:
                pred_year, pred_month = map(int, pred_content.split("-"))
            except Exception as e:
                print("Error parsing prediction date:", model_prediction)
                continue
            
            # Analyze the year and month of the actual release time
            true_year, true_month = parse_year_month_from_true(true_pub_date)
            if true_year is None or true_month is None:
                print("Error parsing true_pub_date:", true_pub_date)
                continue
            
            # Calculate the month difference between the forecast and the truth (negative value indicates the forecast is ahead of schedule)
            diff = (pred_year - true_year) * 12 + (pred_month - true_month)
            
            if diff == 0:
                metrics["Same month"] += 1
            elif diff < 0:
                diff_abs = abs(diff)
                if diff_abs < 3:
                    metrics["Early less than 3 months"] += 1
                elif 3 <= diff_abs < 6:
                    metrics["Early between 3 and 6 months"] += 1
                elif 6 <= diff_abs < 12:
                    metrics["Early between 6 and 12 months"] += 1
                elif 12 <= diff_abs < 24:
                    metrics["Early between 12 and 24 months"] += 1
                else:
                    metrics["Early more than 24 months"] += 1
            else:  # diff > 0, prediction lag
                if diff < 3:
                    metrics["Late less than 3 months"] += 1
                elif 3 <= diff < 6:
                    metrics["Late between 3 and 6 months"] += 1
                elif 6 <= diff < 12:
                    metrics["Late between 6 and 12 months"] += 1
                elif 12 <= diff < 24:
                    metrics["Late between 12 and 24 months"] += 1
                else:
                    metrics["Late more than 24 months"] += 1
    return total, metrics

# Process data from 2016 to 2025
years = range(2016, 2026)
all_years_results = {}

for year in years:
    filename = f"Time-R1/preliminary/original_ability_result/{year}-0.jsonl"
    try:
        total, metrics = process_file(filename)
        all_years_results[year] = {"total": total, "metrics": metrics}
    except Exception as e:
        print(f"Error processing file for year {year}: {e}")

# Output statistics for each year
for year in sorted(all_years_results.keys()):
    print(f"\nYear: {year}")
    total = all_years_results[year]["total"]
    metrics = all_years_results[year]["metrics"]
    print(f"Total results: {total}")
    for key, count in metrics.items():
        percent = (count / total * 100) if total > 0 else 0
        print(f"{key}: {count} ({percent:.2f}%)")