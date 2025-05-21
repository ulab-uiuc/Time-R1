# import json
# from datetime import datetime

# # 文件名，可根据实际情况修改
# filename = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result/2023-0.jsonl"

# # 初始化各个指标的计数器
# metrics = {
#     "Prediction: No event": 0,                             # 预测不会发生
#     "Same month": 0,                                       # 预测与真实发生月份一致
#     "Early less than 3 months": 0,                         # 预测相比真实提前了小于3个月
#     "Late less than 3 months": 0,                          # 预测相比真实滞后了小于3个月
#     "Early between 3 and 6 months": 0,                     # 预测相比真实提前了小于6个月，但超过了3个月
#     "Late between 3 and 6 months": 0,                      # 预测相比真实滞后了小于6个月，但超过了3个月
#     "Early between 6 and 12 months": 0,                    # 预测相比真实提前了小于12个月，但超过了6个月
#     "Late between 6 and 12 months": 0,                     # 预测相比真实滞后了小于12个月，但超过了6个月
#     "Early between 12 and 24 months": 0,                   # 预测相比真实提前了小于24个月，但超过了12个月
#     "Late between 12 and 24 months": 0,                    # 预测相比真实滞后了小于24个月，但超过了12个月
#     "Early more than 24 months": 0,                        # 预测相比真实提前了超过了24个月
#     "Late more than 24 months": 0                          # 预测相比真实滞后了超过了24个月
# }

# total = 0

# def parse_year_month_from_true(date_str):
#     """
#     解析真实发布时间字符串，期望格式为 "YYYY-MM"。
#     返回一个元组 (year, month)，若解析失败则返回 (None, None)。
#     """
#     try:
#         dt = datetime.strptime(date_str, "%Y-%m")
#     except Exception as e:
#         dt = None
#     return (dt.year, dt.month) if dt is not None else (None, None)

# def parse_prediction(pred_str):
#     """
#     解析模型预测字段，去掉 "Prediction:" 前缀，并返回后续内容
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
        
#         # 尝试解析预测的年份和月份（格式应为 "YYYY-MM"）
#         try:
#             pred_year, pred_month = map(int, pred_content.split("-"))
#         except Exception as e:
#             print("Error parsing prediction date:", model_prediction)
#             continue
        
#         # 解析真实的发布时间的年份和月份
#         true_year, true_month = parse_year_month_from_true(true_pub_date)
#         if true_year is None or true_month is None:
#             print("Error parsing true_pub_date:", true_pub_date)
#             continue
        
#         # 计算预测与真实日期之间的月份差值
#         # diff < 0 表示预测提前，diff > 0 表示预测滞后
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

# # 输出统计结果
# print(f"Total results: {total}\n")
# for key, count in metrics.items():
#     percent = (count / total * 100) if total > 0 else 0
#     print(f"{key}: {count} ({percent:.2f}%)")


import json
from datetime import datetime

def parse_year_month_from_true(date_str):
    """
    解析真实发布时间字符串，期望格式为 "YYYY-MM"。
    返回一个元组 (year, month)，若解析失败则返回 (None, None)。
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m")
    except Exception as e:
        dt = None
    return (dt.year, dt.month) if dt is not None else (None, None)

def parse_prediction(pred_str):
    """
    解析模型预测字段，去掉 "Prediction:" 前缀，并返回后续内容
    """
    return pred_str.replace("Prediction:", "").strip()

def process_file(filename):
    """
    处理单个年份的数据文件，返回该文件的总条数和各指标的计数。
    """
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
            
            # 尝试解析预测日期，格式应为 "YYYY-MM"
            try:
                pred_year, pred_month = map(int, pred_content.split("-"))
            except Exception as e:
                print("Error parsing prediction date:", model_prediction)
                continue
            
            # 解析真实发布时间的年份和月份
            true_year, true_month = parse_year_month_from_true(true_pub_date)
            if true_year is None or true_month is None:
                print("Error parsing true_pub_date:", true_pub_date)
                continue
            
            # 计算预测与真实之间的月份差值（负值表示预测提前）
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
            else:  # diff > 0，预测滞后
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

# 处理 2016 到 2025 年的数据
years = range(2016, 2026)
all_years_results = {}

for year in years:
    filename = f"/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result/{year}-0.jsonl"
    try:
        total, metrics = process_file(filename)
        all_years_results[year] = {"total": total, "metrics": metrics}
    except Exception as e:
        print(f"Error processing file for year {year}: {e}")

# 输出各年的统计结果
for year in sorted(all_years_results.keys()):
    print(f"\nYear: {year}")
    total = all_years_results[year]["total"]
    metrics = all_years_results[year]["metrics"]
    print(f"Total results: {total}")
    for key, count in metrics.items():
        percent = (count / total * 100) if total > 0 else 0
        print(f"{key}: {count} ({percent:.2f}%)")