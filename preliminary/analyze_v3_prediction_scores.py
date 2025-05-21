import json
import re
import os
import sys
from tqdm import tqdm
from verl.utils.reward_score.time_prediction import compute_score, extract_answer_format, is_valid_date_format
from collections import defaultdict

# # 确保可以导入评分函数
# # 假设 time_prediction.py 在 verl/utils/reward_score/ 目录下
# sys.path.append('/data/zliu331/temporal_reasoning/TinyZero')
# from verl.utils.reward_score.time_prediction import compute_score, extract_answer_format, is_valid_date_format

def extract_true_date_from_custom_id(custom_id):
    """从custom_id中提取日期部分 (YYYY-MM)"""
    match = re.match(r"(\d{4}-\d{2})_\d+", custom_id)
    if match:
        return match.group(1)
    return None

def analyze_v3_predictions(jsonl_file_path):
    """分析DeepSeek V3模型在时间预测任务上的表现"""
    # 初始化分数累计器
    total_scores_all = []
    accuracy_scores_all = []
    format_bonuses_all = []
    tag_format_scores_all = []
    tag_count_scores_all = []
    
    # 初始化计数器
    processed_count_total = 0
    valid_count_total = 0
    correct_predictions_total = 0
    content_filter_count_total = 0
    skipped_count_total = 0

    # 定义目标月份范围
    target_months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
    monthly_scores = {month: {"total": [], "accuracy": [], "count": 0} for month in target_months}
    
    print(f"正在分析文件: {jsonl_file_path}")
    
    # 读取并处理每条记录
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines() # 一次性读取所有行以便使用tqdm

    print(f"共读取 {len(lines)} 行数据，开始处理...")
    
    for line in tqdm(lines):
        if not line.strip():
            skipped_count_total += 1
            continue
            
        custom_id_for_error = "未知"
        try:
            data = json.loads(line)
            
            # 提取真实答案
            custom_id = data.get('custom_id')
            custom_id_for_error = custom_id if custom_id else "未知"
            if not custom_id:
                skipped_count_total += 1
                continue
                
            true_date = extract_true_date_from_custom_id(custom_id)

            # 如果无法提取真实日期，跳过该记录
            if not true_date or not is_valid_date_format(true_date):
                # print(f"无法从custom_id '{custom_id}'提取有效的日期格式") # 减少打印
                skipped_count_total += 1
                continue
            
            # 提取模型回答
            response = data.get('response', {})
            body = response.get('body', {})
            choices = body.get('choices', [])
            
            if not choices:
                # print(f"记录中没有模型回答: {custom_id}") # 减少打印
                skipped_count_total += 1
                continue

            # 检查是否被内容过滤
            if choices[0].get('finish_reason') == 'content_filter':
                content_filter_count_total += 1
                continue

            message = choices[0].get('message', {})
            content = message.get('content', '')
            # V3模型似乎将思考过程直接包含在content的<think>标签中
            # reasoning = message.get('reasoning_content', '') # V3可能没有这个字段

            # 提取预测日期
            predicted_date = extract_answer_format(content)
            
            # 检查预测是否正确
            if predicted_date == true_date:
                correct_predictions_total += 1
            
            # 构建compute_score函数需要的格式
            # 由于V3将<think>包含在content中，直接使用content作为solution_str
            # 注意：如果V3的content末尾没有<|im_end|>，tag_format_score可能为0
            solution_str = content 
            if not solution_str.strip().endswith("<|im_end|>"): # 确保有im_end, compute_score中的format_reward_single需要
                if solution_str.strip().endswith("</answer>"):
                    solution_str = solution_str.strip() + "<|im_end|>" 
                # else: # 如果连answer标签都没有，格式可能完全不对，但还是尝试加上
                #     solution_str = solution_str.strip() + "\n<answer></answer><|im_end|>"


            ground_truth = {"event_pub_date": true_date}
            
            # 计算得分
            scores_tuple = compute_score(solution_str, ground_truth)
            # 解包分数，注意compute_score返回7个值
            total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, _, _ = scores_tuple
            
            # 累加总分数
            total_scores_all.append(total_score)
            accuracy_scores_all.append(accuracy_score)
            format_bonuses_all.append(format_bonus)
            tag_format_scores_all.append(tag_format_score)
            tag_count_scores_all.append(tag_count_score)

            # 按月份累加分数
            if true_date in monthly_scores:
                monthly_scores[true_date]["total"].append(total_score)
                monthly_scores[true_date]["accuracy"].append(accuracy_score)
                monthly_scores[true_date]["count"] += 1
            
            processed_count_total += 1 # 记录实际处理并评分的样本数
            valid_count_total += 1 # 假设所有能处理的都是有效的
            
        except Exception as e:
            print(f"处理记录 {custom_id_for_error} 时出错: {str(e)}")
            skipped_count_total += 1

    # 计算总体平均分数
    if valid_count_total > 0:
        avg_total_score_all = sum(total_scores_all) / valid_count_total
        avg_accuracy_score_all = sum(accuracy_scores_all) / valid_count_total
        avg_format_bonus_all = sum(format_bonuses_all) / valid_count_total
        avg_tag_format_score_all = sum(tag_format_scores_all) / valid_count_total
        avg_tag_count_score_all = sum(tag_count_scores_all) / valid_count_total
        accuracy_all = correct_predictions_total / valid_count_total
        
        print("\n===== DeepSeek V3模型时间预测任务总体得分统计 =====")
        print(f"总共处理记录: {processed_count_total} 条, 有效记录: {valid_count_total} 条")
        print(f"跳过记录: {skipped_count_total} 条, 内容过滤: {content_filter_count_total} 条")
        print(f"准确率（完全匹配）: {accuracy_all:.4f} ({correct_predictions_total}/{valid_count_total})")
        print(f"平均总分 (avg_total_score): {avg_total_score_all:.4f}")
        print(f"平均准确度分数 (avg_accuracy_score): {avg_accuracy_score_all:.4f}")
        print(f"平均格式奖励 (avg_format_bonus): {avg_format_bonus_all:.4f}")
        print(f"平均标签格式分数 (avg_tag_format_score): {avg_tag_format_score_all:.4f}")
        print(f"平均标签计数分数 (avg_tag_count_score): {avg_tag_count_score_all:.4f}")

        print("\n===== DeepSeek V3模型 每月得分统计 (2024-07 至 2025-02) =====")
        for month in target_months:
            month_data = monthly_scores[month]
            if month_data["count"] > 0:
                avg_month_total = sum(month_data["total"]) / month_data["count"]
                avg_month_accuracy = sum(month_data["accuracy"]) / month_data["count"]
                print(f"月份: {month}")
                print(f"  有效记录数: {month_data['count']}")
                print(f"  平均总分: {avg_month_total:.4f}")
                print(f"  平均准确度分数: {avg_month_accuracy:.4f}")
            else:
                print(f"月份: {month} - 无有效记录")
        
        results_summary = {
            "avg_total_score_all": avg_total_score_all,
            "avg_accuracy_score_all": avg_accuracy_score_all,
            "avg_format_bonus_all": avg_format_bonus_all,
            "avg_tag_format_score_all": avg_tag_format_score_all,
            "avg_tag_count_score_all": avg_tag_count_score_all,
            "accuracy_all": accuracy_all,
            "valid_count_total": valid_count_total,
            "processed_count_total": processed_count_total,
            "skipped_count_total": skipped_count_total,
            "content_filter_count_total": content_filter_count_total,
            "monthly_averages": {}
        }
        for month in target_months:
            month_data = monthly_scores[month]
            if month_data["count"] > 0:
                results_summary["monthly_averages"][month] = {
                    "count": month_data["count"],
                    "avg_total_score": sum(month_data["total"]) / month_data["count"],
                    "avg_accuracy_score": sum(month_data["accuracy"]) / month_data["count"]
                }
            else:
                results_summary["monthly_averages"][month] = {
                    "count": 0,
                    "avg_total_score": 0.0,
                    "avg_accuracy_score": 0.0
                }
        return results_summary
    else:
        print("没有发现有效记录")
        return None

if __name__ == "__main__":
    # 修改为V3结果文件的路径
    file_path = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/prediction_v3_results.jsonl" 
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
    else:
        results = analyze_v3_predictions(file_path)

        output_json_path = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/v3/prediction_v3_scores_with_monthly.json"
        if results:
            output_dir = os.path.dirname(output_json_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"已创建目录: {output_dir}")
                
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n得分结果已保存到: {output_json_path}")





# import json
# import re
# import os
# import sys
# from tqdm import tqdm
# from verl.utils.reward_score.time_prediction import compute_score, extract_answer_format, is_valid_date_format


# # # 确保可以导入评分函数
# # # 假设 time_prediction.py 在 verl/utils/reward_score/ 目录下
# # sys.path.append('/data/zliu331/temporal_reasoning/TinyZero')
# # from verl.utils.reward_score.time_prediction import compute_score, extract_answer_format, is_valid_date_format

# def extract_true_date_from_custom_id(custom_id):
#     """从custom_id中提取日期部分 (YYYY-MM)"""
#     match = re.match(r"(\d{4}-\d{2})_\d+", custom_id)
#     if match:
#         return match.group(1)
#     return None

# def analyze_v3_predictions(jsonl_file_path):
#     """分析DeepSeek V3模型在时间预测任务上的表现"""
#     # 初始化分数累计器
#     total_scores = []
#     accuracy_scores = []
#     format_bonuses = []
#     tag_format_scores = []
#     tag_count_scores = []
    
#     # 初始化计数器
#     processed_count = 0
#     valid_count = 0
#     correct_predictions = 0
#     content_filter_count = 0
#     skipped_count = 0
    
#     print(f"正在分析文件: {jsonl_file_path}")
    
#     # 读取并处理每条记录
#     with open(jsonl_file_path, 'r') as f:
#         lines = f.readlines() # 一次性读取所有行以便使用tqdm

#     print(f"共读取 {len(lines)} 行数据，开始处理...")
    
#     for line in tqdm(lines):
#         if not line.strip():
#             skipped_count += 1
#             continue
            
#         try:
#             data = json.loads(line)
            
#             # 提取真实答案
#             custom_id = data.get('custom_id')
#             if not custom_id:
#                 skipped_count += 1
#                 continue
                
#             true_date = extract_true_date_from_custom_id(custom_id)
            
#             # 如果无法提取真实日期，跳过该记录
#             if not true_date or not is_valid_date_format(true_date):
#                 print(f"无法从custom_id '{custom_id}'提取有效的日期格式")
#                 skipped_count += 1
#                 continue
            
#             # 提取模型回答
#             response = data.get('response', {})
#             body = response.get('body', {})
#             choices = body.get('choices', [])
            
#             if not choices:
#                 print(f"记录中没有模型回答: {custom_id}")
#                 skipped_count += 1
#                 continue

#             # 检查是否被内容过滤
#             if choices[0].get('finish_reason') == 'content_filter':
#                 content_filter_count += 1
#                 continue

#             message = choices[0].get('message', {})
#             content = message.get('content', '')
#             # V3模型似乎将思考过程直接包含在content的<think>标签中
#             # reasoning = message.get('reasoning_content', '') # V3可能没有这个字段

#             # 提取预测日期
#             predicted_date = extract_answer_format(content)
            
#             # 检查预测是否正确
#             if predicted_date == true_date:
#                 correct_predictions += 1
            
#             # 构建compute_score函数需要的格式
#             # 由于V3将<think>包含在content中，直接使用content作为solution_str
#             solution_str = content 
#             ground_truth = {"event_pub_date": true_date}
            
#             # 计算得分
#             scores_tuple = compute_score(solution_str, ground_truth)
#             # 解包分数，注意compute_score返回7个值
#             total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, _, _ = scores_tuple
            
#             # 累加分数
#             total_scores.append(total_score)
#             accuracy_scores.append(accuracy_score)
#             format_bonuses.append(format_bonus)
#             tag_format_scores.append(tag_format_score)
#             tag_count_scores.append(tag_count_score)
            
#             processed_count += 1 # 记录实际处理并评分的样本数
#             valid_count += 1 # 假设所有能处理的都是有效的
            
#         except Exception as e:
#             print(f"处理记录 {custom_id} 时出错: {str(e)}")
#             skipped_count += 1
    
#     # 计算平均分数
#     if valid_count > 0:
#         avg_total_score = sum(total_scores) / valid_count
#         avg_accuracy_score = sum(accuracy_scores) / valid_count
#         avg_format_bonus = sum(format_bonuses) / valid_count
#         avg_tag_format_score = sum(tag_format_scores) / valid_count
#         avg_tag_count_score = sum(tag_count_scores) / valid_count
#         accuracy = correct_predictions / valid_count
        
#         print("\n===== DeepSeek V3模型时间预测任务得分统计 =====")
#         print(f"总共处理记录: {processed_count} 条, 有效记录: {valid_count} 条")
#         print(f"跳过记录: {skipped_count} 条, 内容过滤: {content_filter_count} 条")
#         print(f"准确率（完全匹配）: {accuracy:.4f} ({correct_predictions}/{valid_count})")
#         print(f"平均总分 (avg_total_score): {avg_total_score:.4f}")
#         print(f"平均准确度分数 (avg_accuracy_score): {avg_accuracy_score:.4f}")
#         print(f"平均格式奖励 (avg_format_bonus): {avg_format_bonus:.4f}")
#         print(f"平均标签格式分数 (avg_tag_format_score): {avg_tag_format_score:.4f}")
#         print(f"平均标签计数分数 (avg_tag_count_score): {avg_tag_count_score:.4f}")
        
#         return {
#             "avg_total_score": avg_total_score,
#             "avg_accuracy_score": avg_accuracy_score,
#             "avg_format_bonus": avg_format_bonus,
#             "avg_tag_format_score": avg_tag_format_score,
#             "avg_tag_count_score": avg_tag_count_score,
#             "accuracy": accuracy,
#             "valid_count": valid_count,
#             "processed_count": processed_count,
#             "skipped_count": skipped_count,
#             "content_filter_count": content_filter_count
#         }
#     else:
#         print("没有发现有效记录")
#         return None

# if __name__ == "__main__":
#     # 修改为V3结果文件的路径
#     file_path = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/prediction_v3_results.jsonl" 
    
#     if not os.path.exists(file_path):
#         print(f"文件不存在: {file_path}")
#     else:
#         results = analyze_v3_predictions(file_path)
#         # 可以选择将结果保存到文件
#         # output_json_path = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/prediction_v3_scores.json"
#         # if results:
#         #     with open(output_json_path, 'w') as f:
#         #         json.dump(results, f, indent=2)
#         #     print(f"\n得分结果已保存到: {output_json_path}")
