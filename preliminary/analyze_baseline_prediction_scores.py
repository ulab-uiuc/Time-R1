import json
import re
import sys
import os # Added for directory creation
import numpy as np
import math as builtin_math
from tqdm import tqdm # Added for progress bar
from collections import defaultdict # Added for consistency

# 从time_prediction.py复制必要的函数 (或者确保可以正确导入)
# For this script, functions are copied directly.

def extract_answer_format(solution_str):
    """从解答文本中提取出 <answer>...</answer> 标签中间的内容"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if match:
        return match.group(1).strip()
    return None

def is_valid_date_format(date_str):
    """验证日期格式是否符合 YYYY-MM"""
    if not date_str: return False # Handle None or empty string
    pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
    return re.match(pattern, date_str) is not None

def date_prediction_reward(prediction, target, alpha=0.1):
    """根据预测日期与真实日期之间的月数差距计算奖励"""
    try:
        pred_year, pred_month = map(int, prediction.split("-"))
        target_year, target_month = map(int, target.split("-"))
    except Exception:
        return 0.0
    
    diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
    reward = builtin_math.exp(-alpha * diff_in_months)
    return reward

def format_reward_single(solution_str):
    """检查solution_str是否严格按照要求包含标签"""
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer><\|im_end\|>$" # Corrected pattern
    return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

def tag_count_reward_single(solution_str):
    """根据solution_str中标签的出现次数计算分值"""
    count = 0.0
    # Check for complete tags to avoid partial matches
    if "<think>" in solution_str and "</think>" in solution_str and solution_str.count("</think>") == 1 : # Ensure <think> is also present
        count += 0.33 # Assuming <think> is also desired
    if "<answer>" in solution_str and solution_str.count("<answer>") == 1:
        count += 0.33
    if "</answer>" in solution_str and solution_str.count("</answer>") == 1:
        count += 0.33
    return count

def compute_length_repetition_penalty(solution_str):
    """计算回答过长和内容重复的惩罚值"""
    length_penalty = 0.0
    repetition_penalty = 0.0
    tokens = solution_str.split()
    token_count = len(tokens)

    if token_count > 900:
        excess_ratio = min(1.0, (token_count - 900) / 124)
        length_penalty = excess_ratio * 0.3
    
    # For repetition, consider tokens up to a limit if very long
    eval_tokens = tokens[:min(token_count, 1000)] # Evaluate on first 1000 tokens for performance
    eval_token_count = len(eval_tokens)


    if eval_token_count > 50:
        max_repeat_count = 1
        current_word = None
        current_count = 0
        for word in eval_tokens:
            if word == current_word:
                current_count += 1
                max_repeat_count = max(max_repeat_count, current_count)
            else:
                current_word = word
                current_count = 1
        if max_repeat_count >= 5:
            repetition_penalty = 0.1 * min(5, max_repeat_count - 4)
    
    if eval_token_count > 100:
        for window_size in [3,5,7,9]:
            for i in range(eval_token_count - window_size * 3): # Check within eval_tokens
                phrase_tokens = eval_tokens[i:i+window_size]
                if not phrase_tokens: continue # Should not happen with loop condition
                phrase = ' '.join(phrase_tokens)
                # Check for repeats in the text immediately following the phrase
                next_text_tokens = eval_tokens[i+window_size : i+window_size*4]
                if not next_text_tokens: continue
                next_text = ' '.join(next_text_tokens)
                
                repeat_count = next_text.count(phrase)
                if repeat_count >= 2:
                    repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)
    
    if eval_token_count > 200:
        # Consider up to first 500 tokens for n-gram diversity for performance
        ngram_tokens = eval_tokens[:min(eval_token_count, 500)]
        chunks = [' '.join(ngram_tokens[i:i+5]) for i in range(0, len(ngram_tokens)-4)] # Ensure i+5 is valid
        if chunks:
            unique_chunks = set(chunks)
            unique_ratio = len(unique_chunks) / len(chunks)
            if unique_ratio < 0.5:
                repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)
    
    total_penalty = max(length_penalty, repetition_penalty)
    return total_penalty

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """计算时间预测任务的总评分"""
    answer = extract_answer_format(solution_str)

    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.2
    else:
        no_event_penalty = 0.3

    format_bonus, pred_reward = 0.0, 0.0
    if answer and is_valid_date_format(answer):
        format_bonus = bonus
        true_pub_date = ground_truth.get("event_pub_date")
        if true_pub_date and is_valid_date_format(true_pub_date):
            pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
    
    accuracy_score = pred_reward

    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each # tag_bonus_each is 0.025
    
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            length_repetition_penalty, "time_prediction") # Return penalty instead of 1.0

def analyze_model_predictions(results_file, model_name="Baseline Model"):
    """分析模型在时间预测任务上的表现，并按月份统计"""
    total_scores_all = []
    accuracy_scores_all = []
    format_bonuses_all = []
    tag_format_scores_all = []
    tag_count_scores_all = []
    penalties_all = [] # To store length_repetition_penalty
    
    processed_count_total = 0
    valid_count_total = 0
    correct_predictions_total = 0 # For overall accuracy calculation
    skipped_due_no_gt_total = 0
    skipped_due_json_error_total = 0
    skipped_due_other_error_total = 0

    target_months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
    # target_months = ["2025-03", "2025-04"]
    monthly_scores = {month: {"total": [], "accuracy": [], "count": 0, "correct_preds": 0} for month in target_months}
    
    print(f"正在分析文件: {results_file} ({model_name})")
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文件未找到 {results_file}")
        return None

    print(f"共读取 {len(lines)} 行数据，开始处理...")

    for line in tqdm(lines):
        processed_count_total +=1
        if not line.strip():
            continue
        try:
            result = json.loads(line)
            response_str = result.get('response', '')
            ground_truth_dict = result.get('ground_truth', {})
            
            if not ground_truth_dict or 'event_pub_date' not in ground_truth_dict:
                skipped_due_no_gt_total += 1
                continue
            
            true_date = ground_truth_dict.get('event_pub_date')
            if not is_valid_date_format(true_date): # Ensure true_date is valid before proceeding
                skipped_due_no_gt_total +=1
                continue

            # Modify response_str for better tag_format_score
            # Qwen3B output usually is "<think>...</think>\n<answer>...</answer>"
            # We need to add <|im_end|>
            solution_str_for_score = response_str.strip()
            if solution_str_for_score.endswith("</answer>"):
                 # Check if <think> is present and correctly formatted for format_reward_single
                if solution_str_for_score.startswith("<think>") and "</think>\n<answer>" in solution_str_for_score:
                    solution_str_for_score += "<|im_end|>"
                # else, it might not match format_reward_single anyway, but compute_score will run

            scores_tuple = compute_score(solution_str_for_score, ground_truth_dict)
            total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, penalty, _ = scores_tuple
            
            total_scores_all.append(total_score)
            accuracy_scores_all.append(accuracy_score)
            format_bonuses_all.append(format_bonus)
            tag_format_scores_all.append(tag_format_score)
            tag_count_scores_all.append(tag_count_score)
            penalties_all.append(penalty)
            
            predicted_answer_date = extract_answer_format(response_str)
            if predicted_answer_date and is_valid_date_format(predicted_answer_date) and predicted_answer_date == true_date:
                correct_predictions_total += 1

            if true_date in monthly_scores:
                monthly_scores[true_date]["total"].append(total_score)
                monthly_scores[true_date]["accuracy"].append(accuracy_score)
                monthly_scores[true_date]["count"] += 1
                if predicted_answer_date and is_valid_date_format(predicted_answer_date) and predicted_answer_date == true_date:
                     monthly_scores[true_date]["correct_preds"] += 1
            
            valid_count_total += 1
            
        except json.JSONDecodeError:
            # print(f"警告: 无法解析JSON行: {line[:100]}...")
            skipped_due_json_error_total += 1
            continue
        except Exception as e:
            # print(f"处理样本时出错: {e} - Line: {line[:100]}...")
            skipped_due_other_error_total += 1
            continue

    results_summary = {
        "model_name": model_name,
        "total_records_in_file": len(lines),
        "records_processed_loop": processed_count_total,
        "valid_records_scored": valid_count_total,
        "skipped_due_no_gt": skipped_due_no_gt_total,
        "skipped_due_json_error": skipped_due_json_error_total,
        "skipped_due_other_error": skipped_due_other_error_total,
        "overall_accuracy_manual": 0,
        "avg_total_score_all": 0,
        "avg_accuracy_score_all": 0,
        "avg_format_bonus_all": 0,
        "avg_tag_format_score_all": 0,
        "avg_tag_count_score_all": 0,
        "avg_penalty_all": 0,
        "monthly_averages": {}
    }

    if valid_count_total > 0:
        results_summary.update({
            "overall_accuracy_manual": correct_predictions_total / valid_count_total,
            "avg_total_score_all": sum(total_scores_all) / valid_count_total,
            "avg_accuracy_score_all": sum(accuracy_scores_all) / valid_count_total,
            "avg_format_bonus_all": sum(format_bonuses_all) / valid_count_total,
            "avg_tag_format_score_all": sum(tag_format_scores_all) / valid_count_total,
            "avg_tag_count_score_all": sum(tag_count_scores_all) / valid_count_total,
            "avg_penalty_all": sum(penalties_all) / valid_count_total,
        })

        print(f"\n===== {model_name} 总体评分结果 =====")
        print(f"文件总行数: {len(lines)}, 循环处理行数: {processed_count_total}")
        print(f"有效评分样本数: {valid_count_total}")
        print(f"跳过 (无GT/无效GT): {skipped_due_no_gt_total}, (JSON错误): {skipped_due_json_error_total}, (其他错误): {skipped_due_other_error_total}")
        print(f"总体准确率 (预测日期==真实日期): {results_summary['overall_accuracy_manual']:.4f} ({correct_predictions_total}/{valid_count_total})")
        print(f"平均 total_score: {results_summary['avg_total_score_all']:.4f}")
        print(f"平均 accuracy_score (预测奖励): {results_summary['avg_accuracy_score_all']:.4f}")
        print(f"平均 format_bonus: {results_summary['avg_format_bonus_all']:.4f}")
        print(f"平均 tag_format_score: {results_summary['avg_tag_format_score_all']:.4f}")
        print(f"平均 tag_count_score: {results_summary['avg_tag_count_score_all']:.4f}")
        print(f"平均 penalty: {results_summary['avg_penalty_all']:.4f}")

        print(f"\n===== {model_name} 每月得分统计 (2024-07 至 2025-02) =====")
        for month in target_months:
            month_data = monthly_scores[month]
            month_avg_total = 0
            month_avg_accuracy_score = 0
            month_accuracy_manual = 0
            if month_data["count"] > 0:
                month_avg_total = sum(month_data["total"]) / month_data["count"]
                month_avg_accuracy_score = sum(month_data["accuracy"]) / month_data["count"]
                month_accuracy_manual = month_data["correct_preds"] / month_data["count"]
                print(f"月份: {month}")
                print(f"  有效记录数: {month_data['count']}")
                print(f"  准确率 (预测日期==真实日期): {month_accuracy_manual:.4f} ({month_data['correct_preds']}/{month_data['count']})")
                print(f"  平均总分: {month_avg_total:.4f}")
                print(f"  平均准确度分数 (预测奖励): {month_avg_accuracy_score:.4f}")
            else:
                print(f"月份: {month} - 无有效记录")
            results_summary["monthly_averages"][month] = {
                "count": month_data["count"],
                "correct_preds": month_data["correct_preds"],
                "accuracy_manual": month_accuracy_manual,
                "avg_total_score": month_avg_total,
                "avg_accuracy_score": month_avg_accuracy_score
            }
    else:
        print("没有发现有效记录或所有记录处理失败")
    
    return results_summary

def main():
    # 分析你的Time-R1模型在3月和4月数据上的结果
    # time_r1_results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/time_r1_results/time_prediction_march_april_results.jsonl"
    time_r1_results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/time_r1_results/time_prediction_results.jsonl"
    
    if not os.path.exists(time_r1_results_file):
        print(f"错误: Time-R1模型结果文件未找到 {time_r1_results_file}")
        return

    time_r1_scores = analyze_model_predictions(time_r1_results_file, model_name="Time-R1 (Ours, 3B)")
    
    if time_r1_scores:
        # 将分析结果保存到 time_r1_results 目录中
        output_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/time_r1_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")
        
        # output_json_path = os.path.join(output_dir, "time_r1_prediction_scores_march_april.json")
        output_json_path = os.path.join(output_dir, "time_r1_prediction_scores.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(time_r1_scores, f, indent=2, ensure_ascii=False)
        print(f"\nTime-R1模型得分结果已保存到: {output_json_path}")



    # # 确保这里的路径是正确的 llama3.1-8b 结果文件
    # llama31_results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/time_prediction_llama31_results.jsonl"
    
    # if not os.path.exists(llama31_results_file):
    #     print(f"错误: Llama3.1-8B 结果文件未找到 {llama31_results_file}")
    #     return

    # llama31_scores = analyze_model_predictions(llama31_results_file, model_name="Llama3.1-8B-Instruct")
    
    # if llama31_scores:
    #     # 将分析结果保存到 analysis_results 目录中
    #     output_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #         print(f"已创建目录: {output_dir}")
        
    #     output_json_path = os.path.join(output_dir, "llama31_8b_prediction_scores_with_monthly.json")
    #     with open(output_json_path, 'w', encoding='utf-8') as f:
    #         json.dump(llama31_scores, f, indent=2, ensure_ascii=False)
    #     print(f"\nLlama3.1-8B 得分结果已保存到: {output_json_path}")



    # # 确保这里的路径是正确的 qwen3b 结果文件
    # qwen3b_results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/time_prediction_march_april_results.jsonl"
    
    # if not os.path.exists(qwen3b_results_file):
    #     print(f"错误: Qwen3B 结果文件未找到 {qwen3b_results_file}")
    #     return

    # qwen3b_scores = analyze_model_predictions(qwen3b_results_file, model_name="Qwen2.5-3B-Chat")
    
    # if qwen3b_scores:
    #     output_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #         print(f"已创建目录: {output_dir}")
        
    #     output_json_path = os.path.join(output_dir, "qwen3b_prediction_scores_march_april.json")
    #     with open(output_json_path, 'w', encoding='utf-8') as f:
    #         json.dump(qwen3b_scores, f, indent=2, ensure_ascii=False)
    #     print(f"\nQwen3B 得分结果已保存到: {output_json_path}")

if __name__ == "__main__":
    main()



# import json
# import re
# import sys
# import numpy as np
# import math as builtin_math

# # 从time_prediction.py复制必要的函数
# def extract_answer_format(solution_str):
#     """从解答文本中提取出 <answer>...</answer> 标签中间的内容"""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if match:
#         return match.group(1).strip()
#     return None

# def is_valid_date_format(date_str):
#     """验证日期格式是否符合 YYYY-MM"""
#     pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
#     return re.match(pattern, date_str) is not None

# def date_prediction_reward(prediction, target, alpha=0.1):
#     """根据预测日期与真实日期之间的月数差距计算奖励"""
#     try:
#         pred_year, pred_month = map(int, prediction.split("-"))
#         target_year, target_month = map(int, target.split("-"))
#     except Exception:
#         return 0.0
    
#     diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
#     reward = builtin_math.exp(-alpha * diff_in_months)
#     return reward

# def format_reward_single(solution_str):
#     """检查solution_str是否严格按照要求包含标签"""
#     pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
#     return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

# def tag_count_reward_single(solution_str):
#     """根据solution_str中标签的出现次数计算分值"""
#     count = 0.0
#     if solution_str.count("</think>") == 1:
#         count += 0.33
#     if solution_str.count("<answer>") == 1:
#         count += 0.33
#     if solution_str.count("</answer>") == 1:
#         count += 0.33
#     return count

# def compute_length_repetition_penalty(solution_str):
#     """计算回答过长和内容重复的惩罚值"""
#     # 初始无惩罚
#     length_penalty = 0.0
#     repetition_penalty = 0.0
    
#     # 分词处理
#     tokens = solution_str.split()
#     token_count = len(tokens)

#     # 长度惩罚
#     if token_count > 900:
#         excess_ratio = min(1.0, (token_count - 900) / 124)
#         length_penalty = excess_ratio * 0.3

#     if token_count > 400:
#         tokens = tokens[:400]  # 截断到400个词
    
#     # 单词级别的连续重复检测
#     if token_count > 50:
#         max_repeat_count = 1
#         current_word = None
#         current_count = 0
        
#         for word in tokens:
#             if word == current_word:
#                 current_count += 1
#                 max_repeat_count = max(max_repeat_count, current_count)
#             else:
#                 current_word = word
#                 current_count = 1
        
#         if max_repeat_count >= 5:
#             repetition_penalty = 0.1 * min(5, max_repeat_count - 4)
    
#     # 短语级别的连续重复检测
#     if token_count > 100:
#         for window_size in [3,5,7,9]:
#             for i in range(len(tokens) - window_size * 3):
#                 phrase = ' '.join(tokens[i:i+window_size])
#                 next_text = ' '.join(tokens[i+window_size:i+window_size*4])
#                 repeat_count = next_text.count(phrase)
                
#                 if repeat_count >= 2:
#                     repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)
    
#     # 全局n-gram多样性检测
#     if token_count > 200:
#         chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]
#         if chunks:
#             unique_chunks = set(chunks)
#             unique_ratio = len(unique_chunks) / len(chunks)
            
#             if unique_ratio < 0.5:
#                 repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)
    
#     # 结合长度惩罚和重复惩罚
#     total_penalty = max(length_penalty, repetition_penalty)
    
#     return total_penalty

# def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
#     """计算时间预测任务的总评分"""
#     answer = extract_answer_format(solution_str)

#     # "No event"惩罚
#     no_event_penalty = 0
#     if answer:
#         if "no event" in answer.lower() or "none" in answer.lower():
#             no_event_penalty = 0.2
#     else:
#         no_event_penalty = 0.3

#     # 如果提取到了答案且符合 "YYYY-MM" 格式，则先获得格式奖励
#     format_bonus, pred_reward = 0.0, 0.0
#     if answer and is_valid_date_format(answer):
#         format_bonus = bonus
#         true_pub_date = ground_truth.get("event_pub_date")
#         if true_pub_date and is_valid_date_format(true_pub_date):
#             pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
    
#     accuracy_score = pred_reward

#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 应用长度和重复惩罚
#     length_repetition_penalty = compute_length_repetition_penalty(solution_str)

#     # 总分计算
#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             1.0, "time_prediction")

# def calculate_average_scores(results_file):
#     """计算结果文件中所有预测的平均分数"""
#     total_scores = []
#     accuracy_scores = []
#     format_bonuses = []
#     tag_format_scores = []
#     tag_count_scores = []
#     penalties = []
    
#     # 读取结果文件
#     with open(results_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 result = json.loads(line)
#                 response = result.get('response', '')
#                 ground_truth = result.get('ground_truth', {})
                
#                 if not ground_truth or 'event_pub_date' not in ground_truth:
#                     continue
                
#                 # 计算分数
#                 total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, _, _ = compute_score(
#                     response, ground_truth
#                 )
                
#                 total_scores.append(total_score)
#                 accuracy_scores.append(accuracy_score)
#                 format_bonuses.append(format_bonus)
#                 tag_format_scores.append(tag_format_score)
#                 tag_count_scores.append(tag_count_score)
                
#                 # 打印示例评分（前10个）
#                 if len(total_scores) <= 10:
#                     true_date = ground_truth.get('event_pub_date', 'unknown')
#                     answer = extract_answer_format(response) or "未找到答案"
#                     print(f"样本 #{len(total_scores)-1}: 预测={answer}, 真实={true_date}, "
#                           f"total_score={total_score:.4f}, accuracy_score={accuracy_score:.4f}")
                
#             except json.JSONDecodeError:
#                 print(f"警告: 无法解析JSON行: {line[:50]}...")
#                 continue
#             except Exception as e:
#                 print(f"处理样本时出错: {e}")
#                 continue
    
#     # 计算平均分数
#     avg_total_score = sum(total_scores) / len(total_scores) if total_scores else 0
#     avg_accuracy_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
#     avg_format_bonus = sum(format_bonuses) / len(format_bonuses) if format_bonuses else 0
#     avg_tag_format_score = sum(tag_format_scores) / len(tag_format_scores) if tag_format_scores else 0
#     avg_tag_count_score = sum(tag_count_scores) / len(tag_count_scores) if tag_count_scores else 0
    
#     return {
#         'avg_total_score': avg_total_score,
#         'avg_accuracy_score': avg_accuracy_score,
#         'avg_format_bonus': avg_format_bonus,
#         'avg_tag_format_score': avg_tag_format_score,
#         'avg_tag_count_score': avg_tag_count_score,
#         'sample_count': len(total_scores)
#     }

# def main():
#     results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen7b_results/time_prediction_qwen7b_results.jsonl"
#     # results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/time_prediction_qwen3b_results.jsonl"
#     # results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/time_prediction_llama31_results.jsonl"
#     results = calculate_average_scores(results_file)
    
#     print("\n===== 总体评分结果 =====")
#     print(f"总样本数: {results['sample_count']}")
#     print(f"平均 total_score: {results['avg_total_score']:.4f}")
#     print(f"平均 accuracy_score: {results['avg_accuracy_score']:.4f}")
#     print(f"平均 format_bonus: {results['avg_format_bonus']:.4f}")
#     print(f"平均 tag_format_score: {results['avg_tag_format_score']:.4f}")
#     print(f"平均 tag_count_score: {results['avg_tag_count_score']:.4f}")

# if __name__ == "__main__":
#     main()