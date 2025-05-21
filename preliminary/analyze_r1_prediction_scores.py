import json
import re
import os
from verl.utils.reward_score.time_prediction import compute_score, extract_answer_format, is_valid_date_format
from collections import defaultdict

def extract_true_date_from_custom_id(custom_id):
    """从custom_id中提取日期部分 (YYYY-MM)"""
    match = re.match(r"(\d{4}-\d{2})_\d+", custom_id)
    if match:
        return match.group(1)
    return None

def analyze_r1_predictions(jsonl_file_path):
    """分析r1模型在时间预测任务上的表现"""
    # 初始化分数累计器
    total_scores_all = []
    accuracy_scores_all = []
    format_bonuses_all = []
    tag_format_scores_all = []
    tag_count_scores_all = []
    
    # 初始化按月分数累计器
    # 使用 defaultdict 来方便地添加新月份的数据
    monthly_scores = defaultdict(lambda: {"total_scores": [], "accuracy_scores": [], "correct_predictions": 0, "valid_count": 0})
    
    # 初始化计数器
    processed_count = 0
    valid_count_all = 0
    correct_predictions_all = 0
    
    print(f"正在分析文件: {jsonl_file_path}")
    
    # 读取并处理每条记录
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 提取真实答案
                custom_id = data.get('custom_id')
                true_date = extract_true_date_from_custom_id(custom_id)
                
                # 如果无法提取真实日期，跳过该记录
                if not true_date or not is_valid_date_format(true_date):
                    print(f"无法从custom_id '{custom_id}'提取有效的日期格式")
                    continue
                
                # 提取模型回答
                response = data.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if not choices:
                    print(f"记录中没有模型回答: {custom_id}")
                    continue
                
                content = choices[0].get('message', {}).get('content', '')
                reasoning = choices[0].get('message', {}).get('reasoning_content', '')
                
                # 提取预测日期
                predicted_date = extract_answer_format(content)
                
                # 检查预测是否正确 (总体)
                if predicted_date == true_date:
                    correct_predictions_all += 1
                
                # 构建compute_score函数需要的格式
                # solution_str = f"<think>{reasoning}</think>\n{content}"
                solution_str = f"<think>{reasoning}</think>\n{content.strip()}<|im_end|>"
                ground_truth = {"event_pub_date": true_date}
                
                # 计算得分
                scores_tuple = compute_score(solution_str, ground_truth)
                total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, _, _ = scores_tuple
                
                # 累加总分数
                total_scores_all.append(total_score)
                accuracy_scores_all.append(accuracy_score)
                format_bonuses_all.append(format_bonus)
                tag_format_scores_all.append(tag_format_score)
                tag_count_scores_all.append(tag_count_score)
                
                processed_count += 1
                valid_count_all += 1

                # 按月累加分数
                month_key = true_date[:7] # YYYY-MM
                monthly_scores[month_key]["total_scores"].append(total_score)
                monthly_scores[month_key]["accuracy_scores"].append(accuracy_score)
                monthly_scores[month_key]["valid_count"] += 1
                if predicted_date == true_date:
                    monthly_scores[month_key]["correct_predictions"] += 1
                
            except Exception as e:
                print(f"处理记录时出错: {str(e)}")
    
    # 计算总体平均分数
    results_summary = {}
    if valid_count_all > 0:
        avg_total_score_all = sum(total_scores_all) / valid_count_all
        avg_accuracy_score_all = sum(accuracy_scores_all) / valid_count_all
        avg_format_bonus_all = sum(format_bonuses_all) / valid_count_all
        avg_tag_format_score_all = sum(tag_format_scores_all) / valid_count_all
        avg_tag_count_score_all = sum(tag_count_scores_all) / valid_count_all
        accuracy_all = correct_predictions_all / valid_count_all
        
        print("\n===== R1模型时间预测任务总体得分统计 =====")
        print(f"总共处理记录: {processed_count} 条, 有效记录: {valid_count_all} 条")
        print(f"总体准确率（完全匹配）: {accuracy_all:.4f} ({correct_predictions_all}/{valid_count_all})")
        print(f"总体平均总分 (avg_total_score): {avg_total_score_all:.4f}")
        print(f"总体平均准确度分数 (avg_accuracy_score): {avg_accuracy_score_all:.4f}")
        print(f"总体平均格式奖励 (avg_format_bonus): {avg_format_bonus_all:.4f}")
        print(f"总体平均标签格式分数 (avg_tag_format_score): {avg_tag_format_score_all:.4f}")
        print(f"总体平均标签计数分数 (avg_tag_count_score): {avg_tag_count_score_all:.4f}")
        
        results_summary["overall"] = {
            "avg_total_score": avg_total_score_all,
            "avg_accuracy_score": avg_accuracy_score_all,
            "avg_format_bonus": avg_format_bonus_all,
            "avg_tag_format_score": avg_tag_format_score_all,
            "avg_tag_count_score": avg_tag_count_score_all,
            "accuracy": accuracy_all,
            "valid_count": valid_count_all
        }
    else:
        print("没有发现有效记录")
        return None

    # 计算并打印按月平均分数
    print("\n===== R1模型时间预测任务按月得分统计 =====")
    # 定义期望的月份顺序
    expected_months = [f"{year}-{month:02d}" for year in [2024, 2025] for month in range(1, 13)]
    target_months = [m for m in expected_months if "2024-07" <= m <= "2025-02"]
    
    results_summary["monthly"] = {}

    for month_key in sorted(target_months): # 保证按时间顺序输出
        month_data = monthly_scores.get(month_key) # 使用 .get 以防某个月份无数据
        if month_data and month_data["valid_count"] > 0:
            avg_total_score_month = sum(month_data["total_scores"]) / month_data["valid_count"]
            avg_accuracy_score_month = sum(month_data["accuracy_scores"]) / month_data["valid_count"]
            accuracy_month = month_data["correct_predictions"] / month_data["valid_count"]
            
            print(f"\n月份: {month_key}")
            print(f"  有效记录: {month_data['valid_count']} 条")
            print(f"  准确率（完全匹配）: {accuracy_month:.4f} ({month_data['correct_predictions']}/{month_data['valid_count']})")
            print(f"  平均总分: {avg_total_score_month:.4f}")
            print(f"  平均准确度分数: {avg_accuracy_score_month:.4f}")
            results_summary["monthly"][month_key] = {
                "avg_total_score": avg_total_score_month,
                "avg_accuracy_score": avg_accuracy_score_month,
                "accuracy": accuracy_month,
                "valid_count": month_data['valid_count']
            }
        elif month_key in target_months: # 如果是目标月份但无数据，也打印出来
            print(f"\n月份: {month_key}")
            print(f"  有效记录: 0 条")
            results_summary["monthly"][month_key] = {
                "avg_total_score": 0,
                "avg_accuracy_score": 0,
                "accuracy": 0,
                "valid_count": 0
            }
            
    return results_summary

if __name__ == "__main__":
    file_path = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/prediction_r1_results.jsonl"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
    else:
        results = analyze_r1_predictions(file_path)
        
    output_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/r1/"
    os.makedirs(output_dir, exist_ok=True)

    if results:
        with open(output_dir+"r1_prediction_scores_summary.json", "w") as outfile:
            json.dump(results, outfile, indent=4)
        print("\n详细结果已保存到 r1_prediction_scores_summary.json")
