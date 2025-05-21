import re
import sys

# 临时移除当前工作目录（通常是 sys.path[0]）
orig_path = sys.path.pop(0)
import math as builtin_math  # 这里加载的就是内置的 math 模块
sys.path.insert(0, orig_path)  # 恢复原来的 sys.path


def extract_answer_format(solution_str):
    """
    从解答文本中提取出 <answer>...</answer> 标签中间的内容。
    如果找不到，则返回 None。
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if match:
        return match.group(1).strip()
    return None

def is_valid_date_format(date_str):
    """
    验证 date_str 是否符合 "YYYY-MM" 的格式，其中月份必须为 01 至 12。
    """
    pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
    return re.match(pattern, date_str) is not None

def date_prediction_reward(prediction, target, alpha=0.05):
    """
    根据预测日期与真实日期之间的月数差距计算奖励。
    
    参数:
        prediction (str): 预测的日期，格式为 "YYYY-MM"
        target (str): 真实日期，格式为 "YYYY-MM"
        alpha (float): 衰减速率，默认 0.05
        
    返回:
        float: 奖励值，当两者完全一致时为 1，随着月数差距增加奖励指数衰减。
    """
    try:
        pred_year, pred_month = map(int, prediction.split("-"))
        target_year, target_month = map(int, target.split("-"))
    except Exception:
        return 0.0
    
    diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
    reward = builtin_math.exp(-alpha * diff_in_months)
    return reward

def format_reward_single(solution_str):
    """
    检查 solution_str 是否严格按照要求包含：
    <think> 与 </think> 以及 <answer> 与 </answer> 标签。
    如果完全匹配，则返回 1.0，否则返回 0.0。
    """
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
    return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

def tag_count_reward_single(solution_str):
    """
    根据 solution_str 中标签的出现次数计算分值：
      - 如果 "<think>\n" 出现一次，则奖励 0.25 分；
      - 如果 "\n</think>\n" 出现一次，则奖励 0.25 分；
      - 如果 "\n<answer>\n" 出现一次，则奖励 0.25 分；
      - 如果 "\n</answer>" 出现一次，则奖励 0.25 分；
    返回总分（理想状态为 1.0）。
    """
    count = 0.0
    if solution_str.count("<think>") == 1:
        count += 0.25
    if solution_str.count("</think>") == 1:
        count += 0.25
    if solution_str.count("<answer>") == 1:
        count += 0.25
    if solution_str.count("</answer>") == 1:
        count += 0.25
    return count

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
    """
    计算总奖励：如果解答中提取出的答案严格符合 "YYYY-MM" 格式，则额外加上格式奖励 bonus，
    同时基于预测日期与真实日期（ground_truth['true_pub_date']）之间的月差计算预测准确率奖励。
    如果答案不存在或格式不正确，则预测奖励部分为 0（总奖励直接返回 0）。
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期的字典，格式为 {"true_pub_date": "YYYY-MM"}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于 date_prediction_reward 的衰减速率，默认为 0.05
        
    返回:
        float: 总奖励分数
    """
    answer = extract_answer_format(solution_str)
    # 如果提取到了答案且符合 "YYYY-MM" 格式，则先获得格式奖励
    if answer and is_valid_date_format(answer):
        format_bonus = bonus
        true_pub_date = ground_truth.get("true_pub_date")
        # 确保 ground_truth 中的真实日期也符合 "YYYY-MM" 格式，否则不计算预测奖励
        if true_pub_date and is_valid_date_format(true_pub_date):
            pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
        else:
            pred_reward = 0.0
        format_pred_reward = format_bonus + pred_reward
    else:
        format_pred_reward = 0.0
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    total_score = format_pred_reward + tag_format_score + tag_count_score
    # print(format_pred_reward, tag_format_score, tag_count_score)
    return total_score

# # 示例测试
# if __name__ == "__main__":
#     # 假设预测准确率部分计算结果依赖于真实发布日期
#     ground_truth = {"true_pub_date": "2025-03"}
    
#     # 示例 1：答案存在且格式正确，预测接近目标
#     solution1 = "Some text... <answer>2025-04</answer>"
#     # 示例 2：答案存在但格式错误
#     solution2 = "Some text... <answer>2025/03</answer>"
#     # 示例 3：答案标签不存在
#     solution3 = "Some text without answer tag."
    
#     print(builtin_math.exp(-0.05 * 1))
#     print(builtin_math.exp(-0.05 * 3))
#     print(builtin_math.exp(-0.05 * 6))
#     print(builtin_math.exp(-0.05 * 12))
#     print("Solution1 reward:", compute_total_reward(solution1, ground_truth))
#     print("Solution2 reward:", compute_total_reward(solution2, ground_truth))
#     print("Solution3 reward:", compute_total_reward(solution3, ground_truth))


# if __name__ == "__main__":
#     # 示例1：预测正确、格式正确，且标签齐全
#     solution1 = "<think>Some reasoning...</think>\n<answer>2025-03</answer>"
#     # 示例2：预测错误（日期不同），但格式正确，标签齐全
#     solution2 = "<think>Some reasoning...</think>\n<answer>2024-12</answer>"
#     # 示例3：答案格式错误
#     solution3 = "<think>Some reasoning...</think>\n<answer>2025/03</answer>"
#     # 示例4：标签缺失（但答案可能存在且格式正确）
#     solution4 = "Some text without proper tags <answer>2025-03</answer>"
    
#     ground_truth = {"true_pub_date": "2025-03"}
    
#     for i, sol in enumerate([solution1, solution2, solution3, solution4], 1):
#         score = compute_total_reward(sol, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025)
#         print(f"Solution{i} total score: {score}")