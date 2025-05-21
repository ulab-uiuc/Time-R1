import re
import sys
import numpy as np

# 临时移除当前工作目录（通常是 sys.path[0]）
orig_path = sys.path.pop(0)
import math as builtin_math  # 这里加载的就是内置的 math 模块
sys.path.insert(0, orig_path)  # 恢复原来的 sys.path

#--------------- 通用工具函数 ---------------#

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

def date_prediction_reward(prediction, target, alpha=0.1):
    """
    根据预测日期与真实日期之间的月数差距计算奖励。
    
    参数:
        prediction (str): 预测的日期，格式为 "YYYY-MM"
        target (str): 真实日期，格式为 "YYYY-MM"
        alpha (float): 衰减速率，默认 0.1，比时间推断任务更严格
        
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
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
    return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

def tag_count_reward_single(solution_str):
    """
    根据 solution_str 中标签的出现次数计算分值
    """
    count = 0.0
    if solution_str.count("</think>") == 1:
        count += 0.33
    if solution_str.count("<answer>") == 1:
        count += 0.33
    if solution_str.count("</answer>") == 1:
        count += 0.33
    return count

def compute_length_repetition_penalty(solution_str):
    """
    计算回答过长和内容重复的惩罚值
    返回一个惩罚数值，0表示没有惩罚，值越大表示惩罚越严重
    """
    # 初始无惩罚
    length_penalty = 0.0
    repetition_penalty = 0.0
    
    # 分词处理
    tokens = solution_str.split()
    token_count = len(tokens)

    # 长度惩罚
    if token_count > 900:
        excess_ratio = min(1.0, (token_count - 900) / 124)
        length_penalty = excess_ratio * 0.3

    # if token_count > 400:
    #     tokens = tokens[:400]  
    
    # 1. 检测单词级别的连续重复
    if token_count > 50:
        # 连续相同单词的最大次数
        max_repeat_count = 1
        current_word = None
        current_count = 0
        
        for word in tokens:
            if word == current_word:
                current_count += 1
                max_repeat_count = max(max_repeat_count, current_count)
            else:
                current_word = word
                current_count = 1
        
        # 如果有连续5次以上相同的单词，应用惩罚
        if max_repeat_count >= 5:
            repetition_penalty = 0.1 * min(5, max_repeat_count - 4)  # 最高0.5分惩罚
    
    # 2. 检测短语级别的连续重复
    if token_count > 100:
        # 窗口大小范围内的增量式重复检测
        for window_size in [3,5,7,9]:  # 检测3-9个词的短语重复
            for i in range(len(tokens) - window_size * 3):
                phrase = ' '.join(tokens[i:i+window_size])
                
                # 在接下来的文本中检查是否有连续重复
                next_text = ' '.join(tokens[i+window_size:i+window_size*4])
                repeat_count = next_text.count(phrase)
                
                if repeat_count >= 2:  # 同一短语在其后立即出现2次以上
                    repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)  # 最高0.45惩罚
    
    # 3. 全局n-gram多样性检测
    if token_count > 200:
        chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]  # 滑动窗口
        if chunks:
            unique_chunks = set(chunks)
            unique_ratio = len(unique_chunks) / len(chunks)
            
            # 当重复内容超过50%时开始惩罚
            if unique_ratio < 0.5:
                repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)  # 提高惩罚力度
    
    # 结合长度惩罚和重复惩罚
    total_penalty = max(length_penalty, repetition_penalty)  # 取较大的惩罚
    
    return total_penalty

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    计算时间预测任务的总评分
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期的字典，格式为 {"event_pub_date": "YYYY-MM"}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于 date_prediction_reward 的衰减速率，默认为 0.1 (比推断任务更严格)
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        tuple: (总分, 准确性分数, 格式奖励, 标签格式分数, 标签计数分数, 一致性惩罚, 任务类型)
    """
    answer = extract_answer_format(solution_str)

    # "No event"惩罚
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.2  # 预测任务中更严格地惩罚"No event"回答
    else:
        no_event_penalty = 0.3

    # 如果提取到了答案且符合 "YYYY-MM" 格式，则先获得格式奖励
    format_bonus, pred_reward = 0.0, 0.0
    if answer and is_valid_date_format(answer):
        format_bonus = bonus
        true_pub_date = ground_truth.get("event_pub_date")
        # 确保 ground_truth 中的真实日期也符合 "YYYY-MM" 格式，否则不计算预测奖励
        if true_pub_date and is_valid_date_format(true_pub_date):
            pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
    
    accuracy_score = pred_reward

    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_prediction")

# 测试用例
if __name__ == "__main__":
    # 时间预测测试
    solution = "<think>根据文章描述的技术发展进程和相关行业动态，我预计该事件将在2025年初发生。</think>\n<answer>2025-03</answer><|im_end|>"
    
    ground_truth = {
        "event_pub_date": "2025-03",
    }
    
    print("时间预测分数:", compute_score(solution, ground_truth))