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

def month_diff_reward(pred_diff, true_diff, alpha=0.06):
    """计算月份差值预测的奖励，基于差值的指数衰减"""
    diff = abs(pred_diff - true_diff)
    reward = builtin_math.exp(-alpha * diff)
    return reward

def compute_inconsistency_penalty0(date1, date2, claimed_diff):
    """计算声称的月份差与实际月份差之间的不一致性惩罚"""
    try:
        year1, month1 = map(int, date1.split("-"))
        year2, month2 = map(int, date2.split("-"))
        
        actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
        error = abs(actual_diff - claimed_diff)
        
        # 惩罚系数随误差增加而加重
        if error == 0:
            return 1.0  # 没有误差，不惩罚
        elif error <= 2:
            return 0.7  # 小误差，轻微惩罚
        elif error <= 5:
            return 0.4  # 中等误差，中等惩罚
        else:
            return 0.2  # 大误差，严重惩罚
    except Exception:
        return 0.1  # 出现异常，严重惩罚
    
def compute_inconsistency_penalty(date1, date2, claimed_diff):
    """计算声称的月份差与实际月份差之间的不一致性惩罚，使用指数衰减函数"""
    try:
        year1, month1 = map(int, date1.split("-"))
        year2, month2 = map(int, date2.split("-"))
        
        actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
        error = abs(actual_diff - claimed_diff)
        
        # 基础一致性评估参数
        base_alpha = 0.1  # 基础衰减率
        
        # 对大月份差值应用更宽松的惩罚系数
        if claimed_diff >= 25:
            # 月份差越大，惩罚系数越小
            scaling_factor = min(1.0, 25.0 / claimed_diff)  # 随着月份差增大而减小
            alpha = base_alpha * scaling_factor
        else:
            alpha = base_alpha
            
        # 使用指数函数计算惩罚值(1.0是最好的，不惩罚)
        consistency_score = builtin_math.exp(-alpha * error)
        
        return consistency_score
    except Exception:
        return 0.1  # 出现异常，严重惩罚

def is_valid_order_format(order_str):
    """验证order_str是否为以'-'分隔的包含1,2,3三个不重复数字的字符串"""
    if not re.match(r'\d-\d-\d', order_str):
        return False
    
    # 提取数字并验证是否只包含1,2,3且不重复
    digits = [int(d) for d in order_str.split('-')]
    return sorted(digits) == [1, 2, 3]

def compute_order_accuracy(predicted_order, true_order):
    """
    计算顺序预测的准确性，对比三组两两事件之间的顺序关系
    一组正确得0.333分，两组正确得0.666分，三组全部正确得0.999分
    """
    if predicted_order == true_order:
        return 0.999  # 全部正确，得满分（稍微减少一点以避免浮点数精度问题）
    
    pred_parts = predicted_order.split('-')
    true_parts = true_order.split('-')
    
    # 检查每一对事件之间的相对顺序是否正确
    pairs_correct = 0
    
    # 检查事件1和事件2的相对顺序
    pred_1_before_2 = pred_parts.index('1') < pred_parts.index('2')
    true_1_before_2 = true_parts.index('1') < true_parts.index('2')
    if pred_1_before_2 == true_1_before_2:
        pairs_correct += 1
    
    # 检查事件1和事件3的相对顺序
    pred_1_before_3 = pred_parts.index('1') < pred_parts.index('3')
    true_1_before_3 = true_parts.index('1') < true_parts.index('3')
    if pred_1_before_3 == true_1_before_3:
        pairs_correct += 1
    
    # 检查事件2和事件3的相对顺序
    pred_2_before_3 = pred_parts.index('2') < pred_parts.index('3')
    true_2_before_3 = true_parts.index('2') < true_parts.index('3')
    if pred_2_before_3 == true_2_before_3:
        pairs_correct += 1
    
    # 根据正确的对数计算分数
    return pairs_correct * 0.333

def compute_order_consistency_penalty(event1_date, event2_date, event3_date, claimed_order):
    """计算声称的事件排序与实际日期排序之间的不一致性惩罚"""
    try:
        # 解析三个日期
        year1, month1 = map(int, event1_date.split("-"))
        year2, month2 = map(int, event2_date.split("-"))
        year3, month3 = map(int, event3_date.split("-"))
        
        # 计算每个事件的总月数，用于比较先后顺序
        event1_months = year1 * 12 + month1
        event2_months = year2 * 12 + month2
        event3_months = year3 * 12 + month3
        
        # 根据日期计算实际顺序
        events_by_time = [(1, event1_months), (2, event2_months), (3, event3_months)]
        events_by_time.sort(key=lambda x: x[1])
        actual_order = '-'.join(str(event[0]) for event in events_by_time)
        
        # 比较声称的顺序与实际顺序
        if claimed_order == actual_order:
            return 1.0  # 完全一致，不惩罚
        
        # 计算声称顺序与实际顺序的不一致程度
        # 这里我们可以用compute_order_accuracy来计算两个顺序的相似度
        similarity = compute_order_accuracy(claimed_order, actual_order)
        
        # 根据相似度设置惩罚系数
        if similarity >= 0.666:  # 至少2组正确
            return 0.7
        elif similarity >= 0.333:  # 至少1组正确
            return 0.4
        else:  # 全部错误
            return 0.2
    except Exception:
        return 0.1  # 出现异常，严重惩罚

# 定义标准月份名和它们的所有变体映射
months_and_variants = {
            "January": ["january", "jan", "jan."],
            "February": ["february", "feb", "feb."],
            "March": ["march", "mar", "mar."],
            "April": ["april", "apr", "apr."],
            "May": ["may"],
            "June": ["june", "jun", "jun."],
            "July": ["july", "jul", "jul."],
            "August": ["august", "aug", "aug."],
            "September": ["september", "sept", "sept.", "sep", "sep."],
            "October": ["october", "oct", "oct."],
            "November": ["november", "nov", "nov."],
            "December": ["december", "dec", "dec."]
        }

def entity_match_score(predicted_entity, true_entity, entity_type, alpha=0.3):
    """计算缺失实体预测的匹配得分，使用指数衰减函数"""
    if predicted_entity is None or true_entity is None:
        return 0.0
    
    if entity_type == "year":
        # 对于年份，使用数值差距的指数衰减
        try:
            pred_year = int(predicted_entity)
            true_year = int(true_entity)
            diff = abs(pred_year - true_year)
            return builtin_math.exp(-alpha * diff)
        except ValueError:
            return 0.0
    
    elif entity_type == "month":
        # 对于月份，使用更宽松的匹配标准，包括所有月份变体
        
        # 将月份数字映射到月份名称
        month_numbers = {1: "January", 2: "February", 3: "March", 4: "April", 
                        5: "May", 6: "June", 7: "July", 8: "August", 
                        9: "September", 10: "October", 11: "November", 12: "December"}
        
        # 标准化预测和真实月份为小写
        pred_lower = predicted_entity.lower()
        true_lower = true_entity.lower()
        
        # 尝试解析数字月份
        try:
            if pred_lower.isdigit():
                pred_month_num = int(pred_lower)
                if 1 <= pred_month_num <= 12:
                    pred_lower = month_numbers[pred_month_num].lower()
            
            if true_lower.isdigit():
                true_month_num = int(true_lower)
                if 1 <= true_month_num <= 12:
                    true_lower = month_numbers[true_month_num].lower()
        except (ValueError, KeyError):
            pass
        
        # 对每个标准月份，检查预测和真实值是否匹配其任一变体
        pred_standard_month = None
        true_standard_month = None
        
        for standard_month, variants in months_and_variants.items():
            if pred_lower in variants or pred_lower == standard_month.lower():
                pred_standard_month = standard_month
            
            if true_lower in variants or true_lower == standard_month.lower():
                true_standard_month = standard_month
        
        # 如果预测和真实月份归于同一标准月份，则匹配
        if pred_standard_month and true_standard_month and pred_standard_month == true_standard_month:
            return 1.0
        
        # 如果没有精确匹配，尝试计算月份之间的距离
        month_order = list(months_and_variants.keys())
        
        if pred_standard_month and true_standard_month:
            try:
                pred_idx = month_order.index(pred_standard_month)
                true_idx = month_order.index(true_standard_month)
                
                # 计算环形距离 - 取直接距离和绕一圈距离中的最小值
                direct_diff = abs(pred_idx - true_idx)
                circular_diff = 12 - direct_diff  # 12个月的循环
                month_diff = min(direct_diff, circular_diff)
                
                # 使用月份差距的指数衰减
                return builtin_math.exp(-alpha * month_diff)
            except (ValueError, IndexError):
                pass
    
    return 0.0

def is_valid_year(year_str):
    """验证输入是否是有效的4位数年份"""
    try:
        year = int(year_str)
        return 1900 <= year <= 2100  # 设置合理的年份范围
    except ValueError:
        return False

def is_valid_month(month_str):
    """验证输入是否是有效的月份名称或变体"""
    month_str_lower = month_str.lower()
    
    # 检查是否匹配任何月份或其变体
    for standard_month, variants in months_and_variants.items():
        if month_str_lower in variants or month_str_lower == standard_month.lower():
            return True
    
    # 如果是数字，检查是否在1-12范围内
    try:
        month_num = int(month_str)
        return 1 <= month_num <= 12
    except ValueError:
        pass
    
    return False

def compute_length_repetition_penalty(solution_str):
    """
    计算回答过长和内容重复的惩罚值
    返回一个惩罚数值，0表示没有惩罚，值越大表示惩罚越严重
    """
    # 初始无惩罚
    length_penalty = 0.0
    repetition_penalty = 0.0
    
    # # 提取思考部分文本
    # think_content = ""
    # if "<think>" in solution_str and "</think>" in solution_str:
    #     try:
    #         think_content = solution_str.split("<think>")[1].split("</think>")[0]
    #     except:
    #         pass
    
    # 分词处理
    tokens = solution_str.split()
    token_count = len(tokens)

    # 长度惩罚仍然保留，但单独计算
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
        for window_size in [3,5,7,9]:  # 检测3-7个词的短语重复
            for i in range(len(tokens) - window_size * 3):
                phrase = ' '.join(tokens[i:i+window_size])
                
                # 在接下来的文本中检查是否有连续重复
                next_text = ' '.join(tokens[i+window_size:i+window_size*4])
                repeat_count = next_text.count(phrase)
                
                if repeat_count >= 2:  # 同一短语在其后立即出现2次以上
                    repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)  # 最高0.45惩罚
    
    # 3. 仍然保留全局n-gram多样性检测，但使用滑动窗口而不是跳跃抽样
    if token_count > 200:
        chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]  # 不再每5个词跳一次
        if chunks:
            unique_chunks = set(chunks)
            unique_ratio = len(unique_chunks) / len(chunks)
            
            # 当重复内容超过60%时开始惩罚
            if unique_ratio < 0.5:
                repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)  # 提高惩罚力度
    
    # 结合长度惩罚和重复惩罚
    total_penalty = repetition_penalty * 0.8 # 最高0.4
        
    total_penalty = max(length_penalty, repetition_penalty)  # 取较大的惩罚
    
    return total_penalty                


#--------------- 任务1: 时间差异计算 ---------------#

def extract_time_diff_answer(solution_str):
    """
    从解答文本中提取时间差异答案，包括两个日期和月份差值
    格式：'Event 1: YYYY-MM, Event 2: YYYY-MM. Month difference: XX.'
    返回 (event1_date, event2_date, month_diff) 或者 None
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None, None
    
    answer_text = match.group(1).strip()
    
    # 提取两个日期和月份差值
    date_diff_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2})\. Month difference: (\d{1,3})\.'
    match = re.search(date_diff_pattern, answer_text)
    
    if match:
        event1_date = match.group(1)
        event2_date = match.group(2)
        month_diff = int(match.group(3))
        return event1_date, event2_date, month_diff
    
    return None, None, None

def compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    计算时间差异任务的总评分
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期和月份差的字典，格式为:
                           {"event1_pub_date": "YYYY-MM", 
                            "event2_pub_date": "YYYY-MM",
                            "month_difference": XX}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于date_prediction_reward的衰减速率，默认为 0.05
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        float: 总评分
    """
    # 解析答案
    event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
    # "No event"惩罚
    no_event_penalty = 0
    if event1_date and event2_date:
        if "no event" in event1_date.lower() or "no event" in event2_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 格式奖励部分
    format_bonus = 0.0
    if (event1_date and event2_date and month_diff is not None and 
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
        format_bonus = bonus
    
    # 准确性计算部分
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    diff_accuracy = 0.0
    consistency_penalty = 1.0
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_month_diff = ground_truth.get("month_difference")
        
        if true_event1_date and true_event2_date and true_month_diff is not None:
            # 计算两个日期预测的准确性
            if is_valid_date_format(true_event1_date):
                event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.25
            
            if is_valid_date_format(true_event2_date):
                event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.25
            
            # # 计算月份差值预测的准确性
            # diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha) * 0.5
            # 计算月份差值预测的准确性，根据预测值大小使用不同alpha
            if month_diff >= 25:
                # 大月份差使用较小alpha=0.05，提供更宽松的评分
                diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.05) * 0.5
            else:
                # 小月份差使用较大alpha=0.1，要求更严格的准确性
                diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.1) * 0.5
            
            # 检查一致性并应用惩罚
            consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
            # 应用惩罚
            accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            consistency_penalty, "time_difference")

def compute_time_diff_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """计算时间差异任务的总评分，支持每个事件使用不同的alpha值"""
    # 解析答案
    event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
    # "No event"惩罚
    no_event_penalty = 0
    if event1_date and event2_date:
        if "no event" in event1_date.lower() or "no event" in event2_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 格式奖励部分
    format_bonus = 0.0
    if (event1_date and event2_date and month_diff is not None and 
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
        format_bonus = bonus
    
    # 准确性计算部分
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    diff_accuracy = 0.0
    consistency_penalty = 1.0

    # 获取训练进度信息
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_month_diff = ground_truth.get("month_difference")
        
        # 获取难度信息
        difficulty_info = ground_truth.get("difficulty", {})
        events_difficulty = difficulty_info.get("events_difficulty", [])
        
        # 设置默认alpha值
        alpha1 = alpha
        alpha2 = alpha
        diff_alpha = alpha
        
        # 如果有难度信息，根据每个事件的难度调整alpha
        if len(events_difficulty) == 2:
            # alpha1 = 0.1 if events_difficulty[0] else 0.07
            # alpha2 = 0.1 if events_difficulty[1] else 0.07
            if events_difficulty[0]:  # 简单样本
                alpha1 = 0.1
            else:  # 困难样本
                # 如果有current_alpha信息，使用渐进值，否则使用原始值0.07
                alpha1 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[1]:  # 简单样本
                alpha2 = 0.1
            else:  # 困难样本
                # 如果有current_alpha信息，使用渐进值，否则使用原始值0.07
                alpha2 = current_alpha if current_alpha is not None else 0.07
            # 对月份差使用平均alpha或默认alpha
            if month_diff >= 25:
                diff_alpha = 0.05  # 大月份差使用更宽松的alpha
            else:
                diff_alpha = (alpha1 + alpha2) / 2  # 使用两个事件alpha的平均值
        
        if true_event1_date and true_event2_date and true_month_diff is not None:
            # 计算两个日期预测的准确性，使用各自的alpha
            if is_valid_date_format(true_event1_date):
                event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha1) * 0.25
            
            if is_valid_date_format(true_event2_date):
                event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha2) * 0.25
            
            # 计算月份差值预测的准确性，使用diff_alpha
            diff_accuracy = month_diff_reward(month_diff, true_month_diff, diff_alpha) * 0.5
            
            # 检查一致性并应用惩罚
            consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
            # 应用惩罚
            accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each

    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)
    
    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            consistency_penalty, "time_difference")


#--------------- 任务2: 时间排序 ---------------#

def extract_time_order_answer(solution_str):
    """
    从解答文本中提取时间顺序答案，包括三个日期和时间顺序
    格式：'Event 1: YYYY-MM, Event 2: YYYY-MM, Event 3: YYYY-MM. Event order: X-X-X.'
    返回 (event1_date, event2_date, event3_date, event_order) 或者 None
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None, None, None
    
    answer_text = match.group(1).strip()
    
    # 提取三个日期和顺序
    date_order_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2}), Event 3: (\d{4}-\d{2})\. Event order: (\d-\d-\d)\.'
    match = re.search(date_order_pattern, answer_text)
    
    if match:
        event1_date = match.group(1)
        event2_date = match.group(2)
        event3_date = match.group(3)
        event_order = match.group(4)
        return event1_date, event2_date, event3_date, event_order
    
    return None, None, None, None

def compute_date_diversity_penalty(event1_date, event2_date, event3_date, event_order):
    """检查日期是否具有多样性，以及事件顺序是否不是简单的默认顺序1-2-3"""
    try:
        # 解析日期
        year1, month1 = map(int, event1_date.split("-"))
        year2, month2 = map(int, event2_date.split("-"))
        year3, month3 = map(int, event3_date.split("-"))
        
        # 将日期转换为月份总数
        total_months1 = year1 * 12 + month1
        total_months2 = year2 * 12 + month2
        total_months3 = year3 * 12 + month3
        
        # 检查日期是否全部相同
        if total_months1 == total_months2 == total_months3:
            return 0.2  # 严重惩罚 - 所有日期都相同
        
        # 检查是否是简单的连续月份模式(例如 2018-11, 2018-12, 2019-01)
        is_sequential_0 = False  # 递增模式初始化
        is_sequential_1 = False  # 递减模式初始化
        
        # 检查是否是每月递增模式
        if (total_months2 == total_months1 + 1 and total_months3 == total_months2 + 1):
            is_sequential_0 = True
        # 或每月递减模式
        elif (total_months2 == total_months1 - 1 and total_months3 == total_months2 - 1):
            is_sequential_1 = True
        
        # 检查事件顺序是否只是默认的1-2-3
        is_default_order_0 = (event_order == "1-2-3")
        is_default_order_1 = (event_order == "3-2-1")
        
        # 组合惩罚
        if is_sequential_0 and is_default_order_0:
            return 0.2  # 中度惩罚 - 连续月份和默认顺序
        elif is_sequential_1 and is_default_order_1:
            return 0.2  # 轻度惩罚 - 只是连续月份
        # elif is_default_order and abs(total_months1 - total_months2) < 3 and abs(total_months2 - total_months3) < 3:
        #     return 0.7  # 很轻的惩罚 - 默认顺序且日期接近
        
        # 日期有多样性，且顺序不是默认的
        return 1.0  # 不惩罚
        
    except Exception:
        # 出现解析错误等情况，也应当惩罚
        return 0.5

def compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    计算时间顺序任务的总评分
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期和时间顺序的字典，格式为:
                           {"event1_pub_date": "YYYY-MM", 
                            "event2_pub_date": "YYYY-MM",
                            "event3_pub_date": "YYYY-MM",
                            "event_order": "X-X-X"}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于date_prediction_reward的衰减速率，默认为 0.05
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        float: 总评分
    """
    # 解析答案
    event1_date, event2_date, event3_date, event_order = extract_time_order_answer(solution_str)
    
    # "No event"惩罚
    no_event_penalty = 0
    if event1_date and event2_date and event3_date:
        if ("no event" in event1_date.lower() or "no event" in event2_date.lower() or 
            "no event" in event3_date.lower()):
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 格式奖励部分 - 使用新的is_valid_order_format函数
    format_bonus = 0.0
    if (event1_date and event2_date and event3_date and event_order and
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date) and 
        is_valid_date_format(event3_date) and is_valid_order_format(event_order)):
        format_bonus = bonus
    
    # 准确性计算部分
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    event3_accuracy = 0.0
    order_accuracy = 0.0
    consistency_penalty = 1.0
    combined_penalty = 1.0
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_event3_date = ground_truth.get("event3_pub_date")
        true_event_order = ground_truth.get("event_order")
        
        if (true_event1_date and true_event2_date and true_event3_date and true_event_order and
            is_valid_date_format(true_event1_date) and is_valid_date_format(true_event2_date) and 
            is_valid_date_format(true_event3_date)):
            
            # 计算三个日期预测的准确性 (各占27%)
            event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.2
            event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.2
            event3_accuracy = date_prediction_reward(event3_date, true_event3_date, alpha) * 0.2
            
            # 计算顺序预测的准确性 (占19%) - 使用新的compute_order_accuracy函数
            order_accuracy = compute_order_accuracy(event_order, true_event_order) * 0.4
            
            # 添加一致性惩罚 - 使用新的compute_order_consistency_penalty函数
            consistency_penalty = compute_order_consistency_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # 添加日期多样性惩罚
            diversity_penalty = compute_date_diversity_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # 组合两种惩罚
            combined_penalty = consistency_penalty * diversity_penalty
            
            # 应用惩罚计算总准确性分数
            accuracy_score = (event1_accuracy + event2_accuracy + event3_accuracy + order_accuracy) * combined_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，包括一致性惩罚
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            combined_penalty, "time_ordering")

def compute_time_order_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """计算时间顺序任务的总评分，支持每个事件使用不同的alpha值"""
    # 解析答案
    event1_date, event2_date, event3_date, event_order = extract_time_order_answer(solution_str)
    
    # "No event"惩罚
    no_event_penalty = 0
    if event1_date and event2_date and event3_date:
        if ("no event" in event1_date.lower() or "no event" in event2_date.lower() or 
            "no event" in event3_date.lower()):
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 格式奖励部分
    format_bonus = 0.0
    if (event1_date and event2_date and event3_date and event_order and
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date) and 
        is_valid_date_format(event3_date) and is_valid_order_format(event_order)):
        format_bonus = bonus
    
    # 准确性计算部分
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    event3_accuracy = 0.0
    order_accuracy = 0.0
    consistency_penalty = 1.0
    combined_penalty = 1.0
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_event3_date = ground_truth.get("event3_pub_date")
        true_event_order = ground_truth.get("event_order")
        
        # 获取难度信息
        difficulty_info = ground_truth.get("difficulty", {})
        events_difficulty = difficulty_info.get("events_difficulty", [])

        # 获取训练进度信息
        current_alpha = ground_truth.get("current_alpha", None)
        global_step = ground_truth.get("global_step", 0)
        
        # 设置默认alpha值
        alpha1 = alpha
        alpha2 = alpha
        alpha3 = alpha
        
        # 如果有难度信息，根据每个事件的难度调整alpha
        if len(events_difficulty) == 3:
            # alpha1 = 0.1 if events_difficulty[0] else 0.07
            # alpha2 = 0.1 if events_difficulty[1] else 0.07
            # alpha3 = 0.1 if events_difficulty[2] else 0.07

            # 新增的动态alpha代码
            if events_difficulty[0]:  # 简单样本
                alpha1 = 0.1
            else:  # 困难样本
                # 如果有current_alpha信息，使用渐进值，否则使用原始值0.07
                alpha1 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[1]:  # 简单样本
                alpha2 = 0.1
            else:  # 困难样本
                # 如果有current_alpha信息，使用渐进值，否则使用原始值0.07
                alpha2 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[2]:  # 简单样本
                alpha3 = 0.1
            else:  # 困难样本
                # 如果有current_alpha信息，使用渐进值，否则使用原始值0.07
                alpha3 = current_alpha if current_alpha is not None else 0.07
        
        if (true_event1_date and true_event2_date and true_event3_date and true_event_order and
            is_valid_date_format(true_event1_date) and is_valid_date_format(true_event2_date) and 
            is_valid_date_format(true_event3_date)):
            
            # 计算三个日期预测的准确性，每个使用对应的alpha
            event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha1) * 0.2
            event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha2) * 0.2
            event3_accuracy = date_prediction_reward(event3_date, true_event3_date, alpha3) * 0.2
            
            # 计算顺序预测的准确性
            order_accuracy = compute_order_accuracy(event_order, true_event_order) * 0.4
            
            # 添加一致性惩罚
            consistency_penalty = compute_order_consistency_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # 添加日期多样性惩罚
            diversity_penalty = compute_date_diversity_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # 组合两种惩罚
            combined_penalty = consistency_penalty * diversity_penalty
            
            # 应用惩罚计算总准确性分数
            accuracy_score = (event1_accuracy + event2_accuracy + event3_accuracy + order_accuracy) * combined_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each

    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)
    
    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，包括一致性惩罚
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            combined_penalty, "time_ordering")

#--------------- 任务3: 时间补全 ---------------#

def extract_time_completion_answer(solution_str):
    """
    从解答文本中提取时间补全答案，包括事件日期和缺失实体
    格式：'Event: YYYY-MM. Missing entity: XXXXX.'
    返回 (event_date, missing_entity) 或者 None
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None
    
    answer_text = match.group(1).strip()
    
    # 提取事件日期和缺失实体
    completion_pattern = r'Event: (\d{4}-\d{2})\. Missing entity: (.+?)\.'
    match = re.search(completion_pattern, answer_text)
    
    if match:
        event_date = match.group(1)
        missing_entity = match.group(2).strip()
        return event_date, missing_entity
    
    return None, None

def compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    计算时间补全任务的总评分
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期和缺失实体的字典，格式为:
                           {"event_pub_date": "YYYY-MM", 
                            "mask_type": "year"或"month",
                            "masked_entity": "XXXX"}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于date_prediction_reward的衰减速率，默认为 0.05
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        float: 总评分
    """
    # 解析答案
    event_date, missing_entity = extract_time_completion_answer(solution_str)
    
    # "No event"惩罚
    no_event_penalty = 0
    if event_date:
        if "no event" in event_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 获取掩码类型
    mask_type = ground_truth.get("mask_type", "")
    
    # 格式奖励部分 - 增加对missing_entity类型的验证
    format_bonus = 0.0
    if event_date and missing_entity and is_valid_date_format(event_date):
        # 根据mask_type验证missing_entity的格式
        if mask_type == "year" and is_valid_year(missing_entity):
            format_bonus = bonus
        elif mask_type == "month" and is_valid_month(missing_entity):
            format_bonus = bonus
        elif not mask_type:  # 如果没有mask_type，宽松处理
            format_bonus = bonus
    
    # 准确性计算部分
    date_accuracy = 0.0
    entity_accuracy = 0.0
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event_date = ground_truth.get("event_pub_date")
        masked_entity = ground_truth.get("masked_entity", "")
        
        if true_event_date and mask_type and masked_entity and is_valid_date_format(true_event_date):
            # 计算日期预测的准确性 (占50%)
            date_accuracy = date_prediction_reward(event_date, true_event_date, alpha) * 0.5
            
            # 计算缺失实体预测的准确性 (占50%)
            entity_accuracy = entity_match_score(missing_entity, masked_entity, mask_type, alpha*3) * 0.5
            
            # 总的准确性分数
            accuracy_score = date_accuracy + entity_accuracy
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_completion")  # 一致性惩罚为1.0，因为这个任务不需要一致性检查

def compute_time_completion_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """计算时间补全任务的总评分，支持动态alpha"""
    # 解析答案
    event_date, missing_entity = extract_time_completion_answer(solution_str)
    
    # 获取训练进度信息
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)

    # 获取难度信息并确定alpha值
    difficulty_info = ground_truth.get("difficulty", {})
    if difficulty_info:
        # 使用事件自身的alpha值覆盖默认参数
        event_alpha = difficulty_info.get("alpha", alpha)
        # 如果原始alpha是0.07（困难样本），应用动态alpha
        if abs(event_alpha - 0.07) < 0.001 and current_alpha is not None:
            event_alpha = current_alpha  # 使用动态alpha
        alpha = event_alpha  # 使用动态alpha
    
    # "No event"惩罚
    no_event_penalty = 0
    if event_date:
        if "no event" in event_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # 获取掩码类型
    mask_type = ground_truth.get("mask_type", "")
    
    # 格式奖励部分
    format_bonus = 0.0
    if event_date and missing_entity and is_valid_date_format(event_date):
        # 根据mask_type验证missing_entity的格式
        if mask_type == "year" and is_valid_year(missing_entity):
            format_bonus = bonus
        elif mask_type == "month" and is_valid_month(missing_entity):
            format_bonus = bonus
        elif not mask_type:  # 如果没有mask_type，宽松处理
            format_bonus = bonus
    
    # 准确性计算部分
    date_accuracy = 0.0
    entity_accuracy = 0.0
    
    if format_bonus > 0:  # 只有格式正确才计算准确性
        true_event_date = ground_truth.get("event_pub_date")
        masked_entity = ground_truth.get("masked_entity", "")
        
        if true_event_date and mask_type and masked_entity and is_valid_date_format(true_event_date):
            # 计算日期预测的准确性 (占50%)，使用动态alpha
            date_accuracy = date_prediction_reward(event_date, true_event_date, alpha) * 0.5
            
            # 计算缺失实体预测的准确性 (占50%)，使用相同的动态alpha
            entity_accuracy = entity_match_score(missing_entity, masked_entity, mask_type, alpha*3) * 0.5
            
            # 总的准确性分数
            accuracy_score = date_accuracy + entity_accuracy
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag 奖励部分
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # 应用长度和重复惩罚
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # 总分计算
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_completion")

#--------------- 任务4: 时间推断 ---------------#

def compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    计算时间推断任务的总评分（与现有的compute_score完全相同）
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实日期的字典，格式为 {"event_pub_date": "YYYY-MM"}
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于 date_prediction_reward 的衰减速率，默认为 0.05
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        float: 总评分
    """
    answer = extract_answer_format(solution_str)

    # "No event"惩罚
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.1  
    else:
        no_event_penalty = 0.2

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

    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_inferring")

def compute_time_inferring_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """计算时间推断任务的总评分，支持动态alpha"""
    answer = extract_answer_format(solution_str)

    # 获取训练进度信息
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)

    # 获取事件难度信息并确定alpha值
    difficulty_info = ground_truth.get("difficulty", {})
    if difficulty_info:
        # 使用事件自身的alpha值覆盖默认参数
        event_alpha = difficulty_info.get("alpha", alpha)
        # 如果原始alpha是0.07（困难样本），应用动态alpha
        if abs(event_alpha - 0.07) < 0.001 and current_alpha is not None:
            event_alpha = current_alpha  # 使用动态alpha
        alpha = event_alpha  # 使用动态alpha
    
    # "No event"惩罚
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.1  
    else:
        no_event_penalty = 0.2

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

    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # 返回总分及各部分分数，便于调试
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_inferring")

#--------------- 统一入口函数 ---------------#

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """
    统一的评分入口函数，根据任务类型自动选择合适的评分函数
    
    参数:
        solution_str (str): 解答文本
        ground_truth (dict): 包含真实情况的字典
        bonus (float): 格式正确时的额外奖励，默认为 0.05
        alpha (float): 用于date_prediction_reward的衰减速率，默认为 0.05
        tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
    返回:
        tuple: (总分, 准确性分数, 格式奖励, 标签格式分数, 标签计数分数, 一致性惩罚, 任务类型)
    """
    # 从ground_truth的extra_info里找任务类型
    task_type = ""
    
    # 首先尝试从non_tensor_batch的extra_info中获取task类型
    if isinstance(ground_truth, dict) and "task" in ground_truth:
        task_type = ground_truth.get("task", "")
    
    # 如果没有找到，尝试从ground_truth本身的特征来判断任务类型
    if not task_type:
        if "event1_pub_date" in ground_truth and "event2_pub_date" in ground_truth:
            if "event3_pub_date" in ground_truth:
                task_type = "time_ordering"
            else:
                task_type = "time_difference"
        elif "mask_type" in ground_truth and "masked_entity" in ground_truth:
            task_type = "time_completion"
        elif "event_pub_date" in ground_truth:
            task_type = "time_inferring"
    
    # # 根据任务类型选择合适的评分函数
    # if task_type == "time_difference":
    #     return compute_time_diff_score(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    # elif task_type == "time_ordering":
    #     return compute_time_order_score(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    # elif task_type == "time_completion":
    #     return compute_time_completion_score(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    # elif task_type == "time_inferring":
    #     return compute_time_inferring_score(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    # else:
    #     # 默认使用时间推断评分
    #     return compute_time_inferring_score(solution_str, ground_truth, bonus, alpha, tag_bonus_each)

    # 根据任务类型选择合适的评分函数
    if task_type == "time_difference":
        return compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_ordering":
        return compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_completion":
        return compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_inferring":
        return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    else:
        # 默认使用时间推断评分
        return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)


# 测试用例
if __name__ == "__main__":
    # 任务1：时间差异计算测试
    solution1 = """<think>分析两个事件的时间...经判断第一个事件发生在2019年3月，第二个事件发生在2020年1月，相差10个月。</think>
<answer>Event 1: 2019-03, Event 2: 2020-01. Month difference: 10.</answer><|im_end|>"""
    
    ground_truth1 = {
        "event1_pub_date": "2019-03",
        "event2_pub_date": "2020-01", 
        "month_difference": 10,
        "extra_info": {"task": "time_difference"}
    }
    
    # 任务2：时间排序测试
    solution2 = """<think>分析三个事件的时间...</think>
<answer>Event 1: 2021-05, Event 2: 2019-11, Event 3: 2022-08. Event order: 2-1-3.</answer><|im_end|>"""
    
    ground_truth2 = {
        "event1_pub_date": "2021-05",
        "event2_pub_date": "2019-11",
        "event3_pub_date": "2022-08",
        "event_order": "2-1-3",
        "extra_info": {"task": "time_ordering"}
    }
    
    # 任务3：时间补全测试
    solution3 = """<think>分析事件时间和缺失信息...</think>
<answer>Event: 2021-03. Missing entity: 2019.</answer><|im_end|>"""
    
    ground_truth3 = {
        "event_pub_date": "2021-03",
        "mask_type": "year",
        "masked_entity": "2019",
        "extra_info": {"task": "time_completion"}
    }
    
    # 任务4：时间推断测试
    solution4 = "<think>Some reasoning...</think>\n<answer>2025-03</answer><|im_end|>"
    
    ground_truth4 = {
        "event_pub_date": "2025-03",
        "extra_info": {"task": "time_inferring"}
    }
    
    print("任务1 分数:", compute_score(solution1, ground_truth1))
    print("任务2 分数:", compute_score(solution2, ground_truth2))
    print("任务3 分数:", compute_score(solution3, ground_truth3))
    print("任务4 分数:", compute_score(solution4, ground_truth4))












# import re
# import sys

# # 临时移除当前工作目录（通常是 sys.path[0]）
# orig_path = sys.path.pop(0)
# import math as builtin_math  # 这里加载的就是内置的 math 模块
# sys.path.insert(0, orig_path)  # 恢复原来的 sys.path


# def extract_answer_format(solution_str):
#     """
#     从解答文本中提取出 <answer>...</answer> 标签中间的内容。
#     如果找不到，则返回 None。
#     """
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if match:
#         return match.group(1).strip()
#     return None

# def is_valid_date_format(date_str):
#     """
#     验证 date_str 是否符合 "YYYY-MM" 的格式，其中月份必须为 01 至 12。
#     """
#     pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
#     return re.match(pattern, date_str) is not None

# def date_prediction_reward(prediction, target, alpha=0.05):
#     """
#     根据预测日期与真实日期之间的月数差距计算奖励。
    
#     参数:
#         prediction (str): 预测的日期，格式为 "YYYY-MM"
#         target (str): 真实日期，格式为 "YYYY-MM"
#         alpha (float): 衰减速率，默认 0.05
        
#     返回:
#         float: 奖励值，当两者完全一致时为 1，随着月数差距增加奖励指数衰减。
#     """
#     try:
#         pred_year, pred_month = map(int, prediction.split("-"))
#         target_year, target_month = map(int, target.split("-"))
#     except Exception:
#         return 0.0
    
#     diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
#     reward = builtin_math.exp(-alpha * diff_in_months)
#     return reward

# def format_reward_single(solution_str):
#     """
#     检查 solution_str 是否严格按照要求包含：
#     <think> 与 </think> 以及 <answer> 与 </answer> 标签。
#     如果完全匹配，则返回 1.0，否则返回 0.0。
#     """
#     pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
#     # pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
#     return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

# def tag_count_reward_single(solution_str):
#     """
#     根据 solution_str 中标签的出现次数计算分值：
#       - 如果 "<think>\n" 出现一次，则奖励 0.25 分；
#       - 如果 "\n</think>\n" 出现一次，则奖励 0.25 分；
#       - 如果 "\n<answer>\n" 出现一次，则奖励 0.25 分；
#       - 如果 "\n</answer>" 出现一次，则奖励 0.25 分；
#     返回总分（理想状态为 1.0）。
#     """
#     count = 0.0
#     # if solution_str.count("<think>") == 1:
#     #     count += 0.25
#     if solution_str.count("</think>") == 1:
#         count += 0.33
#     if solution_str.count("<answer>") == 1:
#         count += 0.33
#     if solution_str.count("</answer>") == 1:
#         count += 0.33
#     return count

# def tag_count_reward_single0(solution_str):
#     """
#     改进的标签计数奖励，更强调格式正确性和标签顺序
#     """
#     count = 0.0
    
#     # 基础标签检查
#     if solution_str.count("<think>") == 1:
#         count += 0.1
#         print(count, 0)
#     if solution_str.count("</think>") == 1:
#         count += 0.15
#         print(count, 1)
#     if solution_str.count("<answer>") == 1:
#         count += 0.15
#         print(count, 2)
#     if solution_str.count("</answer>") == 1:
#         count += 0.15
#         print(count, 3)
    
#     # 标签顺序奖励
#     if "<think>" in solution_str and "</think>" in solution_str:
#         if solution_str.find("<think>") < solution_str.find("</think>"):
#             count += 0.15
#             print(count, 4)
    
#     if "</think>" in solution_str and "<answer>" in solution_str:
#         if solution_str.find("</think>") < solution_str.find("<answer>"):
#             count += 0.15
#             print(count, 5)
    
#     if "<answer>" in solution_str and "</answer>" in solution_str:
#         if solution_str.find("<answer>") < solution_str.find("</answer>"):
#             count += 0.15
#             print(count, 6)
            
#     # 对重复内容的惩罚
#     repetition_penalty = 0
#     if "<think>" in solution_str and "</think>" in solution_str:
#         think_content = solution_str.split("<think>")[1].split("</think>")[0]
#         words = think_content.split()
#         if len(words) > 20:
#             # 计算重复词比例
#             unique_words = set(words)
#             repetition_ratio = 1 - (len(unique_words) / len(words))
#             if repetition_ratio > 0.4:  # 超过40%的词是重复的
#                 repetition_penalty = 1
                
#     return count - repetition_penalty

# # def fluency_reward(solution_str):
# #     rewards = []
# #     allowed_chars = r"a-zA-Z0-9_\s" + re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
# #     pattern = rf'^[{allowed_chars}]+$'
    
# #     for completion in solution_str:
# #         if re.fullmatch(pattern, completion, flags=re.ASCII):
# #             rewards.append(1.0)
# #         else:
# #             rewards.append(0.0)
# #     return rewards

# def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """
#     计算总奖励：如果解答中提取出的答案严格符合 "YYYY-MM" 格式，则额外加上格式奖励 bonus，
#     同时基于预测日期与真实日期（ground_truth['true_pub_date']）之间的月差计算预测准确率奖励。
#     如果答案不存在或格式不正确，则预测奖励部分为 0（总奖励直接返回 0）。
    
#     参数:
#         solution_str (str): 解答文本
#         ground_truth (dict): 包含真实日期的字典，格式为 {"true_pub_date": "YYYY-MM"}
#         bonus (float): 格式正确时的额外奖励，默认为 0.05
#         alpha (float): 用于 date_prediction_reward 的衰减速率，默认为 0.05
        
#     返回:
#         float: 总奖励分数
#     """
#     answer = extract_answer_format(solution_str)

#     # "No event"惩罚
#     no_event_penalty = 0
#     if answer:
#         if "no event" in answer.lower() or "none" in answer.lower():
#             no_event_penalty = 0.1  
#     else:
#         no_event_penalty = 0.2

#     # 如果提取到了答案且符合 "YYYY-MM" 格式，则先获得格式奖励
#     format_bonus, pred_reward = 0.0, 0.0
#     if answer and is_valid_date_format(answer):
#         format_bonus = bonus
#         true_pub_date = ground_truth.get("true_pub_date")
#         # 确保 ground_truth 中的真实日期也符合 "YYYY-MM" 格式，否则不计算预测奖励
#         if true_pub_date and is_valid_date_format(true_pub_date):
#             pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
#     format_pred_reward = format_bonus + pred_reward

#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     total_score = format_pred_reward + tag_format_score + tag_count_score - no_event_penalty
#     # print(format_pred_reward, tag_format_score, tag_count_score, no_event_penalty)
#     # print(format_pred_reward, tag_format_score, tag_count_score)
#     return total_score, pred_reward, format_bonus, tag_format_score, tag_count_score

# def extract_time_diff_answer(solution_str):
#     """
#     从解答文本中提取时间差异答案，包括两个日期和月份差值
#     格式：'Event 1: YYYY-MM, Event 2: YYYY-MM. Month difference: XX.'
#     返回 (event1_date, event2_date, month_diff) 或者 None
#     """
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if not match:
#         return None, None, None
    
#     answer_text = match.group(1).strip()
    
#     # 提取两个日期和月份差值
#     date_diff_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2})\. Month difference: (\d{1,3})\.'
#     match = re.search(date_diff_pattern, answer_text)
    
#     if match:
#         event1_date = match.group(1)
#         event2_date = match.group(2)
#         month_diff = int(match.group(3))
#         return event1_date, event2_date, month_diff
    
#     return None, None, None

# def month_diff_reward(pred_diff, true_diff, alpha=0.05):
#     """计算月份差值预测的奖励，基于差值的指数衰减"""
#     diff = abs(pred_diff - true_diff)
#     reward = builtin_math.exp(-alpha * diff)
#     return reward

# def compute_inconsistency_penalty(date1, date2, claimed_diff):
#     """计算声称的月份差与实际月份差之间的不一致性惩罚"""
#     try:
#         year1, month1 = map(int, date1.split("-"))
#         year2, month2 = map(int, date2.split("-"))
        
#         actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
#         error = abs(actual_diff - claimed_diff)
        
#         # 惩罚系数随误差增加而加重
#         if error == 0:
#             return 1.0  # 没有误差，不惩罚
#         elif error <= 2:
#             return 0.7  # 小误差，轻微惩罚
#         elif error <= 5:
#             return 0.4  # 中等误差，中等惩罚
#         else:
#             return 0.2  # 大误差，严重惩罚
#     except Exception:
#         return 0.1  # 出现异常，严重惩罚

# def compute_time_diff_score(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """
#     计算时间差异任务的总评分
    
#     参数:
#         solution_str (str): 解答文本
#         ground_truth (dict): 包含真实日期和月份差的字典，格式为:
#                            {"event1_pub_date": "YYYY-MM", 
#                             "event2_pub_date": "YYYY-MM",
#                             "month_difference": XX}
#         bonus (float): 格式正确时的额外奖励，默认为 0.05
#         alpha (float): 用于date_prediction_reward的衰减速率，默认为 0.05
#         tag_bonus_each (float): 每个标签格式正确的奖励，默认为 0.025
        
#     返回:
#         float: 总评分
#     """
#     # 解析答案
#     event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
#     # "No event"惩罚
#     no_event_penalty = 0
#     if event1_date and event2_date:
#         if "no event" in event1_date.lower() or "no event" in event2_date.lower():
#             no_event_penalty = 0.1
#     else:
#         no_event_penalty = 0.2
    
#     # 格式奖励部分
#     format_bonus = 0.0
#     if (event1_date and event2_date and month_diff is not None and 
#         is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
#         format_bonus = bonus
    
#     # 准确性计算部分
#     event1_accuracy = 0.0
#     event2_accuracy = 0.0
#     diff_accuracy = 0.0
    
#     if format_bonus > 0:  # 只有格式正确才计算准确性
#         true_event1_date = ground_truth.get("event1_pub_date")
#         true_event2_date = ground_truth.get("event2_pub_date")
#         true_month_diff = ground_truth.get("month_difference")
        
#         if true_event1_date and true_event2_date and true_month_diff is not None:
#             # 计算两个日期预测的准确性
#             if is_valid_date_format(true_event1_date):
#                 event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.4
            
#             if is_valid_date_format(true_event2_date):
#                 event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.4
            
#             # 计算月份差值预测的准确性
#             diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha) * 0.2
            
#             # 检查一致性并应用惩罚
#             consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
#             # 应用惩罚
#             accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
#         else:
#             accuracy_score = 0.0
#     else:
#         accuracy_score = 0.0
    
#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 总分计算
#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty
    
#     # 返回总分及各部分分数，便于调试
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             consistency_penalty if 'consistency_penalty' in locals() else 0.0)



# # # 示例测试
# # if __name__ == "__main__":
# #     # 假设预测准确率部分计算结果依赖于真实发布日期
# #     ground_truth = {"true_pub_date": "2025-03"}
    
# #     # 示例 1：答案存在且格式正确，预测接近目标
# #     solution1 = "Some text... <answer>2025-04</answer>"
# #     # 示例 2：答案存在但格式错误
# #     solution2 = "Some text... <answer>2025/03</answer>"
# #     # 示例 3：答案标签不存在
# #     solution3 = "Some text without answer tag."
    
# #     print(builtin_math.exp(-0.05 * 1))
# #     print(builtin_math.exp(-0.05 * 3))
# #     print(builtin_math.exp(-0.05 * 6))
# #     print(builtin_math.exp(-0.05 * 12))
# #     print("Solution1 reward:", compute_total_reward(solution1, ground_truth))
# #     print("Solution2 reward:", compute_total_reward(solution2, ground_truth))
# #     print("Solution3 reward:", compute_total_reward(solution3, ground_truth))


# if __name__ == "__main__":
#     # 示例1：预测正确、格式正确，且标签齐全
#     solution1 = "<think>Some reasoning...</think>\n<answer>2025-03</answer><|im_end|>"
#     # 示例2：预测错误（日期不同），但格式正确，标签齐全
#     solution2 = "<think> the first half of the year. </think>\n<answer>2021-01</answer>\nThe specific date of January 2021 is inferred as the most likely occurrence based on the context provided.  The event aligns with the timing of large-scale events during the early stages of the pandemic when online ticketing systems were still being tested and could be prone to crashes.<|im_end|>"
#     # 示例3：答案格式错误
#     solution3 = "<think>At first glance, headlines mentioning 'Donald Trump's final days in office' suggests the likely approximate timeframe.  However, time precision within a 6-month span (from January to June, 2017 to 2018) is requested and thus the information has to be further narrowed.  It is given that there was a 'Tough Sanctions' that was carried out 'in 2017'.  These sanctions were then 'rolled back with no explanation' in 'Donald Trump’s final days in office'.  Let us reason: Despite the 'no explanation', there certainly was no notion of immediate rollback.  Political decisions about sanctions are not made quickly, especially not on a Monday in a time when a new President is inaugurating on January 20th.  <answer>2025-03</answer><|im_end|>"
#     # 示例4：标签缺失（但答案可能存在且格式正确）
#     solution4 = "Some text without proper tags <answer>no event</answer>"
#     # 示例4：标签缺失（但答案可能存在且格式正确）
#     solution5 = "<|im_start|>system You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.- Output the event's occurrence date in the format 'YYYY-MM'.Show your work in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>2025-03</answer>. <think> At first glance, headlines mentioning 'Donald Trump's final days in office' suggests the likely approximate timeframe. However, time precision within a 6-month span (from January to June, 2017 to 2018) is requested and thus the information has to be further narrowed. It is given that there was a 'Tough Sanctions' that was carried out 'in 2017'. These sanctions were then 'rolled back with no explanation' in 'Donald Trump’s final days in office'. Let us reason: Despite the 'no explanation', there certainly was no notion of immediate rollback. Political decisions about sanctions are not made quickly, especially not on a Monday in a time when a new President is inaugurating on January 20th. <answer>2017-12</answer><|im_end|>"
    
#     ground_truth = {"true_pub_date": "2025-03"}
    
#     print(compute_score(solution1, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025))
#     print(compute_score(solution3, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025))
#     print(compute_score(solution5, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025))
#     # for i, sol in enumerate([solution1, solution2, solution3, solution4, solution5], 1):
#     #     score = compute_score(sol, ground_truth, bonus=0.1, alpha=0.05, tag_bonus_each=0.05)
#     #     print(f"Solution{i} total score: {score}")