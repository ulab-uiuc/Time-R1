import os
import re
import json
import pickle
import random
from datetime import datetime
from collections import Counter

# 定义月份列表及其变体
MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

# 包括月份的简写和变体
MONTH_VARIANTS = {
    "January": ["January", "Jan", "Jan."],
    "February": ["February", "Feb", "Feb."],
    "March": ["March", "Mar."],
    "April": ["April", "Apr", "Apr."],
    "May": ["May"],
    "June": ["June", "Jun", "Jun."],
    "July": ["July", "Jul", "Jul."],
    "August": ["August", "Aug", "Aug."],
    "September": ["September", "Sept", "Sept.", "Sep", "Sep."],
    "October": ["October", "Oct", "Oct."],
    "November": ["November", "Nov", "Nov."],
    "December": ["December", "Dec", "Dec."]
}

# 创建月份别名到标准月份的映射
MONTH_MAPPING = {}
for standard, variants in MONTH_VARIANTS.items():
    for variant in variants:
        MONTH_MAPPING[variant.lower()] = standard

def parse_year_month_from_true(date_str: str):
    """
    Parse a date string in the format 'YYYY-MM'. Returns (year, month) or (None, None) if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

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
    
    # Split events into train and test sets - 与event_order_nyt.py完全相同
    random.seed(random_seed)
    random.shuffle(all_events)
    split_index = int(split_ratio * len(all_events))
    
    return all_events[:split_index], all_events[split_index:]

def extract_years(text):
    """提取文本中的年份（四位数字），使用上下文规则过滤非年份的四位数字"""
    years = []
    
    # 查找形如yyyy的四位数字
    year_matches = re.finditer(r'\b(19\d\d|20\d\d)\b', text)
    
    for match in year_matches:
        year = match.group(1)
        start, end = match.span()
        
        # 获取上下文 (前后30个字符)
        context_start = max(0, start - 15)
        context_end = min(len(text), end + 15)
        context = text[context_start:context_end].lower()
        
        # 检查上下文是否支持这是一个年份
        # 例如，上下文中有时间词汇或介词暗示时间
        is_year = (
            # 时间词汇
            any(word in context for word in ["year""in", "on", "of", "since", "during", "before", "after", "through", "until", "for", "by", "at", "from"]) or
            # 月份名
            any(month.lower() in context for month in MONTHS) or
            # 季节名
            any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"])
        )
        
        if is_year:
            years.append(year)
    
    return years

def extract_months_0(text):
    """提取文本中的月份名称，使用增强的上下文规则过滤非月份含义的词"""
    found_months = []
    text_lower = text.lower()
    
    # 具有歧义的月份单词
    ambiguous_months = ["may", "march", "august"]
    
    # 情态动词"may"的常见模式
    modal_may_patterns = [
        r'may\s+(be|have|not|also|include|seem|become|need|want|make|take|get|help|cause|vary|\w+\s+to)',  # may后跟动词或不定式
        r'(it|this|that|they|we|you|he|she|which|who)\s+may\s+\w+',  # 代词+may+动词
        r'(and|but|or|then|however|therefore)\s+may\s+\w+',  # 连词+may+动词
        r'may\s+(not|never|only|also|still|yet)'  # may后跟副词
    ]
    
    # 检查每个月份
    for standard_month, variants in MONTH_VARIANTS.items():
        for variant in variants:
            variant_lower = variant.lower()
            
            # 简单检查该变体是否出现在文本中
            if not re.search(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                continue
                
            # 对于可能有歧义的月份，需要额外的上下文检查
            if variant_lower in ambiguous_months:
                # 提取包含该月份的上下文
                contexts = []
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    # 使用更大的上下文窗口(30个字符)来捕获更多上下文信息
                    context_start = max(0, start - 15)
                    context_end = min(len(text_lower), end + 15)
                    context = text_lower[context_start:context_end]
                    contexts.append(context)
                
                # 检查是否有任何上下文满足时间相关条件
                is_month = False
                for context in contexts:
                    # 对于"may"特别检查是否是情态动词
                    if variant_lower == "may":
                        # 检查是否匹配情态动词may的模式
                        is_modal_verb = any(re.search(pattern, context) for pattern in modal_may_patterns)
                        if is_modal_verb:
                            continue  # 如果是情态动词，跳过这个匹配
                    
                    # 检查上下文中的时间指示
                    is_time_context = (
                        # 明确的时间格式
                        re.search(r'(in|on|of|since|during|before|after|through|until|for|by)\s+' + re.escape(variant_lower), context) or
                        re.search(re.escape(variant_lower) + r'\s+\d{1,2}(st|nd|rd|th)?', context) or
                        # 年份在附近
                        re.search(r'\b(19\d\d|20\d\d)\b.*' + re.escape(variant_lower), context) or
                        re.search(re.escape(variant_lower) + r'.*\b(19\d\d|20\d\d)\b', context) or
                        # 其他季节或月份在附近
                        any(month.lower() != variant_lower and month.lower() in context for month in MONTHS) or
                        any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"])
                    )
                    
                    if is_time_context:
                        is_month = True
                        break
                
                if not is_month:
                    continue  # 如果没有时间上下文，跳过这个可能有歧义的月份
            
            # 添加到找到的月份列表
            found_months.append(standard_month)
            break  # 找到一个变体就停止检查其他变体
    
    return found_months

def get_word_context(text, position, n_words=3):
    """获取指定位置前后n个单词的上下文"""
    # 分割文本为单词列表及其位置
    word_positions = []
    for match in re.finditer(r'\b\w+\b', text):
        word_positions.append((match.start(), match.end(), match.group()))
    
    if not word_positions:
        return [], "", []
    
    # 找到包含目标位置的单词索引
    target_index = -1
    for i, (start, end, _) in enumerate(word_positions):
        if start <= position < end:
            target_index = i
            break
    
    # 如果没找到精确匹配，找最接近的单词
    if target_index == -1:
        for i, (start, end, _) in enumerate(word_positions):
            if start > position:
                target_index = i
                break
        if target_index == -1:
            target_index = len(word_positions) - 1
    
    # 获取前后n个单词
    start_index = max(0, target_index - n_words)
    end_index = min(len(word_positions), target_index + n_words + 1)
    
    before_words = [word for _, _, word in word_positions[start_index:target_index]]
    current_word = word_positions[target_index][2] if target_index < len(word_positions) else ""
    after_words = [word for _, _, word in word_positions[target_index+1:end_index]]
    
    return before_words, current_word, after_words

def extract_months(text):
    """提取文本中的月份名称，使用基于单词的上下文规则进行严格过滤"""
    found_months = []
    text_lower = text.lower()
    
    # 严格过滤的月份
    strict_months = ["may", "march", "august"]
    
    for standard_month, variants in MONTH_VARIANTS.items():
        for variant in variants:
            variant_lower = variant.lower()
            
            # 简单检查该变体是否出现在文本中
            if not re.search(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                continue
            
            # 对需要严格过滤的月份应用特殊规则
            if variant_lower in strict_months:
                is_month = False
                
                # 查找所有匹配位置
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # 获取单词级上下文（前后各3个单词）
                    before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                    
                    # 排除特殊情况 - Mar-a-Lago
                    if variant_lower == "mar" and "a-lago" in " ".join(after_words):
                        continue
                    
                    # 条件1: 介词+月份格式
                    if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                        is_month = True
                        break
                    
                    # 条件2: 月份+数字格式
                    if after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                        is_month = True
                        break
                    
                    # 条件3: 月份+年份格式
                    if after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                        is_month = True
                        break
                    
                    # # 条件4: 年份+月份格式 (新增)
                    # if before_words and re.match(r'(19|20)\d{2}', before_words[-1]):
                    #     is_month = True
                    #     break
                
                # 如果没有找到符合条件的证据，跳过这个月份
                if not is_month:
                    continue
            else:
                # 对于其他月份，使用较宽松的规则
                is_month = True  # 默认接受非歧义月份
            
            # 添加到找到的月份列表
            found_months.append(standard_month)
            break  # 找到一个变体就停止检查其他变体
    
    return found_months

def count_time_entity_distribution(events):
    """统计每条新闻中时间实体出现次数的分布，确保每次月份出现都经过严格验证"""
    
    # 用于存储每条新闻中时间实体出现次数
    year_counts_per_article = []
    month_counts_per_article = []  # 月份出现的总次数（严格验证）
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # 合并标题和摘要以便分析
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # 提取年份
        years = extract_years(full_text)
        year_counts_per_article.append(len(years))
        
        # 统计月份出现次数（使用严格规则）
        month_count = 0
        
        # 遍历所有月份及其变体
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # 找到该变体的所有匹配位置
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # 应用验证规则
                    is_valid_month = False
                    
                    # 严格月份需要特殊处理
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # 获取单词级上下文
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # 排除Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # 验证规则
                        # 1. 介词+月份
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. 月份+数字
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. 月份+年份
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                        
                        # 4. 年份+月份(可选激活)
                        # elif before_words and re.match(r'(19|20)\d{2}', before_words[-1]):
                        #     is_valid_month = True
                    else:
                        # 非严格月份可以直接计入
                        is_valid_month = True
                    
                    # 如果验证通过，计数加1
                    if is_valid_month:
                        month_count += 1
        
        # 记录此条新闻中的有效月份次数
        month_counts_per_article.append(month_count)
    
    # 统计分布
    year_count_distribution = Counter(year_counts_per_article)
    month_count_distribution = Counter(month_counts_per_article)
    
    return year_count_distribution, month_count_distribution

def count_combined_time_entity_distribution(events):
    """统计每条新闻中时间实体(年份+月份)总出现次数的分布"""
    
    # 用于存储每条新闻中时间实体总出现次数
    combined_counts_per_article = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # 合并标题和摘要以便分析
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # 提取年份
        years = extract_years(full_text)
        year_count = len(years)
        
        # 统计月份出现次数（使用严格规则）
        month_count = 0
        
        # 遍历所有月份及其变体
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # 找到该变体的所有匹配位置
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # 应用验证规则
                    is_valid_month = False
                    
                    # 严格月份需要特殊处理
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # 获取单词级上下文
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # 排除Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # 验证规则
                        # 1. 介词+月份
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. 月份+数字
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. 月份+年份
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    else:
                        # 非严格月份可以直接计入
                        is_valid_month = True
                    
                    # 如果验证通过，计数加1
                    if is_valid_month:
                        month_count += 1
        
        # 记录此条新闻中年份和月份的总次数
        combined_counts_per_article.append(year_count + month_count)
    
    # 统计分布
    combined_count_distribution = Counter(combined_counts_per_article)
    
    return combined_count_distribution

def analyze_time_entities(events):
    """分析事件中的时间实体"""
    year_counter = Counter()
    month_counter = Counter()
    
    # 记录包含时间实体的事件数
    events_with_years = 0
    events_with_months = 0
    
    # 保存包含时间实体的示例
    examples_with_years = []
    examples_with_months = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # 合并标题和摘要以便分析
        full_text = headline + " " + abstract
        
        # 提取年份和月份
        years = extract_years(full_text)
        months = extract_months(full_text)
        
        # 更新计数器
        year_counter.update(years)
        month_counter.update(months)
        
        # 记录包含时间实体的事件数
        if years:
            events_with_years += 1
            # 存储所有例子，不再限制数量
            examples_with_years.append({
                "text": full_text,
                "years": years
            })
        if months:
            events_with_months += 1
            # 存储所有例子，不再限制数量
            examples_with_months.append({
                "text": full_text,
                "months": months
            })

        # # 记录包含时间实体的事件数
        # if years:
        #     events_with_years += 1
        #     if len(examples_with_years) < 5:  # 保存前5个示例
        #         examples_with_years.append({
        #             "text": full_text,
        #             "years": years
        #         })
        # if months:
        #     events_with_months += 1
        #     if len(examples_with_months) < 5:  # 保存前5个示例
        #         examples_with_months.append({
        #             "text": full_text,
        #             "months": months
        #         })
    
    return year_counter, month_counter, events_with_years, events_with_months, examples_with_years, examples_with_months

def print_results(train_years, train_months, test_years, test_months):
    """打印统计结果"""
    print("=== 年份统计 ===")
    print("训练集:")
    for year, count in sorted(train_years.items()):
        print(f"  {year}: {count}")
    
    print("\n测试集:")
    for year, count in sorted(test_years.items()):
        print(f"  {year}: {count}")
    
    print("\n=== 月份统计 ===")
    print("训练集:")
    for month in MONTHS:
        print(f"  {month}: {train_months.get(month, 0)}")
    
    print("\n测试集:")
    for month in MONTHS:
        print(f"  {month}: {test_months.get(month, 0)}")

# 在main函数中添加，print_results之后
def print_random_examples(examples_with_years, examples_with_months, count=100):
    """随机打印指定数量的年份和月份识别示例"""
    print(f"\n=== 随机年份识别示例 ({count}个) ===")
    if len(examples_with_years) > 0:
        # 随机选择最多count个例子
        sample_size = min(count, len(examples_with_years))
        sampled_year_examples = random.sample(examples_with_years, sample_size)
        
        for i, example in enumerate(sampled_year_examples, 1):
            # 截断过长的文本以提高可读性
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"示例 {i}/{sample_size}:")
            print(f"文本: {text_sample}")
            print(f"识别到的年份: {example['years']}")
            
            # 打印上下文，帮助验证识别正确性
            for year in example['years']:
                # 查找年份周围的上下文
                year_pos = example['text'].find(year)
                if year_pos != -1:
                    context_start = max(0, year_pos - 30)
                    context_end = min(len(example['text']), year_pos + len(year) + 30)
                    print(f"  '{year}'上下文: ...{example['text'][context_start:context_end]}...")
            print()
    else:
        print("没有找到包含年份的例子")
        
    print(f"\n=== 随机月份识别示例 ({count}个) ===")
    if len(examples_with_months) > 0:
        # 随机选择最多count个例子
        sample_size = min(count, len(examples_with_months))
        sampled_month_examples = random.sample(examples_with_months, sample_size)
        
        for i, example in enumerate(sampled_month_examples, 1):
            # 截断过长的文本以提高可读性
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"示例 {i}/{sample_size}:")
            print(f"文本: {text_sample}")
            print(f"识别到的月份: {example['months']}")
            
            # 打印上下文，帮助验证识别正确性
            for month in example['months']:
                # 尝试查找月份及其变体周围的上下文
                variants = MONTH_VARIANTS[month]
                for variant in variants:
                    if variant.lower() in example['text'].lower():
                        pos = example['text'].lower().find(variant.lower())
                        if pos != -1:
                            context_start = max(0, pos - 30)
                            context_end = min(len(example['text']), pos + len(variant) + 30)
                            print(f"  '{month}'上下文: ...{example['text'][context_start:context_end]}...")
                            break
            print()
    else:
        print("没有找到包含月份的例子")

def print_specific_month_examples(examples_with_months, month_name, count=30):
    """打印特定月份的随机示例"""
    # 筛选包含指定月份的例子
    matching_examples = [example for example in examples_with_months 
                        if month_name in example['months']]
    
    print(f"\n=== 随机 {month_name} 识别示例 ({count}个) ===")
    if len(matching_examples) > 0:
        # 随机选择最多count个例子
        sample_size = min(count, len(matching_examples))
        sampled_examples = random.sample(matching_examples, sample_size)
        
        for i, example in enumerate(sampled_examples, 1):
            # 截断过长的文本以提高可读性
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"示例 {i}/{sample_size}:")
            print(f"文本: {text_sample}")
            
            # 打印上下文，帮助验证识别正确性
            # 尝试查找月份及其变体周围的上下文
            variants = MONTH_VARIANTS[month_name]
            for variant in variants:
                if variant.lower() in example['text'].lower():
                    pos = example['text'].lower().find(variant.lower())
                    if pos != -1:
                        context_start = max(0, pos - 30)
                        context_end = min(len(example['text']), pos + len(variant) + 30)
                        print(f"  '{month_name}'上下文: ...{example['text'][context_start:context_end]}...")
                        break
            print()
    else:
        print(f"没有找到包含月份 {month_name} 的例子")

def create_masked_dataset0(events, output_file):
    """对事件集合进行时间实体掩码处理并保存为jsonl格式"""
    
    masked_events = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # 合并标题和摘要用于分析
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # 提取年份和月份及其在文本中的位置
        year_entities = []
        month_entities = []
        
        # 查找年份
        for match in re.finditer(r'\b(19\d\d|20\d\d)\b', full_text):
            year_value = match.group(1)
            start, end = match.span()
            
            # 验证是否为实际年份
            is_year = True
            
            # 获取上下文
            context_start = max(0, start - 15)
            context_end = min(len(full_text), end + 15)
            context = full_text[context_start:context_end].lower()
            
            # 简单验证上下文
            if any(word in context for word in ["year", "in", "on", "of", "since", "during", "before", "after"]) or \
               any(month.lower() in context for month in MONTHS) or \
               any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"]):
                year_entities.append({
                    "value": year_value,
                    "start_in_full": start,
                    "end_in_full": end,
                    "type": "year"
                })
        
        # 查找月份
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                for match in re.finditer(r'\b' + re.escape(variant) + r'\b', full_text, re.IGNORECASE):
                    start, end = match.span()
                    
                    # 验证是否为真实月份，特别是对于有歧义的月份
                    is_valid_month = True
                    variant_lower = variant.lower()
                    
                    if variant_lower in ["may", "march", "august"]:
                        # 获取上下文
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # 排除Mar-a-Lago等特殊情况
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            is_valid_month = False
                            continue
                        
                        # 应用验证规则
                        is_valid_month = False
                        
                        # 验证规则：介词+月份、月份+数字、月份+年份
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    
                    if is_valid_month:
                        # 确定实际月份在哪个部分(headline或abstract)
                        if start < len(headline):
                            # 在标题中
                            start_in_component = start
                            component = "headline"
                        else:
                            # 在摘要中
                            start_in_component = start - len(headline) - 1  # 减1是因为中间有空格
                            component = "abstract"
                        
                        end_in_component = start_in_component + (end - start)
                        
                        month_entities.append({
                            "value": standard_month,
                            "start_in_full": start,
                            "end_in_full": end,
                            "start_in_component": start_in_component,
                            "end_in_component": end_in_component,
                            "component": component,
                            "type": "month"
                        })
                    
                    # 找到一个有效月份变体后就跳过该月份的其他变体
                    if is_valid_month:
                        break
        
        # 合并所有时间实体
        all_entities = year_entities + month_entities
        
        # 如果没有时间实体，则跳过该事件
        if not all_entities:
            # masked_events.append(event)
            continue
        
        # 随机选择一个时间实体进行掩码
        entity_to_mask = random.choice(all_entities)
        
        # 创建掩码后的新闻对象
        masked_event = event.copy()
        
        # 确定掩码标记
        mask_token = "<YEAR>" if entity_to_mask["type"] == "year" else "<MONTH>"
        
        # 应用掩码
        if "start_in_component" in entity_to_mask:
            # 月份情况 - 已确定在哪个组件中
            component = entity_to_mask["component"]
            start = entity_to_mask["start_in_component"]
            end = entity_to_mask["end_in_component"]
            
            text = masked_event[component]
            masked_text = text[:start] + mask_token + text[end:]
            masked_event[component] = masked_text
        else:
            # 年份情况 - 需要确定在哪个组件中
            start = entity_to_mask["start_in_full"]
            end = entity_to_mask["end_in_full"]
            
            if start < len(headline):
                # 在标题中
                masked_text = headline[:start] + mask_token + headline[end:]
                masked_event["headline"] = masked_text
            else:
                # 在摘要中
                start_in_abstract = start - len(headline) - 1
                end_in_abstract = end - len(headline) - 1
                masked_text = abstract[:start_in_abstract] + mask_token + abstract[end_in_abstract:]
                masked_event["abstract"] = masked_text
        
        # 添加掩码信息
        masked_event["masked_info"] = entity_to_mask["value"]
        masked_event["mask_type"] = entity_to_mask["type"]  # 可选：记录掩码类型
        
        masked_events.append(masked_event)
    
    # 保存为jsonl格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in masked_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"已保存掩码处理后的数据集到 {output_file}，共 {len(masked_events)} 条记录")
    return masked_events

def create_masked_dataset(events, output_file):
    """对事件集合进行时间实体掩码处理并保存为jsonl格式"""
    
    masked_events = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # 合并标题和摘要用于分析
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # 使用统一的函数提取年份和月份
        years = extract_years(full_text)
        
        # 提取年份和月份及其在文本中的位置
        year_entities = []
        month_entities = []
        
        # 查找年份及其位置
        for year in years:
            # 对每个年份找到其在文本中的位置
            for match in re.finditer(r'\b' + re.escape(year) + r'\b', full_text):
                start, end = match.span()
                
                # 确定年份在哪个组件中
                if start < len(headline):
                    # 在标题中
                    start_in_component = start
                    end_in_component = end
                    component = "headline"
                else:
                    # 在摘要中
                    start_in_component = start - len(headline) - 1
                    end_in_component = end - len(headline) - 1
                    component = "abstract"
                
                year_entities.append({
                    "value": year,
                    "start_in_full": start,
                    "end_in_full": end,
                    "start_in_component": start_in_component,
                    "end_in_component": end_in_component,
                    "component": component,
                    "type": "year"
                })
        
        # 查找月份及其位置
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # 找到该变体的所有匹配位置
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # 应用验证规则
                    is_valid_month = False
                    
                    # 严格月份需要特殊处理
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # 获取单词级上下文
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # 排除Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # 验证规则
                        # 1. 介词+月份
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. 月份+数字
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. 月份+年份
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    else:
                        # 非严格月份可以直接计入
                        is_valid_month = True
                    
                    # 如果验证通过，添加到月份实体列表
                    if is_valid_month:
                        # 确定实际月份在哪个部分(headline或abstract)
                        if start < len(headline):
                            # 在标题中
                            start_in_component = start
                            end_in_component = end
                            component = "headline"
                        else:
                            # 在摘要中
                            start_in_component = start - len(headline) - 1  # 减1是因为中间有空格
                            end_in_component = end - len(headline) - 1
                            component = "abstract"
                        
                        month_entities.append({
                            "value": standard_month,
                            "start_in_full": start,
                            "end_in_full": end,
                            "start_in_component": start_in_component,
                            "end_in_component": end_in_component,
                            "component": component,
                            "type": "month"
                        })
                
                # # 一个月份的所有匹配都处理完后，跳出内层循环
                # break
        
        # 合并所有时间实体
        all_entities = year_entities + month_entities
        
        # 如果没有时间实体，跳过该事件
        if not all_entities:
            continue
        
        # 随机选择一个时间实体进行掩码
        entity_to_mask = random.choice(all_entities)
        
        # 创建掩码后的新闻对象
        masked_event = event.copy()
        
        # 确定掩码标记
        mask_token = "<YEAR>" if entity_to_mask["type"] == "year" else "<MONTH>"
        
        # 应用掩码
        component = entity_to_mask["component"]
        start = entity_to_mask["start_in_component"]
        end = entity_to_mask["end_in_component"]
        
        text = masked_event[component]
        masked_text = text[:start] + mask_token + text[end:]
        masked_event[component] = masked_text
        
        # 添加掩码信息
        masked_event["masked_info"] = entity_to_mask["value"]
        masked_event["mask_type"] = entity_to_mask["type"]  # 可选：记录掩码类型
        
        masked_events.append(masked_event)
    
    # 保存为jsonl格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in masked_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"已保存掩码处理后的数据集到 {output_file}，共 {len(masked_events)} 条记录")
    return masked_events

def main():
    # 使用与event_order_nyt.py相同的数据加载逻辑
    input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"
    years = range(2016, 2024)  # Using data from 2016 to 2023
    
    # 检查是否存在已保存的训练/测试数据划分
    split_cache_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_train_test_split.pkl"
    
    if os.path.exists(split_cache_file):
        # 如果存在缓存文件，直接加载已分割的数据
        print(f"Loading train/test split from cache: {split_cache_file}")
        with open(split_cache_file, 'rb') as f:
            train_events, test_events = pickle.load(f)
    else:
        # 如果不存在缓存文件，创建并保存分割
        print(f"Creating new train/test split and saving to: {split_cache_file}")
        train_events, test_events = load_events(input_dir, years)
        
        # 保存分割结果供未来使用
        with open(split_cache_file, 'wb') as f:
            pickle.dump((train_events, test_events), f)
    
    # 在统计输出之后添加
    print("\n=== 创建时间实体掩码数据集 ===")
    
    # 创建输出目录
    masked_output_dir = "/data/zliu331/temporal_reasoning/TinyZero/datasets/masked_time_entity"
    os.makedirs(masked_output_dir, exist_ok=True)
    
    # 处理训练集和测试集
    train_output_file = os.path.join(masked_output_dir, "train_masked.jsonl")
    test_output_file = os.path.join(masked_output_dir, "test_masked.jsonl")
    
    create_masked_dataset(train_events, train_output_file)
    create_masked_dataset(test_events, test_output_file)
    
    print(f"\n掩码数据集已保存到 {masked_output_dir}")



    # print(f"分析 {len(train_events)} 个训练事件和 {len(test_events)} 个测试事件...")
    
    # train_results = analyze_time_entities(train_events)
    # test_results = analyze_time_entities(test_events)
    
    # train_years, train_months, train_events_with_years, train_events_with_months, train_year_examples, train_month_examples = train_results
    # test_years, test_months, test_events_with_years, test_events_with_months, test_year_examples, test_month_examples = test_results

    # # 打印结果
    # print_results(train_years, train_months, test_years, test_months)
    
    # # # 保存结果
    # # output_dir = "/data/zliu331/temporal_reasoning/TinyZero/datasets/time_entity_stats"
    # # os.makedirs(output_dir, exist_ok=True)
    
    # # output_file = os.path.join(output_dir, "time_entity_statistics.pkl")
    
    # # results = {
    # #     'train_years': dict(train_years),
    # #     'train_months': dict(train_months),
    # #     'test_years': dict(test_years),
    # #     'test_months': dict(test_months)
    # # }
    
    # # with open(output_file, 'wb') as f:
    # #     pickle.dump(results, f)
    
    # # print(f"\n结果已保存到 {output_file}")
    
    # # 输出总体统计信息
    # print("\n=== 总体统计 ===")
    # print(f"训练集中提到年份的总次数: {sum(train_years.values())}")
    # print(f"训练集中提到月份的总次数: {sum(train_months.values())}")
    # print(f"测试集中提到年份的总次数: {sum(test_years.values())}")
    # print(f"测试集中提到月份的总次数: {sum(test_months.values())}")
    
    # print(f"\n训练集中包含年份的事件数: {train_events_with_years} ({train_events_with_years/len(train_events)*100:.2f}%)")
    # print(f"训练集中包含月份的事件数: {train_events_with_months} ({train_events_with_months/len(train_events)*100:.2f}%)")
    # print(f"测试集中包含年份的事件数: {test_events_with_years} ({test_events_with_years/len(test_events)*100:.2f}%)")
    # print(f"测试集中包含月份的事件数: {test_events_with_months} ({test_events_with_months/len(test_events)*100:.2f}%)")
    
    # print("\n最常见的年份:")
    # for year, count in train_years.most_common(5):
    #     print(f"  {year}: {count}")
    
    # print("\n最常见的月份:")
    # for month, count in train_months.most_common(5):
    #     print(f"  {month}: {count}")

    # # # 打印识别示例
    # # print("\n=== 年份识别示例 ===")
    # # for i, example in enumerate(train_year_examples[:3], 1):
    # #     print(f"示例 {i}:")
    # #     print(f"文本: {example['text']}...")
    # #     print(f"识别到的年份: {example['years']}")
    # #     print()
    
    # # print("\n=== 月份识别示例 ===")
    # # for i, example in enumerate(train_month_examples[:3], 1):
    # #     print(f"示例 {i}:")
    # #     print(f"文本: {example['text']}...")
    # #     print(f"识别到的月份: {example['months']}")
    # #     print()
    # # print("\n正在随机选择和打印时间实体识别示例...")
    # # print_random_examples(train_year_examples, train_month_examples, count=100) 


    # print("\n正在选择和打印特定月份的识别示例...")
    # print_specific_month_examples(train_month_examples, "June", count=30)  
    # print_specific_month_examples(train_month_examples, "July", count=30)


    # print("\n=== 每条新闻中时间实体(年份+月份)总出现次数统计 ===")
    # train_combined_dist = count_combined_time_entity_distribution(train_events)
    # test_combined_dist = count_combined_time_entity_distribution(test_events)

    # print("\n每条新闻中时间实体(年份+月份)总出现次数分布:")
    # print("训练集:")
    # for count in sorted(train_combined_dist.keys()):
    #     print(f"  出现{count}次时间实体的新闻数: {train_combined_dist[count]} ({train_combined_dist[count]/len(train_events)*100:.2f}%)")

    # print("\n测试集:")
    # for count in sorted(test_combined_dist.keys()):
    #     print(f"  出现{count}次时间实体的新闻数: {test_combined_dist[count]} ({test_combined_dist[count]/len(test_events)*100:.2f}%)")

    # # print("\n=== 每条新闻中时间实体出现次数统计 ===")
    # # train_year_dist, train_month_dist = count_time_entity_distribution(train_events)
    # # test_year_dist, test_month_dist = count_time_entity_distribution(test_events)
    
    # # print("\n每条新闻中年份出现次数分布:")
    # # print("训练集:")
    # # for count in sorted(train_year_dist.keys()):
    # #     print(f"  出现{count}次年份的新闻数: {train_year_dist[count]} ({train_year_dist[count]/len(train_events)*100:.2f}%)")
    
    # # print("\n测试集:")
    # # for count in sorted(test_year_dist.keys()):
    # #     print(f"  出现{count}次年份的新闻数: {test_year_dist[count]} ({test_year_dist[count]/len(test_events)*100:.2f}%)")
    
    # # print("\n每条新闻中月份出现次数分布(严格验证后):")
    # # print("训练集:")
    # # for count in sorted(train_month_dist.keys()):
    # #     print(f"  出现{count}次月份的新闻数: {train_month_dist[count]} ({train_month_dist[count]/len(train_events)*100:.2f}%)")
    
    # # print("\n测试集:")
    # # for count in sorted(test_month_dist.keys()):
    # #     print(f"  出现{count}次月份的新闻数: {test_month_dist[count]} ({test_month_dist[count]/len(test_events)*100:.2f}%)")


def print_random_masked_examples(file_path, count=30):
    """从掩码数据集中随机打印指定数量的示例"""
    # 读取JSONL文件
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"文件共有 {len(examples)} 条记录")
    
    print(examples[0], examples[1])
    # # 随机选择指定数量的示例
    # sample_count = min(count, len(examples))
    # sampled_examples = random.sample(examples, sample_count)
    
    # # 打印选中的示例
    # for i, example in enumerate(sampled_examples, 1):
    #     headline = example.get("headline", "")
    #     abstract = example.get("abstract", "")
    #     masked_info = example.get("masked_info", "")
    #     mask_type = example.get("mask_type", "")
        
    #     print(f"\n示例 {i}/{sample_count}:")
    #     print(f"标题: {headline}")
    #     print(f"摘要: {abstract}")
    #     print(f"掩码信息: {masked_info}")
    #     print(f"掩码类型: {mask_type}")
    #     print("-" * 80)



if __name__ == "__main__":
    # main()

    train_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/masked_time_entity/train_masked.jsonl"
    test_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/masked_time_entity/test_masked.jsonl"

    print("=== 训练集随机示例 ===")
    print_random_masked_examples(train_file, count=30)

    print("\n=== 测试集随机示例 ===")
    print_random_masked_examples(test_file, count=30)