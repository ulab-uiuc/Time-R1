import os
import re
import json
import random
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

def construct_prefix(event):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Please carefully read the following news article information:\n"
        f"Headline: {event['headline']}\n"
        f"Abstract: {event['abstract']}\n"
        "For the purpose of this prediction, assume that the event described in the article definitely will occur within the next few months or years. "
        "Based on the information provided and your general knowledge, determine the most likely specific future occurrence date of the event.\n"
        "- You can recall relevant and similar events in the past and their occurrence dates and identify the development patterns to help you predict.\n"
        "- Output the event's predicted occurrence date in the format 'YYYY-MM'.\n"
        "- Do not output 'No event' under any circumstances. Always provide your best prediction, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>2025-03</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def parse_year_month_from_true(date_str: str):
    """
    Parse a date string in the format 'YYYY-MM'. Returns (year, month) or (None, None) if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def month_difference(year1, month1, year2, month2):
    """
    计算两个日期之间的月份差值
    """
    return (year2 - year1) * 12 + (month2 - month1)

def calculate_time_distance(event_date, reference_date="2024-01"):
    """
    计算事件日期与参考日期的距离（以月为单位）
    正值表示事件在参考日期之后，负值表示事件在参考日期之前
    """
    ref_year, ref_month = parse_year_month_from_true(reference_date)
    event_year, event_month = event_date["year"], event_date["month"]
    
    return month_difference(ref_year, ref_month, event_year, event_month)

def load_events_by_year_month(input_dir, years_range=(2024, 2026)):
    """
    加载指定年份范围的事件，并按年月组织
    返回一个字典，格式为: {(year, month): [events]}
    """
    events_by_year_month = defaultdict(list)
    
    for year in range(years_range[0], years_range[1]):
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
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
                events_by_year_month[(year_val, month_val)].append(event)
    
    return events_by_year_month

def calculate_sampling_rate(month, start_month=7, end_month=10, start_rate=1.0, end_rate=0.5):
    """
    计算过渡层的采样率，随月份增加线性递减
    """
    if month < start_month:
        return start_rate
    if month > end_month:
        return end_rate
    
    # 线性插值计算采样率
    progress = (month - start_month) / (end_month - start_month)
    return start_rate - progress * (start_rate - end_rate)

def build_time_prediction_datasets(input_dir, output_train_file, output_test_file):
    """
    构建时间预测任务的训练集和测试集
    按照分层策略:
    - 基础层（70%）：24年1-6月完整数据
    - 过渡层（20%）：24年7-10月采样数据（月份越靠后采样率越低）
    - 示例层（10%）：24年11月-25年2月少量精选数据（各占总训练集的3.33%）
    """
    # 加载2024-2025年的所有事件数据
    events_by_year_month = load_events_by_year_month(input_dir, (2024, 2026))
    
    # 统计每个月的数据量
    stats = {ym: len(events) for ym, events in events_by_year_month.items()}
    print("数据统计:", stats)
    
    # 计算基础层(2024年1-6月)的总数量
    base_layer_count = 0
    for year in [2024]:
        for month in range(1, 7):
            base_layer_count += len(events_by_year_month.get((year, month), []))
    
    # 根据基础层数量计算其他层的目标数量
    target_total = base_layer_count / 0.7  # 基础层占70%
    transition_layer_target = target_total * 0.2  # 过渡层占20%
    example_layer_target = target_total * 0.1  # 示例层占10%
    
    print(f"基础层数量: {base_layer_count}")
    print(f"过渡层目标数量: {transition_layer_target}")
    print(f"示例层目标数量: {example_layer_target}")
    
    # 初始化训练集和测试集
    train_samples = []
    test_samples = []
    index_counter = 0
    
    # 1. 基础层: 2024年1-6月全部数据
    for month in range(1, 7):
        for event in events_by_year_month.get((2024, month), []):
            # 计算与参考日期(2024-01)的时间距离
            time_distance = calculate_time_distance(event)
            
            prefix = construct_prefix(event)
            ground_truth = {
                "event_pub_date": event["true_pub_date"],
                "time_distance": time_distance
            }
            
            sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": prefix
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": "train",
                    "index": index_counter,
                    "layer": "base",
                    "time_distance": time_distance,
                    "year": event["year"],
                    "month": event["month"]
                }
            }
            train_samples.append(sample)
            index_counter += 1
    
    # 2. 过渡层: 2024年7-10月，采样率随月份递减
    transition_samples = []
    # for month in range(7, 11):
    #     sampling_rate = calculate_sampling_rate(month)
    #     print(f"月份 {month} 的采样率: {sampling_rate}")
    #     events = events_by_year_month.get((2024, month), [])
        
    #     # 计算本月需要采样的数量
    #     sample_count = int(len(events) * sampling_rate)
    #     if sample_count > 0:
    #         sampled_events = random.sample(events, sample_count)

    # 预先计算每个月的目标采样量
    # total_events_in_transition = sum(len(events_by_year_month.get((2024, m), [])) for m in range(7, 11))
    monthly_weights = {m: calculate_sampling_rate(m) for m in range(7, 11)}
    weight_sum = sum(monthly_weights.values())

    for month in range(7, 11):
        events = events_by_year_month.get((2024, month), [])
        # 按权重比例计算目标样本数
        target_count = int(transition_layer_target * monthly_weights[month] / weight_sum)
        target_count = min(target_count, len(events))  # 不能超过实际有的数量
        
        # 直接采样正确数量
        sampled_events = random.sample(events, target_count)
            
        # 将采样的事件添加到过渡层
        for event in sampled_events:
            time_distance = calculate_time_distance(event)
            prefix = construct_prefix(event)
            ground_truth = {
                "event_pub_date": event["true_pub_date"],
                "time_distance": time_distance
            }
            
            sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": prefix
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": "train",
                    "index": index_counter,
                    "layer": "transition",
                    "time_distance": time_distance,
                    "year": event["year"],
                    "month": event["month"]
                }
            }
            transition_samples.append(sample)
            index_counter += 1
        
        # 将未采样的事件添加到测试集
        for event in [e for e in events if e not in sampled_events]:
            time_distance = calculate_time_distance(event)
            prefix = construct_prefix(event)
            ground_truth = {
                "event_pub_date": event["true_pub_date"],
                "time_distance": time_distance
            }
            
            sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": prefix
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": "test",
                    "index": len(test_samples),
                    "time_distance": time_distance,
                    "year": event["year"],
                    "month": event["month"]
                }
            }
            test_samples.append(sample)
    
    # # 调整过渡层样本数量，如果超出目标，进行随机下采样
    # if len(transition_samples) > transition_layer_target:
    #     transition_samples = random.sample(transition_samples, int(transition_layer_target))
    
    # 3. 示例层: 2024年11月-2025年2月，每个月份平均分配示例层的目标数量
    example_months = [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    samples_per_month = example_layer_target / len(example_months)
    
    example_samples = []
    for year, month in example_months:
        events = events_by_year_month.get((year, month), [])
        if not events:
            continue
            
        # 每个月份采样相同比例的数据
        sample_count = min(len(events), int(samples_per_month))
        if sample_count > 0:
            sampled_events = random.sample(events, sample_count)
            
            # 将采样的事件添加到示例层
            for event in sampled_events:
                time_distance = calculate_time_distance(event)
                prefix = construct_prefix(event)
                ground_truth = {
                    "event_pub_date": event["true_pub_date"],
                    "time_distance": time_distance
                }
                
                sample = {
                    "data_source": "new_york_times",
                    "prompt": [{
                        "role": "user",
                        "content": prefix
                    }],
                    "ability": "news_time_prediction",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth
                    },
                    "extra_info": {
                        "split": "train",
                        "index": index_counter,
                        "layer": "example",
                        "time_distance": time_distance,
                        "year": event["year"],
                        "month": event["month"]
                    }
                }
                example_samples.append(sample)
                index_counter += 1
            
            # 将未采样的事件添加到测试集
            for event in [e for e in events if e not in sampled_events]:
                time_distance = calculate_time_distance(event)
                prefix = construct_prefix(event)
                ground_truth = {
                    "event_pub_date": event["true_pub_date"],
                    "time_distance": time_distance
                }
                
                sample = {
                    "data_source": "new_york_times",
                    "prompt": [{
                        "role": "user",
                        "content": prefix
                    }],
                    "ability": "news_time_prediction",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth
                    },
                    "extra_info": {
                        "split": "test",
                        "index": len(test_samples),
                        "time_distance": time_distance,
                        "year": event["year"],
                        "month": event["month"]
                    }
                }
                test_samples.append(sample)
    
    # 合并所有训练样本
    train_samples.extend(transition_samples)
    train_samples.extend(example_samples)
    
    # 随机打乱训练集和测试集
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    # 统计各层数量与比例
    base_count = sum(1 for s in train_samples if s["extra_info"].get("layer") == "base")
    transition_count = sum(1 for s in train_samples if s["extra_info"].get("layer") == "transition")
    example_count = sum(1 for s in train_samples if s["extra_info"].get("layer") == "example")
    total_count = len(train_samples)
    
    print(f"训练集总样本数: {total_count}")
    print(f"基础层: {base_count} ({base_count/total_count*100:.2f}%)")
    print(f"过渡层: {transition_count} ({transition_count/total_count*100:.2f}%)")
    print(f"示例层: {example_count} ({example_count/total_count*100:.2f}%)")
    print(f"测试集样本数: {len(test_samples)}")
    
    # 检查月份分布
    train_month_dist = defaultdict(int)
    test_month_dist = defaultdict(int)
    
    for sample in train_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        train_month_dist[(year, month)] += 1
    
    for sample in test_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        test_month_dist[(year, month)] += 1
    
    print("\n训练集月份分布:")
    for (year, month), count in sorted(train_month_dist.items()):
        print(f"{year}-{month:02d}: {count} ({count/total_count*100:.2f}%)")
    
    print("\n测试集月份分布:")
    for (year, month), count in sorted(test_month_dist.items()):
        print(f"{year}-{month:02d}: {count} ({count/len(test_samples)*100:.2f}%)")
    
    # 保存为 Parquet 文件
    df_train = pd.DataFrame(train_samples)
    df_test = pd.DataFrame(test_samples)
    
    df_train.to_parquet(output_train_file, index=False)
    df_test.to_parquet(output_test_file, index=False)
    
    print(f"已保存训练集到: {output_train_file}")
    print(f"已保存测试集到: {output_test_file}")
    
    return train_samples, test_samples

def create_small_test_set(test_file_path, output_file_path, sample_count=1024, random_seed=1024):
    """
    从测试集中抽取指定数量的样本，确保每个月份的样本数量相等
    
    参数:
        test_file_path: 原始测试集的路径
        output_file_path: 输出的小测试集路径
        sample_count: 需要抽取的样本总数，默认1024
        random_seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取原始测试集
    df_test = pd.read_parquet(test_file_path)
    test_data = df_test.to_dict('records')
    
    # 按月份分组
    samples_by_month = defaultdict(list)
    for sample in test_data:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        samples_by_month[(year, month)].append(sample)
    
    # 确认有哪些月份
    available_months = sorted(samples_by_month.keys())
    print(f"测试集中包含的月份: {available_months}")
    
    # 计算每个月份需要的样本数
    months_count = len(available_months)  # 应该是8个月份
    samples_per_month = sample_count // months_count  # 每个月份128条
    
    print(f"每个月份需要抽取 {samples_per_month} 条数据")
    
    # 抽取样本
    selected_samples = []
    insufficient_months = []
    
    for month_key in available_months:
        month_samples = samples_by_month[month_key]
        
        if len(month_samples) >= samples_per_month:
            # 如果样本足够，随机抽取指定数量
            month_selected = random.sample(month_samples, samples_per_month)
        else:
            # 如果样本不足，全部使用并记录不足的月份
            month_selected = month_samples
            insufficient_months.append((month_key, samples_per_month - len(month_samples)))
            print(f"警告: {month_key}月样本不足，只有{len(month_samples)}条，还需{samples_per_month - len(month_samples)}条")
        
        selected_samples.extend(month_selected)
    
    # 处理样本不足的情况
    if insufficient_months:
        shortage = sum(shortage for _, shortage in insufficient_months)
        print(f"总计缺少 {shortage} 条样本，将从其他月份补充")
        
        # 找出有富余样本的月份
        surplus_months = []
        for month_key in available_months:
            remaining = len(samples_by_month[month_key]) - samples_per_month
            if remaining > 0:
                surplus_months.append((month_key, remaining))
        
        # 计算从每个有富余样本的月份中额外抽取的数量
        extra_samples = []
        if surplus_months:
            # 按富余样本比例分配需要额外抽取的数量
            total_surplus = sum(surplus for _, surplus in surplus_months)
            
            for month_key, surplus in surplus_months:
                # 计算这个月需要额外抽取多少样本
                extra_count = min(shortage * surplus // total_surplus, surplus)
                if extra_count > 0:
                    # 已选样本
                    already_selected = [s for s in selected_samples if 
                                       s["extra_info"]["year"] == month_key[0] and 
                                       s["extra_info"]["month"] == month_key[1]]
                    
                    # 可选样本 (排除已选样本)
                    available = [s for s in samples_by_month[month_key] if s not in already_selected]
                    
                    # 额外抽样
                    extra = random.sample(available, extra_count)
                    extra_samples.extend(extra)
                    shortage -= extra_count
                    
                    print(f"从 {month_key} 月额外抽取 {extra_count} 条样本")
        
        # 添加额外抽取的样本
        selected_samples.extend(extra_samples)
    
    # 最终确认样本总数
    final_count = len(selected_samples)
    print(f"最终抽取的样本总数: {final_count}")
    
    # 随机打乱顺序
    random.shuffle(selected_samples)
    
    # 统计每个月份最终的样本数
    final_month_dist = defaultdict(int)
    for sample in selected_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        final_month_dist[(year, month)] += 1
    
    print("\n最终月份分布:")
    for month_key in sorted(final_month_dist.keys()):
        count = final_month_dist[month_key]
        print(f"{month_key[0]}-{month_key[1]:02d}: {count} ({count/final_count*100:.2f}%)")
    
    # 保存为parquet文件
    df_small_test = pd.DataFrame(selected_samples)
    df_small_test.to_parquet(output_file_path, index=False)
    print(f"已保存小型测试集到: {output_file_path}")
    
    return selected_samples

def create_march_april_test_set(input_file, output_file, samples_per_month=128, random_seed=1024):
    """
    从2025年3月和4月数据中抽取指定数量的样本，构建测试集
    
    参数:
        input_file: 输入文件路径，包含3月和4月的数据
        output_file: 输出的小测试集路径
        samples_per_month: 每月需要抽取的样本数，默认128
        random_seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 按月份分组存储数据
    samples_by_month = defaultdict(list)
    
    # 读取原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                pub_date = data.get("pub_date", "")
                
                # 尝试解析日期
                if pub_date:
                    dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z")
                    year, month = dt.year, dt.month
                    
                    # 只保留2025年3月和4月的数据
                    if year == 2025 and month in [3, 4]:
                        event = {
                            "headline": data.get("headline", ""),
                            "abstract": data.get("abstract", ""),
                            # "lead_paragraph": data.get("lead_paragraph", ""),
                            "true_pub_date": f"{year}-{month:02d}",
                            "year": year,
                            "month": month
                        }
                        samples_by_month[(year, month)].append(event)
            except Exception as e:
                print(f"处理数据行时出错: {e}")
                continue

    # 如果3月或4月的数据不足，则生成模拟数据补充
    for month in [3, 4]:
        if len(samples_by_month.get((2025, month), [])) < samples_per_month:
            print(f"2025年{month}月数据不足，将生成模拟数据补充")
            needed = samples_per_month - len(samples_by_month.get((2025, month), []))
            
            # 生成模拟数据
            for i in range(needed):
                event = {
                    "headline": f"模拟头条新闻 #2025-{month:02d}-{i+1:03d}",
                    "abstract": f"这是2025年{month}月的模拟新闻摘要 #{i+1:03d}，用于测试时间预测模型。",
                    # "lead_paragraph": f"这是2025年{month}月的模拟新闻导语 #{i+1:03d}，提供更详细的内容用于时间预测测试。",
                    "true_pub_date": f"2025-{month:02d}",
                    "year": 2025,
                    "month": month
                }
                samples_by_month[(2025, month)].append(event)
    
    # 从每个月随机抽取指定数量的样本
    selected_samples = []
    
    for month in [3, 4]:
        month_key = (2025, month)
        month_samples = samples_by_month[month_key]
        
        if len(month_samples) >= samples_per_month:
            # 随机抽取
            month_selected = random.sample(month_samples, samples_per_month)
        else:
            # 如果样本不足，全部使用
            month_selected = month_samples
        
        # 为每个样本添加额外信息
        for i, sample in enumerate(month_selected):
            # 计算与参考日期(2024-01)的时间距离
            time_distance = (sample["year"] - 2024) * 12 + (sample["month"] - 1)
            
            prepared_sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": construct_prefix(sample)
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "event_pub_date": sample["true_pub_date"],
                        "time_distance": time_distance
                    }
                },
                "extra_info": {
                    "split": "test",
                    "index": len(selected_samples),
                    "time_distance": time_distance,
                    "year": sample["year"],
                    "month": sample["month"]
                }
            }
            selected_samples.append(prepared_sample)
    
    # 随机打乱顺序
    random.shuffle(selected_samples)

    # 统计每个月份最终的样本数
    final_month_dist = defaultdict(int)
    for sample in selected_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        final_month_dist[(year, month)] += 1
    
    print("\n最终月份分布:")
    total_count = len(selected_samples)
    for month_key in sorted(final_month_dist.keys()):
        count = final_month_dist[month_key]
        print(f"{month_key[0]}-{month_key[1]:02d}: {count} ({count/total_count*100:.2f}%)")
    
    # 保存为parquet文件
    df_test = pd.DataFrame(selected_samples)
    df_test.to_parquet(output_file, index=False)
    print(f"已保存测试集到: {output_file}")
    
    return selected_samples

# def parse_articles_from_v3_generation(jsonl_file):
#     """从v3生成结果中解析文章"""
#     articles_by_month = defaultdict(list)
    
#     with open(jsonl_file, 'r') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
                
#                 # 获取custom_id并提取月份和新闻类别
#                 custom_id = data.get('custom_id', '')
#                 if not custom_id or '_' not in custom_id:
#                     continue
                    
#                 # month, desk, _ = custom_id.split('_', 2)
#                 date_str, desk, _ = custom_id.split('_', 2)
                
#                 # 直接从date_str (格式如"2025-01")提取年份和月份
#                 if '-' in date_str:
#                     year, month_num = map(int, date_str.split('-'))
#                 else:
#                     print(f"警告: 日期格式不正确: {date_str}")
#                     continue
                
#                 # 获取生成内容
#                 content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                
#                 # 解析各篇文章
#                 article_pattern = r'ARTICLE (\d+):\s*(?:\*\*)?Headline:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Abstract:(?:\*\*)?\s*(.*?)(?=\n\nARTICLE \d+:|$)'
#                 for match in re.finditer(article_pattern, content, re.DOTALL):
#                     article_num = match.group(1)
#                     headline = match.group(2).strip()
#                     abstract = match.group(3).strip()
                    
#                     # 对headline和abstract进行清理
#                     headline = headline.replace('\n', ' ').strip()
#                     abstract = abstract.replace('\n', ' ').strip()
                    
#                     # 去掉可能的markdown格式
#                     headline = re.sub(r'\*\*|\*', '', headline)
#                     abstract = re.sub(r'\*\*|\*', '', abstract)
                    
#                     # 如果headline或abstract为空，跳过此文章
#                     if not headline or not abstract:
#                         continue

#                     # 构建标准格式的true_pub_date
#                     true_pub_date = f"{year}-{month_num:02d}"
                    
#                     # 创建文章记录
#                     article = {
#                         'headline': headline,
#                         'abstract': abstract,
#                         'desk': desk,
#                         'true_pub_date': true_pub_date,
#                         'year': year,
#                         'month': month_num
#                     }
                    
#                     articles_by_month[month_num].append(article)
#             except Exception as e:
#                 print(f"处理行时出错: {e}")
    
#     return articles_by_month

# def build_new_training_dataset(original_parquet, v3_jsonl, output_parquet):
#     """构建新的训练数据集"""
#     # 1. 读取原始训练数据
#     print("读取原始训练数据...")
#     df_original = pd.read_parquet(original_parquet)
#     original_samples = df_original.to_dict('records')
    
#     # 2. 按月份分组并每月随机抽取1000条
#     print("按月份分组并抽取数据...")
#     samples_by_month = defaultdict(list)
#     for sample in original_samples:
#         year = sample["extra_info"]["year"]
#         month = sample["extra_info"]["month"]
        
#         # 只保留2024年1-7月的数据
#         if year == 2024 and 1 <= month <= 7:
#             samples_by_month[(year, month)].append(sample)
    
#     # 随机抽取每月1000条
#     selected_original_samples = []
#     for (year, month), samples in samples_by_month.items():
#         if len(samples) > 1000:
#             selected = random.sample(samples, 1000)
#         else:
#             selected = samples
#         selected_original_samples.extend(selected)
#         print(f"从{year}-{month:02d}抽取了{len(selected)}条数据")
    
#     # 3. 解析v3生成的数据
#     print("解析v3生成的数据...")
#     articles_by_month = parse_articles_from_v3_generation(v3_jsonl)
    
#     # 4. 将v3生成的文章转换为训练样本
#     print("将v3生成的文章转换为训练样本...")
#     v3_samples = []
#     index_counter = len(selected_original_samples)
    
#     for month, articles in articles_by_month.items():
#         # year, month_num = article['year'], article['month'] #parse_year_month_from_true(month)
#         # if year is None or month_num is None:
#         #     continue
            
#         for article in articles:
#             year, month_num = article['year'], article['month'] #parse_year_month_from_true(month)
#             if year is None or month_num is None:
#                 continue
#             # 计算与参考日期的时间距离
#             time_distance = calculate_time_distance(article)
            
#             # 构造前缀
#             event = {
#                 'headline': article['headline'],
#                 'abstract': article['abstract']
#             }
#             prefix = construct_prefix(event)

#             # 创建训练样本
#             sample = {
#                 "data_source": "new_york_times",
#                 "prompt": [{
#                     "role": "user",
#                     "content": prefix
#                 }],
#                 "ability": "news_time_prediction",
#                 "reward_model": {
#                     "style": "rule",
#                     "ground_truth": {
#                         "event_pub_date": article['true_pub_date'],
#                         "time_distance": time_distance
#                     }
#                 },
#                 "extra_info": {
#                     "split": "train",
#                     "index": index_counter,
#                     "layer": "generated",  # 标记为生成的数据
#                     "time_distance": time_distance,
#                     "year": year,
#                     "month": month_num,
#                     "desk": article['desk']  # 添加新闻类别信息
#                 }
#             }
#             v3_samples.append(sample)
#             index_counter += 1
    
#     print(f"从v3生成结果中提取了{len(v3_samples)}条样本")
    
#     # 5. 合并数据
#     all_samples = selected_original_samples + v3_samples
#     random.shuffle(all_samples)  # 随机打乱
    
#     # 6. 保存为新的train_time_prediction.parquet
#     df_new = pd.DataFrame(all_samples)
#     df_new.to_parquet(output_parquet, index=False)
    
#     # 7. 统计数据集情况
#     print("\n新训练集统计:")
#     print(f"总样本数: {len(all_samples)}")
#     print(f"原始数据样本数: {len(selected_original_samples)}")
#     print(f"v3生成样本数: {len(v3_samples)}")

#     # 按月份统计
#     month_stats = defaultdict(int)
#     for sample in all_samples:
#         year = sample["extra_info"]["year"]
#         month = sample["extra_info"]["month"]
#         month_stats[(year, month)] += 1
    
#     print("\n月份分布:")
#     for (year, month), count in sorted(month_stats.items()):
#         print(f"{year}-{month:02d}: {count}")
    
#     # 如果有v3生成的样本，统计新闻类别分布
#     if v3_samples:
#         desk_stats = defaultdict(int)
#         for sample in v3_samples:
#             desk = sample["extra_info"].get("desk", "Unknown")
#             desk_stats[desk] += 1
        
#         print("\nv3生成样本新闻类别分布:")
#         for desk, count in sorted(desk_stats.items(), key=lambda x: x[1], reverse=True):
#             print(f"{desk}: {count}")
    
#     return all_samples

# if __name__ == "__main__":
#     # 设置随机种子，确保结果可复现
#     random.seed(1024)
    
#     # 文件路径
#     original_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_prediction.parquet"
#     v3_jsonl = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/v3_generation_4prediction_results.jsonl"
#     output_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_prediction_with_generated.parquet"
    
#     # 构建新的训练数据集
#     build_new_training_dataset(original_parquet, v3_jsonl, output_parquet)


# ...existing code...

# def parse_articles_from_v3_generation(jsonl_file):
#     """从v3生成结果中解析文章，并记录被过滤的条目"""
#     articles_by_month = defaultdict(list)
#     filtered_out_articles = [] # 用于存储被过滤掉的文章的原始信息
    
#     with open(jsonl_file, 'r') as f:
#         for line_num, line in enumerate(f, 1): # 添加行号以便追踪
#             try:
#                 data = json.loads(line)
                
#                 custom_id = data.get('custom_id', '')
#                 if not custom_id or '_' not in custom_id:
#                     filtered_out_articles.append({
#                         "line_num": line_num,
#                         "custom_id": custom_id,
#                         "reason": "Invalid or missing custom_id",
#                         "original_content": data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
#                     })
#                     continue
                    
#                 date_str, desk, _ = custom_id.split('_', 2)
                
#                 if '-' in date_str:
#                     year, month_num = map(int, date_str.split('-'))
#                 else:
#                     filtered_out_articles.append({
#                         "line_num": line_num,
#                         "custom_id": custom_id,
#                         "reason": f"Incorrect date format in custom_id: {date_str}",
#                         "original_content": data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
#                     })
#                     continue
                
#                 content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                
#                 article_pattern = r'ARTICLE (\d+):\s*(?:\*\*)?Headline:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Abstract:(?:\*\*)?\s*(.*?)(?=\n\nARTICLE \d+:|$)'
                
#                 # 首先用简单模式统计一下理论上的文章数，用于对比
#                 potential_articles_count = len(re.findall(r'ARTICLE \d+:', content))
#                 extracted_articles_count = 0

#                 for match in re.finditer(article_pattern, content, re.DOTALL):
#                     article_num_match = match.group(1) # 提取的文章编号
#                     headline = match.group(2).strip()
#                     abstract = match.group(3).strip()
                    
#                     original_headline = headline # 保留清理前的用于记录
#                     original_abstract = abstract # 保留清理前的用于记录

#                     headline = headline.replace('\n', ' ').strip()
#                     abstract = abstract.replace('\n', ' ').strip()
                    
#                     headline = re.sub(r'\*\*|\*', '', headline)
#                     abstract = re.sub(r'\*\*|\*', '', abstract)
                    
#                     if not headline or not abstract:
#                         filtered_out_articles.append({
#                             "line_num": line_num,
#                             "custom_id": custom_id,
#                             "article_num_in_source": article_num_match,
#                             "reason": "Missing headline or abstract after cleaning",
#                             "original_headline": original_headline,
#                             "original_abstract": original_abstract,
#                             "cleaned_headline": headline,
#                             "cleaned_abstract": abstract,
#                             "full_article_text_match": match.group(0) # 记录匹配到的完整文章文本
#                         })
#                         continue

#                     extracted_articles_count += 1
#                     true_pub_date = f"{year}-{month_num:02d}"
                    
#                     article = {
#                         'headline': headline,
#                         'abstract': abstract,
#                         'desk': desk,
#                         'true_pub_date': true_pub_date,
#                         'year': year,
#                         'month': month_num
#                     }
#                     # 使用 true_pub_date 作为 key，确保年月正确分组
#                     articles_by_month[true_pub_date].append(article)

#                 # 如果简单模式找到的文章数和复杂模式提取到的不一致，也记录下来
#                 if potential_articles_count > extracted_articles_count:
#                     # 这种情况比较复杂，可能是有ARTICLE N但后续格式不对，导致复杂正则没匹配上
#                     # 可以选择在这里记录整个content，或者更细致的diff
#                     pass # 暂时不详细记录这种差异，主要关注headline/abstract为空的情况

#             except Exception as e:
#                 filtered_out_articles.append({
#                     "line_num": line_num,
#                     "custom_id": custom_id if 'custom_id' in locals() else "Unknown",
#                     "reason": f"Exception during processing: {str(e)}",
#                     "original_content": line # 记录原始行数据
#                 })
    
#     return articles_by_month, filtered_out_articles

# ...existing code...

# def parse_articles_from_v3_generation(jsonl_file):
    """从v3生成结果中解析文章，并记录被过滤的条目，特别是那些简单匹配成功但详细匹配失败的。"""
    articles_by_month = defaultdict(list)
    filtered_out_articles = [] 
    extracted_articles_count = 0 # <--- 正确的初始化位置
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id', '')
                response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if not custom_id or '_' not in custom_id:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": "Invalid or missing custom_id",
                        "original_content_block": response_content
                    })
                    continue
                    
                date_str, desk, _ = custom_id.split('_', 2)
                
                if '-' in date_str:
                    year, month_num = map(int, date_str.split('-'))
                else:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": f"Incorrect date format in custom_id: {date_str}",
                        "original_content_block": response_content
                    })
                    continue
                
                # 详细解析模式
                article_pattern_detailed = r'ARTICLE (\d+):\s*(?:\*\*)?Headline:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Abstract:(?:\*\*)?\s*(.*?)(?=\n\nARTICLE \d+:|$)'
                
                # 首先，将整个 content 按 "ARTICLE \d+:" 分割，尝试识别出所有潜在的文章块
                # 使用 re.split 来保留分隔符，方便定位
                # 加一个捕获组到分隔符中，使其被保留
                potential_article_blocks = re.split(r'(ARTICLE \d+:)', response_content)
                
                # 第一个元素如果为空或者不是 "ARTICLE \d+:" 开头，则移除
                if potential_article_blocks and not potential_article_blocks[0].strip().startswith("ARTICLE"):
                    potential_article_blocks.pop(0)

                processed_article_indices_in_block = set() # 记录在这个 custom_id 的 content 中已经被成功处理的文章索引

                # 先用详细模式处理一遍，记录成功提取的文章
                for match_detailed in re.finditer(article_pattern_detailed, response_content, re.DOTALL):
                    article_num_detailed = match_detailed.group(1)
                    headline_detailed = match_detailed.group(2).strip()
                    abstract_detailed = match_detailed.group(3).strip()

                    cleaned_headline = headline_detailed.replace('\n', ' ').strip()
                    cleaned_headline = re.sub(r'\*\*|\*', '', cleaned_headline)
                    cleaned_abstract = abstract_detailed.replace('\n', ' ').strip()
                    cleaned_abstract = re.sub(r'\*\*|\*', '', cleaned_abstract)

                    if not cleaned_headline or not cleaned_abstract:
                        # 这种情况是详细模式匹配上了，但清理后内容为空
                        filtered_out_articles.append({
                            "line_num": line_num,
                            "custom_id": custom_id,
                            "article_num_in_source": article_num_detailed,
                            "reason": "Detailed match but missing headline/abstract after cleaning",
                            "original_headline": headline_detailed,
                            "original_abstract": abstract_detailed,
                            "full_article_text_match": match_detailed.group(0)
                        })
                        # 即使被过滤，也标记为已处理（按详细模式）
                        processed_article_indices_in_block.add(article_num_detailed)
                        continue

                    # 成功提取
                    extracted_articles_count += 1 # 确保这个计数器在函数开始时初始化为0
                    true_pub_date = f"{year}-{month_num:02d}"
                    article = {
                        'headline': cleaned_headline,
                        'abstract': cleaned_abstract,
                        'desk': desk,
                        'true_pub_date': true_pub_date,
                        'year': year,
                        'month': month_num
                    }
                    articles_by_month[true_pub_date].append(article)
                    processed_article_indices_in_block.add(article_num_detailed)

                # 现在遍历简单模式识别出的文章，找出那些未被详细模式处理的
                # 将 potential_article_blocks 两两组合成 (标识, 内容)
                simple_matches = []
                for i in range(0, len(potential_article_blocks) -1, 2):
                    identifier = potential_article_blocks[i]
                    text_block = potential_article_blocks[i+1]
                    article_num_match_simple = re.match(r'ARTICLE (\d+):', identifier)
                    if article_num_match_simple:
                        simple_matches.append({
                            "id_text": identifier.strip(),
                            "num": article_num_match_simple.group(1),
                            "text_content": text_block.strip() 
                        })
                
                for pot_article in simple_matches:
                    article_num_simple = pot_article["num"]
                    if article_num_simple not in processed_article_indices_in_block:
                        # 这个 ARTICLE N: 被简单模式找到，但未被详细模式成功处理或因内容为空被过滤
                        # 尝试从 pot_article["text_content"] 中提取原始的 Headline 和 Abstract (如果格式类似但不完全匹配详细正则)
                        headline_match_raw = re.search(r'(?i)Headline:(.*?)(?=\nAbstract:|\n\n|$)', pot_article["text_content"], re.DOTALL)
                        abstract_match_raw = re.search(r'(?i)Abstract:(.*?)(?=\n\nARTICLE \d+:|$)', pot_article["text_content"], re.DOTALL)
                        
                        original_headline_raw = headline_match_raw.group(1).strip() if headline_match_raw else "N/A (Raw Headline not found)"
                        original_abstract_raw = abstract_match_raw.group(1).strip() if abstract_match_raw else "N/A (Raw Abstract not found)"

                        filtered_out_articles.append({
                            "line_num": line_num,
                            "custom_id": custom_id,
                            "article_num_in_source": article_num_simple,
                            "reason": "Simple match (ARTICLE N:) found, but not successfully processed by detailed pattern or content missing.",
                            "potential_article_identifier": pot_article["id_text"],
                            "following_text_block": pot_article["text_content"],
                            "raw_headline_attempt": original_headline_raw,
                            "raw_abstract_attempt": original_abstract_raw
                        })
            
            except Exception as e:
                filtered_out_articles.append({
                    "line_num": line_num,
                    "custom_id": custom_id if 'custom_id' in locals() else "Unknown",
                    "reason": f"Exception during processing: {str(e)}",
                    "original_content_block": line 
                })
    
    # 在函数开始处初始化 extracted_articles_count
    # extracted_articles_count = 0 # 或者移到循环外部，如果它是全局的或类成员

    return articles_by_month, filtered_out_articles

def parse_articles_from_v3_generation(jsonl_file):
    """从v3生成结果中解析文章，并记录被过滤的条目，特别是那些简单匹配成功但详细匹配失败的。"""
    articles_by_month = defaultdict(list)
    filtered_out_articles = [] 
    extracted_articles_count = 0 
    
    with open(jsonl_file, 'r', encoding='utf-8') as f: # Added encoding
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id', '')
                response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if not custom_id or '_' not in custom_id:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": "Invalid or missing custom_id",
                        "original_content_block": response_content
                    })
                    continue
                    
                date_str, desk, _ = custom_id.split('_', 2)
                
                if '-' in date_str:
                    year, month_num = map(int, date_str.split('-'))
                else:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": f"Incorrect date format in custom_id: {date_str}",
                        "original_content_block": response_content
                    })
                    continue
                
                article_pattern_detailed = r'ARTICLE (\d+):\s*(?:\*\*)?Headline:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Abstract:(?:\*\*)?\s*(.*?)(?=\n\nARTICLE \d+:|$)'
                
                potential_article_blocks = re.split(r'(ARTICLE \d+:)', response_content)
                
                if potential_article_blocks and not potential_article_blocks[0].strip().startswith("ARTICLE"):
                    potential_article_blocks.pop(0)

                processed_article_indices_in_block = set()

                for match_detailed in re.finditer(article_pattern_detailed, response_content, re.DOTALL):
                    article_num_detailed = match_detailed.group(1)
                    headline_detailed = match_detailed.group(2).strip()
                    abstract_detailed = match_detailed.group(3).strip()

                    cleaned_headline = headline_detailed.replace('\n', ' ').strip()
                    cleaned_headline = re.sub(r'\*\*|\*', '', cleaned_headline).strip() # Remove all * and **
                    cleaned_abstract = abstract_detailed.replace('\n', ' ').strip()
                    cleaned_abstract = re.sub(r'\*\*|\*', '', cleaned_abstract).strip() # Remove all * and **

                    if not cleaned_headline or not cleaned_abstract:
                        filtered_out_articles.append({
                            "line_num": line_num,
                            "custom_id": custom_id,
                            "article_num_in_source": article_num_detailed,
                            "reason": "Detailed match but missing headline/abstract after cleaning",
                            "original_headline": headline_detailed,
                            "original_abstract": abstract_detailed,
                            "full_article_text_match": match_detailed.group(0)
                        })
                        processed_article_indices_in_block.add(article_num_detailed)
                        continue

                    extracted_articles_count += 1
                    true_pub_date = f"{year}-{month_num:02d}"
                    article = {
                        'headline': cleaned_headline,
                        'abstract': cleaned_abstract,
                        'desk': desk,
                        'true_pub_date': true_pub_date,
                        'year': year,
                        'month': month_num
                    }
                    articles_by_month[true_pub_date].append(article)
                    processed_article_indices_in_block.add(article_num_detailed)

                simple_matches = []
                for i in range(0, len(potential_article_blocks) -1, 2):
                    identifier = potential_article_blocks[i]
                    text_block = potential_article_blocks[i+1]
                    article_num_match_simple = re.match(r'ARTICLE (\d+):', identifier)
                    if article_num_match_simple:
                        simple_matches.append({
                            "id_text": identifier.strip(),
                            "num": article_num_match_simple.group(1),
                            "text_content": text_block.strip() 
                        })
                
                for pot_article in simple_matches:
                    article_num_simple = pot_article["num"]
                    if article_num_simple not in processed_article_indices_in_block:
                        text_block_after_id = pot_article["text_content"]
                        
                        headline_extracted_raw = ""
                        abstract_extracted_raw = ""

                        # Attempt 1: Match format like "**Headline:** content"
                        h_match_bold_keyword = re.search(r'\*\*Headline:\*\*\s*(.*?)(?=\s*\*\*Abstract:\*\*|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                        if h_match_bold_keyword:
                            headline_extracted_raw = h_match_bold_keyword.group(1).strip()

                        a_match_bold_keyword = re.search(r'\*\*Abstract:\*\*\s*(.*?)(?=\n\nARTICLE \d+:|\n\n---|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                        if a_match_bold_keyword:
                            abstract_extracted_raw = a_match_bold_keyword.group(1).strip()
                        
                        # Attempt 2: Fallback for formats like "Headline: content" (keywords not bold)
                        if not headline_extracted_raw: # Only if first attempt failed for headline
                            h_match_plain_keyword = re.search(r'(?<!\*)\bHeadline:\s*(.*?)(?=(?<!\*)\bAbstract:|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                            if h_match_plain_keyword:
                                headline_extracted_raw = h_match_plain_keyword.group(1).strip()
                        
                        if not abstract_extracted_raw: # Only if first attempt failed for abstract
                            a_match_plain_keyword = re.search(r'(?<!\*)\bAbstract:\s*(.*?)(?=\n\nARTICLE \d+:|\n\n---|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                            if a_match_plain_keyword:
                                abstract_extracted_raw = a_match_plain_keyword.group(1).strip()
                        
                        final_headline = ""
                        if headline_extracted_raw:
                            final_headline = headline_extracted_raw.replace('\n', ' ').strip()
                            final_headline = re.sub(r'\*\*|\*', '', final_headline).strip()

                        final_abstract = ""
                        if abstract_extracted_raw:
                            final_abstract = abstract_extracted_raw.replace('\n', ' ').strip()
                            final_abstract = re.sub(r'\*\*|\*', '', final_abstract).strip()
                        
                        if final_headline and final_abstract:
                            extracted_articles_count += 1
                            true_pub_date = f"{year}-{month_num:02d}"
                            article = {
                                'headline': final_headline,
                                'abstract': final_abstract,
                                'desk': desk,
                                'true_pub_date': true_pub_date,
                                'year': year,
                                'month': month_num
                            }
                            articles_by_month[true_pub_date].append(article)
                            # Mark as processed to avoid re-logging if it somehow appeared again
                            processed_article_indices_in_block.add(article_num_simple) 
                        else:
                            # Log if still not parsable
                            filtered_out_articles.append({
                                "line_num": line_num,
                                "custom_id": custom_id,
                                "article_num_in_source": article_num_simple,
                                "reason": "Simple match (ARTICLE N:) found, but content extraction failed with new logic.",
                                "potential_article_identifier": pot_article["id_text"],
                                "following_text_block": text_block_after_id,
                                "attempted_headline_raw": headline_extracted_raw,
                                "attempted_abstract_raw": abstract_extracted_raw 
                            })
            
            except Exception as e:
                filtered_out_articles.append({
                    "line_num": line_num,
                    "custom_id": custom_id if 'custom_id' in locals() else "Unknown",
                    "reason": f"Exception during processing: {str(e)}",
                    "original_content_block": line 
                })
    
    return articles_by_month, filtered_out_articles

def build_new_training_dataset(original_parquet, v3_jsonl, output_parquet, filtered_output_jsonl="filtered_v3_samples.jsonl"):
    """构建新的训练数据集"""
    # 1. 读取原始训练数据
    print("读取原始训练数据...")
    df_original = pd.read_parquet(original_parquet)
    original_samples = df_original.to_dict('records')
    
    # 2. 按月份分组并每月随机抽取1000条
    print("按月份分组并抽取数据...")
    samples_by_month = defaultdict(list)
    for sample in original_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        
        # 只保留2024年1-7月的数据
        if year == 2024 and 1 <= month <= 7:
            samples_by_month[(year, month)].append(sample)
    
    # 随机抽取每月1000条
    selected_original_samples = []
    for (year, month), samples in samples_by_month.items():
        if len(samples) > 1000:
            selected = random.sample(samples, 1000)
        else:
            selected = samples
        selected_original_samples.extend(selected)
        print(f"从{year}-{month:02d}抽取了{len(selected)}条数据")

    # 3. 解析v3生成的数据
    print("解析v3生成的数据...")
    articles_by_month, filtered_samples = parse_articles_from_v3_generation(v3_jsonl) # 获取被过滤的样本
    
    # 将被过滤的样本保存到文件
    if filtered_samples:
        print(f"记录了 {len(filtered_samples)} 条被过滤的v3生成条目到: {filtered_output_jsonl}")
        with open(filtered_output_jsonl, 'w', encoding='utf-8') as f_out:
            for item in filtered_samples:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 4. 将v3生成的文章转换为训练样本
    print("将v3生成的文章转换为训练样本...")
    v3_samples = []
    index_counter = len(selected_original_samples)
    
    for month, articles in articles_by_month.items():
            
        for article in articles:
            year, month_num = article['year'], article['month'] 
            # if year is None or month_num is None: # 这个检查其实在parse_articles_from_v3_generation中已经做了
            #     continue
            # 计算与参考日期的时间距离
            # time_distance = calculate_time_distance(article) # 确保calculate_time_distance接受的是字典
            time_distance = month_difference(2024, 1, year, month_num) # 直接使用 month_difference
            # time_distance = calculate_time_distance(article)
            
            # 构造前缀
            event = {
                'headline': article['headline'],
                'abstract': article['abstract']
            }
            prefix = construct_prefix(event)

            # 创建训练样本
            sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": prefix
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "event_pub_date": article['true_pub_date'],
                        "time_distance": time_distance
                    }
                },
                "extra_info": {
                    "split": "train",
                    "index": index_counter,
                    "layer": "generated",  # 标记为生成的数据
                    "time_distance": time_distance,
                    "year": year,
                    "month": month_num,
                    "desk": article['desk']  # 添加新闻类别信息
                }
            }
            v3_samples.append(sample)
            index_counter += 1
    
    print(f"从v3生成结果中提取了{len(v3_samples)}条样本")
    
    # 5. 合并数据
    all_samples = selected_original_samples + v3_samples
    random.shuffle(all_samples)  # 随机打乱
    
    # 6. 保存为新的train_time_prediction.parquet
    df_new = pd.DataFrame(all_samples)
    df_new.to_parquet(output_parquet, index=False)
    
    # 7. 统计数据集情况
    print("\n新训练集统计:")
    print(f"总样本数: {len(all_samples)}")
    print(f"原始数据样本数: {len(selected_original_samples)}")
    print(f"v3生成样本数: {len(v3_samples)}")

    # 按月份统计
    month_stats = defaultdict(int)
    for sample in all_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        month_stats[(year, month)] += 1
    
    print("\n月份分布:")
    for (year, month), count in sorted(month_stats.items()):
        print(f"{year}-{month:02d}: {count}")
    
    # 如果有v3生成的样本，统计新闻类别分布
    if v3_samples:
        desk_stats = defaultdict(int)
        for sample in v3_samples:
            desk = sample["extra_info"].get("desk", "Unknown")
            desk_stats[desk] += 1
        
        print("\nv3生成样本新闻类别分布:")
        for desk, count in sorted(desk_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{desk}: {count}")
    
    return all_samples
            

if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    random.seed(1024)
    
    # 文件路径
    original_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_prediction.parquet"
    v3_jsonl = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/v3_generation_4prediction_results.jsonl"
    output_parquet = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_prediction_with_generated_1.parquet"
    filtered_output_jsonl = "/data/zliu331/temporal_reasoning/TinyZero/examples/data_preprocess/filtered_v3_samples.jsonl" # 指定输出路径
    
    # 构建新的训练数据集
    build_new_training_dataset(original_parquet, v3_jsonl, output_parquet, filtered_output_jsonl) # 传递新参数











# if __name__ == "__main__":
#     # 设置随机种子，确保结果可复现
#     random.seed(1024)
    
#     # input_dir = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result"
#     # output_train_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_prediction.parquet"
#     # output_test_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_prediction.parquet"
    
#     # # # 构建时间预测数据集
#     # # train_samples, test_samples = build_time_prediction_datasets(input_dir, output_train_file, output_test_file)

#     # small_test_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_time_prediction.parquet"
    
#     # # 创建小型测试集
#     # create_small_test_set(output_test_file, small_test_file)

#     input_dir = "/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years"
#     input_file = os.path.join(input_dir, "2025_until_apr.jsonl")
#     output_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_march_april_prediction.parquet"
    
#     # 如果没有3月和4月的原始数据，可以直接生成模拟数据
#     if not os.path.exists(input_file):
#         print(f"未找到3月和4月的原始数据文件: {input_file}")
#         print("将直接生成模拟数据...")
#         # 创建一个空文件作为输入，内部逻辑会自动生成模拟数据
#         with open(input_file, "w") as f:
#             pass
    
#     # 创建测试集
#     create_march_april_test_set(input_file, output_file)