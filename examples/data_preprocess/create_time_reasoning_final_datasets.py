import pandas as pd
import random
import os
from typing import List, Dict, Any

def random_sample_from_parquet(file_path: str, sample_size: int) -> pd.DataFrame:
    """
    从parquet文件中随机抽取指定数量的样本
    
    参数:
        file_path: parquet文件路径
        sample_size: 要抽取的样本数量
        
    返回:
        包含随机抽样结果的DataFrame
    """
    print(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)
    
    total_samples = len(df)
    if total_samples < sample_size:
        print(f"Warning: Requested {sample_size} samples but only {total_samples} available in {file_path}.")
        sample_size = total_samples
    
    print(f"Randomly sampling {sample_size} from {total_samples} samples in {file_path}")
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    return sampled_df

def create_combined_dataset(
    task_files: Dict[str, str], 
    sample_sizes: Dict[str, int],
    output_file: str,
    shuffle: bool = True
) -> None:
    """
    从多个任务数据集中抽样并合并为一个综合数据集
    
    参数:
        task_files: 任务名称到数据文件路径的映射
        sample_sizes: 任务名称到样本数量的映射
        output_file: 输出文件路径
        shuffle: 是否打乱最终数据集
    """
    combined_df = pd.DataFrame()
    
    for task_name, file_path in task_files.items():
        if task_name not in sample_sizes:
            print(f"Warning: No sample size specified for {task_name}. Skipping.")
            continue
            
        sample_size = sample_sizes[task_name]
        sampled_df = random_sample_from_parquet(file_path, sample_size)
        
        # 记录每个数据集的大小，以便统计
        print(f"  - {task_name}: {len(sampled_df)} samples")
        
        # 合并到总数据集
        combined_df = pd.concat([combined_df, sampled_df], ignore_index=True)
    
    # 是否打乱数据
    if shuffle:
        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 保存合并后的数据集
    print(f"Saving combined dataset with {len(combined_df)} samples to {output_file}")
    combined_df.to_parquet(output_file, index=False)
    
    # 打印每种任务在最终数据集中的分布
    task_counts = combined_df['extra_info'].apply(lambda x: x.get('task', 'unknown')).value_counts()
    print("\nTask distribution in the final dataset:")
    for task, count in task_counts.items():
        print(f"  - {task}: {count} samples ({count/len(combined_df)*100:.2f}%)")

def main():
    # 数据集根目录
    dataset_dir = "/data/zliu331/temporal_reasoning/TinyZero/datasets"
    
    # 各子任务训练数据集
    train_files = {
        "time_difference": os.path.join(dataset_dir, "train_time_difference.parquet"),
        "time_ordering": os.path.join(dataset_dir, "train_time_ordering.parquet"),
        "time_completion": os.path.join(dataset_dir, "train_time_completion.parquet"),
        "time_inferring": os.path.join(dataset_dir, "train_time_inferring.parquet")
    }
    
    # 各子任务测试数据集
    test_files = {
        "time_difference": os.path.join(dataset_dir, "test_time_difference.parquet"),
        "time_ordering": os.path.join(dataset_dir, "test_time_ordering.parquet"),
        "time_completion": os.path.join(dataset_dir, "test_time_completion.parquet"),
        "time_inferring": os.path.join(dataset_dir, "test_time_inferring.parquet")
    }
    
    # 训练集样本数
    train_samples = {
        "time_difference": 13000,
        "time_ordering": 13000,
        "time_completion": 13000,
        "time_inferring": 10000
    }
    
    # 测试集样本数
    test_samples = {
        "time_difference": 256,
        "time_ordering": 256,
        "time_completion": 256,
        "time_inferring": 256
    }
    
    # 输出文件路径
    train_output = os.path.join(dataset_dir, "train_time_reasoning_combined.parquet")
    test_output = os.path.join(dataset_dir, "test_time_reasoning_combined.parquet")
    
    # 创建训练集
    print("\n===== Creating Combined Training Dataset =====")
    create_combined_dataset(train_files, train_samples, train_output)
    
    # 创建测试集
    print("\n===== Creating Combined Test Dataset =====")
    create_combined_dataset(test_files, test_samples, test_output)
    
    print("\nProcess completed successfully!")

# import pandas as pd
# import os

def create_time_inferring_easy_dataset():
    """
    从train_time_inferring.parquet中提取与train_easy_nyt.parquet匹配的样本，
    创建train_time_inferring_easy.parquet用于第一阶段训练。
    """
    # 文件路径
    easy_samples_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_easy_nyt.parquet"
    all_samples_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_inferring.parquet"
    output_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_inferring_easy.parquet"
    
    # 加载简单样本数据集
    easy_df = pd.read_parquet(easy_samples_path)
    
    # 创建标题和摘要的匹配集
    easy_samples_set = set()
    for _, row in easy_df.iterrows():
        content = row['prompt'][0]['content']
        
        # 从content中提取headline和abstract
        try:
            headline_start = content.find("Headline: ") + len("Headline: ")
            headline_end = content.find("\n", headline_start)
            headline = content[headline_start:headline_end].strip()
            
            abstract_start = content.find("Abstract: ") + len("Abstract: ")
            abstract_end = content.find("\n", abstract_start)
            abstract = content[abstract_start:abstract_end].strip()
            
            easy_samples_set.add((headline, abstract))
        except Exception:
            continue
    
    print(f"Loaded {len(easy_samples_set)} unique samples from easy dataset.")
    
    # 加载所有时间推断样本
    all_df = pd.read_parquet(all_samples_path)
    print(f"Loaded {len(all_df)} samples from all time_inferring dataset.")
    
    # 筛选匹配的样本
    matched_indices = []
    
    for idx, row in all_df.iterrows():
        content = row['prompt'][0]['content']
        
        try:
            headline_start = content.find("Headline: ") + len("Headline: ")
            headline_end = content.find("\n", headline_start)
            headline = content[headline_start:headline_end].strip()
            
            abstract_start = content.find("Abstract: ") + len("Abstract: ")
            abstract_end = content.find("\n", abstract_start)
            abstract = content[abstract_start:abstract_end].strip()
            
            # 检查是否在简单样本集合中
            if (headline, abstract) in easy_samples_set:
                matched_indices.append(idx)
        except Exception:
            continue
    
    # 创建新的DataFrame并保存
    matched_df = all_df.iloc[matched_indices].copy()
    
    # 更新extra_info中的split字段
    for i in range(len(matched_df)):
        matched_df.iloc[i]['extra_info']['split'] = 'train_easy'
    
    # 保存为Parquet文件
    matched_df.to_parquet(output_path, index=False)
    print(f"Finished! {len(matched_df)} matched samples saved to {output_path}.")

# if __name__ == "__main__":
#     create_time_inferring_easy_dataset()


# if __name__ == "__main__":
#     # 设置随机种子以确保可重复性
#     random.seed(1024)
#     main()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_month_difference_distribution():
    """
    分析train_time_reasoning_combined.parquet中time_difference任务的月份差分布
    """
    # 文件路径
    dataset_path = "/data/zliu331/temporal_reasoning/TinyZero/datasets/train_time_reasoning_combined.parquet"
    
    # 读取数据集
    print(f"Reading {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # 筛选time_difference任务
    time_diff_df = df[df['extra_info'].apply(lambda x: x.get('task') == 'time_difference')]
    print(f"Found {len(time_diff_df)} time_difference samples in the dataset.")
    
    # 提取月份差值
    month_diffs = []
    for _, row in time_diff_df.iterrows():
        ground_truth = row['reward_model']['ground_truth']
        if isinstance(ground_truth, dict) and 'month_difference' in ground_truth:
            month_diff = ground_truth['month_difference']
            if month_diff is not None:
                month_diffs.append(float(month_diff))
    
    month_diffs = np.array(month_diffs)
    print(f"Extracted {len(month_diffs)} valid month difference values.")
    
    # 统计基本信息
    print("\n--- Month Difference Statistics ---")
    print(f"Mean: {np.mean(month_diffs):.2f}")
    print(f"Median: {np.median(month_diffs):.2f}")
    print(f"Min: {np.min(month_diffs):.2f}")
    print(f"Max: {np.max(month_diffs):.2f}")
    print(f"Std Dev: {np.std(month_diffs):.2f}")
    
    # 区间分布
    ranges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, np.inf]
    range_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', 
                    '41-50', '51-75', '76-100', '100+']
    
    counts = []
    for i in range(len(ranges)-1):
        count = np.sum((month_diffs >= ranges[i]) & (month_diffs < ranges[i+1]))
        counts.append(count)
        percentage = count / len(month_diffs) * 100
        print(f"{range_labels[i]}: {count} samples ({percentage:.2f}%)")
    
    # 返回统计结果
    return {
        'month_diffs': month_diffs,
        'ranges': ranges,
        'range_labels': range_labels,
        'counts': counts
    }

# if __name__ == "__main__":
#     stats = analyze_month_difference_distribution()
    
    # 如果想要可视化，取消下面注释
    # plt.figure(figsize=(12, 6))
    # plt.bar(stats['range_labels'], stats['counts'])
    # plt.title('Month Difference Distribution in time_difference Task')
    # plt.xlabel('Month Difference Range')
    # plt.ylabel('Number of Samples')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('month_diff_distribution.png')


def extract_headline_abstract(content):
    """从提示内容中提取标题和摘要"""
    try:
        headline_start = content.find("Headline: ") + len("Headline: ")
        headline_end = content.find("\n", headline_start)
        headline = content[headline_start:headline_end].strip()
        
        abstract_start = content.find("Abstract: ") + len("Abstract: ")
        abstract_end = content.find("\n", abstract_start)
        abstract = content[abstract_start:abstract_end].strip()
        
        return headline, abstract
    except Exception:
        return None, None

def extract_two_events(content):
    """从time_difference任务提示中提取两个事件的标题和摘要"""
    try:
        # 提取第一个事件
        event1_headline_start = content.find("News article 1:\nHeadline: ") + len("News article 1:\nHeadline: ")
        event1_headline_end = content.find("\n", event1_headline_start)
        event1_headline = content[event1_headline_start:event1_headline_end].strip()
        
        event1_abstract_start = content.find("Abstract: ", event1_headline_end) + len("Abstract: ")
        event1_abstract_end = content.find("\n", event1_abstract_start)
        event1_abstract = content[event1_abstract_start:event1_abstract_end].strip()
        
        # 提取第二个事件
        event2_headline_start = content.find("News article 2:\nHeadline: ") + len("News article 2:\nHeadline: ")
        event2_headline_end = content.find("\n", event2_headline_start)
        event2_headline = content[event2_headline_start:event2_headline_end].strip()
        
        event2_abstract_start = content.find("Abstract: ", event2_headline_end) + len("Abstract: ")
        event2_abstract_end = content.find("\n", event2_abstract_start)
        event2_abstract = content[event2_abstract_start:event2_abstract_end].strip()
        
        return (event1_headline, event1_abstract), (event2_headline, event2_abstract)
    except Exception:
        return None, None
    
def add_difficulty_tags_to_dataset(combined_path, easy_samples_path, output_path):
    """为混合数据集中的每个样本添加难度标签"""
    # 加载数据集
    combined_df = pd.read_parquet(combined_path)
    easy_df = pd.read_parquet(easy_samples_path)
    
    # 建立简单样本的标识集合（标题+摘要）
    easy_identifiers = set()
    for _, row in easy_df.iterrows():
        content = row['prompt'][0]['content']
        try:
            headline, abstract = extract_headline_abstract(content)
            if headline and abstract:
                easy_identifiers.add((headline, abstract))
        except Exception:
            continue
            
    print(f"Loaded {len(easy_identifiers)} unique easy samples for matching")
    
    # 记录各类型任务难度统计
    difficulty_stats = {
        "time_inferring": {"easy": 0, "hard": 0},
        "time_difference": {"easy": 0, "hard": 0},
        "time_ordering": {"easy": 0, "hard": 0},
        "time_completion": {"easy": 0, "hard": 0}
    }
    
    # 遍历混合数据集的每个样本，添加难度标签
    for idx, row in combined_df.iterrows():
        task_type = row['extra_info']['task']
        content = row['prompt'][0]['content']
        
        # 初始化默认为"困难"
        difficulty = {
            "is_easy": False,
            "alpha": 0.07  # 困难样本使用宽松alpha
        }
        
        # 根据任务类型分别处理
        if task_type == "time_inferring":
            try:
                headline, abstract = extract_headline_abstract(content)
                if headline and abstract and (headline, abstract) in easy_identifiers:
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1  # 简单样本使用严格alpha
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
                
        elif task_type == "time_difference":
            # 提取两个事件的标题和摘要
            try:
                event1, event2 = extract_two_events(content)
                if event1 and event2:
                    event1_easy = (event1[0], event1[1]) in easy_identifiers
                    event2_easy = (event2[0], event2[1]) in easy_identifiers
                    
                    # 记录每个事件的难度
                    difficulty["events_difficulty"] = [int(event1_easy), int(event2_easy)]
                    # 如果两个事件都简单，整体任务也简单
                    if event1_easy and event2_easy:
                        difficulty["is_easy"] = True
                        difficulty["alpha"] = 0.1
                        difficulty_stats[task_type]["easy"] += 1
                    else:
                        difficulty_stats[task_type]["hard"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
                
        elif task_type == "time_ordering":
            # 处理三个事件的情况
            try:
                # 提取三个事件信息
                events_easy = []
                
                # 提取第一个事件
                event1_headline, event1_abstract = extract_event_info(content, 1)
                event1_easy = (event1_headline, event1_abstract) in easy_identifiers if event1_headline and event1_abstract else False
                events_easy.append(int(event1_easy))
                
                # 提取第二个事件
                event2_headline, event2_abstract = extract_event_info(content, 2)
                event2_easy = (event2_headline, event2_abstract) in easy_identifiers if event2_headline and event2_abstract else False
                events_easy.append(int(event2_easy))
                
                # 提取第三个事件
                event3_headline, event3_abstract = extract_event_info(content, 3)
                event3_easy = (event3_headline, event3_abstract) in easy_identifiers if event3_headline and event3_abstract else False
                events_easy.append(int(event3_easy))
                
                # 记录每个事件的难度
                difficulty["events_difficulty"] = events_easy
                
                # 如果全部事件都简单，整体任务也简单
                if all(events_easy):
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
            
        elif task_type == "time_completion":
            # 特殊处理time_completion任务
            try:
                # 提取标题和摘要
                headline, abstract = extract_headline_abstract(content)
                
                # 由于难以还原原始内容，只要标题或摘要在简单样本中就标记为简单
                # 遍历简单样本集合
                found_match = False
                for easy_headline, easy_abstract in easy_identifiers:
                    # 检查标题或摘要是否有高度匹配
                    if (headline and easy_headline and (headline in easy_headline or easy_headline in headline)) or \
                       (abstract and easy_abstract and (abstract in easy_abstract or easy_abstract in abstract)):
                        found_match = True
                        break
                
                if found_match:
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
            
        # 将难度信息添加到ground_truth中
        combined_df.at[idx, 'reward_model']['ground_truth']['difficulty'] = difficulty
    
    # 打印难度统计
    print("\n=== Difficulty Statistics ===")
    for task_type, stats in difficulty_stats.items():
        total = stats["easy"] + stats["hard"]
        if total > 0:
            easy_percent = stats["easy"] / total * 100
            print(f"{task_type}: {stats['easy']} easy ({easy_percent:.1f}%), {stats['hard']} hard ({100-easy_percent:.1f}%)")
    
    # 保存添加了难度标签的数据集
    combined_df.to_parquet(output_path, index=False)
    print(f"Saved dataset with difficulty tags to {output_path}")

def extract_event_info(content, event_num):
    """从time_ordering任务提示中提取指定事件的标题和摘要"""
    try:
        event_headline_start = content.find(f"News article {event_num}:\nHeadline: ") + len(f"News article {event_num}:\nHeadline: ")
        event_headline_end = content.find("\n", event_headline_start)
        event_headline = content[event_headline_start:event_headline_end].strip()
        
        event_abstract_start = content.find("Abstract: ", event_headline_end) + len("Abstract: ")
        event_abstract_end = content.find("\n", event_abstract_start)
        event_abstract = content[event_abstract_start:event_abstract_end].strip()
        
        return event_headline, event_abstract
    except Exception:
        return None, None
    
def main_difficulty_tagging():
    """
    为时间推理训练数据集添加难度标签和动态alpha值，
    用于实现多阶段训练策略中的第二阶段混合训练
    """
    # 设置文件路径
    dataset_dir = "/data/zliu331/temporal_reasoning/TinyZero/datasets"
    
    # 输入文件
    combined_path = os.path.join(dataset_dir, "train_time_reasoning_combined.parquet")
    easy_samples_path = os.path.join(dataset_dir, "train_easy_nyt.parquet")
    
    # 输出文件
    output_path = os.path.join(dataset_dir, "train_time_reasoning_dynamic_alpha.parquet")
    
    print("\n===== Adding Difficulty Tags and Dynamic Alpha Values =====")
    print(f"Input combined dataset: {combined_path}")
    print(f"Easy samples reference: {easy_samples_path}")
    print(f"Output dataset: {output_path}")
    
    # 调用难度标签添加函数
    add_difficulty_tags_to_dataset(combined_path, easy_samples_path, output_path)
    
    print("\nDifficulty tagging completed successfully!")
    
    # # 对测试集也做同样处理
    # test_combined_path = os.path.join(dataset_dir, "test_time_reasoning_combined.parquet")
    # test_output_path = os.path.join(dataset_dir, "test_time_reasoning_dynamic_alpha.parquet")
    
    # print("\n===== Adding Difficulty Tags to Test Dataset =====")
    # add_difficulty_tags_to_dataset(test_combined_path, easy_samples_path, test_output_path)
    
    # print("\nTest dataset difficulty tagging completed!")


if __name__ == "__main__":
    # 根据需要取消下面函数的注释来运行不同的数据集生成任务
    
    # 生成简单时间推断数据集
    # create_time_inferring_easy_dataset()
    
    # 生成混合数据集
    # random.seed(1024)
    # main()
    
    # 分析月份差分布
    # stats = analyze_month_difference_distribution()
    
    # 添加难度标签和动态alpha值
    main_difficulty_tagging()