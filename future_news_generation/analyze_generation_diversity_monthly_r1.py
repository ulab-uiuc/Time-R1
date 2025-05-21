import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import os

def load_jsonl(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def save_jsonl(data, file_path):
    """保存JSONL格式数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def compute_similarity_matrix(items):
    """使用SentenceTransformer计算语义相似度矩阵"""
    if len(items) <= 1:
        return np.zeros((1, 1))
    
    # 合并标题和摘要比较
    texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
    # 使用SentenceTransformer计算嵌入向量
    try:
        # 加载语义模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 使用GPU如果可用
        model.to("cuda:6")  # 指定GPU号
        
        # 计算嵌入向量
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        print(f"计算语义嵌入向量时出错: {str(e)}")
        # 如果无法计算，返回零矩阵
        return np.zeros((len(texts), len(texts)))

def select_diverse_news(items, count=5):
    """贪心选择相互最不相似的新闻"""
    if len(items) <= count:
        return items
    
    print("  计算语义相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(items)
    n = len(items)
    
    # 初始化已选和未选索引
    selected_indices = []
    remaining_indices = list(range(n))
    
    # 选择第一篇文章（可以是最长的，或者随机选择）
    # 这里我们选择摘要最长的，假设其可能包含更多信息
    text_lengths = [len(items[i]['abstract']) for i in range(n)]
    first_idx = text_lengths.index(max(text_lengths))
    
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    print(f"  已选择初始文章: {items[first_idx]['headline'][:50]}...")
    
    # 贪心选择剩余的文章
    while len(selected_indices) < count and remaining_indices:
        # 计算每个候选文章与已选文章的总相似度
        min_sim = float('inf')
        next_idx = -1
        
        for idx in remaining_indices:
            # 计算与已选文章的平均相似度
            avg_sim = sum(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices) / len(selected_indices)
            
            if avg_sim < min_sim:
                min_sim = avg_sim
                next_idx = idx
        
        if next_idx != -1:
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
            print(f"  选择第{len(selected_indices)}篇文章 (相似度: {min_sim:.4f}): {items[next_idx]['headline'][:50]}...")
        else:
            break
    
    # 返回选中的新闻
    return [items[i] for i in selected_indices]

def main():
    # 加载原始新闻数据
    input_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_multi_month_r1.jsonl"
    output_dir = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_r1/"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    news_data = load_jsonl(input_file)
    print(f"共加载了 {len(news_data)} 条新闻")
    
    # 按月份和主题分组
    month_topic_groups = defaultdict(lambda: defaultdict(list))
    months = set()
    
    for item in news_data:
        month = item.get('month', 'unknown')
        topic = item.get('topic', 'unknown')
        month_topic_groups[month][topic].append(item)
        months.add(month)
    
    # 按月份统计
    print("\n按月份统计信息:")
    for month in sorted(months):
        topic_count = sum(len(items) for items in month_topic_groups[month].values())
        print(f"  {month}: {topic_count}条新闻, {len(month_topic_groups[month])}个主题")
    
    # 按月份处理
    all_selected_news = []
    
    for month in sorted(months):
        print(f"\n处理月份: {month}")
        month_selected_news = []
        
        for topic, items in month_topic_groups[month].items():
            print(f"  处理主题 '{topic}'，共 {len(items)} 条新闻")
            diverse_selection = select_diverse_news(items, count=5)
            month_selected_news.extend(diverse_selection)
            print(f"  选择了 {len(diverse_selection)} 条多样性新闻")
        
        # 保存当月筛选结果
        month_output_file = os.path.join(output_dir, f"diverse_news_{month}.jsonl")
        save_jsonl(month_selected_news, month_output_file)
        print(f"  已保存 {len(month_selected_news)} 条多样性新闻到 {month_output_file}")
        
        all_selected_news.extend(month_selected_news)
    
    # 保存所有月份的筛选结果
    all_output_file = os.path.join(output_dir, "all_diverse_news.jsonl")
    save_jsonl(all_selected_news, all_output_file)
    print(f"\n筛选完成！已保存总计 {len(all_selected_news)} 条多样性新闻到 {all_output_file}")
    
    # 按月份和主题统计筛选结果
    print("\n按月份和主题统计筛选结果：")
    month_topic_counts = defaultdict(lambda: defaultdict(int))
    
    for item in all_selected_news:
        month = item.get('month', 'unknown')
        topic = item.get('topic', 'unknown')
        month_topic_counts[month][topic] += 1
    
    for month in sorted(month_topic_counts.keys()):
        print(f"\n月份: {month}")
        for topic, count in sorted(month_topic_counts[month].items()):
            print(f"  {topic}: {count}条新闻")

if __name__ == "__main__":
    main()