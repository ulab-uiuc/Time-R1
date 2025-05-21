import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import os
import argparse
import re
from langdetect import detect, LangDetectException

def load_jsonl(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                continue
    return data

# 正则表达式匹配中文字符（包括简体和繁体的主要范围）
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
def is_mainly_english(text, chinese_threshold=0):
    """
    检查文本是否主要为英文。
    如果 langdetect 检测失败、检测为非英文，或中文字符比例超过阈值，则返回 False。
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return False # 处理空或非字符串输入

    try:
        # 1. 使用 langdetect 进行初步检测
        detected_lang = detect(text)
        if detected_lang != 'en':
            # print(f"  Langdetect identified non-English: {detected_lang} in '{text[:50]}...'") # 可选：调试信息
            return False

        # 2. 检查中文字符比例
        chinese_chars = chinese_char_pattern.findall(text)
        chinese_char_count = sum(len(s) for s in chinese_chars)
        total_chars = len(text)

        if total_chars > 0:
            chinese_ratio = chinese_char_count / total_chars
            if chinese_ratio > chinese_threshold:
                # print(f"  Chinese char ratio exceeded threshold: {chinese_ratio:.2f} in '{text[:50]}...'") # 可选：调试信息
                return False

        # 通过所有检查
        return True

    except LangDetectException:
        # langdetect 检测失败，保守处理，返回 False
        # print(f"  Langdetect failed for text: {text[:50]}...") # 可选：调试信息
        return False
    except Exception as e:
        # 捕获其他潜在错误
        print(f"  Error in is_mainly_english for text '{text[:50]}...': {e}")
        return False

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
        model.to("cuda")  # 指定GPU号
        
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

def process_monthly_news(input_dir, output_dir, count_per_topic=5):
    """
    处理多个月份的新闻数据，每个月份每个主题选择多样性最高的n条新闻
    
    参数:
        input_dir: 包含月度新闻数据的目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有月份的新闻文件
    news_files = [f for f in os.listdir(input_dir) if f.startswith("future_news_") and f.endswith(".jsonl")]
    
    # 按月份排序
    news_files.sort()
    
    print(f"找到{len(news_files)}个月份的新闻文件")
    
    # 处理每个月份
    all_selected_news = []
    
    for file_name in news_files:
        # 从文件名解析月份
        month_match = re.search(r'future_news_(\d{4}-\d{2})', file_name)
        if not month_match:
            continue
            
        month = month_match.group(1)
        
        print(f"\n处理月份: {month}")
        
        # 加载当月新闻数据
        file_path = os.path.join(input_dir, file_name)
        news_data = load_jsonl(file_path)
        
        print(f"  加载了 {len(news_data)} 条新闻")
        
        # 过滤非英文新闻并清理标题
        processed_news = []
        skipped_non_english = 0
        for item in news_data:
            item['month'] = month
            headline = item.get('headline', '')
            abstract = item.get('abstract', '')
            content = f"{headline} {abstract}"

            if is_mainly_english(content):
                # 清理标题中的 '[' 和 ']'
                if isinstance(headline, str):
                    item['headline'] = headline.replace('[', '').replace(']', '')
                processed_news.append(item)
            else:
                skipped_non_english += 1
                # print(f"  跳过非英文新闻: {headline[:30]}...") # 可选：调试信息

        print(f"  跳过了 {skipped_non_english} 条非英文或含过多中文的新闻")
        print(f"  处理 {len(processed_news)} 条英文新闻")
        
        # 按主题分组
        topic_groups = defaultdict(list)
        for item in news_data:
            topic = item.get('topic', 'unknown')
            topic_groups[topic].append(item)
        
        # 选择每个主题最多样的n条新闻
        month_selected_news = []
        
        for topic, items in topic_groups.items():
            print(f"  处理主题 '{topic}'，共 {len(items)} 条新闻")
            diverse_selection = select_diverse_news(items, count=count_per_topic)
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
        topic_total = 0
        for topic, count in sorted(month_topic_counts[month].items()):
            print(f"  {topic}: {count}条新闻")
            topic_total += count
        print(f"  总计: {topic_total}条新闻")
    
    return all_selected_news

def main():
    # # 读取CSV文件
    # df = pd.read_csv('/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/llama31_results/future_news_llama31_parsed.csv')

    # # 创建输出目录
    # output_dir = '/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/llama31_results/monthly_data/'
    # os.makedirs(output_dir, exist_ok=True)

    # # 按月份分组
    # monthly_data = defaultdict(list)
    # for _, row in df.iterrows():
    #     month = row['target_date']
    #     item = {
    #         'headline': row['headline'],
    #         'abstract': row['abstract'],
    #         'topic': row['topic'],
    #         'custom_id': row['custom_id']
    #     }
    #     monthly_data[month].append(item)

    # # 保存按月份分组的JSONL文件
    # for month, items in monthly_data.items():
    #     output_file = os.path.join(output_dir, f'future_news_{month}.jsonl')
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for item in items:
    #             f.write(json.dumps(item, ensure_ascii=False) + '\n')
    #     print(f'已保存 {len(items)} 条新闻到 {output_file}')

    # print('转换完成！')




    parser = argparse.ArgumentParser(description="Select diverse news from monthly generated news")
    parser.add_argument("--output_dir", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_test/", help="Directory containing monthly news files")
    parser.add_argument("--input_dir", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_test", help="Output directory for diverse news")
    parser.add_argument("--count_per_topic", type=int, default=5, help="Number of news to select per topic")
    
    args = parser.parse_args()
    
    # 处理按月生成的新闻
    process_monthly_news(
        args.input_dir,
        args.output_dir,
        count_per_topic=args.count_per_topic
    )

if __name__ == "__main__":
    main()






# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict
# import torch

# def load_jsonl(file_path):
#     """加载JSONL格式数据"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return data

# def save_jsonl(data, file_path):
#     """保存JSONL格式数据"""
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# def compute_similarity_matrix_fast(items):
#     """计算相似度矩阵"""
#     if len(items) <= 1:
#         return np.zeros((1, 1))
    
#     # 合并标题和摘要比较
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
#     # 计算TF-IDF向量
#     vectorizer = TfidfVectorizer(stop_words='english')
#     try:
#         tfidf_matrix = vectorizer.fit_transform(texts)
#         similarity_matrix = cosine_similarity(tfidf_matrix)
#         return similarity_matrix
#     except ValueError:
#         # 如果无法计算，返回零矩阵
#         return np.zeros((len(texts), len(texts)))

# def compute_similarity_matrix(items):
#     """使用SentenceTransformer计算语义相似度矩阵"""
#     if len(items) <= 1:
#         return np.zeros((1, 1))
    
#     # 合并标题和摘要比较
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
#     # 使用SentenceTransformer计算嵌入向量
#     try:
#         # 加载语义模型
#         model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # # 使用GPU如果可用
#         # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # model.to(device)
#         model.to("cuda:6")  # 指定GPU号9
        
#         # 计算嵌入向量
#         embeddings = model.encode(texts, show_progress_bar=False)
        
#         # 计算余弦相似度
#         similarity_matrix = cosine_similarity(embeddings)
#         return similarity_matrix
#     except Exception as e:
#         print(f"计算语义嵌入向量时出错: {str(e)}")
#         # 如果无法计算，返回零矩阵
#         return np.zeros((len(texts), len(texts)))

# def select_diverse_news(items, count=5):
#     """贪心选择相互最不相似的新闻"""
#     if len(items) <= count:
#         return items
    
#     print("  计算语义相似度矩阵...")
#     similarity_matrix = compute_similarity_matrix(items)
#     n = len(items)
    
#     # 初始化已选和未选索引
#     selected_indices = []
#     remaining_indices = list(range(n))
    
#     # 选择第一篇文章（可以是最长的，或者随机选择）
#     # 这里我们选择摘要最长的，假设其可能包含更多信息
#     text_lengths = [len(items[i]['abstract']) for i in range(n)]
#     first_idx = text_lengths.index(max(text_lengths))
    
#     selected_indices.append(first_idx)
#     remaining_indices.remove(first_idx)
    
#     print(f"  已选择初始文章: {items[first_idx]['headline'][:50]}...")
    
#     # 贪心选择剩余的文章
#     while len(selected_indices) < count and remaining_indices:
#         # 计算每个候选文章与已选文章的总相似度
#         min_sim = float('inf')
#         next_idx = -1
        
#         for idx in remaining_indices:
#             # 计算与已选文章的平均相似度
#             avg_sim = sum(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices) / len(selected_indices)
            
#             if avg_sim < min_sim:
#                 min_sim = avg_sim
#                 next_idx = idx
        
#         if next_idx != -1:
#             selected_indices.append(next_idx)
#             remaining_indices.remove(next_idx)
#             print(f"  选择第{len(selected_indices)}篇文章 (相似度: {min_sim:.4f}): {items[next_idx]['headline'][:50]}...")
#         else:
#             break
    
#     # 返回选中的新闻
#     return [items[i] for i in selected_indices]

# def main():
#     # 加载原始新闻数据
#     input_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1_multi.jsonl"
#     output_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1_multi_selection5.jsonl"
    
#     news_data = load_jsonl(input_file)
#     print(f"共加载了 {len(news_data)} 条新闻")
    
#     # 按主题分组
#     topic_groups = defaultdict(list)
#     for item in news_data:
#         topic = item.get('topic', 'unknown')
#         topic_groups[topic].append(item)
    
#     # 选择每个主题最多样的5条新闻
#     selected_news = []
    
#     for topic, items in topic_groups.items():
#         print(f"处理主题 '{topic}'，共 {len(items)} 条新闻")
#         diverse_selection = select_diverse_news(items, count=5)
#         selected_news.extend(diverse_selection)
#         print(f"  选择了 {len(diverse_selection)} 条多样性新闻")
    
#     # 保存筛选结果
#     save_jsonl(selected_news, output_file)
#     print(f"筛选完成！已保存 {len(selected_news)} 条多样性新闻到 {output_file}")
    
#     # 按主题统计筛选结果
#     topic_counts = defaultdict(int)
#     for item in selected_news:
#         topic_counts[item.get('topic', 'unknown')] += 1
    
#     print("\n筛选结果统计：")
#     for topic, count in sorted(topic_counts.items()):
#         print(f"{topic}: {count}条新闻")

# if __name__ == "__main__":
#     main()








# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict

# def load_jsonl(file_path):
#     """加载JSONL格式数据"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return data

# def find_similar_news(items, similarity_threshold=0.75):
#     """查找相似度高的新闻"""
#     if len(items) <= 1:
#         return [], len(items)
    
#     # 合并标题和摘要比较
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
#     # 计算TF-IDF向量
#     vectorizer = TfidfVectorizer(stop_words='english')
#     try:
#         tfidf_matrix = vectorizer.fit_transform(texts)
#     except ValueError:
#         return [], len(texts)
    
#     # 计算余弦相似度
#     similarity_matrix = cosine_similarity(tfidf_matrix)
    
#     # 标记相似新闻组
#     similar_groups = []
#     processed = set()
    
#     for i in range(len(texts)):
#         if i in processed:
#             continue
        
#         group = [i]
#         processed.add(i)
        
#         for j in range(i+1, len(texts)):
#             if j not in processed and similarity_matrix[i, j] >= similarity_threshold:
#                 group.append(j)
#                 processed.add(j)
        
#         if len(group) > 1:
#             similar_groups.append(group)
    
#     # 计算不重复新闻数
#     unique_count = len(texts) - sum(len(group) - 1 for group in similar_groups)
    
#     return similar_groups, unique_count

# def analyze_news_diversity(news_data, similarity_threshold=0.75):
#     """分析每个主题的新闻多样性"""
#     # 按主题分组
#     topic_groups = defaultdict(list)
#     for item in news_data:
#         topic = item.get('topic', 'unknown')
#         topic_groups[topic].append(item)
    
#     # 初始化结果
#     results = {}
#     total_news = 0
#     total_unique = 0
    
#     # 分析每个主题
#     for topic, items in topic_groups.items():
#         total = len(items)
#         total_news += total
        
#         similar_groups, unique_count = find_similar_news(items, similarity_threshold)
#         total_unique += unique_count
        
#         # 计算重复率
#         dup_rate = (total - unique_count) / total * 100 if total > 0 else 0
        
#         results[topic] = {
#             'total': total,
#             'unique': unique_count,
#             'dup_rate': dup_rate,
#             'similar_groups': similar_groups,
#             'items': items
#         }
    
#     # 汇总数据
#     summary = {
#         'by_topic': results,
#         'total_news': total_news,
#         'total_unique': total_unique,
#         'overall_dup_rate': (total_news - total_unique) / total_news * 100 if total_news > 0 else 0
#     }
    
#     return summary

# def print_diversity_results(results):
#     """打印多样性分析结果"""
#     print("\n==== 每个主题的新闻多样性分析 ====")
#     print(f"{'主题':<15} {'总条数':<10} {'不重复条数':<15} {'重复率':<10}")
#     print("-" * 50)
    
#     for topic, data in sorted(results['by_topic'].items()):
#         print(f"{topic:<15} {data['total']:<10} {data['unique']:<15} {data['dup_rate']:.2f}%")
    
#     print("-" * 50)
#     print(f"{'总计':<15} {results['total_news']:<10} {results['total_unique']:<15} {results['overall_dup_rate']:.2f}%")

# # 主函数
# file_path = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1_multi.jsonl"
# news_data = load_jsonl(file_path)
# print(f"共加载了 {len(news_data)} 条新闻")
# results = analyze_news_diversity(news_data)
# print_diversity_results(results)

# # 输出每个主题的相似新闻组数量
# print("\n==== 重复新闻组统计 ====")
# for topic, data in sorted(results['by_topic'].items()):
#     similar_count = len(data['similar_groups'])
#     if similar_count > 0:
#         print(f"{topic:<15} 有 {similar_count} 组相似新闻")

# # 显示详细的重复新闻组
# def print_similar_groups(results, topic=None):
#     """打印指定主题的相似新闻组"""
#     topics = [topic] if topic else sorted(results['by_topic'].keys())
    
#     for t in topics:
#         data = results['by_topic'][t]
#         similar_groups = data['similar_groups']
#         if len(similar_groups) > 0:
#             print(f"\n主题: {t} (共 {len(similar_groups)} 组相似新闻)")
            
#             for i, group in enumerate(similar_groups[:3]):  # 只显示前3组
#                 print(f"\n  相似组 #{i+1}:")
#                 for idx in group:
#                     print(f"    - {data['items'][idx]['headline']}")
            
#             if len(similar_groups) > 3:
#                 print(f"    ... 还有 {len(similar_groups)-3} 组未显示")

# print_similar_groups(results)