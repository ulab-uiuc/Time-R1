# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import os

# def load_jsonl(file_path):
#     """加载JSONL格式数据"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def preprocess_news(news, is_generated=False):
#     """预处理新闻数据，只保留标题"""
#     if is_generated:
#         # 生成的新闻格式
#         return {
#             'headline': news.get('headline', ''),
#             'abstract': news.get('abstract', ''),
#             'text': news.get('headline', '')  # 只使用标题
#         }
#     else:
#         # 真实新闻格式
#         return {
#             'headline': news.get('headline', ''),
#             'abstract': news.get('abstract', ''),
#             'text': news.get('headline', '')  # 只使用标题
#         }

# def compute_embeddings(texts, model):
#     """使用预训练模型计算文本嵌入向量"""
#     print(f"Computing embeddings for {len(texts)} texts...")
#     return model.encode(texts, show_progress_bar=True)

# def find_similar_news(generated_news, real_news, threshold=0.7, top_k=10):
#     """找出生成新闻与真实新闻中标题相似度高的对应关系"""
#     print("正在加载嵌入模型...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # 预处理新闻数据
#     print("预处理新闻数据...")
#     processed_generated = [preprocess_news(news, is_generated=True) for news in generated_news]
#     processed_real = [preprocess_news(news) for news in real_news]
    
#     # 准备文本数据
#     generated_texts = [news['text'] for news in processed_generated]
#     real_texts = [news['text'] for news in processed_real]
    
#     # 计算嵌入向量
#     generated_embeddings = compute_embeddings(generated_texts, model)
#     real_embeddings = compute_embeddings(real_texts, model)
    
#     # 计算相似度
#     print("计算相似度矩阵...")
#     similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
#     # 找出相似度高的新闻对
#     similar_pairs = []
    
#     print("查找高相似度新闻对...")
#     for i, gen_embed in enumerate(tqdm(generated_embeddings)):
#         # 找出与当前生成新闻最相似的前K个真实新闻
#         similarities = similarity_matrix[i]
#         top_indices = np.argsort(similarities)[::-1][:top_k]
        
#         for idx in top_indices:
#             similarity = similarities[idx]
#             if similarity >= threshold:
#                 similar_pairs.append({
#                     'generated_index': i,
#                     'real_index': idx,
#                     'similarity': similarity,
#                     'generated_headline': processed_generated[i]['headline'],
#                     'generated_abstract': processed_generated[i]['abstract'],
#                     'real_headline': processed_real[idx]['headline'],
#                     'real_abstract': processed_real[idx]['abstract'],
#                 })
    
#     # 按相似度降序排序
#     similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
#     return similar_pairs

# def analyze_similarity_results(similar_pairs, output_dir="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/similarity_analysis_headline"):
#     """分析相似度结果并生成报告"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 转换为DataFrame便于分析
#     df = pd.DataFrame(similar_pairs)
    
#     if df.empty:
#         print("没有找到相似度高于阈值的新闻对!")
#         stats = {
#             'count': 0,
#             'mean_similarity': 0.0,
#             'median_similarity': 0.0,
#             'max_similarity': 0.0,
#             'min_similarity': 0.0,
#             'similarity_threshold': 0.0,
#         }
#         with open(f"{output_dir}/similarity_stats.json", "w") as f:
#             json.dump(stats, f, indent=4)
#         return stats
    
#     # 保存详细结果
#     df.to_csv(f"{output_dir}/similar_headlines.csv", index=False)
    
#     # 创建相似度分布图
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['similarity'], bins=20, kde=True)
#     plt.title('Distribution of Headline Similarity Scores')
#     plt.xlabel('Similarity Score')
#     plt.ylabel('Count')
#     plt.savefig(f"{output_dir}/headline_similarity_distribution.png", dpi=500, bbox_inches='tight')
    
#     # 计算统计数据
#     stats = {
#         'count': int(len(similar_pairs)),
#         'mean_similarity': float(df['similarity'].mean()),
#         'median_similarity': float(df['similarity'].median()),
#         'max_similarity': float(df['similarity'].max()),
#         'min_similarity': float(df['similarity'].min()),
#         'similarity_threshold': float(df['similarity'].iloc[-1]) if len(df) > 0 else 0.0,
#     }
    
#     # 将统计数据保存为JSON
#     with open(f"{output_dir}/similarity_stats.json", "w") as f:
#         json.dump(stats, f, indent=4)
    
#     # 创建前20对最相似的新闻标题报告
#     top_count = min(20, len(df))
#     top_similar = df.head(top_count)
#     with open(f"{output_dir}/top_similar_headlines.md", "w") as f:
#         f.write(f"# 最相似的{top_count}对新闻标题\n\n")
#         for _, row in top_similar.iterrows():
#             f.write(f"## 相似度: {float(row['similarity']):.4f}\n\n")
#             f.write("### 生成的新闻\n")
#             f.write(f"**标题:** {row['generated_headline']}\n\n")
#             f.write(f"**摘要:** {row['generated_abstract']}\n\n")
#             f.write("### 真实新闻\n")
#             f.write(f"**标题:** {row['real_headline']}\n\n")
#             f.write(f"**摘要:** {row['real_abstract']}\n\n")
#             f.write("---\n\n")
    
#     return stats

# def main():
#     # 加载数据
#     print("加载生成的新闻数据...")
#     generated_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1.2.jsonl')
    
#     print("加载真实的新闻数据...")
#     real_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025-1.jsonl')
    
#     print(f"加载完成。生成新闻: {len(generated_news)}条, 真实新闻: {len(real_news)}条")
    
#     # 找出相似的新闻对，提高阈值
#     similar_pairs = find_similar_news(generated_news, real_news, threshold=0.75, top_k=5)
    
#     # 分析结果
#     stats = analyze_similarity_results(similar_pairs)
    
#     print("\n分析完成!")
#     print(f"共找到{stats['count']}对标题相似度较高的新闻对")
#     print(f"平均相似度: {stats['mean_similarity']:.4f}")
#     print(f"最高相似度: {stats['max_similarity']:.4f}")
#     print(f"详细结果已保存到headline_similarity目录")

# if __name__ == "__main__":
#     main()








import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def load_jsonl(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_news(news, is_generated=False):
    """预处理新闻数据，返回统一格式"""
    if is_generated:
        # 生成的新闻格式
        return {
            'headline': news.get('headline', ''),
            'abstract': news.get('abstract', ''),
            'text': f"{news.get('headline', '')} {news.get('abstract', '')}"
        }
    else:
        # 真实新闻格式
        return {
            'headline': news.get('headline', ''),
            'abstract': news.get('abstract', ''),
            'text': f"{news.get('headline', '')} {news.get('abstract', '')}"
        }

def compute_embeddings(texts, model):
    """使用预训练模型计算文本嵌入向量"""
    print(f"Computing embeddings for {len(texts)} texts...")
    return model.encode(texts, show_progress_bar=True)

def find_similar_news(generated_news, real_news, threshold=0.7, top_k=10):
    """找出生成新闻与真实新闻中相似度高的对应关系"""
    print("正在加载嵌入模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 预处理新闻数据
    print("预处理新闻数据...")
    processed_generated = [preprocess_news(news, is_generated=True) for news in generated_news]
    processed_real = [preprocess_news(news) for news in real_news]
    
    # 准备文本数据
    generated_texts = [news['text'] for news in processed_generated]
    real_texts = [news['text'] for news in processed_real]
    
    # 计算嵌入向量
    generated_embeddings = compute_embeddings(generated_texts, model)
    real_embeddings = compute_embeddings(real_texts, model)
    
    # 计算相似度
    print("计算相似度矩阵...")
    similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
    # 找出相似度高的新闻对
    similar_pairs = []
    
    print("查找高相似度新闻对...")
    for i, gen_embed in enumerate(tqdm(generated_embeddings)):
        # 找出与当前生成新闻最相似的前K个真实新闻
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                similar_pairs.append({
                    'generated_index': i,
                    'real_index': idx,
                    'similarity': similarity,
                    'generated_headline': processed_generated[i]['headline'],
                    'generated_abstract': processed_generated[i]['abstract'],
                    'real_headline': processed_real[idx]['headline'],
                    'real_abstract': processed_real[idx]['abstract'],
                })
    
    # 按相似度降序排序
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_pairs

def analyze_similarity_results(similar_pairs, output_dir="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/similarity_analysis_2025_01_r1_batch"):
    """分析相似度结果并生成报告"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 转换为DataFrame便于分析
    df = pd.DataFrame(similar_pairs)
    
    # 保存详细结果
    df.to_csv(f"{output_dir}/similar_news_pairs.csv", index=False)
    
    # 创建相似度分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['similarity'], bins=20, kde=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/similarity_distribution.png", dpi=500, bbox_inches='tight')
    
    # # 计算统计数据
    # stats = {
    #     'count': len(similar_pairs),
    #     'mean_similarity': df['similarity'].mean(),
    #     'median_similarity': df['similarity'].median(),
    #     'max_similarity': df['similarity'].max(),
    #     'min_similarity': df['similarity'].min(),
    #     'similarity_threshold': df['similarity'].iloc[-1],  # 最低的相似度
    # }
    # 计算统计数据
    stats = {
        'count': int(len(similar_pairs)),
        'mean_similarity': float(df['similarity'].mean()),
        'median_similarity': float(df['similarity'].median()),
        'max_similarity': float(df['similarity'].max()),
        'min_similarity': float(df['similarity'].min()),
        'similarity_threshold': float(df['similarity'].iloc[-1]) if len(df) > 0 else 0.0,
    }
    
    # 将统计数据保存为JSON
    with open(f"{output_dir}/similarity_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    # 创建前10对最相似的新闻的报告
    top_similar = df.head(10)
    with open(f"{output_dir}/top_similar_news.md", "w") as f:
        f.write("# 最相似的10对新闻\n\n")
        for _, row in top_similar.iterrows():
            f.write(f"## 相似度: {row['similarity']:.4f}\n\n")
            f.write("### 生成的新闻\n")
            f.write(f"**标题:** {row['generated_headline']}\n\n")
            f.write(f"**摘要:** {row['generated_abstract']}\n\n")
            f.write("### 真实新闻\n")
            f.write(f"**标题:** {row['real_headline']}\n\n")
            f.write(f"**摘要:** {row['real_abstract']}\n\n")
            f.write("---\n\n")
    
    return stats

def main():
    # 加载数据
    print("加载生成的新闻数据...")
    # generated_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1.2.jsonl')
    generated_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_r1_batch.jsonl')
    
    print("加载真实的新闻数据...")
    real_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025-1.jsonl')
    
    print(f"加载完成。生成新闻: {len(generated_news)}条, 真实新闻: {len(real_news)}条")
    
    # 找出相似的新闻对
    similar_pairs = find_similar_news(generated_news, real_news, threshold=0.6, top_k=5)
    
    # 分析结果
    stats = analyze_similarity_results(similar_pairs)
    
    print("\n分析完成!")
    print(f"共找到{stats['count']}对相似度较高的新闻对")
    print(f"平均相似度: {stats['mean_similarity']:.4f}")
    print(f"最高相似度: {stats['max_similarity']:.4f}")
    print(f"详细结果已保存到similarity_analysis目录")

if __name__ == "__main__":
    main()






# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import os
# from datetime import datetime

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

# def filter_news_by_month(news_list, year_month="2024-11"):
#     """从新闻列表中筛选指定年月的新闻"""
#     filtered_news = []
#     for news in news_list:
#         pub_date = news.get("pub_date", "")
#         if pub_date.startswith(year_month):
#             filtered_news.append(news)
#     return filtered_news

# def preprocess_news(news, is_generated=False):
#     """预处理新闻数据，返回统一格式"""
#     if is_generated:
#         # 生成的新闻格式
#         return {
#             'headline': news.get('headline', ''),
#             'abstract': news.get('abstract', ''),
#             'text': f"{news.get('headline', '')} {news.get('abstract', '')}"
#         }
#     else:
#         # 真实新闻格式
#         return {
#             'headline': news.get('headline', ''),
#             'abstract': news.get('abstract', ''),
#             'text': f"{news.get('headline', '')} {news.get('abstract', '')}"
#         }

# def compute_embeddings(texts, model):
#     """使用预训练模型计算文本嵌入向量"""
#     print(f"Computing embeddings for {len(texts)} texts...")
#     return model.encode(texts, show_progress_bar=True)

# def find_similar_news(generated_news, real_news, threshold=0.7, top_k=10):
#     """找出生成新闻与真实新闻中相似度高的对应关系"""
#     print("正在加载嵌入模型...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # 预处理新闻数据
#     print("预处理新闻数据...")
#     processed_generated = [preprocess_news(news, is_generated=True) for news in generated_news]
#     processed_real = [preprocess_news(news) for news in real_news]
    
#     # 准备文本数据
#     generated_texts = [news['text'] for news in processed_generated]
#     real_texts = [news['text'] for news in processed_real]
    
#     # 计算嵌入向量
#     generated_embeddings = compute_embeddings(generated_texts, model)
#     real_embeddings = compute_embeddings(real_texts, model)
    
#     # 计算相似度
#     print("计算相似度矩阵...")
#     similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
#     # 找出相似度高的新闻对
#     similar_pairs = []
    
#     print("查找高相似度新闻对...")
#     for i, gen_embed in enumerate(tqdm(generated_embeddings)):
#         # 找出与当前生成新闻最相似的前K个真实新闻
#         similarities = similarity_matrix[i]
#         top_indices = np.argsort(similarities)[::-1][:top_k]
        
#         for idx in top_indices:
#             similarity = similarities[idx]
#             if similarity >= threshold:
#                 similar_pairs.append({
#                     'generated_index': i,
#                     'real_index': idx,
#                     'similarity': similarity,
#                     'generated_headline': processed_generated[i]['headline'],
#                     'generated_abstract': processed_generated[i]['abstract'],
#                     'real_headline': processed_real[idx]['headline'],
#                     'real_abstract': processed_real[idx]['abstract'],
#                 })
    
#     # 按相似度降序排序
#     similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
#     return similar_pairs

# def analyze_similarity_results(similar_pairs, output_dir="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/similarity_analysis_2024_11_t1"):
#     """分析相似度结果并生成报告"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 转换为DataFrame便于分析
#     df = pd.DataFrame(similar_pairs)
    
#     if df.empty:
#         print("没有找到相似度高于阈值的新闻对!")
#         stats = {
#             'count': 0,
#             'mean_similarity': 0.0,
#             'median_similarity': 0.0,
#             'max_similarity': 0.0,
#             'min_similarity': 0.0,
#             'similarity_threshold': 0.0,
#         }
#         with open(f"{output_dir}/similarity_stats.json", "w") as f:
#             json.dump(stats, f, indent=4)
#         return stats
    
#     # 保存详细结果
#     df.to_csv(f"{output_dir}/similar_news_pairs.csv", index=False)
    
#     # 创建相似度分布图
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['similarity'], bins=20, kde=True)
#     plt.title('Distribution of Similarity Scores (2024-11)')
#     plt.xlabel('Similarity Score')
#     plt.ylabel('Count')
#     plt.savefig(f"{output_dir}/similarity_distribution.png", dpi=500, bbox_inches='tight')
    
#     # 计算统计数据
#     stats = {
#         'count': int(len(similar_pairs)),
#         'mean_similarity': float(df['similarity'].mean()),
#         'median_similarity': float(df['similarity'].median()),
#         'max_similarity': float(df['similarity'].max()),
#         'min_similarity': float(df['similarity'].min()),
#         'similarity_threshold': float(df['similarity'].iloc[-1]) if len(df) > 0 else 0.0,
#     }
    
#     # 将统计数据保存为JSON
#     with open(f"{output_dir}/similarity_stats.json", "w") as f:
#         json.dump(stats, f, indent=4)
    
#     # 创建前10对最相似的新闻的报告
#     top_count = min(10, len(df))
#     top_similar = df.head(top_count)
#     with open(f"{output_dir}/top_similar_news.md", "w") as f:
#         f.write(f"# 最相似的{top_count}对新闻 (2024-11)\n\n")
#         for _, row in top_similar.iterrows():
#             f.write(f"## 相似度: {float(row['similarity']):.4f}\n\n")
#             f.write("### 生成的新闻\n")
#             f.write(f"**标题:** {row['generated_headline']}\n\n")
#             f.write(f"**摘要:** {row['generated_abstract']}\n\n")
#             f.write("### 真实新闻\n")
#             f.write(f"**标题:** {row['real_headline']}\n\n")
#             f.write(f"**摘要:** {row['real_abstract']}\n\n")
#             f.write("---\n\n")
    
#     return stats

# def main():
#     # 加载生成的2024-11新闻数据
#     print("加载生成的2024-11新闻数据...")
#     generated_news = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2024_11_360_t1.jsonl')
    
#     # 加载2024年真实新闻数据并筛选11月份的
#     print("加载2024年真实新闻数据...")
#     real_news_2024 = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl')
    
#     print("筛选2024年11月的新闻...")
#     real_news_2024_11 = filter_news_by_month(real_news_2024, "2024-11")
    
#     print(f"加载完成。生成新闻: {len(generated_news)}条, 真实新闻(2024-11): {len(real_news_2024_11)}条")
    
#     # 如果没有找到2024年11月的真实新闻，使用替代方案
#     if len(real_news_2024_11) == 0:
#         print("警告: 未找到2024年11月的真实新闻数据，使用可用的2024年数据替代")
#         real_news_2024_11 = real_news_2024
#         print(f"使用全部2024年的{len(real_news_2024)}条新闻数据进行比较...")
    
#     # 找出相似的新闻对
#     similar_pairs = find_similar_news(generated_news, real_news_2024_11, threshold=0.6, top_k=5)
    
#     # 分析结果
#     stats = analyze_similarity_results(similar_pairs)
    
#     print("\n分析完成!")
#     print(f"共找到{stats['count']}对相似度较高的新闻对")
#     print(f"平均相似度: {stats['mean_similarity']:.4f}")
#     print(f"最高相似度: {stats['max_similarity']:.4f}")
#     print(f"详细结果已保存到similarity_analysis_2024_11目录")

# if __name__ == "__main__":
#     main()