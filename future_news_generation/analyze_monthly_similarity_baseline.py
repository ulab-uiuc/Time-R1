import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
from collections import defaultdict
import argparse
import torch

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

def extract_year_month(date_str):
    """从日期字符串中提取年月"""
    if not date_str:
        return None
    
    # 对于格式为 YYYY-MM 的日期
    if re.match(r'^\d{4}-\d{2}$', date_str):
        return date_str
    
    # 对于包含完整日期的格式 (如 YYYY-MM-DD...)
    match = re.search(r'(\d{4}-\d{2})', date_str)
    if match:
        return match.group(1)
    
    return None

def filter_news_by_month(news_list, year_month):
    """从新闻列表中筛选指定年月的新闻"""
    filtered_news = []
    for news in news_list:
        pub_date = news.get("pub_date", "")
        if pub_date.startswith(year_month):
            filtered_news.append(news)
    return filtered_news

def convert_numpy_types(obj):
    """递归地将NumPy类型转换为Python原生类型"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def find_most_similar_pairs(generated_news, real_news, model):
    """找出每条生成新闻与真实新闻中最相似的对应关系"""
    if not generated_news or not real_news:
        return [], 0.0
    
    # 预处理新闻文本
    generated_texts = []
    real_texts = []
    
    # 提取标题和摘要组合文本
    for news in generated_news:
        text = f"{news.get('headline', '')} {news.get('abstract', '')}"
        generated_texts.append(text)
    
    for news in real_news:
        text = f"{news.get('headline', '')} {news.get('abstract', '')}"
        real_texts.append(text)
    
    # 计算嵌入向量
    print(f"计算 {len(generated_texts)} 条生成新闻的嵌入向量...")
    generated_embeddings = model.encode(generated_texts, show_progress_bar=True)
    
    print(f"计算 {len(real_texts)} 条真实新闻的嵌入向量...")
    real_embeddings = model.encode(real_texts, show_progress_bar=True)
    
    # 计算相似度
    print("计算相似度矩阵...")
    similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
    # 输出调试信息
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    if similarity_matrix.shape[0] > 0 and similarity_matrix.shape[1] > 0:
        print(f"相似度矩阵示例(前5x5):")
        print(similarity_matrix[:min(5, similarity_matrix.shape[0]), :min(5, similarity_matrix.shape[1])])
        print(f"最大相似度: {np.max(similarity_matrix)}")
        print(f"平均相似度: {np.mean(similarity_matrix)}")
    
    # 找出每条生成新闻最相似的真实新闻
    most_similar_pairs = []
    
    print("查找最相似的新闻对...")
    for i in tqdm(range(len(generated_news))):
        similarities = similarity_matrix[i]
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        # 确保相似度不是零或NaN
        if np.isnan(max_similarity):
            max_similarity = 0.0
        
        most_similar_pairs.append({
            'generated_index': i,
            'real_index': int(max_idx),
            'similarity': float(max_similarity),
            'generated_headline': generated_news[i].get('headline', ''),
            'generated_abstract': generated_news[i].get('abstract', ''),
            'real_headline': real_news[max_idx].get('headline', ''),
            'real_abstract': real_news[max_idx].get('abstract', '')
        })
    
    # 计算平均相似度
    similarities = [pair['similarity'] for pair in most_similar_pairs]
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # 输出调试信息
    print(f"相似度值前5个: {similarities[:5]}")
    print(f"计算得到的平均相似度: {avg_similarity}")
    
    return most_similar_pairs, avg_similarity

def analyze_monthly_similarity(diverse_news_file, real_news_2024_file, real_news_2025_file, output_dir):
    """
    按月份分析生成新闻与真实新闻的相似度
    
    参数:
        diverse_news_file: 包含多样性筛选后的生成新闻的文件
        real_news_2024_file: 2024年真实新闻文件
        real_news_2025_file: 2025年真实新闻文件
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型
    print("加载嵌入模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to("cuda")  # 使用GPU加速
    
    # 加载生成的新闻数据
    print(f"加载生成的新闻数据: {diverse_news_file}")
    generated_news = load_jsonl(diverse_news_file)
    
    # 按月份分组
    generated_by_month = defaultdict(list)
    for news in generated_news:
        month = news.get('month', '') or news.get('target_date', '')
        if month:
            generated_by_month[month].append(news)
    
    print(f"生成新闻按月份分组完成，共 {len(generated_by_month)} 个月份")
    for month, news_list in sorted(generated_by_month.items()):
        print(f"  {month}: {len(news_list)} 条")
    
    # 加载真实新闻数据
    print(f"加载2024年真实新闻: {real_news_2024_file}")
    real_news_2024 = load_jsonl(real_news_2024_file)
    
    print(f"加载2025年真实新闻: {real_news_2025_file}")
    real_news_2025 = load_jsonl(real_news_2025_file)
    
    print(f"2024年真实新闻: {len(real_news_2024)}条")
    print(f"2025年真实新闻: {len(real_news_2025)}条")
    
    # 分析每个月份的相似度
    monthly_results = []
    
    for month, gen_news_list in sorted(generated_by_month.items()):
        print(f"\n=====================")
        print(f"分析 {month} 月份的新闻，共 {len(gen_news_list)} 条生成新闻")
        
        # 确定真实新闻的年份
        year = month.split('-')[0]
        real_news_source = real_news_2024 if year == '2024' else real_news_2025
        
        # 筛选同月份的真实新闻
        real_news_month = filter_news_by_month(real_news_source, month)
        print(f"找到 {len(real_news_month)} 条 {month} 的真实新闻")
        
        # 如果没有找到同月份的真实新闻，使用对应年份的所有新闻
        if len(real_news_month) == 0:
            print(f"警告: 未找到 {month} 的真实新闻，使用 {year} 年的所有新闻代替")
            real_news_month = real_news_source
        
        # 找出相似的新闻对
        similar_pairs, avg_similarity = find_most_similar_pairs(gen_news_list, real_news_month, model)
        
        # 记录结果
        monthly_results.append({
            'month': month,
            'generated_count': len(gen_news_list),
            'real_count': len(real_news_month),
            'avg_similarity': avg_similarity,
            'pairs': similar_pairs
        })
        
        # 保存每个月的详细结果
        month_output_dir = os.path.join(output_dir, f"similarity_{month}")
        if not os.path.exists(month_output_dir):
            os.makedirs(month_output_dir)
        
        # 保存相似对的CSV
        pairs_df = pd.DataFrame(convert_numpy_types(similar_pairs))
        pairs_df.to_csv(os.path.join(month_output_dir, "similar_pairs.csv"), index=False)
        
        # 创建相似度分布图
        plt.figure(figsize=(10, 6))
        sns.histplot([pair['similarity'] for pair in similar_pairs], bins=20, kde=True)
        plt.title(f'Distribution of Similarity Scores ({month})')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(month_output_dir, "similarity_distribution.png"), dpi=500, bbox_inches='tight')
        plt.close()
        
        # 创建包含前5对最相似新闻的报告
        top_count = min(5, len(similar_pairs))
        top_similar = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:top_count]
        
        with open(os.path.join(month_output_dir, "top_similar_news.md"), "w") as f:
            f.write(f"# {month} 月份最相似的 {top_count} 对新闻\n\n")
            for pair in top_similar:
                f.write(f"## 相似度: {pair['similarity']:.4f}\n\n")
                f.write("### 生成的新闻\n")
                f.write(f"**标题:** {pair['generated_headline']}\n\n")
                f.write(f"**摘要:** {pair['generated_abstract']}\n\n")
                f.write("### 真实新闻\n")
                f.write(f"**标题:** {pair['real_headline']}\n\n")
                f.write(f"**摘要:** {pair['real_abstract']}\n\n")
                f.write("---\n\n")
    
    # 生成汇总报告
    summary_df = pd.DataFrame([{
        'month': result['month'],
        'generated_count': result['generated_count'],
        'real_count': result['real_count'],
        'avg_similarity': result['avg_similarity']
    } for result in monthly_results])
    
    # 保存汇总数据
    summary_df.to_csv(os.path.join(output_dir, "monthly_similarity_summary.csv"), index=False)
    
    # 绘制月份相似度对比图
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='month', y='avg_similarity', data=summary_df)
    plt.title('平均相似度得分比较 (按月份)')
    plt.xlabel('月份')
    plt.ylabel('平均相似度')
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(i, row.avg_similarity + 0.01, f'{row.avg_similarity:.4f}', 
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_similarity_comparison.png"), dpi=500, bbox_inches='tight')
    
    # 生成月度汇总报告
    with open(os.path.join(output_dir, "monthly_similarity_report.md"), "w") as f:
        f.write("# 生成新闻与真实新闻相似度月度分析报告\n\n")
        f.write("## 月度相似度汇总\n\n")
        
        f.write("| 月份 | 生成新闻数量 | 真实新闻数量 | 平均相似度 |\n")
        f.write("|------|------------|--------------|----------|\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"| {row['month']} | {row['generated_count']} | {row['real_count']} | {row['avg_similarity']:.4f} |\n")
        
        f.write("\n## 月度相似度分析\n\n")
        f.write("![月度相似度比较](monthly_similarity_comparison.png)\n\n")
        
        f.write("## 各月份详细结果\n\n")
        for month in sorted(summary_df['month']):
            f.write(f"- [{month} 月份分析结果](similarity_{month}/top_similar_news.md)\n")
        
        f.write("\n## 数据说明\n\n")
        f.write("本分析比较了 deepseek r1 模型生成的未来新闻与真实新闻数据集的相似度。")
        f.write("相似度反映了生成内容与真实新闻在主题和风格上的接近程度，帮助评估模型生成的新闻在内容上的真实性。")
    
    print("\n分析完成!")
    print(f"报告已保存到 {output_dir} 目录")
    
    return monthly_results

def main():
    # 设置参数
    parser = argparse.ArgumentParser(description="分析 deepseek r1 模型生成新闻与真实新闻的相似度")
    
    parser.add_argument("--diverse_news_file", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_r1/all_diverse_news.jsonl", 
                       help="多样性筛选后的生成新闻文件")
    
    parser.add_argument("--real_news_2024_file", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl", 
                       help="2024年真实新闻数据文件")
    
    parser.add_argument("--real_news_2025_file", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl", 
                       help="2025年真实新闻数据文件")
    
    parser.add_argument("--output_dir", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/deepseekr1_results/similarity_analysis", 
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始分析 deepseek r1 模型生成新闻与真实新闻的相似度...")
    print(f"生成新闻文件: {args.diverse_news_file}")
    print(f"2024年真实新闻文件: {args.real_news_2024_file}")
    print(f"2025年真实新闻文件: {args.real_news_2025_file}")
    print(f"输出目录: {args.output_dir}")
    
    # 调用分析函数
    monthly_results = analyze_monthly_similarity(
        args.diverse_news_file,
        args.real_news_2024_file,
        args.real_news_2025_file,
        args.output_dir
    )
    
    # 输出总体结果
    if monthly_results:
        avg_similarities = [result['avg_similarity'] for result in monthly_results]
        overall_avg = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
        print(f"\n总体平均相似度: {overall_avg:.4f}")
        
        # 输出每个月份的平均相似度
        print("\n各月份平均相似度:")
        for result in sorted(monthly_results, key=lambda x: x['month']):
            print(f"  {result['month']}: {result['avg_similarity']:.4f}")
    
    print("\n分析完成！详细结果已保存到输出目录。")
    print("请查看输出目录中的 monthly_similarity_report.md 获取完整分析报告。")

if __name__ == "__main__":
    main()