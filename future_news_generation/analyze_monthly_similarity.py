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

def extract_headline_abstract_from_prompt(prompt_content):
    """从测试集的prompt字段提取headline和abstract"""
    headline_pattern = r"Headline:\s*(.*?)(?:\n|$)"
    abstract_pattern = r"Abstract:\s*(.*?)(?:\n|$)"
    
    headline_match = re.search(headline_pattern, prompt_content)
    abstract_match = re.search(abstract_pattern, prompt_content)
    
    headline = headline_match.group(1).strip() if headline_match else ""
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    return {
        "headline": headline,
        "abstract": abstract
    }

def load_test_news_data(parquet_file):
    """
    从test_time_prediction.parquet文件加载测试集数据,
    提取headline和abstract，并按月份组织
    """
    # 加载parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 按月份组织新闻数据
    news_by_month = defaultdict(list)
    
    # 遍历每个样本
    for _, row in df.iterrows():
        # 提取年月信息
        year = row['extra_info']['year']
        month = row['extra_info']['month']
        month_key = f"{year}-{month:02d}"  # 格式化为 YYYY-MM
        
        # 提取headline和abstract
        prompt_content = row['prompt'][0]['content']
        news_item = extract_headline_abstract_from_prompt(prompt_content)
        
        # 添加发布日期信息
        news_item['pub_date'] = month_key
        
        # 将新闻添加到对应月份
        news_by_month[month_key].append(news_item)
    
    return news_by_month

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

def analyze_monthly_similarity_with_test_data(diverse_news_file, test_data_file, output_dir):
    """
    按月份分析生成新闻与测试集真实新闻的相似度
    
    参数:
        diverse_news_file: 包含多样性筛选后的生成新闻的文件
        test_data_file: 测试集数据文件(parquet格式)
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型
    print("加载嵌入模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 使用GPU(如果可用)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")
    model.to("cuda")
    # model.to("cuda:0")
    
    # 加载生成的新闻数据
    print("加载生成的新闻数据...")
    generated_news = load_jsonl(diverse_news_file)
    
    # 按月份分组
    generated_by_month = defaultdict(list)
    for news in generated_news:
        month = news.get('month', '')
        if month:
            generated_by_month[month].append(news)
    
    print(f"生成新闻按月份分组完成，共 {len(generated_by_month)} 个月份")
    for month, news_list in generated_by_month.items():
        print(f"  {month}: {len(news_list)} 条")
    
    # 加载测试集真实新闻数据
    print(f"从测试集加载真实新闻数据: {test_data_file}")
    real_news_by_month = load_test_news_data(test_data_file)
    
    print("测试集真实新闻数量统计:")
    for month, news_list in sorted(real_news_by_month.items()):
        print(f"  {month}: {len(news_list)} 条")
    
    # 分析每个月份的相似度
    monthly_results = []
    
    for month, gen_news_list in sorted(generated_by_month.items()):
        print(f"\n=====================")
        print(f"分析 {month} 月份的新闻，共 {len(gen_news_list)} 条生成新闻")
        
        # 获取对应月份的真实新闻
        real_news_month = real_news_by_month.get(month, [])
        print(f"找到 {len(real_news_month)} 条 {month} 的测试集真实新闻")
        
        # 如果没有找到同月份的真实新闻，使用相近月份的数据
        if len(real_news_month) == 0:
            print(f"警告: 未找到 {month} 的测试集真实新闻，尝试使用相近月份")
            
            # 解析年月
            year_str, month_str = month.split('-')
            year = int(year_str)
            month_num = int(month_str)
            
            # 尝试相近月份
            candidates = []
            # 向前查找一个月
            prev_month = month_num - 1
            prev_year = year
            if prev_month < 1:
                prev_month = 12
                prev_year -= 1
            candidates.append(f"{prev_year}-{prev_month:02d}")
            
            # 向后查找一个月
            next_month = month_num + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            candidates.append(f"{next_year}-{next_month:02d}")
            
            # 尝试各候选月份
            for candidate in candidates:
                if candidate in real_news_by_month and len(real_news_by_month[candidate]) > 0:
                    real_news_month = real_news_by_month[candidate]
                    print(f"使用 {candidate} 月份的 {len(real_news_month)} 条测试集真实新闻替代")
                    break
            
            # 如果仍未找到，使用所有测试集数据
            if len(real_news_month) == 0:
                all_test_news = []
                for m_news in real_news_by_month.values():
                    all_test_news.extend(m_news)
                real_news_month = all_test_news
                print(f"未找到相近月份的测试集数据，使用所有 {len(real_news_month)} 条测试集真实新闻")
        
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
        f.write("# 生成新闻与测试集真实新闻相似度月度分析报告\n\n")
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
        f.write("本分析使用的真实新闻数据来自测试集，确保模型在训练时未接触过这些数据，以避免数据泄露问题。")
    
    print("\n分析完成!")
    print(f"报告已保存到 {output_dir} 目录")
    
    return monthly_results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Analyze monthly news similarity using test data")
#     parser.add_argument("--diverse_news_file", type=str, 
#                         default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_test/all_diverse_news_hpx2_filtered.jsonl", 
#                         help="Combined file with all diverse generated news")
#     parser.add_argument("--test_data_file", type=str, 
#                         default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_prediction.parquet", 
#                         help="Test dataset file in parquet format")
#     parser.add_argument("--output_dir", type=str, 
#                         default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/monthly_similarity_analysis_test_data_test/test2", 
#                         help="Output directory for similarity analysis")
    
#     args = parser.parse_args()
    
#     # 分析生成新闻与测试集真实新闻的相似度
#     analyze_monthly_similarity_with_test_data(
#         args.diverse_news_file,
#         args.test_data_file,
#         args.output_dir
#     )



def main():
    # 设置参数
    parser = argparse.ArgumentParser(description="分析 llama 8B 生成新闻与测试集真实新闻的相似度")
    
    parser.add_argument("--diverse_news_file", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/llama31_results/diverse_news/all_diverse_news.jsonl", 
                       help="多样性筛选后的生成新闻文件")
    
    parser.add_argument("--test_data_file", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_prediction.parquet", 
                       help="测试集数据文件(parquet格式)")
    
    parser.add_argument("--output_dir", type=str, 
                       default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/llama31_results/similarity_analysis", 
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始分析 llama 8B 生成新闻与测试集真实新闻的相似度...")
    print(f"生成新闻文件: {args.diverse_news_file}")
    print(f"测试集数据文件: {args.test_data_file}")
    print(f"输出目录: {args.output_dir}")
    
    # 调用分析函数
    monthly_results = analyze_monthly_similarity_with_test_data(
        args.diverse_news_file,
        args.test_data_file,
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









# import json
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import os
# from collections import defaultdict
# import re
# import argparse

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

# def extract_year_month(date_str):
#     """从日期字符串中提取年月"""
#     if not date_str:
#         return None
    
#     # 对于格式为 YYYY-MM 的日期
#     if re.match(r'^\d{4}-\d{2}$', date_str):
#         return date_str
    
#     # 对于包含完整日期的格式 (如 YYYY-MM-DD...)
#     match = re.search(r'(\d{4}-\d{2})', date_str)
#     if match:
#         return match.group(1)
    
#     return None

# def filter_news_by_month(news_list, year_month):
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
#     print(f"计算 {len(texts)} 条文本的嵌入向量...")
#     return model.encode(texts, show_progress_bar=True)

# def find_most_similar_pairs(generated_news, real_news, model):
#     """找出每条生成新闻与真实新闻中最相似的对应关系"""
#     if not generated_news or not real_news:
#         return [], 0.0
    
#     # 预处理新闻数据
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
    
#     # 找出每条生成新闻最相似的真实新闻
#     most_similar_pairs = []
    
#     print("查找最相似的新闻对...")
#     for i, gen_embed in enumerate(tqdm(generated_embeddings)):
#         # 找出与当前生成新闻最相似的真实新闻
#         similarities = similarity_matrix[i]
#         max_idx = np.argmax(similarities)
#         max_similarity = similarities[max_idx]
        
#         most_similar_pairs.append({
#             'generated_index': i,
#             'real_index': max_idx,
#             'similarity': float(max_similarity),
#             'generated_headline': processed_generated[i]['headline'],
#             'generated_abstract': processed_generated[i]['abstract'],
#             'real_headline': processed_real[max_idx]['headline'],
#             'real_abstract': processed_real[max_idx]['abstract'],
#         })
    
#     # 计算平均相似度
#     avg_similarity = np.mean([pair['similarity'] for pair in most_similar_pairs])
    
#     return most_similar_pairs, avg_similarity

# def analyze_monthly_similarity(diverse_news_file, output_dir):
#     """按月份分析生成新闻与真实新闻的相似度"""
#     # 创建输出目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 加载模型
#     print("加载嵌入模型...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # 加载生成的新闻数据
#     print("加载生成的新闻数据...")
#     generated_news = load_jsonl(diverse_news_file)
    
#     # 按月份分组
#     generated_by_month = defaultdict(list)
#     for news in generated_news:
#         month = news.get('month', '')
#         if month:
#             generated_by_month[month].append(news)
    
#     print(f"生成新闻按月份分组完成，共 {len(generated_by_month)} 个月份")
    
#     # 加载真实新闻数据
#     real_news_2024 = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl')
#     real_news_2025 = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl')
    
#     print(f"加载真实新闻完成，2024年: {len(real_news_2024)}条, 2025年: {len(real_news_2025)}条")
    
#     # 分析每个月份的相似度
#     monthly_results = []
    
#     for month, news_list in sorted(generated_by_month.items()):
#         print(f"\n=====================")
#         print(f"分析 {month} 月份的新闻，共 {len(news_list)} 条")
        
#         # 确定真实新闻的年份
#         year = month.split('-')[0]
#         real_news_source = real_news_2024 if year == '2024' else real_news_2025
        
#         # 筛选同月份的真实新闻
#         real_news_month = filter_news_by_month(real_news_source, month)
#         print(f"找到 {len(real_news_month)} 条 {month} 的真实新闻")
        
#         # 如果没有找到同月份的真实新闻，使用对应年份的所有新闻
#         if len(real_news_month) == 0:
#             print(f"警告: 未找到 {month} 的真实新闻，使用 {year} 年的所有新闻代替")
#             real_news_month = real_news_source
        
#         # 找出相似的新闻对
#         similar_pairs, avg_similarity = find_most_similar_pairs(news_list, real_news_month, model)
        
#         # 记录结果
#         monthly_results.append({
#             'month': month,
#             'generated_count': len(news_list),
#             'real_count': len(real_news_month),
#             'avg_similarity': avg_similarity,
#             'pairs': similar_pairs
#         })
        
#         # 保存每个月的详细结果
#         month_output_dir = os.path.join(output_dir, f"similarity_{month}")
#         if not os.path.exists(month_output_dir):
#             os.makedirs(month_output_dir)
        
#         # 保存相似对的CSV
#         pairs_df = pd.DataFrame(similar_pairs)
#         pairs_df.to_csv(os.path.join(month_output_dir, "similar_pairs.csv"), index=False)
        
#         # 创建相似度分布图
#         plt.figure(figsize=(10, 6))
#         sns.histplot([pair['similarity'] for pair in similar_pairs], bins=20, kde=True)
#         plt.title(f'Distribution of Similarity Scores ({month})')
#         plt.xlabel('Similarity Score')
#         plt.ylabel('Count')
#         plt.savefig(os.path.join(month_output_dir, "similarity_distribution.png"), dpi=500, bbox_inches='tight')
#         plt.close()
        
#         # 创建包含前5对最相似新闻的报告
#         top_count = min(5, len(similar_pairs))
#         top_similar = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:top_count]
        
#         with open(os.path.join(month_output_dir, "top_similar_news.md"), "w") as f:
#             f.write(f"# {month} 月份最相似的 {top_count} 对新闻\n\n")
#             for pair in top_similar:
#                 f.write(f"## 相似度: {pair['similarity']:.4f}\n\n")
#                 f.write("### 生成的新闻\n")
#                 f.write(f"**标题:** {pair['generated_headline']}\n\n")
#                 f.write(f"**摘要:** {pair['generated_abstract']}\n\n")
#                 f.write("### 真实新闻\n")
#                 f.write(f"**标题:** {pair['real_headline']}\n\n")
#                 f.write(f"**摘要:** {pair['real_abstract']}\n\n")
#                 f.write("---\n\n")
    
#     # 生成汇总报告
#     summary_df = pd.DataFrame([{
#         'month': result['month'],
#         'generated_count': result['generated_count'],
#         'real_count': result['real_count'],
#         'avg_similarity': result['avg_similarity']
#     } for result in monthly_results])
    
#     # 保存汇总数据
#     summary_df.to_csv(os.path.join(output_dir, "monthly_similarity_summary.csv"), index=False)
    
#     # 绘制月份相似度对比图
#     plt.figure(figsize=(12, 8))
#     ax = sns.barplot(x='month', y='avg_similarity', data=summary_df)
#     plt.title('平均相似度得分比较 (按月份)')
#     plt.xlabel('月份')
#     plt.ylabel('平均相似度')
#     plt.xticks(rotation=45)
    
#     # 在柱状图上添加数值标签
#     for i, row in enumerate(summary_df.itertuples()):
#         ax.text(i, row.avg_similarity + 0.01, f'{row.avg_similarity:.4f}', 
#                 ha='center', va='bottom', rotation=0, fontsize=9)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "monthly_similarity_comparison.png"), dpi=500, bbox_inches='tight')
    
#     # 生成月度汇总报告
#     with open(os.path.join(output_dir, "monthly_similarity_report.md"), "w") as f:
#         f.write("# 生成新闻与真实新闻相似度月度分析报告\n\n")
#         f.write("## 月度相似度汇总\n\n")
        
#         f.write("| 月份 | 生成新闻数量 | 真实新闻数量 | 平均相似度 |\n")
#         f.write("|------|------------|--------------|----------|\n")
        
#         for _, row in summary_df.iterrows():
#             f.write(f"| {row['month']} | {row['generated_count']} | {row['real_count']} | {row['avg_similarity']:.4f} |\n")
        
#         f.write("\n## 月度相似度分析\n\n")
#         f.write("![月度相似度比较](monthly_similarity_comparison.png)\n\n")
        
#         f.write("## 各月份详细结果\n\n")
#         for month in sorted(summary_df['month']):
#             f.write(f"- [{month} 月份分析结果](similarity_{month}/top_similar_news.md)\n")
    
#     print("\n分析完成!")
#     print(f"报告已保存到 {output_dir} 目录")
    
#     return monthly_results

# if __name__ == "__main__":
#     # 输入输出文件和目录
#     diverse_news_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month/all_diverse_news.jsonl"
#     output_dir = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/monthly_similarity_analysis"
    
#     analyze_monthly_similarity(diverse_news_file, output_dir)









# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import os

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

# def preprocess_news(news, is_generated=False):
#     """预处理新闻数据，返回统一格式的文本"""
#     return f"{news.get('headline', '')} {news.get('abstract', '')}"

# def calculate_similarity(generated_news, real_news, model_name='all-MiniLM-L6-v2'):
#     """计算生成新闻与真实新闻之间的相似度"""
#     # 加载模型
#     print("加载嵌入模型...")
#     model = SentenceTransformer(model_name)
#     model.to("cuda:1")  # 使用GPU加速
    
#     # 预处理新闻文本
#     generated_texts = [preprocess_news(news, is_generated=True) for news in generated_news]
#     real_texts = [preprocess_news(news) for news in real_news]
    
#     # 计算嵌入向量
#     print(f"计算 {len(generated_texts)} 条生成新闻的嵌入向量...")
#     generated_embeddings = model.encode(generated_texts, show_progress_bar=True)
    
#     print(f"计算 {len(real_texts)} 条真实新闻的嵌入向量...")
#     real_embeddings = model.encode(real_texts, show_progress_bar=True)
    
#     # 计算相似度矩阵
#     print("计算相似度矩阵...")
#     similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
#     # 找出每条生成新闻的最大相似度
#     max_similarities = []
    
#     print("查找最相似新闻对...")
#     for i in tqdm(range(len(generated_news))):
#         max_sim = np.max(similarity_matrix[i])
#         max_idx = np.argmax(similarity_matrix[i])
        
#         max_similarities.append({
#             'generated_index': i,
#             'real_index': max_idx,
#             'similarity': float(max_sim),
#             'generated_headline': generated_news[i]['headline'],
#             'real_headline': real_news[max_idx].get('headline', ''),
#             'generated_abstract': generated_news[i]['abstract'],
#             'real_abstract': real_news[max_idx].get('abstract', '')
#         })
    
#     # 计算平均相似度
#     avg_similarity = np.mean([item['similarity'] for item in max_similarities])
    
#     return max_similarities, avg_similarity

# def find_most_similar_pairs_0(generated_news, real_news, model):
#     """找出每条生成新闻与真实新闻中最相似的对应关系"""
#     if not generated_news or not real_news:
#         return [], 0.0
    
#     # 预处理新闻数据 - 直接获取文本字符串
#     generated_texts = [preprocess_news(news, is_generated=True) for news in generated_news]
#     real_texts = [preprocess_news(news) for news in real_news]
    
#     # 不需要再从processed_generated中提取text字段，因为preprocess_news直接返回文本
    
#     # 计算嵌入向量
#     print(f"计算 {len(generated_texts)} 条生成新闻的嵌入向量...")
#     generated_embeddings = model.encode(generated_texts, show_progress_bar=True)
    
#     print(f"计算 {len(real_texts)} 条真实新闻的嵌入向量...")
#     real_embeddings = model.encode(real_texts, show_progress_bar=True)
    
#     # 计算相似度
#     print("计算相似度矩阵...")
#     similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
#     # 找出每条生成新闻最相似的真实新闻
#     most_similar_pairs = []
    
#     print("查找最相似的新闻对...")
#     for i, gen_embed in enumerate(tqdm(generated_embeddings)):
#         # 找出与当前生成新闻最相似的真实新闻
#         similarities = similarity_matrix[i]
#         max_idx = np.argmax(similarities)
#         max_similarity = similarities[max_idx]
        
#         most_similar_pairs.append({
#             'generated_index': i,
#             'real_index': max_idx,
#             'similarity': float(max_similarity),
#             'generated_headline': generated_news[i].get('headline', ''),
#             'generated_abstract': generated_news[i].get('abstract', ''),
#             'real_headline': real_news[max_idx].get('headline', ''),
#             'real_abstract': real_news[max_idx].get('abstract', '')
#         })
    
#     # 计算平均相似度
#     avg_similarity = np.mean([pair['similarity'] for pair in most_similar_pairs])
    
#     return most_similar_pairs, avg_similarity

# def find_most_similar_pairs(generated_news, real_news, model):
#     """找出每条生成新闻与真实新闻中最相似的对应关系"""
#     if not generated_news or not real_news:
#         return [], 0.0
    
#     # 预处理新闻文本
#     generated_texts = []
#     real_texts = []
    
#     # 提取标题和摘要组合文本
#     for news in generated_news:
#         text = f"{news.get('headline', '')} {news.get('abstract', '')}"
#         generated_texts.append(text)
    
#     for news in real_news:
#         text = f"{news.get('headline', '')} {news.get('abstract', '')}"
#         real_texts.append(text)
    
#     # 计算嵌入向量
#     print(f"计算 {len(generated_texts)} 条生成新闻的嵌入向量...")
#     generated_embeddings = model.encode(generated_texts, show_progress_bar=True)
    
#     print(f"计算 {len(real_texts)} 条真实新闻的嵌入向量...")
#     real_embeddings = model.encode(real_texts, show_progress_bar=True)
    
#     # 计算相似度
#     print("计算相似度矩阵...")
#     similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
#     # 找出每条生成新闻最相似的真实新闻
#     most_similar_pairs = []
    
#     print("查找最相似的新闻对...")
#     for i in tqdm(range(len(generated_news))):
#         similarities = similarity_matrix[i]
#         max_idx = np.argmax(similarities)
#         max_similarity = similarities[max_idx]
        
#         # 确保相似度不是零
#         if np.isnan(max_similarity):
#             max_similarity = 0.0
        
#         most_similar_pairs.append({
#             'generated_index': i,
#             'real_index': int(max_idx),  # 确保是Python原生int类型
#             'similarity': float(max_similarity),  # 确保是Python原生float类型
#             'generated_headline': generated_news[i].get('headline', ''),
#             'generated_abstract': generated_news[i].get('abstract', ''),
#             'real_headline': real_news[max_idx].get('headline', ''),
#             'real_abstract': real_news[max_idx].get('abstract', '')
#         })
    
#     # 计算平均相似度并添加调试信息
#     similarities = [pair['similarity'] for pair in most_similar_pairs]
#     avg_similarity = np.mean(similarities) if similarities else 0.0
    
#     # 输出调试信息
#     print(f"相似度值示例: {similarities[:5]}")
#     print(f"计算得到的平均相似度: {avg_similarity}")
    
#     return most_similar_pairs, avg_similarity

# def convert_numpy_types(obj):
#     """递归地将NumPy类型转换为Python原生类型"""
#     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
#                          np.uint8, np.uint16, np.uint32, np.uint64)):
#         return int(obj)
#     elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.bool_)):
#         return bool(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {key: convert_numpy_types(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_types(item) for item in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_numpy_types(item) for item in obj)
#     else:
#         return obj

# def analyze_monthly_similarity(diverse_news_dir, output_dir):
#     """
#     按月份分析生成新闻与真实新闻的相似度
    
#     参数:
#         diverse_news_dir: 包含多样性筛选后的月度新闻的目录
#         output_dir: 输出目录
#     """
#     # 创建输出目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # 加载模型
#     print("加载嵌入模型...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     model.to("cuda:1")  # 使用GPU加速
    
#     # 查找所有月份的筛选后新闻文件
#     news_files = [f for f in os.listdir(diverse_news_dir) 
#                  if f.startswith("diverse_news_") and f.endswith(".jsonl") 
#                  and not f == "all_diverse_news.jsonl"]
    
#     # 按月份排序
#     news_files.sort()
    
#     print(f"找到{len(news_files)}个月份的多样性新闻文件")
    
#     # 加载真实新闻数据
#     real_news_2024 = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2024.jsonl')
#     real_news_2025 = load_jsonl('/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl')
    
#     print(f"加载真实新闻完成，2024年: {len(real_news_2024)}条, 2025年: {len(real_news_2025)}条")
    
#     # 分析每个月份的相似度
#     monthly_results = []
    
#     for file_name in news_files:
#         # 从文件名解析月份
#         month_match = re.search(r'diverse_news_(\d{4}-\d{2})', file_name)
#         if not month_match:
#             continue
            
#         month = month_match.group(1)
        
#         print(f"\n=====================")
#         print(f"分析 {month} 月份的新闻")
        
#         # 加载当月筛选后的新闻
#         file_path = os.path.join(diverse_news_dir, file_name)
#         news_list = load_jsonl(file_path)
        
#         print(f"加载了 {len(news_list)} 条 {month} 月份的多样性新闻")
        
#         # 确定真实新闻的年份
#         year = month.split('-')[0]
#         real_news_source = real_news_2024 if year == '2024' else real_news_2025
        
#         # 筛选同月份的真实新闻
#         real_news_month = filter_news_by_month(real_news_source, month)
#         print(f"找到 {len(real_news_month)} 条 {month} 的真实新闻")
        
#         # 如果没有找到同月份的真实新闻，使用对应年份的所有新闻
#         if len(real_news_month) == 0:
#             print(f"警告: 未找到 {month} 的真实新闻，使用 {year} 年的所有新闻代替")
#             real_news_month = real_news_source
        
#         # 找出相似的新闻对
#         similar_pairs, avg_similarity = find_most_similar_pairs(news_list, real_news_month, model)
        
#         # 记录结果
#         monthly_results.append({
#             'month': month,
#             'generated_count': len(news_list),
#             'real_count': len(real_news_month),
#             'avg_similarity': avg_similarity,
#             'pairs': similar_pairs
#         })
        
#         # 创建月份输出目录
#         month_output_dir = os.path.join(output_dir, f"similarity_{month}")
#         if not os.path.exists(month_output_dir):
#             os.makedirs(month_output_dir)
        
#         # 保存相似对的CSV和JSON (注意处理NumPy类型)
#         pairs_df = pd.DataFrame(convert_numpy_types(similar_pairs))
#         pairs_df.to_csv(os.path.join(month_output_dir, "similar_pairs.csv"), index=False)
        
#         # 创建相似度分布图
#         plt.figure(figsize=(10, 6))
#         sns.histplot([pair['similarity'] for pair in similar_pairs], bins=20, kde=True)
#         plt.title(f'Distribution of Similarity Scores ({month})')
#         plt.xlabel('Similarity Score')
#         plt.ylabel('Count')
#         plt.savefig(os.path.join(month_output_dir, "similarity_distribution.png"), dpi=300, bbox_inches='tight')
#         plt.close()

#         with open(os.path.join(month_output_dir, "similarity_results.json"), "w") as f:
#             json_data = {
#                 "month": month,
#                 "average_similarity": avg_similarity,
#                 "generated_count": len(news_list),
#                 "real_count": len(real_news_month),
#                 "similar_pairs": similar_pairs
#             }
#             # 转换NumPy类型后再保存
#             json_data = convert_numpy_types(json_data)
#             json.dump(json_data, f, indent=2)
        
#     # 生成汇总报告
#     summary_df = pd.DataFrame([{
#         'month': result['month'],
#         'generated_count': result['generated_count'],
#         'real_count': result['real_count'],
#         'avg_similarity': result['avg_similarity']
#     } for result in monthly_results])
    
#     # 保存汇总数据
#     summary_df.to_csv(os.path.join(output_dir, "monthly_similarity_summary.csv"), index=False)
    
#     # 绘制月份相似度对比图
#     plt.figure(figsize=(12, 8))
#     ax = sns.barplot(x='month', y='avg_similarity', data=summary_df)
#     plt.title('平均相似度得分比较 (按月份)')
#     plt.xlabel('月份')
#     plt.ylabel('平均相似度')
#     plt.xticks(rotation=45)
    
#     # 在柱状图上添加数值标签
#     for i, row in enumerate(summary_df.itertuples()):
#         ax.text(i, row.avg_similarity + 0.01, f'{row.avg_similarity:.4f}', 
#                 ha='center', va='bottom', rotation=0, fontsize=9)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "monthly_similarity_comparison.png"), dpi=300, bbox_inches='tight')
    
#     # 生成月度汇总报告
#     with open(os.path.join(output_dir, "monthly_similarity_report.md"), "w") as f:
#         f.write("# 生成新闻与真实新闻相似度月度分析报告\n\n")
#         f.write("## 月度相似度汇总\n\n")
        
#         f.write("| 月份 | 生成新闻数量 | 真实新闻数量 | 平均相似度 |\n")
#         f.write("|------|------------|--------------|----------|\n")
        
#         for _, row in summary_df.iterrows():
#             f.write(f"| {row['month']} | {row['generated_count']} | {row['real_count']} | {row['avg_similarity']:.4f} |\n")
        
#         f.write("\n## 月度相似度分析\n\n")
#         f.write("![月度相似度比较](monthly_similarity_comparison.png)\n\n")
        
#         f.write("## 各月份详细结果\n\n")
#         for month in sorted(summary_df['month']):
#             f.write(f"- [{month} 月份分析结果](similarity_{month}/top_similar_news.md)\n")
    
#     print("\n分析完成!")
#     print(f"报告已保存到 {output_dir} 目录")
    
#     return monthly_results

# def main():
#     parser = argparse.ArgumentParser(description="Analyze monthly news similarity")
#     parser.add_argument("--diverse_news_dir", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month", help="Directory containing diverse news by month")
#     parser.add_argument("--output_dir", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/monthly_similarity_analysis", help="Output directory for similarity analysis")
    
#     args = parser.parse_args()
    
#     # 分析多样性新闻与真实新闻的相似度
#     analyze_monthly_similarity(
#         args.diverse_news_dir,
#         args.output_dir
#     )

# def main_single_month():
#     # 输出目录
#     output_dir = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/monthly_similarity_analysis/2025_01"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 加载生成的未来新闻
#     generated_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_360_t1_multi_selection5.jsonl"
#     generated_news = load_jsonl(generated_file)
#     print(f"加载了 {len(generated_news)} 条生成的2025-01新闻")
    
#     # 加载2025年的真实新闻
#     real_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl"
#     all_real_news = load_jsonl(real_file)
#     print(f"加载了 {len(all_real_news)} 条2025年真实新闻")
    
#     # 筛选2025-01月的真实新闻
#     real_news_2025_01 = [news for news in all_real_news if news.get('pub_date', '').startswith('2025-01')]
#     print(f"找到 {len(real_news_2025_01)} 条2025年1月的真实新闻")
    
#     # 如果没有2025-01的真实新闻，使用所有2025年的新闻
#     if len(real_news_2025_01) == 0:
#         print(f"警告: 未找到2025-01的真实新闻，使用所有2025年的新闻")
#         real_news_2025_01 = all_real_news
    
#     # 计算相似度
#     similar_pairs, avg_similarity = calculate_similarity(generated_news, real_news_2025_01)
    
#     # 输出结果
#     print(f"\n生成新闻与真实新闻的平均最大相似度: {avg_similarity:.4f}")
    
#     # 保存相似度结果
#     with open(os.path.join(output_dir, "similarity_results.json"), "w") as f:
#         json_data = {
#             "average_similarity": avg_similarity,
#             "similar_pairs": similar_pairs
#         }
#         # 转换NumPy类型后再保存
#         json_data = convert_numpy_types(json_data)
#         json.dump(json_data, f, indent=2)
    
#     # 生成详细报告
#     with open(os.path.join(output_dir, "similarity_report.md"), "w") as f:
#         f.write("# 2025年1月生成新闻与真实新闻相似度分析\n\n")
#         f.write(f"## 平均最大相似度: {avg_similarity:.4f}\n\n")
#         f.write("## 相似度分布\n\n")
        
#         # 按相似度分组统计
#         ranges = {
#             "0.9-1.0": 0,
#             "0.8-0.9": 0,
#             "0.7-0.8": 0,
#             "0.6-0.7": 0,
#             "0.5-0.6": 0,
#             "<0.5": 0
#         }
        
#         for pair in similar_pairs:
#             sim = pair['similarity']
#             if sim >= 0.9:
#                 ranges["0.9-1.0"] += 1
#             elif sim >= 0.8:
#                 ranges["0.8-0.9"] += 1
#             elif sim >= 0.7:
#                 ranges["0.7-0.8"] += 1
#             elif sim >= 0.6:
#                 ranges["0.6-0.7"] += 1
#             elif sim >= 0.5:
#                 ranges["0.5-0.6"] += 1
#             else:
#                 ranges["<0.5"] += 1
        
#         f.write("| 相似度范围 | 新闻对数量 | 百分比 |\n")
#         f.write("|------------|------------|--------|\n")
#         for range_name, count in ranges.items():
#             percent = count / len(similar_pairs) * 100
#             f.write(f"| {range_name} | {count} | {percent:.2f}% |\n")
        
#         # 相似度最高的10对新闻
#         f.write("\n## 相似度最高的10对新闻\n\n")
#         top_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:10]
#         for i, pair in enumerate(top_pairs):
#             f.write(f"### {i+1}. 相似度: {pair['similarity']:.4f}\n\n")
#             f.write("#### 生成新闻\n")
#             f.write(f"**标题:** {pair['generated_headline']}\n\n")
#             f.write(f"**摘要:** {pair['generated_abstract']}\n\n")
#             f.write("#### 真实新闻\n")
#             f.write(f"**标题:** {pair['real_headline']}\n\n")
#             f.write(f"**摘要:** {pair['real_abstract']}\n\n")
#             f.write("---\n\n")
        
#         # 相似度最低的5对新闻
#         f.write("\n## 相似度最低的5对新闻\n\n")
#         bottom_pairs = sorted(similar_pairs, key=lambda x: x['similarity'])[:5]
#         for i, pair in enumerate(bottom_pairs):
#             f.write(f"### {i+1}. 相似度: {pair['similarity']:.4f}\n\n")
#             f.write("#### 生成新闻\n")
#             f.write(f"**标题:** {pair['generated_headline']}\n\n")
#             f.write(f"**摘要:** {pair['generated_abstract']}\n\n")
#             f.write("#### 真实新闻\n")
#             f.write(f"**标题:** {pair['real_headline']}\n\n")
#             f.write(f"**摘要:** {pair['real_abstract']}\n\n")
#             f.write("---\n\n")
    
#     # 显示相似度最高的5对新闻
#     print("\n相似度最高的5对新闻:")
#     for i, pair in enumerate(top_pairs[:5]):
#         print(f"{i+1}. 相似度: {pair['similarity']:.4f}")
#         print(f"   生成: {pair['generated_headline']}")
#         print(f"   真实: {pair['real_headline']}")
#         print("")

# if __name__ == "__main__":
#     main()