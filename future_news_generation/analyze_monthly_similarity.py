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
    """Load JSONL format data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def extract_year_month(date_str):
    """Extracting the year and month from the date string"""
    if not date_str:
        return None
    
    # For dates with format YYYY-MM
    if re.match(r'^\d{4}-\d{2}$', date_str):
        return date_str
    
    # For formats containing full dates (such as YYYY-MM-DD...)
    match = re.search(r'(\d{4}-\d{2})', date_str)
    if match:
        return match.group(1)
    
    return None

def filter_news_by_month(news_list, year_month):
    """Filter news from a specified year and month from a news list"""
    filtered_news = []
    for news in news_list:
        pub_date = news.get("pub_date", "")
        if pub_date.startswith(year_month):
            filtered_news.append(news)
    return filtered_news

def convert_numpy_types(obj):
    """Convert NumPy type to Python native type recursively"""
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
    """Find out the most similar correspondence between each generated news and real news"""
    if not generated_news or not real_news:
        return [], 0.0
    
    # Preprocess news text
    generated_texts = []
    real_texts = []
    
    # Extract title and abstract text
    for news in generated_news:
        text = f"{news.get('headline', '')} {news.get('abstract', '')}"
        generated_texts.append(text)
    
    for news in real_news:
        text = f"{news.get('headline', '')} {news.get('abstract', '')}"
        real_texts.append(text)
    
    # Compute embed vectors
    print(f"Compute the embed vector of {len(generated_texts)} that generates news...")
    generated_embeddings = model.encode(generated_texts, show_progress_bar=True)
    
    print(f"Compute the embedded vector of {len(real_texts)} real news...")
    real_embeddings = model.encode(real_texts, show_progress_bar=True)
    
    # Calculate similarity
    print("Calculate the similarity matrix...")
    similarity_matrix = cosine_similarity(generated_embeddings, real_embeddings)
    
    # Output debugging information
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    if similarity_matrix.shape[0] > 0 and similarity_matrix.shape[1] > 0:
        print(f"Similarity matrix example (first 5x5):")
        print(similarity_matrix[:min(5, similarity_matrix.shape[0]), :min(5, similarity_matrix.shape[1])])
        print(f"Maximum similarity: {np.max(similarity_matrix)}")
        print(f"Average similarity: {np.mean(similarity_matrix)}")
    
    # Find out the most similar real news for each generated news
    most_similar_pairs = []
    
    print("Find the most similar news pairs...")
    for i in tqdm(range(len(generated_news))):
        similarities = similarity_matrix[i]
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        # Make sure the similarity is not zero or NaN
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
    
    # Calculate the average similarity
    similarities = [pair['similarity'] for pair in most_similar_pairs]
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # Output debugging information
    print(f"The first 5 similarity values: {similarities[:5]}")
    print(f"The calculated average similarity: {avg_similarity}")
    
    return most_similar_pairs, avg_similarity

def analyze_monthly_similarity(diverse_news_file, real_news_2024_file, real_news_2025_file, output_dir):
    """Analyze the similarity between generated news and real news by month
    
    parameter:
        diverse_news_file: A file that contains the generated news after diversity filtering
        real_news_2024_file: 2024 real news files
        real_news_2025_file: 2025 real news files
        output_dir: output directory"""
    # Create an output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loading the model
    print("Loading the embed model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to("cuda")  # Use GPU to speed up
    
    # Load the generated news data
    print(f"Load the generated news data: {diverse_news_file}")
    generated_news = load_jsonl(diverse_news_file)
    
    # Group by month
    generated_by_month = defaultdict(list)
    for news in generated_news:
        month = news.get('month', '') or news.get('target_date', '')
        if month:
            generated_by_month[month].append(news)
    
    print(f"Generate news is grouped by month, totaling {len(generated_by_month)} months")
    for month, news_list in sorted(generated_by_month.items()):
        print(f"{month}: {len(news_list)}")
    
    # Load real news data
    print(f"Loading 2024 real news: {real_news_2024_file}")
    real_news_2024 = load_jsonl(real_news_2024_file)
    
    print(f"Loading 2025 real news: {real_news_2025_file}")
    real_news_2025 = load_jsonl(real_news_2025_file)
    
    print(f"2024 real news: {len(real_news_2024)}")
    print(f"2025 real news: {len(real_news_2025)}")
    
    # Analyze the similarity of each month
    monthly_results = []
    
    for month, gen_news_list in sorted(generated_by_month.items()):
        print(f"\n=====================")
        print(f"Analyze the news in {month} months, a total of {len(gen_news_list)} generated news")
        
        # Determine the year of real news
        year = month.split('-')[0]
        real_news_source = real_news_2024 if year == '2024' else real_news_2025
        
        # Filter real news from the same month
        real_news_month = filter_news_by_month(real_news_source, month)
        print(f"Found {len(real_news_month)} {month} real news")
        
        # If no real news for the same month is found, use all news for the corresponding year
        if len(real_news_month) == 0:
            print(f"Warning: {month}'s real news was not found, use all news for {year} years instead")
            real_news_month = real_news_source
        
        # Find similar news pairs
        similar_pairs, avg_similarity = find_most_similar_pairs(gen_news_list, real_news_month, model)
        
        # Record the results
        monthly_results.append({
            'month': month,
            'generated_count': len(gen_news_list),
            'real_count': len(real_news_month),
            'avg_similarity': avg_similarity,
            'pairs': similar_pairs
        })
        
        # Save detailed results for each month
        month_output_dir = os.path.join(output_dir, f"similarity_{month}")
        if not os.path.exists(month_output_dir):
            os.makedirs(month_output_dir)
        
        # Save CSVs of similar pairs
        pairs_df = pd.DataFrame(convert_numpy_types(similar_pairs))
        pairs_df.to_csv(os.path.join(month_output_dir, "similar_pairs.csv"), index=False)
        
        # Create a similarity distribution map
        plt.figure(figsize=(10, 6))
        sns.histplot([pair['similarity'] for pair in similar_pairs], bins=20, kde=True)
        plt.title(f'Distribution of Similarity Scores ({month})')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(month_output_dir, "similarity_distribution.png"), dpi=500, bbox_inches='tight')
        plt.close()
        
        # Create a report with the top 5 pairs of the most similar news
        top_count = min(5, len(similar_pairs))
        top_similar = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:top_count]
        
        with open(os.path.join(month_output_dir, "top_similar_news.md"), "w") as f:
            f.write(f"# {month} The most similar month {top_count} to news\n\n")
            for pair in top_similar:
                f.write(f"## Similarity: {pair['similarity']:.4f}\n\n")
                f.write("### Generated News\n")
                f.write(f"**Title:** {pair['generated_headline']}\n\n")
                f.write(f"**Summary:** {pair['generated_abstract']}\n\n")
                f.write("###Real News\n")
                f.write(f"**Title:** {pair['real_headline']}\n\n")
                f.write(f"**Summary:** {pair['real_abstract']}\n\n")
                f.write("---\n\n")
    
    # Generate a summary report
    summary_df = pd.DataFrame([{
        'month': result['month'],
        'generated_count': result['generated_count'],
        'real_count': result['real_count'],
        'avg_similarity': result['avg_similarity']
    } for result in monthly_results])
    
    # Save summary data
    summary_df.to_csv(os.path.join(output_dir, "monthly_similarity_summary.csv"), index=False)
    
    # Draw a comparison chart of the similarity of months
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='month', y='avg_similarity', data=summary_df)
    plt.title('Comparison of Average Similarity Scores (by Month)')
    plt.xlabel('month')
    plt.ylabel('average similarity')
    plt.xticks(rotation=45)
    
    # Add a numerical label to the bar chart
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(i, row.avg_similarity + 0.01, f'{row.avg_similarity:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"monthly_similarity_comparison.png"), dpi=500, bbox_inches='tight')
    
    # Generate monthly summary reports
    with open(os.path.join(output_dir, "monthly_similarity_report.md"), "w") as f:
        f.write("# Generate monthly analysis report on news and real news similarity\n\n")
        f.write("## Monthly similarity summary\n\n")
        
        f.write("| Month | Number of news generated | Number of real news | Average similarity |\n")
        f.write("|------|------------|--------------|----------|\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"| {row['month']} | {row['generated_count']} | {row['real_count']} | {row['avg_similarity']:.4f} |\n")
        
        f.write("\n## Monthly Similarity Analysis\n\n")
        f.write("![Monthly_similarity_comparison.png)\n\n")
        
        f.write("## Detailed results for each month\n\n")
        for month in sorted(summary_df['month']):
            f.write(f"- [{month} Month Analysis Results](similarity_{month}/top_similar_news.md)\n")
        
        f.write("\n## Data Description\n\n")
        f.write("This analysis compares the similarity between future news generated by deepseek r1 model and real news datasets.")
        f.write("Similarity reflects the proximity of generated content to real news in terms of theme and style, helping to evaluate the authenticity of the news generated by the model in terms of content.")
    
    print("\nThe analysis is completed!")
    print(f"Reports have been saved to the {output_dir} directory")
    
    return monthly_results

def main():
    # Set parameters
    parser = argparse.ArgumentParser(description="Analysis of deepseek r1 model-generated news and real news")
    
    parser.add_argument("--diverse_news_file", type=str, 
                       default="Time-R1/future_news_generation/results/diverse_news_by_month_r1/all_diverse_news.jsonl", 
                       help="Generate news files after diversity filtering")
    
    parser.add_argument("--real_news_2024_file", type=str, 
                       default="Time-R1/datasets/nyt_years/2024.jsonl", 
                       help="2024 real news data files")
    
    parser.add_argument("--real_news_2025_file", type=str, 
                       default="Time-R1/datasets/nyt_years/2025.jsonl", 
                       help="2025 real news data files")
    
    parser.add_argument("--output_dir", type=str, 
                       default="Time-R1/future_news_generation/deepseekr1_results/similarity_analysis", 
                       help="Analysis results output directory")
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Start analyzing the similarity between deepseek r1 model generated news and real news...")
    print(f"Generate news files: {args.diverse_news_file}")
    print(f"2024 real news files: {args.real_news_2024_file}")
    print(f"2025 real news files: {args.real_news_2025_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Call the analysis function
    monthly_results = analyze_monthly_similarity(
        args.diverse_news_file,
        args.real_news_2024_file,
        args.real_news_2025_file,
        args.output_dir
    )
    
    # Output overall results
    if monthly_results:
        avg_similarities = [result['avg_similarity'] for result in monthly_results]
        overall_avg = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
        print(f"\nOverall average similarity: {overall_avg:.4f}")
        
        # Output average similarity for each month
        print("\nAverage similarity of each month:")
        for result in sorted(monthly_results, key=lambda x: x['month']):
            print(f"  {result['month']}: {result['avg_similarity']:.4f}")
    
    print("\nThe analysis is completed! The detailed results have been saved to the output directory.")
    print("Please check the monthly_similarity_report.md in the output directory for the complete analysis report.")

if __name__ == "__main__":
    main()