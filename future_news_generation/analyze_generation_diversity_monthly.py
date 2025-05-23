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
    """Load JSONL format data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                continue
    return data

# Regular expressions match Chinese characters (including the main ranges of simplified and traditional Chinese)
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
def is_mainly_english(text, chinese_threshold=0):
    """Check if the text is mainly in English.
    Returns False if langdetect detection fails, is detected in non-English, or the Chinese character ratio exceeds the threshold."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return False # Handle empty or non-string input

    try:
        # 1. Use langdetect for preliminary testing
        detected_lang = detect(text)
        if detected_lang != 'en':
            # print(f" Langdetect identified non-English: {detected_lang} in '{text[:50]}...'") # Optional: Debugging information
            return False

        # 2. Check the Chinese character ratio
        chinese_chars = chinese_char_pattern.findall(text)
        chinese_char_count = sum(len(s) for s in chinese_chars)
        total_chars = len(text)

        if total_chars > 0:
            chinese_ratio = chinese_char_count / total_chars
            if chinese_ratio > chinese_threshold:
                # print(f" Chinese char ratio exceeded threshold: {chinese_ratio:.2f} in '{text[:50]}...'") # Optional: Debugging information
                return False

        # Pass all checks
        return True

    except LangDetectException:
        # langdetect detection failed, conservative processing, return False
        # print(f" Langdetect failed for text: {text[:50]}...") # Optional: debug information
        return False
    except Exception as e:
        # Catch other potential errors
        print(f"  Error in is_mainly_english for text '{text[:50]}...': {e}")
        return False

def save_jsonl(data, file_path):
    """Save JSONL format data"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def compute_similarity_matrix(items):
    """Calculate semantic similarity matrix using SentenceTransformer"""
    if len(items) <= 1:
        return np.zeros((1, 1))
    
    # Merge Title and Summary Comparison
    texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
    # Use SentenceTransformer to compute embedded vectors
    try:
        # Loading semantic model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use the GPU if available
        model.to("cuda")  # Specify the GPU number
        
        # Compute embed vectors
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        print(f"An error occurred while calculating semantic embedding vector: {str(e)}")
        # If it cannot be calculated, return the zero matrix
        return np.zeros((len(texts), len(texts)))

def select_diverse_news(items, count=5):
    """Greedy chooses the most dissimilar news"""
    if len(items) <= count:
        return items
    
    print("Calculate semantic similarity matrix...")
    similarity_matrix = compute_similarity_matrix(items)
    n = len(items)
    
    # Initialize selected and unselected indexes
    selected_indices = []
    remaining_indices = list(range(n))
    
    # Select the first post (can be the longest, or randomly selected)
    # Here we choose the one with the longest summary, assuming it may contain more information
    text_lengths = [len(items[i]['abstract']) for i in range(n)]
    first_idx = text_lengths.index(max(text_lengths))
    
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    print(f"Initial article selected: {items[first_idx]['headline'][:50]}...")
    
    # Greedy to choose the remaining articles
    while len(selected_indices) < count and remaining_indices:
        # Calculate the total similarity between each candidate article and the selected article
        min_sim = float('inf')
        next_idx = -1
        
        for idx in remaining_indices:
            # Calculate the average similarity to the selected article
            avg_sim = sum(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices) / len(selected_indices)
            
            if avg_sim < min_sim:
                min_sim = avg_sim
                next_idx = idx
        
        if next_idx != -1:
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
            print(f"Select the {len(selected_indices)} article (similarity: {min_sim:.4f}): {items[next_idx]['headline'][:50]}...")
        else:
            break
    
    # Return to selected news
    return [items[i] for i in selected_indices]

def process_monthly_news(input_dir, output_dir, count_per_topic=5):
    """Process news data for multiple months, select the most diverse n news for each topic each month
    
    parameter:
        input_dir: directory containing monthly news data
        output_dir: output directory"""
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find news files for all months
    news_files = [f for f in os.listdir(input_dir) if f.startswith("future_news_") and f.endswith(".jsonl")]
    
    # Sort by month
    news_files.sort()
    
    print(f"Find the news file for the month {len(news_files)}")
    
    # Process each month
    all_selected_news = []
    
    for file_name in news_files:
        # parse month from file name
        month_match = re.search(r'future_news_(\d{4}-\d{2})', file_name)
        if not month_match:
            continue
            
        month = month_match.group(1)
        
        print(f"\nProcessing month: {month}")
        
        # Load news data for the month
        file_path = os.path.join(input_dir, file_name)
        news_data = load_jsonl(file_path)
        
        print(f"Loaded {len(news_data)} news")
        
        # Filter non-English news and clean up titles
        processed_news = []
        skipped_non_english = 0
        for item in news_data:
            item['month'] = month
            headline = item.get('headline', '')
            abstract = item.get('abstract', '')
            content = f"{headline} {abstract}"

            if is_mainly_english(content):
                # Clean up '[' and ']' in the title
                if isinstance(headline, str):
                    item['headline'] = headline.replace('[', '').replace(']', '')
                processed_news.append(item)
            else:
                skipped_non_english += 1
                # print(f" Skip non-English news: {headline[:30]}...") # Optional: Debugging information

        print(f"Skipped {skipped_non_english} news that is not in English or contains too much Chinese")
        print(f"Process {len(processed_news)} English news")
        
        # Group by topic
        topic_groups = defaultdict(list)
        for item in news_data:
            topic = item.get('topic', 'unknown')
            topic_groups[topic].append(item)
        
        # Choose the most diverse n news for each topic
        month_selected_news = []
        
        for topic, items in topic_groups.items():
            print(f"Handle topic '{topic}', total {len(items)} news")
            diverse_selection = select_diverse_news(items, count=count_per_topic)
            month_selected_news.extend(diverse_selection)
            print(f"{len(diverse_selection)} diversity news selected")
        
        # Save the filter results for the month
        month_output_file = os.path.join(output_dir, f"diverse_news_{month}.jsonl")
        save_jsonl(month_selected_news, month_output_file)
        print(f"Saved {len(month_selected_news)} Diversity news to {month_output_file}")
        
        all_selected_news.extend(month_selected_news)
    
    # Save filter results for all months
    all_output_file = os.path.join(output_dir, "all_diverse_news.jsonl")
    save_jsonl(all_selected_news, all_output_file)
    print(f"\nFiltering is complete! Saved Total {len(all_selected_news)} Diversity News to {all_output_file}")
    
    # Filter results by month and topic statistics
    print("\nFilter results by month and topic statistics:")
    month_topic_counts = defaultdict(lambda: defaultdict(int))
    
    for item in all_selected_news:
        month = item.get('month', 'unknown')
        topic = item.get('topic', 'unknown')
        month_topic_counts[month][topic] += 1
    
    for month in sorted(month_topic_counts.keys()):
        print(f"\nMonth: {month}")
        topic_total = 0
        for topic, count in sorted(month_topic_counts[month].items()):
            print(f"{topic}: {count} News")
            topic_total += count
        print(f"Total: {topic_total} news")
    
    return all_selected_news

def main():
    # # Read CSV file
    # df = pd.read_csv('Time-R1/future_news_generation/llama31_results/future_news_llama31_parsed.csv')

    # # Create an output directory
    # output_dir = 'Time-R1/future_news_generation/llama31_results/monthly_data/'
    # os.makedirs(output_dir, exist_ok=True)

    # # Group by month
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

    # # Save JSONL files grouped by month
    # for month, items in monthly_data.items():
    #     output_file = os.path.join(output_dir, f'future_news_{month}.jsonl')
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for item in items:
    #             f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # print(f'Save {len(items)} News to {output_file}')

    # print('Conversion is completed!')




    parser = argparse.ArgumentParser(description="Select diverse news from monthly generated news")
    parser.add_argument("--output_dir", type=str, default="Time-R1/future_news_generation/results/diverse_news_by_month_test/", help="Directory containing monthly news files")
    parser.add_argument("--input_dir", type=str, default="Time-R1/future_news_generation/results/future_news_test", help="Output directory for diverse news")
    parser.add_argument("--count_per_topic", type=int, default=5, help="Number of news to select per topic")
    
    args = parser.parse_args()
    
    # Process monthly news
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
# """Load JSONL format data"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return data

# def save_jsonl(data, file_path):
# """Save JSONL format data"""
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# def compute_similarity_matrix_fast(items):
# """Computing the similarity matrix"""
#     if len(items) <= 1:
#         return np.zeros((1, 1))
    
# # Merge Title and Summary Comparison
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
# # Calculate TF-IDF vectors
#     vectorizer = TfidfVectorizer(stop_words='english')
#     try:
#         tfidf_matrix = vectorizer.fit_transform(texts)
#         similarity_matrix = cosine_similarity(tfidf_matrix)
#         return similarity_matrix
#     except ValueError:
# # If it cannot be calculated, return the zero matrix
#         return np.zeros((len(texts), len(texts)))

# def compute_similarity_matrix(items):
# """Use SentenceTransformer to calculate semantic similarity matrix""""
#     if len(items) <= 1:
#         return np.zeros((1, 1))
    
# # Merge Title and Summary Comparison
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
# # Use SentenceTransformer to compute embedded vectors
#     try:
# # Loading semantic model
#         model = SentenceTransformer('all-MiniLM-L6-v2')
        
# # # Use GPU if available
#         # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # model.to(device)
# model.to("cuda:6") # Specify GPU number 9
        
# # Compute embedded vectors
#         embeddings = model.encode(texts, show_progress_bar=False)
        
# # Calculate cosine similarity
#         similarity_matrix = cosine_similarity(embeddings)
#         return similarity_matrix
#     except Exception as e:
# print(f"Error calculating semantic embedding vector: {str(e)}")
# # If it cannot be calculated, return the zero matrix
#         return np.zeros((len(texts), len(texts)))

# def select_diverse_news(items, count=5):
# """Greedy chooses the most dissimilar news"""
#     if len(items) <= count:
#         return items
    
# print(" Calculate semantic similarity matrix...")
#     similarity_matrix = compute_similarity_matrix(items)
#     n = len(items)
    
# # Initialize selected and unselected indexes
#     selected_indices = []
#     remaining_indices = list(range(n))
    
# # Select the first post (can be the longest, or randomly selected)
# # Here we choose the one with the longest summary, assuming it may contain more information
#     text_lengths = [len(items[i]['abstract']) for i in range(n)]
#     first_idx = text_lengths.index(max(text_lengths))
    
#     selected_indices.append(first_idx)
#     remaining_indices.remove(first_idx)
    
# print(f" Initial article selected: {items[first_idx]['headline'][:50]}...")
    
# # Greedy to choose the remaining articles
#     while len(selected_indices) < count and remaining_indices:
# # Calculate the total similarity between each candidate article and the selected article
#         min_sim = float('inf')
#         next_idx = -1
        
#         for idx in remaining_indices:
# # Calculate the average similarity to selected articles
#             avg_sim = sum(similarity_matrix[idx][sel_idx] for sel_idx in selected_indices) / len(selected_indices)
            
#             if avg_sim < min_sim:
#                 min_sim = avg_sim
#                 next_idx = idx
        
#         if next_idx != -1:
#             selected_indices.append(next_idx)
#             remaining_indices.remove(next_idx)
# print(f" Select the {len(selected_indices)} article (similarity: {min_sim:.4f}): {items[next_idx]['headline'][:50]}...")
#         else:
#             break
    
# # Return to selected news
#     return [items[i] for i in selected_indices]

# def main():
# # Load original news data
#     input_file = "Time-R1/future_news_generation/results/future_news_2025_01_360_t1_multi.jsonl"
#     output_file = "Time-R1/future_news_generation/results/future_news_2025_01_360_t1_multi_selection5.jsonl"
    
#     news_data = load_jsonl(input_file)
# print(f"Total {len(news_data)} news loaded")
    
# # Group by topic
#     topic_groups = defaultdict(list)
#     for item in news_data:
#         topic = item.get('topic', 'unknown')
#         topic_groups[topic].append(item)
    
# # Choose the 5 news items with the most diverse range of each topic
#     selected_news = []
    
#     for topic, items in topic_groups.items():
# print(f"handling topic '{topic}', total {len(items)} news")
#         diverse_selection = select_diverse_news(items, count=5)
#         selected_news.extend(diverse_selection)
# print(f" {len(diverse_selection)} diversity news")
    
# # Save filter results
#     save_jsonl(selected_news, output_file)
# print(f"Filter completed! Saved {len(selected_news)} Diversity news to {output_file}")
    
# # Filter results by topic statistics
#     topic_counts = defaultdict(int)
#     for item in selected_news:
#         topic_counts[item.get('topic', 'unknown')] += 1
    
# print("\nFilter result statistics:")
#     for topic, count in sorted(topic_counts.items()):
# print(f"{topic}: {count}news")

# if __name__ == "__main__":
#     main()








# import json
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict

# def load_jsonl(file_path):
# """Load JSONL format data"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return data

# def find_similar_news(items, similarity_threshold=0.75):
# """Find news with high similarity"""
#     if len(items) <= 1:
#         return [], len(items)
    
# # Merge Title and Summary Comparison
#     texts = [f"{item['headline']} {item['abstract']}" for item in items]
    
# # Calculate TF-IDF vectors
#     vectorizer = TfidfVectorizer(stop_words='english')
#     try:
#         tfidf_matrix = vectorizer.fit_transform(texts)
#     except ValueError:
#         return [], len(texts)
    
# # Calculate cosine similarity
#     similarity_matrix = cosine_similarity(tfidf_matrix)
    
# # Tag similar news groups
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
    
# # Calculate the number of non-repetitive news
#     unique_count = len(texts) - sum(len(group) - 1 for group in similar_groups)
    
#     return similar_groups, unique_count

# def analyze_news_diversity(news_data, similarity_threshold=0.75):
# """Analyze news diversity for each topic"""
# # Group by topic
#     topic_groups = defaultdict(list)
#     for item in news_data:
#         topic = item.get('topic', 'unknown')
#         topic_groups[topic].append(item)
    
# # Initialization result
#     results = {}
#     total_news = 0
#     total_unique = 0
    
# # Analyze each topic
#     for topic, items in topic_groups.items():
#         total = len(items)
#         total_news += total
        
#         similar_groups, unique_count = find_similar_news(items, similarity_threshold)
#         total_unique += unique_count
        
# # Calculate the repetition rate
#         dup_rate = (total - unique_count) / total * 100 if total > 0 else 0
        
#         results[topic] = {
#             'total': total,
#             'unique': unique_count,
#             'dup_rate': dup_rate,
#             'similar_groups': similar_groups,
#             'items': items
#         }
    
# # Summary of data
#     summary = {
#         'by_topic': results,
#         'total_news': total_news,
#         'total_unique': total_unique,
#         'overall_dup_rate': (total_news - total_unique) / total_news * 100 if total_news > 0 else 0
#     }
    
#     return summary

# def print_diversity_results(results):
# """Print diversity analysis results"""
# print("\n==== News diversity analysis for each topic =====")
# print(f"{'Theme':<15} {'Total number of entries':<10} {'No number of repetitions':<15} {'Repetition rate':<10}")
#     print("-" * 50)
    
#     for topic, data in sorted(results['by_topic'].items()):
#         print(f"{topic:<15} {data['total']:<10} {data['unique']:<15} {data['dup_rate']:.2f}%")
    
#     print("-" * 50)
# print(f"{'total':<15} {results['total_news']:<10} {results['total_unique']:<15} {results['overall_dup_rate']:.2f}%")

# # Main function
# file_path = "Time-R1/future_news_generation/results/future_news_2025_01_360_t1_multi.jsonl"
# news_data = load_jsonl(file_path)
# print(f"Total {len(news_data)} news loaded")
# results = analyze_news_diversity(news_data)
# print_diversity_results(results)

# # Output the number of similar news groups for each topic
# print("\n==== Repeat news group statistics =====")
# for topic, data in sorted(results['by_topic'].items()):
#     similar_count = len(data['similar_groups'])
#     if similar_count > 0:
# print(f"{topic:<15} has {similar_count} group similar news")

# # Show detailed duplicate news groups
# def print_similar_groups(results, topic=None):
# """Print similar newsgroups for specified topics"""
#     topics = [topic] if topic else sorted(results['by_topic'].keys())
    
#     for t in topics:
#         data = results['by_topic'][t]
#         similar_groups = data['similar_groups']
#         if len(similar_groups) > 0:
# print(f"\nTop: {t} (Total {len(similar_groups)} Group Similar News)")
            
# for i, group in enumerate(similar_groups[:3]): # Only the first 3 groups are displayed
# print(f"\n Similar group #{i+1}:")
#                 for idx in group:
#                     print(f"    - {data['items'][idx]['headline']}")
            
#             if len(similar_groups) > 3:
# print(f" ... and {len(similar_groups)-3} group not displayed")

# print_similar_groups(results)