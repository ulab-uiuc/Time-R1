import json
import re
import os
from tqdm import tqdm

def extract_multiple_headlines_abstracts(content):
    """从API响应的content字段中提取多条headline和abstract"""
    # 查找<answer>标签内容
    match = re.search(r'\u003canswer\u003e(.*?)\u003c/answer\u003e', content, re.DOTALL)
    if not match:
        return []
    
    answer_text = match.group(1).strip()
    
    # 提取所有NEWS块
    news_pattern = r'NEWS\s*(\d+):\s*Headline:\s*(.*?)\s*Abstract:\s*(.*?)(?=NEWS\s*\d+:|$)'
    news_matches = re.findall(news_pattern, answer_text, re.DOTALL)
    
    results = []
    
    for _, headline, abstract in news_matches:
        headline = headline.strip().strip('"\'')
        abstract = abstract.strip()
        
        if headline and abstract:
            results.append({
                'headline': headline,
                'abstract': abstract
            })
    
    # 如果没有找到NEWS格式，尝试直接提取Headline和Abstract
    if not results:
        headline_matches = re.findall(r'Headline:\s*(.*?)(?:\n|$)', answer_text, re.DOTALL)
        abstract_matches = re.findall(r'Abstract:\s*(.*?)(?=Headline:|$)', answer_text, re.DOTALL)
        
        for i in range(min(len(headline_matches), len(abstract_matches))):
            headline = headline_matches[i].strip().strip('"\'')
            abstract = abstract_matches[i].strip()
            
            if headline and abstract:
                results.append({
                    'headline': headline,
                    'abstract': abstract
                })
    
    return results

def process_api_responses(input_file, output_file):
    """处理API响应并生成简洁的JSONL文件，包含多条新闻"""
    processed_count = 0
    error_count = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # 读取并统计输入文件行数，用于进度条
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="处理API响应"):
            try:
                data = json.loads(line)
                # 检查是否存在错误
                if data.get('error') is not None:
                    error_count += 1
                    continue
                
                # 提取custom_id，从中获取月份和主题信息
                custom_id = data.get('custom_id', '')
                parts = custom_id.split('_')
                
                # 假设格式为 "YYYY-MM_Topic_Number"，至少需要3部分
                if len(parts) >= 3:
                    month = parts[0]  # 如 "2024-07"
                    topic = parts[1]  # 如 "Foreign"
                else:
                    month = "unknown"
                    topic = "unknown"
                
                # 提取content内容
                content = data['response']['body']['choices'][0]['message']['content']
                
                # 提取多条新闻
                news_items = extract_multiple_headlines_abstracts(content)
                
                for item in news_items:
                    # 为每条新闻添加月份、主题和自定义ID信息
                    item['month'] = month
                    item['topic'] = topic
                    item['custom_id'] = custom_id
                    
                    # 写入输出文件
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"处理第{processed_count+error_count}行时出错: {str(e)}")
    
    print(f"处理完成! 成功提取 {processed_count} 条新闻，失败 {error_count} 条")
    return processed_count, error_count

if __name__ == "__main__":
    input_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/results_multi_month_r1.jsonl"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_multi_month_r1.jsonl"
    
    processed, errors = process_api_responses(input_file, output_file)