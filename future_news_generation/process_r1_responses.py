import json
import re
import os
from tqdm import tqdm

def extract_headline_abstract(content):
    """从API响应的content字段中提取headline和abstract"""
    # 查找<answer>标签内容
    match = re.search(r'\u003canswer\u003e(.*?)\u003c/answer\u003e', content, re.DOTALL)
    if not match:
        return None, None
    
    answer_text = match.group(1).strip()
    
    # 从答案中提取标题和摘要
    headline_match = re.search(r'Headline:?\s*["\']?(.*?)["\']?(?:\n|$)', answer_text, re.DOTALL)
    abstract_match = re.search(r'Abstract:?\s*(.*?)(?:\n\n|$)', answer_text, re.DOTALL)
    
    headline = headline_match.group(1).strip() if headline_match else ""
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # 删除可能的引号
    headline = headline.strip('"\'')
    
    return headline, abstract

def process_api_responses(input_file, output_file):
    """处理API响应并生成简洁的JSONL文件"""
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
                
                # 提取content内容
                content = data['response']['body']['choices'][0]['message']['content']
                
                # 提取headline和abstract
                headline, abstract = extract_headline_abstract(content)
                
                if headline and abstract:
                    # 创建简化的记录
                    simplified_record = {
                        'headline': headline,
                        'abstract': abstract,
                        'custom_id': data.get('custom_id')
                    }
                    
                    # 写入输出文件
                    f_out.write(json.dumps(simplified_record, ensure_ascii=False) + '\n')
                    processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"处理第{processed_count+error_count}行时出错: {str(e)}")
    
    print(f"处理完成! 成功提取 {processed_count} 条新闻，失败 {error_count} 条")
    return processed_count, error_count

if __name__ == "__main__":
    input_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/results.jsonl"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_2025_01_r1_batch.jsonl"
    
    processed, errors = process_api_responses(input_file, output_file)