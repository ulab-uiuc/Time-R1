import json
import os
import re
import pandas as pd
from collections import defaultdict
from analyze_generation_diversity_monthly import (
    compute_similarity_matrix, 
    select_diverse_news, 
    is_mainly_english,
    save_jsonl
)

def extract_news_from_deepseek_response(item):
    """从Deepseek API响应中提取新闻数据"""
    try:
        # 提取响应内容
        custom_id = item.get('custom_id', '')
        response_body = item.get('response', {}).get('body', {})
        message = response_body.get('choices', [{}])[0].get('message', {})
        content = message.get('content', '')
        
        # 从custom_id解析主题和月份
        parts = custom_id.split('_')
        if len(parts) >= 3:
            target_date = parts[0]  # 格式如2024-07
            topic = parts[1]        # 如Foreign, Metro等
        else:
            target_date = "unknown"
            topic = "unknown"
            
        # 提取news部分
        news_items = []
        if '<answer>' in content:
            # 提取<answer>标签中的内容
            answer_content = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if answer_content:
                answer_text = answer_content.group(1).strip()
                
                # 分割出每条新闻
                news_sections = re.split(r'NEWS\s+\d+:', answer_text)
                # 去除可能的空白部分
                news_sections = [section.strip() for section in news_sections if section.strip()]
                
                for i, section in enumerate(news_sections):
                    # 提取标题和摘要
                    headline_match = re.search(r'Headline:?\s*"?([^"]*)"?', section, re.IGNORECASE)
                    abstract_match = re.search(r'Abstract:?\s*"?([^"]*)"?', section, re.IGNORECASE)
                    
                    if headline_match and abstract_match:
                        headline = headline_match.group(1).strip().strip('"')
                        abstract = abstract_match.group(1).strip().strip('"')
                        
                        news_item = {
                            'custom_id': f"{custom_id}_news{i+1}",
                            'target_date': target_date,
                            'month': target_date,  # 添加month字段用于后续处理
                            'topic': topic,
                            'headline': headline,
                            'abstract': abstract
                        }
                        news_items.append(news_item)
        
        return news_items
    except Exception as e:
        print(f"处理项目时出错: {e}")
        return []

def process_deepseek_results(input_file, output_dir):
    """处理Deepseek API结果文件并生成多样性新闻"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取结果文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"共读取 {len(data)} 条Deepseek API响应")
    
    # 提取所有新闻项
    all_news = []
    for item in data:
        news_items = extract_news_from_deepseek_response(item)
        all_news.extend(news_items)
    
    print(f"共提取出 {len(all_news)} 条新闻")
    
    # 按月份分组
    monthly_data = defaultdict(list)
    for item in all_news:
        month = item.get('month', 'unknown')
        monthly_data[month].append(item)
    
    # 保存按月份分组的JSONL文件
    for month, items in monthly_data.items():
        output_file = os.path.join(output_dir, f'future_news_{month}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'已保存 {len(items)} 条新闻到 {output_file}')
    
    print('按月份转换完成！')
    return monthly_data

def generate_diverse_news(monthly_data, output_dir, count_per_topic=5):
    """为每个月份每个主题选择多样性最高的新闻"""
    # 确保输出目录存在
    diverse_output_dir = os.path.join(output_dir, 'diverse_news')
    os.makedirs(diverse_output_dir, exist_ok=True)
    
    # 处理每个月份
    all_selected_news = []
    
    for month, news_data in monthly_data.items():
        print(f"\n处理月份: {month}")
        
        # 过滤非英文新闻并清理标题
        processed_news = []
        skipped_non_english = 0
        for item in news_data:
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

        print(f"  跳过了 {skipped_non_english} 条非英文或含过多中文的新闻")
        print(f"  处理 {len(processed_news)} 条英文新闻")
        
        # 按主题分组
        topic_groups = defaultdict(list)
        for item in processed_news:
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
        month_output_file = os.path.join(diverse_output_dir, f"diverse_news_{month}.jsonl")
        save_jsonl(month_selected_news, month_output_file)
        print(f"  已保存 {len(month_selected_news)} 条多样性新闻到 {month_output_file}")
        
        all_selected_news.extend(month_selected_news)
    
    # 保存所有月份的筛选结果
    all_output_file = os.path.join(diverse_output_dir, "all_diverse_news.jsonl")
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
    # 配置参数
    input_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/deepseek32b_results/results_32b.jsonl"
    output_dir = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/deepseek32b_results"
    count_per_topic = 5
    
    # 处理结果文件
    monthly_data = process_deepseek_results(input_file, output_dir)
    
    # 生成多样性新闻
    generate_diverse_news(monthly_data, output_dir, count_per_topic)

if __name__ == "__main__":
    main()