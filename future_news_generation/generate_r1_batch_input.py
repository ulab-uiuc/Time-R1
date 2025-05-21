import json
import random
import os
from datetime import datetime

def create_future_news_prompt_multi(target_date, seed_topic=None):
    """创建未来新闻生成的提示，每个prompt生成3条不同新闻"""
    topic_mapping = {
        "Foreign": "international affairs and global politics",
        "Business": "business, economics and financial markets",
        "OpEd": "opinion and editorial commentary", 
        "National": "U.S. national news and domestic affairs",
        "Washington": "U.S. politics and government",
        "Metro": "local news and urban affairs",
        "Science": "science, technology and innovation",
        "Politics": "political developments and elections"
    }
    
    topic_desc = topic_mapping.get(seed_topic, seed_topic) if seed_topic else ""
    topic_instr = f"about {topic_desc}" if seed_topic else ""

    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Given the target future date of {target_date}, generate THREE distinct and plausible news headlines and abstracts {topic_instr} that might be published on that date.\n\n"
        "You can follow these steps in your reasoning:\n"
        f"1. Analyze current trends and development patterns in relevant fields before {target_date}\n" 
        f"2. Infer what stage of development might be reached by {target_date}\n"
        "3. Based on this reasoning, generate THREE credible and DIFFERENT news articles on the same topic\n\n"
        "Your generated news should:\n"
        f"- Be realistic and plausible for publication in {target_date}\n"
        "- Avoid extreme or highly unlikely scenarios\n"
        f"- Be written from the perspective of {target_date}, not as a prediction from the present\n"
        f"- Reflect reasonable developments that could occur between now and {target_date}\n"
        "- Have significant differences from each other - cover different angles, events, or developments within the same topic\n\n"
        f"Show your reasoning process in <think></think> tags, explaining why these news items are likely to occur by {target_date}, then provide your answer in <answer></answer> tags using the following format exactly:\n\n"
        "NEWS 1:\n"
        "Headline: [News headline 1]\n"
        "Abstract: [1-2 sentence news abstract 1]\n\n"
        "NEWS 2:\n"
        "Headline: [News headline 2]\n"
        "Abstract: [1-2 sentence news abstract 2]\n\n"
        "NEWS 3:\n"
        "Headline: [News headline 3]\n"
        "Abstract: [1-2 sentence news abstract 3]\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"Let me carefully consider what news events {topic_instr} might plausibly occur in the target timeframe based on current trends and development patterns and systematically work through the reasoning process.\n"
        "<think>"
    )
    return prompt

def generate_multi_month_batch_input_file(output_file, start_month="2024-07", end_month="2025-02", prompts_per_topic=10, temperature=1, max_tokens=2048, top_p=1, stream=False):
    """生成多个月份的批量推理输入文件"""
    # 生成月份列表
    months = []
    year_start = int(start_month.split("-")[0])
    month_start = int(start_month.split("-")[1])
    year_end = int(end_month.split("-")[0])
    month_end = int(end_month.split("-")[1])
    
    current_year, current_month = year_start, month_start
    
    while current_year < year_end or (current_year == year_end and current_month <= month_end):
        months.append(f"{current_year}-{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    # 所有主题
    topics = [
        "Foreign", "Business", "OpEd", "National", 
        "Washington", "Metro", "Science", "Politics"
    ]
    
    # 生成请求
    requests = []
    id_counter = 0
    month_counts = {month: 0 for month in months}
    
    for target_date in months:
        topic_counts = {topic: 0 for topic in topics}
        
        for topic in topics:
            for _ in range(prompts_per_topic):
                prompt = create_future_news_prompt_multi(target_date, topic)
                request = {
                    "custom_id": f"{target_date}_{topic}_{id_counter}",
                    "body": {
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": stream
                    }
                }
                requests.append(request)
                topic_counts[topic] += 1
                month_counts[target_date] += 1
                id_counter += 1
        
        print(f"已为日期 {target_date} 生成 {sum(topic_counts.values())} 条请求")
    
    # 写入文件
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    total_prompts = len(requests)
    print(f"\n✅ 已生成包含 {total_prompts} 条请求的输入文件: {output_file}")
    print(f"月份分布:")
    for month, count in month_counts.items():
        print(f"  {month}: {count}条请求")
    print(f"预计生成的新闻总数: {total_prompts * 3}")  # 每个prompt生成3条新闻

if __name__ == "__main__":
    # 修改这些参数以适应您的需求
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/future_news_multi_month_r1_batch_input.jsonl"
    start_month = "2024-07"  # 起始月份
    end_month = "2025-02"    # 结束月份
    prompts_per_topic = 10   # 每个主题的提示数量
    temperature = 1.0        # 温度参数
    max_tokens = 2048        # 最大生成token数（增加以容纳3条新闻）
    
    generate_multi_month_batch_input_file(
        output_file=output_file,
        start_month=start_month,
        end_month=end_month,
        prompts_per_topic=prompts_per_topic,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )









# import json
# import random
# import os
# from datetime import datetime

# def create_future_news_prompt(target_date, seed_topic=None):
#     """创建未来新闻生成的提示"""
#     topic_mapping = {
#         "Foreign": "international affairs and global politics",
#         "Business": "business, economics and financial markets",
#         "OpEd": "opinion and editorial commentary", 
#         "National": "U.S. national news and domestic affairs",
#         "Washington": "U.S. politics and government",
#         "Metro": "local news and urban affairs",
#         "Science": "science, technology and innovation",
#         "Politics": "political developments and elections"
#     }
#     topic_desc = topic_mapping.get(seed_topic, seed_topic) if seed_topic else ""
#     topic_instr = f"about {topic_desc}" if seed_topic else ""

#     prompt = (
#         "<|im_start|>system\n"
#         "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
#         "<|im_end|>\n"
#         "<|im_start|>user\n"
#         f"Given the target future date of {target_date}, predict and generate a plausible news headline and abstract {topic_instr} that might be published on that date.\n\n"
#         "You can follow these steps in your reasoning:\n"
#         f"1. Analyze current trends and development patterns in relevant fields before {target_date}\n"
#         f"2. Infer what stage of development might be reached by {target_date}\n"
#         "3. Based on this reasoning, generate a credible news article\n\n"
#         "Your generated news should:\n"
#         f"- Be realistic and plausible for publication in {target_date}\n"
#         "- Avoid extreme or highly unlikely scenarios\n"
#         f"- Be written from the perspective of {target_date}, not as a prediction from the present\n"
#         f"- Reflect reasonable developments that could occur between now and {target_date}\n\n"
#         "Show your reasoning process in <think></think> tags, explaining why this news is likely to occur by "
#         f"{target_date}, then provide your answer in <answer></answer> tags using the following format exactly:\n\n"
#         "Headline: [News headline]\n"
#         "Abstract: [1-2 sentence news abstract]\n"
#         "<|im_end|>\n"
#         "<|im_start|>assistant\n"
#         f"Let me carefully consider what news events {topic_instr} might plausibly occur in the target timeframe based on current trends and development patterns and systematically work through the reasoning process.\n"
#         "<think>"
#     )
#     return prompt

# def generate_batch_input_file(output_file, target_date="2025-01", num_samples=1024, temperature=1, max_tokens=1024, top_p=1, stream=False):
#     """生成批量推理输入文件"""
#     # 主题分布
#     topic_distribution = {
#         "Foreign": 0.22, 
#         "Business": 0.18, 
#         "OpEd": 0.16,
#         "National": 0.12, 
#         "Washington": 0.11, 
#         "Metro": 0.09,
#         "Science": 0.08, 
#         "Politics": 0.04
#     }
    
#     # 计算每个主题的样本数
#     topic_counts = {t: int(num_samples * w) for t, w in topic_distribution.items()}
#     remaining = num_samples - sum(topic_counts.values())
#     if remaining > 0:
#         topic_counts[random.choice(list(topic_counts.keys()))] += remaining
    
#     # 生成请求
#     requests = []
#     id_counter = 0
    
#     for topic, count in topic_counts.items():
#         for _ in range(count):
#             prompt = create_future_news_prompt(target_date, topic)
#             request = {
#                 "custom_id": str(id_counter),
#                 "body": {
#                     "messages": [
#                         {"role": "user", "content": prompt}
#                     ],
#                     "max_tokens": max_tokens,
#                     "temperature": temperature,
#                     "top_p": top_p,
#                     "stream": stream
#                 }
#             }
#             requests.append(request)
#             id_counter += 1
    
#     # 写入文件
#     os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
#     with open(output_file, "w", encoding="utf-8") as f:
#         for req in requests:
#             f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
#     print(f"✅ 已生成包含 {len(requests)} 条请求的输入文件: {output_file}")
#     print(f"主题分布: {', '.join([f'{t}: {c}' for t, c in topic_counts.items()])}")

# if __name__ == "__main__":
#     # 修改这些参数以适应您的需求
#     output_file = "future_news_2025_01_r1_batch_input.jsonl"
#     target_date = "2025-01"
#     num_samples = 1024  # 总样本数
#     temperature = 1.0   # 温度参数
#     max_tokens = 1024   # 最大生成token数
    
#     generate_batch_input_file(
#         output_file=output_file,
#         target_date=target_date,
#         num_samples=num_samples,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stream=True
#     )