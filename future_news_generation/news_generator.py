import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import random
from tqdm import tqdm
import argparse
import re

def load_model_and_tokenizer(model_path):
    """
    加载预训练的模型和分词器
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map="auto"
    )
    return model, tokenizer

def create_future_news_prompt_single(target_date, seed_topic=None):
    """
    创建用于生成未来新闻的提示
    
    参数:
        target_date: 目标日期，格式"YYYY-MM"
        seed_topic: 可选的主题种子词，引导生成特定领域的新闻
    """
#     if seed_topic:
#         prompt = f"""Please generate a realistic news headline and abstract for a news article that might be published in {target_date}. 
# The news should be about {seed_topic}. Make sure it aligns with current trends and likely future developments.
# Respond in the following format exactly:

# Headline: [News headline]
# Abstract: [2-3 sentence news abstract]
# """
#     else:
#         prompt = f"""Please generate a realistic news headline and abstract for a news article that might be published in {target_date}.
# Make sure it aligns with current trends and likely future developments.
# Respond in the following format exactly:

# Headline: [News headline]
# Abstract: [2-3 sentence news abstract]
# """
    # topic_instruction = f"about {seed_topic}" if seed_topic else ""
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

    # 在创建prompt时使用映射
    topic_description = topic_mapping.get(seed_topic, seed_topic) if seed_topic else ""
    topic_instruction = f"about {topic_description}" if seed_topic else ""
    
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Given the target future date of {target_date}, predict and generate a plausible news headline and abstract {topic_instruction} that might be published on that date.\n\n"
        "You can follow these steps in your reasoning:\n"
        f"1. Analyze current trends and development patterns in relevant fields before {target_date}\n" 
        f"2. Infer what stage of development might be reached by {target_date}\n"
        "3. Based on this reasoning, generate a credible news article\n\n"
        "Your generated news should:\n"
        f"- Be realistic and plausible for publication in {target_date}\n"
        "- Avoid extreme or highly unlikely scenarios\n"
        f"- Be written from the perspective of {target_date}, not as a prediction from the present\n"
        f"- Reflect reasonable developments that could occur between now and {target_date}\n\n"
        f"Show your reasoning process in <think></think> tags, explaining why this news is likely to occur by {target_date}, then provide your answer in <answer></answer> tags using the following format exactly:\n\n"
        "Headline: [News headline]\n"
        "Abstract: [1-2 sentence news abstract]\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"Let me carefully consider what news events {topic_instruction} might plausibly occur in the target timeframe based on current trends and development patterns and systematically work through the reasoning process.\n"
        "<think>"
    )
    
    return prompt

def create_future_news_prompt(target_date, seed_topic=None):
    """创建用于生成未来新闻的提示"""
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

    # 在创建prompt时使用映射
    topic_description = topic_mapping.get(seed_topic, seed_topic) if seed_topic else ""
    topic_instruction = f"about {topic_description}" if seed_topic else ""
    
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Given the target future date of {target_date}, generate THREE distinct and plausible news headlines and abstracts {topic_instruction} that might be published on that date.\n\n"
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
        "- Be written ONLY in English, do not use any other languages\n\n"  # new requirement
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
        f"Let me carefully consider what news events {topic_instruction} might plausibly occur in the target timeframe based on current trends and development patterns and systematically work through the reasoning process.\n"
        "<think>"
    )
    
    return prompt

def extract_headline_abstract(text):
    """
    从生成的文本中提取标题和摘要
    """
    headline = None
    abstract = None
    
    # 查找标题
    headline_start = text.find("Headline:")
    if headline_start != -1:
        headline_start += len("Headline:")
        headline_end = text.find("Abstract:", headline_start)
        if headline_end != -1:
            headline = text[headline_start:headline_end].strip()
    
    # 查找摘要
    abstract_start = text.find("Abstract:")
    if abstract_start != -1:
        abstract_start += len("Abstract:")
        abstract = text[abstract_start:].strip()
    
    return headline, abstract

def extract_multiple_headlines_abstracts(text):
    """
    从生成的文本中提取3条新闻的标题和摘要
    """
    results = []
    
    # 查找<answer>部分
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = text  # 如果没有answer标签，使用整个文本
    
    # 定义NEWS块模式
    news_pattern = r'NEWS\s*(\d+):\s*Headline:\s*(.*?)\s*Abstract:\s*(.*?)(?=NEWS\s*\d+:|$)'
    news_matches = re.findall(news_pattern, answer_text, re.DOTALL)
    
    if news_matches:
        # 使用正则表达式找到的NEWS块
        for _, headline, abstract in news_matches:
            headline = headline.strip()
            abstract = abstract.strip()
            if headline and abstract:
                results.append({"headline": headline, "abstract": abstract})
    else:
        # 尝试另一种格式：每个标题和摘要分开
        headline_pattern = r'Headline:\s*(.*?)\s*(?=Abstract:|$)'
        abstract_pattern = r'Abstract:\s*(.*?)(?=Headline:|$)'
        
        headlines = re.findall(headline_pattern, answer_text, re.DOTALL)
        abstracts = re.findall(abstract_pattern, answer_text, re.DOTALL)
        
        # 确保标题和摘要的数量相等
        for i in range(min(len(headlines), len(abstracts))):
            headline = headlines[i].strip()
            abstract = abstracts[i].strip()
            if headline and abstract:
                results.append({"headline": headline, "abstract": abstract})
    
    return results

def generate_future_news(model, tokenizer, prompt, num_samples=1, 
                         temperature=0.7, max_new_tokens=256):
    """
    生成未来新闻
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    results = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 只保留生成的部分
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        headline, abstract = extract_headline_abstract(generated_text)
        
        if headline and abstract:
            results.append({
                "headline": headline,
                "abstract": abstract,
                "original_generation": generated_text
            })
    
    return results

def generate_future_news_batch_single(model, tokenizer, prompts, temperature=1, max_new_tokens=1024):
    """批量生成未来新闻"""
    # 批量编码所有提示
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    results = []
    for i, output in enumerate(outputs):
        # 只保留生成的部分
        input_length = inputs.input_ids.shape[1]
        if inputs.input_ids.shape[0] > 1:  # 批量处理时
            input_length = len(inputs.input_ids[i])
        
        generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        headline, abstract = extract_headline_abstract(generated_text)
        
        if headline and abstract:
            results.append({
                "headline": headline,
                "abstract": abstract,
                "original_generation": generated_text
            })
    
    return results

def generate_future_news_batch(model, tokenizer, prompts, temperature=1, max_new_tokens=1024):
    """批量生成未来新闻，每个prompt生成3条新闻"""
    # 批量编码所有提示
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    all_results = []
    for i, output in enumerate(outputs):
        # 只保留生成的部分
        input_length = inputs.input_ids.shape[1]
        if inputs.input_ids.shape[0] > 1 and len(inputs.input_ids[i]) < input_length:
            input_length = len(inputs.input_ids[i])
        
        generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        
        # 提取多条新闻
        news_items = extract_multiple_headlines_abstracts(generated_text)
        
        # 为每条新闻添加原始生成文本
        for item in news_items:
            item["original_generation"] = generated_text
            item["prompt_index"] = i  # 记录这条新闻来自哪个提示
        
        all_results.extend(news_items)
    
    return all_results

def generate_future_news_dataset_single(model, tokenizer, target_date, output_file, 
                                num_samples=1024, batch_size=16, topic_distribution=None):
    """
    生成未来新闻数据集
    
    参数:
        topic_distribution: 一个字典，指定不同主题的比例，例如
                           {"politics": 0.2, "technology": 0.3, "health": 0.1, 
                            "economy": 0.2, "sports": 0.1, "other": 0.1}
    """
    all_results = []
    
    if topic_distribution:
        # 根据主题分布生成新闻
        topics = list(topic_distribution.keys())
        topic_weights = list(topic_distribution.values())
        
        # 为每个主题分配样本数量
        topic_counts = {topic: int(num_samples * weight) for topic, weight in topic_distribution.items()}
        
        # 确保总数等于num_samples
        total_assigned = sum(topic_counts.values())
        if total_assigned < num_samples:
            topic_counts[random.choice(topics)] += (num_samples - total_assigned)
        
        for topic, count in topic_counts.items():
            if count == 0:
                continue
                
            print(f"Generating {count} samples for topic: {topic}")
            for i in tqdm(range(0, count, batch_size)):
                batch_count = min(batch_size, count - i)
                
                # 一次性准备多个提示
                batch_prompts = [
                    create_future_news_prompt(target_date, seed_topic=topic if topic != "other" else None)
                    for _ in range(batch_count)
                ]
                
                # 批量生成
                batch_results = generate_future_news_batch(model, tokenizer, batch_prompts)
                
                # 处理结果
                for result in batch_results:
                    result["topic"] = topic
                    result["target_date"] = target_date
                    all_results.append(result)
            # for i in tqdm(range(0, count, batch_size)):
            #     batch_count = min(batch_size, count - i)
            #     for _ in range(batch_count):
            #         prompt = create_future_news_prompt(target_date, seed_topic=topic if topic != "other" else None)
            #         results = generate_future_news(model, tokenizer, prompt, num_samples=1)
            #         for result in results:
            #             result["topic"] = topic
            #             result["target_date"] = target_date
            #             all_results.append(result)
    else:
        # 生成通用新闻
        print(f"Generating {num_samples} generic news samples")
        for i in tqdm(range(0, count, batch_size)):
            batch_count = min(batch_size, count - i)
            
            # 一次性准备多个提示
            batch_prompts = [
                create_future_news_prompt(target_date, seed_topic=topic if topic != "other" else None)
                for _ in range(batch_count)
            ]
            
            # 批量生成
            batch_results = generate_future_news_batch(model, tokenizer, batch_prompts)
            
            # 处理结果
            for result in batch_results:
                result["topic"] = topic
                result["target_date"] = target_date
                all_results.append(result)
        # for i in tqdm(range(0, num_samples, batch_size)):
        #     batch_count = min(batch_size, num_samples - i)
        #     for _ in range(batch_count):
        #         prompt = create_future_news_prompt(target_date)
        #         results = generate_future_news(model, tokenizer, prompt, num_samples=1)
        #         for result in results:
        #             result["target_date"] = target_date
        #             all_results.append(result)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_json(output_file, orient="records", lines=True)
    print(f"Generated {len(all_results)} samples, saved to {output_file}")
    
    return all_results

def generate_future_news_dataset(model, tokenizer, target_date, output_file, prompts_per_topic=10, batch_size=80):
    """
    并行生成所有主题的未来新闻数据集
    
    参数:
        prompts_per_topic: 每个主题生成的提示数量
        batch_size: 一次处理的提示数量
    """
    all_results = []
    
    # 所有需要生成的主题
    topics = ["Foreign", "Business", "OpEd", "National", "Washington", "Metro", "Science", "Politics"]
    
    # 为每个主题创建prompts_per_topic个提示
    all_prompts = []
    prompt_topic_map = []  # 记录每个提示对应的主题
    
    print(f"Creating prompts for {len(topics)} topics, {prompts_per_topic} prompts per topic")
    
    for topic in topics:
        for _ in range(prompts_per_topic):
            prompt = create_future_news_prompt(target_date, seed_topic=topic)
            all_prompts.append(prompt)
            prompt_topic_map.append(topic)
    
    # 按批次处理所有提示
    total_prompts = len(all_prompts)
    total_batches = (total_prompts + batch_size - 1) // batch_size  # 向上取整
    
    print(f"Processing {total_prompts} prompts in {total_batches} batches (size {batch_size})")
    
    for batch_idx in tqdm(range(total_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_prompts)
        
        # 提取当前批次的提示
        batch_prompts = all_prompts[start_idx:end_idx]
        
        # 批量生成新闻
        batch_results = generate_future_news_batch(model, tokenizer, batch_prompts, max_new_tokens=1024)
        
        # 为每条新闻添加主题和日期信息
        for result in batch_results:
            prompt_idx = result["prompt_index"] + start_idx
            result["topic"] = prompt_topic_map[prompt_idx]
            result["target_date"] = target_date
            # 删除不需要的中间字段
            if "prompt_index" in result:
                del result["prompt_index"]
            
            all_results.append(result)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_json(output_file, orient="records", lines=True)
    print(f"Generated {len(all_results)} news items from {total_prompts} prompts, saved to {output_file}")
    
    return all_results

def generate_multi_month_news_dataset(model, tokenizer, start_month, end_month, output_dir, prompts_per_topic=10, batch_size=80):
    """
    为多个月份生成未来新闻数据集
    
    参数:
        start_month: 起始月份 (YYYY-MM格式)
        end_month: 结束月份 (YYYY-MM格式)
        output_dir: 输出目录
    """
    # 生成月份列表
    months = []
    start_year, start_month_num = map(int, start_month.split("-"))
    end_year, end_month_num = map(int, end_month.split("-"))
    
    current_year, current_month = start_year, start_month_num
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month_num):
        months.append(f"{current_year}-{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    print(f"将为以下{len(months)}个月份生成数据: {', '.join(months)}")
    
    # 为每个月份生成数据
    all_results = {}
    for target_date in months:
        print(f"\n=== 生成 {target_date} 月份的新闻 ===")
        output_file = os.path.join(output_dir, f"future_news_{target_date}.jsonl")
        
        # 为当前月份生成数据
        results = generate_future_news_dataset(
            model, 
            tokenizer, 
            target_date, 
            output_file,
            prompts_per_topic=prompts_per_topic,
            batch_size=batch_size
        )
        
        all_results[target_date] = len(results)
    
    # 汇总统计
    print("\n=== 生成完成 ===")
    for month, count in all_results.items():
        print(f"{month}: 生成了 {count} 条新闻")
    
    return all_results

def main_single():
    parser = argparse.ArgumentParser(description="Generate future news articles")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--target_date", type=str, default="2025-01", help="Target date in YYYY-MM format")
    parser.add_argument("--num_samples", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation")
    parser.add_argument("--balanced", action="store_true", help="Use balanced topic distribution")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to("cuda:9")  # 指定GPU号9
    model.eval()  # 切换到评估模式
    
    # 设置主题分布（如果需要）
    topic_distribution = None
    if args.balanced:
        topic_distribution = {
            "Foreign": 0.22,
            "Business": 0.18, 
            "OpEd": 0.16,
            "National": 0.12,
            "Washington": 0.11,
            "Metro": 0.09,
            "Science": 0.08,
            "Politics": 0.04
        }
    
    # 生成数据集
    generate_future_news_dataset(
        model, 
        tokenizer, 
        args.target_date, 
        args.output_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        topic_distribution=topic_distribution
    )

def main_multi_news_generation():
    parser = argparse.ArgumentParser(description="Generate future news articles")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--target_date", type=str, default="2025-01", help="Target date in YYYY-MM format")
    parser.add_argument("--prompts_per_topic", type=int, default=10, help="Number of prompts per topic")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size for generation")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to("cuda:6")  # 指定GPU号9
    model.eval()  # 切换到评估模式
    
    # 生成数据集
    generate_future_news_dataset(
        model, 
        tokenizer, 
        args.target_date, 
        args.output_file,
        prompts_per_topic=args.prompts_per_topic,
        batch_size=args.batch_size
    )

def main():
    parser = argparse.ArgumentParser(description="Generate future news articles")
    parser.add_argument("--model_path", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/check_points_time_prediction_zero/time_prediction/zero/actor/global_step_360", help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/future_news_test", help="Output directory")
    parser.add_argument("--start_month", type=str, default="2024-07", help="Start month in YYYY-MM format")
    parser.add_argument("--end_month", type=str, default="2025-02", help="End month in YYYY-MM format")
    parser.add_argument("--prompts_per_topic", type=int, default=10, help="Number of prompts per topic")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size for generation")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to("cuda")  # 指定GPU号
    model.eval()  # 切换到评估模式
    
    # 生成多月份数据集
    generate_multi_month_news_dataset(
        model, 
        tokenizer, 
        args.start_month,
        args.end_month,
        args.output_dir,
        prompts_per_topic=args.prompts_per_topic,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()