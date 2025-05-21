import json
import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            # device_map="auto",  # 自动利用可用GPU
        )
        model = model.to("cuda:0")  # 显式地将模型移动到GPU上
        print(f"模型已加载到设备: {model.device}")
        model.eval()  # 切换到评估模式
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def create_future_news_prompt(target_date, seed_topic):
    """创建未来新闻生成的提示"""
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
        "Abstract: [1-2 sentence news abstract 3]"
    )
    return prompt

def generate_test_samples(start_month="2024-07", end_month="2025-02", samples_per_topic=2):
    """生成测试样本"""
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
    
    # 生成测试样本
    test_samples = []
    
    for target_date in months:
        for topic in topics:
            for i in range(samples_per_topic):
                prompt = create_future_news_prompt(target_date, topic)
                sample = {
                    "target_date": target_date,
                    "topic": topic,
                    "sample_id": f"{target_date}_{topic}_{i}",
                    "prompt": prompt
                }
                test_samples.append(sample)
    
    # 打印样本统计信息
    print(f"共生成 {len(test_samples)} 条测试样本")
    print(f"目标月份: {', '.join(months)}")
    print(f"主题类型: {', '.join(topics)}")
    print(f"每个主题每月样本数: {samples_per_topic}")
    
    return test_samples

def batch_inference(model, tokenizer, test_samples, output_file, batch_size=8, max_new_tokens=2048):
    """批量推理"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    
    results = []
    
    try:
        for i in tqdm(range(0, len(test_samples), batch_size), desc="批量推理"):
            batch = test_samples[i:i+batch_size]
            
            # 构建消息批次
            message_batch = []
            for sample in batch:
                message_batch.append([
                    {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."},
                    {"role": "user", "content": sample["prompt"]}
                ])
            
            # 应用聊天模板
            text_batch = tokenizer.apply_chat_template(
                message_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 对文本进行编码
            model_inputs_batch = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # 根据实际情况调整
            ).to("cuda:0")
            
            # 模型生成
            with torch.no_grad():
                try:
                    generated_ids_batch = model.generate(
                        **model_inputs_batch,
                        max_new_tokens=max_new_tokens,
                        temperature=1.0,  # 增加创造性
                        top_p=0.9,        # 控制生成多样性
                    )
                    
                    # 处理生成结果
                    new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
                    response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    
                    # 记录结果
                    for j, response in enumerate(response_batch):
                        sample_idx = i + j
                        if sample_idx < len(test_samples):
                            sample = test_samples[sample_idx]
                            
                            result = {
                                "custom_id": sample["sample_id"],
                                "target_date": sample["target_date"],
                                "topic": sample["topic"],
                                "prompt": sample["prompt"],
                                "response": response.strip(),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            results.append(result)
                            
                            # 实时写入结果，避免中断后丢失
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                
                except torch.cuda.OutOfMemoryError:
                    print(f"警告: GPU内存不足，尝试减小batch_size或max_new_tokens")
                    # 如果内存不足，处理当前批次的单个样本
                    for sample in batch:
                        process_single_sample(model, tokenizer, sample, output_file, max_new_tokens)
    
    except KeyboardInterrupt:
        print("推理被用户中断")
    
    print(f"\n✅ 测试完成! 结果已保存至: {output_file}")
    print(f"共处理 {len(results)}/{len(test_samples)} 条数据")

def process_single_sample(model, tokenizer, sample, output_file, max_new_tokens):
    """处理单个样本（在批处理内存不足时使用）"""
    # 构建消息
    message = [
        {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."},
        {"role": "user", "content": sample["prompt"]}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 编码
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda:0")
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.9,
        )
        
        # 处理结果
        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        # 记录结果
        result = {
            "custom_id": sample["sample_id"],
            "target_date": sample["target_date"],
            "topic": sample["topic"],
            "prompt": sample["prompt"],
            "response": response.strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 写入结果
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def extract_news_from_responses(output_file, parsed_output_file):
    """从响应中提取新闻并保存到CSV文件"""
    print(f"从 {output_file} 提取新闻...")
    
    # 读取生成结果
    results = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"共读取 {len(results)} 条生成结果")
    
    # 提取新闻
    news_list = []
    
    for result in tqdm(results, desc="提取新闻"):
        response = result["response"]
        
        # 尝试提取<answer>标签中的内容
        answer = ""
        if "<answer>" in response and "</answer>" in response:
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            # 如果没有标签，尝试直接提取NEWS格式
            answer = response
        
        # 提取各条新闻
        try:
            news_parts = answer.split("NEWS ")
            # 去除可能的空字符串
            news_parts = [part for part in news_parts if part.strip()]
            
            for idx, part in enumerate(news_parts):
                if idx >= 3:  # 最多提取3条新闻
                    break
                    
                part = part.strip()
                
                # 提取标题和摘要
                headline = ""
                abstract = ""
                
                if "Headline:" in part and "Abstract:" in part:
                    headline_part = part.split("Headline:")[1].split("Abstract:")[0].strip()
                    abstract_part = part.split("Abstract:")[1].strip()
                    
                    # 清理可能的额外内容
                    headline = headline_part.split("\n")[0].strip()
                    abstract_lines = abstract_part.split("\n")
                    abstract = abstract_lines[0].strip()
                    
                    if headline and abstract:
                        news_item = {
                            "custom_id": result["custom_id"] + f"_news{idx+1}",
                            "target_date": result["target_date"],
                            "topic": result["topic"],
                            "headline": headline,
                            "abstract": abstract,
                        }
                        news_list.append(news_item)
        except Exception as e:
            print(f"处理样本 {result['custom_id']} 时出错: {e}")
    
    # 保存提取的新闻
    df = pd.DataFrame(news_list)
    df.to_csv(parsed_output_file, index=False, encoding="utf-8")
    
    print(f"✅ 共提取 {len(news_list)} 条新闻，已保存至: {parsed_output_file}")
    
    # 打印每个主题的新闻数量
    topic_counts = df["topic"].value_counts()
    print("\n各主题新闻数量:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}条")
    
    # 打印每个月份的新闻数量
    date_counts = df["target_date"].value_counts()
    print("\n各月份新闻数量:")
    for date, count in sorted(date_counts.items()):
        print(f"  {date}: {count}条")

def main():
    # 配置参数
    model_path = "/data/models/Qwen2.5-3B-Instruct"
    output_file = "/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/qwen3b_results/future_news_qwen3b_results.jsonl"
    parsed_output_file = "/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/future_news_generation/qwen3b_results/future_news_qwen3b_parsed.csv"
    
    # 生成参数
    start_month = "2024-07"    # 起始月份
    end_month = "2025-02"      # 结束月份
    samples_per_topic = 10      # 每个主题每月生成的样本数
    batch_size = 80            # 批处理大小
    max_new_tokens = 2048      # 最大生成token数
    
    # 如果输出目录不存在，创建它
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果已存在输出文件，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print(f"开始测试Qwen2.5-3B-Instruct模型在未来新闻生成任务上的表现")
    print(f"模型路径: {model_path}")
    print(f"结果输出: {output_file}")
    print(f"起始月份: {start_month}, 结束月份: {end_month}")
    print(f"每个主题每月样本数: {samples_per_topic}")
    print(f"批处理大小: {batch_size}")
    print(f"最大生成token数: {max_new_tokens}")
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 生成测试样本
        test_samples = generate_test_samples(start_month, end_month, samples_per_topic)
        
        # 批量推理
        batch_inference(model, tokenizer, test_samples, output_file, batch_size, max_new_tokens)
        
        # 从响应中提取新闻
        extract_news_from_responses(output_file, parsed_output_file)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()