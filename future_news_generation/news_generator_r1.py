
import os, json, random, argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI      
import time   

# ---------------------------------------------
# 1. prompt 与解析：完全照旧
# ---------------------------------------------
def create_future_news_prompt(target_date, seed_topic=None):
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
        f"Given the target future date of {target_date}, predict and generate a plausible news headline and abstract {topic_instr} that might be published on that date.\n\n"
        "You can follow these steps in your reasoning:\n"
        f"1. Analyze current trends and development patterns in relevant fields before {target_date}\n"
        f"2. Infer what stage of development might be reached by {target_date}\n"
        "3. Based on this reasoning, generate a credible news article\n\n"
        "Your generated news should:\n"
        f"- Be realistic and plausible for publication in {target_date}\n"
        "- Avoid extreme or highly unlikely scenarios\n"
        f"- Be written from the perspective of {target_date}, not as a prediction from the present\n"
        f"- Reflect reasonable developments that could occur between now and {target_date}\n\n"
        "Show your reasoning process in <think></think> tags, explaining why this news is likely to occur by "
        f"{target_date}, then provide your answer in <answer></answer> tags using the following format exactly:\n\n"
        "Headline: [News headline]\n"
        "Abstract: [1-2 sentence news abstract]\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"Let me carefully consider what news events {topic_instr} might plausibly occur in the target timeframe based on current trends and development patterns and systematically work through the reasoning process.\n"
        "<think>"
    )
    return prompt   

def extract_headline_abstract(text):
    headline = abstract = None
    hs = text.find("Headline:")
    if hs != -1:
        hs += len("Headline:")
        he = text.find("Abstract:", hs)
        if he != -1:
            headline = text[hs:he].strip()
    as_ = text.find("Abstract:")
    if as_ != -1:
        as_ += len("Abstract:")
        abstract = text[as_:].strip()
    return headline, abstract  

# ---------------------------------------------
# 2. r1（DeepSeek）客户端与单条调用
# ---------------------------------------------
def build_client(api_key="xxx", base_url="https://api.deepseek.com"):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请通过 --api_key 或环境变量 DEEPSEEK_API_KEY 提供 DeepSeek key")
    return OpenAI(api_key=api_key, base_url=base_url)

def r1_call_0(client, prompt, model="deepseek-reasoner",
            temperature=1, max_tokens=1024):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[r1 error] {e}")
        return None

def r1_call(client, prompt, model="deepseek-reasoner",
            temperature=1, max_tokens=1024, max_retries=3, retry_delay=1):
    """增强版API调用函数，支持失败重试机制"""
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                print(f"🔄 第{retry_count}次重试...")
            
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return resp.choices[0].message.content.strip()
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            if retry_count <= max_retries:
                print(f"[r1 错误] {error_msg} - 将在{retry_delay}秒后重试 ({retry_count}/{max_retries})")
                time.sleep(retry_delay)  # 添加延迟，避免过快重试
            else:
                print(f"[r1 最终失败] {error_msg} - 已达到最大重试次数")
                return None

# ---------------------------------------------
# 3. 批量生成（沿用旧签名，内部改成循环调用 r1）
# ---------------------------------------------
def generate_future_news_batch_0(client, prompts,
                               temperature=1, max_tokens=512,
                               model="deepseek-reasoner"):
    results = []
    for prompt in prompts:
        text = r1_call(client, prompt, model=model,
                       temperature=temperature, max_tokens=max_tokens)
        if not text:
            continue
        headline, abstract = extract_headline_abstract(text)
        if headline and abstract:
            results.append(
                {"headline": headline,
                 "abstract": abstract,
                 "original_generation": text}
            )
    return results

def generate_future_news_batch(client, prompts,
                               temperature=1, max_tokens=1024,
                               model="deepseek-reasoner"):
    results = []
    for i, prompt in enumerate(prompts):
        print(f"生成 {i+1}/{len(prompts)}...")
        text = r1_call(client, prompt, model=model,
                      temperature=temperature, max_tokens=max_tokens)
        if not text:
            print(f"❌ 生成失败，跳过此条")
            continue
            
        headline, abstract = extract_headline_abstract(text)
        if headline and abstract:
            results.append(
                {"headline": headline,
                 "abstract": abstract,
                 "original_generation": text}
            )
        else:
            print(f"⚠️ 解析失败: {text[:100]}...")
            
    return results

# ---------------------------------------------
# 4. 数据集主循环（逻辑保持一致）
# ---------------------------------------------
def generate_future_news_dataset_0(client, target_date, output_file,
                                 num_samples=1024, batch_size=16,
                                 topic_distribution=None,
                                 temperature=1, model="deepseek-reasoner"):
    all_res = []

    if topic_distribution:
        t_counts = {t: int(num_samples * w) for t, w in topic_distribution.items()}
        diff = num_samples - sum(t_counts.values())
        if diff > 0:
            t, = random.sample(list(t_counts), 1)
            t_counts[t] += diff
    else:
        t_counts = {None: num_samples}

    for topic, total in t_counts.items():
        if total == 0: continue
        print(f"[{topic or 'generic'}] 目标 {total} 条")
        for i in tqdm(range(0, total, batch_size)):
            bs = min(batch_size, total - i)
            batch_prompts = [
                create_future_news_prompt(target_date, seed_topic=topic)
                for _ in range(bs)
            ]
            batch_res = generate_future_news_batch(
                client, batch_prompts,
                temperature=temperature, model=model
            )
            for r in batch_res:
                r["topic"] = topic or "generic"
                r["target_date"] = target_date
                all_res.append(r)

    # 4.3 保存
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    pd.DataFrame(all_res).to_json(output_file, orient="records", lines=True,
                                  force_ascii=False)
    print(f"✅ 完成：{len(all_res)} 条 -> {output_file}")

def generate_future_news_dataset(client, target_date, output_file,
                                 num_samples=1024, batch_size=16,
                                 topic_distribution=None,
                                 temperature=1, model="deepseek-reasoner"):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # 计数器初始化
    total_generated = 0
    
    # 是否需要创建新文件
    file_exists = os.path.exists(output_file)
    file_mode = "a" if file_exists else "w"
    
    if topic_distribution:
        t_counts = {t: int(num_samples * w) for t, w in topic_distribution.items()}
        diff = num_samples - sum(t_counts.values())
        if diff > 0:
            t, = random.sample(list(t_counts), 1)
            t_counts[t] += diff
    else:
        t_counts = {None: num_samples}

    for topic, total in t_counts.items():
        if total == 0: continue
        print(f"[{topic or 'generic'}] 目标 {total} 条")
        for i in tqdm(range(0, total, batch_size)):
            bs = min(batch_size, total - i)
            batch_prompts = [
                create_future_news_prompt(target_date, seed_topic=topic)
                for _ in range(bs)
            ]
            batch_res = generate_future_news_batch(
                client, batch_prompts,
                temperature=temperature, model=model
            )
            
            # 处理本批次结果
            batch_save = []
            for r in batch_res:
                r["topic"] = topic or "generic"
                r["target_date"] = target_date
                batch_save.append(r)
            
            # 增量保存当前批次
            if batch_save:
                with open(output_file, file_mode, encoding="utf-8") as f:
                    for item in batch_save:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                # 首次写入后改为追加模式
                file_mode = "a"
                
                # 更新计数
                total_generated += len(batch_save)
                print(f"✓ 已保存 {len(batch_save)} 条新数据，累计 {total_generated} 条 -> {output_file}")

    print(f"✅ 全部完成：共生成 {total_generated} 条数据 -> {output_file}")
    return total_generated

# ---------------------------------------------
# 5. CLI 与入口
# ---------------------------------------------
def main():
    ap = argparse.ArgumentParser("Generate future news with DeepSeek r1")
    ap.add_argument("--output_file",  required=True, help="输出 JSONL")
    ap.add_argument("--target_date",  default="2025-01", help="YYYY-MM")
    ap.add_argument("--num_samples",  type=int, default=1024)
    ap.add_argument("--batch_size",   type=int, default=256)
    ap.add_argument("--balanced",    action="store_true",
                    help="使用与原脚本相同的 topic_distribution")
    ap.add_argument("--api_key",   default="sk-2c82e0ffa979464084fb5977cd3815f1", help="DeepSeek key (可用 env)")
    ap.add_argument("--temperature", type=float, default=1)
    ap.add_argument("--model",       default="deepseek-reasoner")
    args = ap.parse_args()

    topic_distribution = None
    if args.balanced:
        topic_distribution = {
            "Foreign": 0.22, "Business": 0.18, "OpEd": 0.16,
            "National": 0.12, "Washington": 0.11, "Metro": 0.09,
            "Science": 0.08, "Politics": 0.04
        }   

    client = build_client(args.api_key)

    generate_future_news_dataset(
        client, args.target_date, args.output_file,
        num_samples=args.num_samples, batch_size=args.batch_size,
        topic_distribution=topic_distribution,
        temperature=args.temperature, model=args.model
    )

if __name__ == "__main__":
    main()
