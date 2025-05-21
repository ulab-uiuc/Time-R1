import json
import re
from langdetect import detect, LangDetectException
import argparse

# --- 从 analyze_generation_diversity_monthly.py 复制过来的函数 ---

# 正则表达式匹配中文字符（包括简体和繁体的主要范围）
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')

def load_jsonl(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                continue
    return data

def save_jsonl(data, file_path):
    """保存JSONL格式数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def is_mainly_english(text, chinese_threshold=0.1):
    """
    检查文本是否主要为英文。
    如果 langdetect 检测失败、检测为非英文，或中文字符比例超过阈值，则返回 False。
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return False # 处理空或非字符串输入

    try:
        # 1. 使用 langdetect 进行初步检测
        detected_lang = detect(text)
        if detected_lang != 'en':
            # print(f"  Langdetect identified non-English: {detected_lang} in '{text[:50]}...'") # 可选：调试信息
            return False

        # 2. 检查中文字符比例
        chinese_chars = chinese_char_pattern.findall(text)
        chinese_char_count = sum(len(s) for s in chinese_chars)
        total_chars = len(text)

        if total_chars > 0:
            chinese_ratio = chinese_char_count / total_chars
            if chinese_ratio > chinese_threshold:
                # print(f"  Chinese char ratio exceeded threshold: {chinese_ratio:.2f} in '{text[:50]}...'") # 可选：调试信息
                return False

        # 通过所有检查
        return True

    except LangDetectException:
        # langdetect 检测失败，保守处理，返回 False
        # print(f"  Langdetect failed for text: {text[:50]}...") # 可选：调试信息
        return False
    except Exception as e:
        # 捕获其他潜在错误
        print(f"  Error in is_mainly_english for text '{text[:50]}...': {e}")
        return False

# --- 主过滤逻辑 ---

def filter_and_clean_news(input_file, output_file, chinese_threshold=0.1):
    """加载、过滤、清理并保存新闻数据"""
    print(f"Loading data from: {input_file}")
    news_data = load_jsonl(input_file)
    print(f"Loaded {len(news_data)} items.")

    filtered_news = []
    skipped_count = 0
    processed_count = 0

    for item in news_data:
        processed_count += 1
        headline = item.get('headline', '')
        abstract = item.get('abstract', '')
        content = f"{headline} {abstract}"

        if is_mainly_english(content, chinese_threshold=chinese_threshold):
            # 清理标题中的 '[' 和 ']'
            if isinstance(headline, str):
                item['headline'] = headline.replace('[', '').replace(']', '')
            filtered_news.append(item)
        else:
            skipped_count += 1
            # print(f"Skipping item: {headline[:50]}...") # 可选：调试信息

    print(f"Processed {processed_count} items.")
    print(f"Skipped {skipped_count} items due to language filter.")
    print(f"Kept {len(filtered_news)} items.")

    print(f"Saving filtered data to: {output_file}")
    save_jsonl(filtered_news, output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL news data based on language and clean headlines.")
    parser.add_argument("--input_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_test/all_diverse_news_hpx2.jsonl", help="Input JSONL file path.")
    parser.add_argument("--output_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/results/diverse_news_by_month_test/all_diverse_news_hpx2_filtered.jsonl", help="Output JSONL file path.")
    parser.add_argument("--chinese_threshold", type=float, default=0, help="Maximum allowed ratio of Chinese characters (0.0 to 1.0).")

    args = parser.parse_args()

    filter_and_clean_news(args.input_file, args.output_file, args.chinese_threshold)