import pandas as pd
import json
import os
from tqdm import tqdm

def create_test_batch_input_file(
    test_parquet_file, 
    output_file, 
    max_tokens=1024, 
    temperature=1, 
    top_p=1.0, 
    stream=True
):
    """
    从small_test_time_prediction.parquet创建用于r1评估的批处理输入文件
    
    参数:
        test_parquet_file: 小型测试集文件路径
        output_file: 输出的JSONL批处理文件路径
        max_tokens: 最大生成token数
        temperature: 采样温度
        top_p: top-p采样参数
        stream: 是否流式输出
    """
    print(f"从 {test_parquet_file} 读取测试数据...")
    df = pd.read_parquet(test_parquet_file)
    test_samples = df.to_dict('records')
    
    print(f"共读取 {len(test_samples)} 条测试数据")
    
    # 初始化请求列表
    requests = []
    
    # 处理每个测试样本
    print("生成批处理请求...")
    for idx, sample in enumerate(tqdm(test_samples)):
        # 获取原始prompt
        prompt_content = sample['prompt'][0]['content']
        
        # 记录年月信息用于custom_id
        year = sample['extra_info']['year']
        month = sample['extra_info']['month']
        
        # 创建请求
        request = {
            "custom_id": f"{year}-{month:02d}_{idx}",
            "body": {
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
        }
        requests.append(request)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    
    # 写入JSONL文件
    print(f"写入 {len(requests)} 条请求到 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    # 统计月份分布
    month_counts = {}
    for sample in test_samples:
        year = sample['extra_info']['year']
        month = sample['extra_info']['month']
        month_key = f"{year}-{month:02d}"
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
    
    # 输出统计信息
    print("\n✅ 批处理文件生成完成!")
    print(f"总请求数: {len(requests)}")
    print("月份分布:")
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count}条")

if __name__ == "__main__":
    # 可以根据需要调整这些参数
    test_parquet_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_time_prediction.parquet"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/time_prediction_r1_test_batch_input.jsonl"
    
    create_test_batch_input_file(
        test_parquet_file=test_parquet_file,
        output_file=output_file,
        max_tokens=1024,
        temperature=1,
        top_p=1.0,
        stream=True
    )