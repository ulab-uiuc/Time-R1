import pandas as pd
import json
import os
from tqdm import tqdm

def create_reasoning_test_batch_input_file(
    test_parquet_file, 
    output_file, 
    max_tokens=2048, 
    temperature=1, 
    top_p=1.0, 
    stream=True
):
    """
    从test_time_reasoning_combined.parquet创建用于r1评估的批处理输入文件
    
    参数:
        test_parquet_file: 时间推理测试集文件路径
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
        
        # 获取任务类型
        task_type = sample['extra_info'].get('task', 'unknown')
        
        # 创建自定义ID（包含任务类型）
        custom_id = f"{task_type}_{idx}"
        
        # 创建请求
        request = {
            "custom_id": custom_id,
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
    
    # 统计任务类型分布
    task_counts = {}
    for sample in test_samples:
        task_type = sample['extra_info'].get('task', 'unknown')
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    # 输出统计信息
    print("\n✅ 批处理文件生成完成!")
    print(f"总请求数: {len(requests)}")
    print("任务类型分布:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}条 ({count/len(requests)*100:.2f}%)")

if __name__ == "__main__":
    # 可以根据需要调整这些参数
    test_parquet_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/time_reasoning_r1_test_batch_input.jsonl"
    
    create_reasoning_test_batch_input_file(
        test_parquet_file=test_parquet_file,
        output_file=output_file,
        max_tokens=2048,  # 时间推理任务可能需要更长的输出
        temperature=1,
        top_p=1.0,
        stream=True
    )