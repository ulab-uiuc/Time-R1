import json
import os
import torch
import pandas as pd
from tqdm import tqdm
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
        model = model.to("cuda:6")  # 显式地将模型移动到GPU上
        print(f"模型已加载到设备: {model.device}")
        model.eval()  # 切换到评估模式
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def load_test_data(test_parquet_file):
    """加载测试数据"""
    print(f"从 {test_parquet_file} 读取测试数据...")
    df = pd.read_parquet(test_parquet_file)
    test_samples = df.to_dict('records')
    print(f"共读取 {len(test_samples)} 条测试数据")
    
    # 统计月份分布
    month_counts = {}
    for sample in test_samples:
        year = sample['extra_info']['year']
        month = sample['extra_info']['month']
        month_key = f"{year}-{month:02d}"
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
    
    print("月份分布:")
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count}条")
    
    return test_samples

def batch_inference(model, tokenizer, test_samples, output_file, batch_size=8, max_new_tokens=1024):
    """批量推理"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    
    results = []
    
    try:
        for i in tqdm(range(0, len(test_samples), batch_size), desc="批量推理"):
            batch = test_samples[i:i+batch_size]
            
            # 构建消息批次
            message_batch = []
            for sample in batch:
                prompt_content = sample['prompt'][0]['content']
                message_batch.append([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
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
            ).to("cuda:6")
            
            # 模型生成
            with torch.no_grad():
                try:
                    generated_ids_batch = model.generate(
                        **model_inputs_batch,
                        max_new_tokens=max_new_tokens,
                    )
                    
                    # 处理生成结果
                    new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
                    response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    
                    # 记录结果
                    for j, response in enumerate(response_batch):
                        sample_idx = i + j
                        if sample_idx < len(test_samples):
                            sample = test_samples[sample_idx]
                            year = sample['extra_info']['year']
                            month = sample['extra_info']['month']
                            
                            result = {
                                "custom_id": f"{year}-{month:02d}_{sample_idx}",
                                "prompt": sample['prompt'][0]['content'],
                                "response": response.strip(),
                                "year": year,
                                "month": month,
                                "extra_info": sample['extra_info']
                            }
                            
                            if 'reward_model' in sample and 'ground_truth' in sample['reward_model']:
                                result["ground_truth"] = sample['reward_model']['ground_truth']
                            
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
    prompt_content = sample['prompt'][0]['content']
    year = sample['extra_info']['year']
    month = sample['extra_info']['month']
    
    # 构建消息
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
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
    ).to("cuda:6")
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        
        # 处理结果
        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        # 记录结果
        result = {
            "custom_id": f"{year}-{month:02d}_{sample.get('id', 0)}",
            "prompt": prompt_content,
            "response": response.strip(),
            "year": year,
            "month": month,
            "extra_info": sample['extra_info']
        }
        
        if 'reward_model' in sample and 'ground_truth' in sample['reward_model']:
            result["ground_truth"] = sample['reward_model']['ground_truth']
        
        # 写入结果
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def main_qwen():
    # 配置参数
    model_path = "/mnt/data_from_server1/models/Qwen2.5-3B-Instruct"
    # test_parquet_file = "/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/datasets/small_test_time_prediction.parquet"
    # output_file = "/mnt/data_from_server2/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/time_prediction_qwen3b_results.jsonl"
    test_parquet_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_march_april_prediction.parquet"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/time_prediction_march_april_results.jsonl"
    
    batch_size = 128  # 根据GPU显存调整
    max_new_tokens = 1024
    
    # 如果输出目录不存在，创建它
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果已存在输出文件，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print(f"开始测试Qwen2.5-3B-Instruct模型在时间预测数据集上的表现")
    print(f"模型路径: {model_path}")
    print(f"测试数据: {test_parquet_file}")
    print(f"结果输出: {output_file}")
    print(f"批处理大小: {batch_size}")
    print(f"最大生成token数: {max_new_tokens}")
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 加载测试数据
        test_samples = load_test_data(test_parquet_file)
        
        # 批量推理
        batch_inference(model, tokenizer, test_samples, output_file, batch_size, max_new_tokens)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 配置参数
    model_path = "/data/zliu331/temporal_reasoning/TinyZero/check_points_time_prediction_zero/time_prediction/zero/actor/global_step_360"
    test_parquet_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/small_test_time_prediction.parquet"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/time_r1_results/time_prediction_results.jsonl"
    
    # test_parquet_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_march_april_prediction.parquet"
    # output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/time_r1_results/time_prediction_march_april_results.jsonl"
    batch_size = 128  # 根据GPU显存调整
    max_new_tokens = 1024
    
    # 如果输出目录不存在，创建它
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果已存在输出文件，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print(f"开始测试Time-R1模型在3月和4月时间预测数据集上的表现")
    print(f"模型路径: {model_path}")
    print(f"测试数据: {test_parquet_file}")
    print(f"结果输出: {output_file}")
    print(f"批处理大小: {batch_size}")
    print(f"最大生成token数: {max_new_tokens}")
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 加载测试数据
        test_samples = load_test_data(test_parquet_file)
        
        # 批量推理
        batch_inference(model, tokenizer, test_samples, output_file, batch_size, max_new_tokens)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # main_qwen()
    main()