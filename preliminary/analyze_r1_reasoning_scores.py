import json
import re
import pandas as pd
import os
import sys
from tqdm import tqdm
from verl.utils.reward_score.time_reasoning import compute_score

def analyze_r1_reasoning_scores(
    results_file,
    test_data_file,
    output_file=None
):
    """
    分析R1模型在time reasoning任务上的表现
    
    参数:
        results_file: R1模型的结果文件路径
        test_data_file: 原始测试数据文件路径
        output_file: 可选的结果保存路径
    """
    print(f"加载测试数据集: {test_data_file}")
    # 读取原始测试数据
    test_df = pd.read_parquet(test_data_file)
    
    # 创建索引到测试样本的映射
    test_samples = {}
    task_type_counts = {}
    
    # 转换为记录列表并按照与创建请求时相同的方式进行遍历
    test_records = test_df.to_dict('records')
    for idx, row in enumerate(test_records):
        task_type = row['extra_info'].get('task', 'unknown')
        
        # 统计每种任务类型的数量
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # 使用与创建请求时相同的ID构造方式
        key = f"{task_type}_{idx}"
        test_samples[key] = row
    
    print(f"读取完成，共加载 {len(test_samples)} 个测试样本")
    print("测试集任务类型分布:")
    for task, count in sorted(task_type_counts.items()):
        print(f"  - {task}: {count}条")
    
    print(f"\n加载R1回答: {results_file}")
    # 读取R1的回答
    r1_results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    r1_results.append(data)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行: {line[:50]}...")
    
    print(f"读取完成，共加载 {len(r1_results)} 个R1回答")
    
    # 初始化得分统计
    scores = {
        'all': {'total': [], 'accuracy': []},
        'time_inferring': {'total': [], 'accuracy': []},
        'time_difference': {'total': [], 'accuracy': []},
        'time_ordering': {'total': [], 'accuracy': []},
        'time_completion': {'total': [], 'accuracy': []}
    }
    
    processed_count = 0
    skipped_count = 0
    content_filter_count = 0
    not_found_ids = set()
    
    # 处理每个R1回答
    print("\n开始计算得分...")
    for result in tqdm(r1_results):
        custom_id = result.get('custom_id')
        if not custom_id:
            skipped_count += 1
            continue
        
        # 检查是否被内容过滤
        response = result.get('response', {})
        choices = response.get('body', {}).get('choices', [])
        if choices and choices[0].get('finish_reason') == 'content_filter':
            content_filter_count += 1
            continue
            
        # 获取回答内容和推理过程
        content = ''
        reasoning = ''
        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            content = message.get('content', '')
            reasoning = message.get('reasoning_content', '')
        
        # 获取对应的测试样本 - 修复此处的判断
        test_sample = test_samples.get(custom_id)
        if test_sample is None:  # 使用明确的None判断
            skipped_count += 1
            not_found_ids.add(custom_id)
            continue
        
        # 提取任务类型
        match = re.match(r'(\w+)_\d+', custom_id)
        task_type = match.group(1) if match else 'unknown'
        
        # 构建评分输入
        solution_str = f"<think>{reasoning}</think>\n{content}"
        ground_truth = test_sample['reward_model']['ground_truth']
        ground_truth['task'] = task_type  # 添加任务类型到ground_truth
        
        # 计算得分
        try:
            score_results = compute_score(solution_str, ground_truth)
            total_score, accuracy_score = score_results[0], score_results[1]
            
            # 记录得分
            scores['all']['total'].append(total_score)
            scores['all']['accuracy'].append(accuracy_score)
            
            if task_type in scores:
                scores[task_type]['total'].append(total_score)
                scores[task_type]['accuracy'].append(accuracy_score)
            
            processed_count += 1
        except Exception as e:
            print(f"评分错误 ({custom_id}): {str(e)}")
            skipped_count += 1
    
    # 计算平均得分
    results = {}
    for task, task_scores in scores.items():
        if task_scores['total']:
            avg_total = sum(task_scores['total']) / len(task_scores['total'])
            avg_accuracy = sum(task_scores['accuracy']) / len(task_scores['accuracy'])
            results[task] = {
                'avg_total_score': avg_total,
                'avg_accuracy_score': avg_accuracy,
                'count': len(task_scores['total'])
            }
    
    # 输出结果
    print("\n===== R1模型在Time Reasoning任务上的表现 =====")
    print(f"处理完成: {processed_count}条，跳过: {skipped_count}条，内容过滤: {content_filter_count}条")
    if not_found_ids:
        print(f"未找到对应测试样本的ID数量: {len(not_found_ids)}")
        if len(not_found_ids) < 10:
            print(f"未找到的ID: {', '.join(list(not_found_ids))}")
    
    print("\n总体平均得分:")
    print(f"  平均总分 (Total Score): {results['all']['avg_total_score']:.4f}")
    print(f"  平均准确度 (Accuracy Score): {results['all']['avg_accuracy_score']:.4f}")
    
    print("\n按任务类型的平均得分:")
    for task in ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']:
        if task in results:
            print(f"\n{task}:")
            print(f"  样本数量: {results[task]['count']}")
            print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
            print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    # 文件路径
    results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_r1_results.jsonl"
    test_data_file = "/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet"
    output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_r1_scores.json"
    
    # 运行分析
    analyze_r1_reasoning_scores(results_file, test_data_file, output_file)