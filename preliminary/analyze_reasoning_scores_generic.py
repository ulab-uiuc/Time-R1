import json
import re
import pandas as pd
import os
import sys
from tqdm import tqdm
from verl.utils.reward_score.time_reasoning import compute_score
import argparse # 导入 argparse

def analyze_reasoning_scores(
    results_file,
    test_data_file,
    output_file=None,
    model_name="Model" # 添加模型名称参数
):
    """
    分析模型在time reasoning任务上的表现 (通用版本)

    参数:
        results_file: 模型的结果文件路径 (JSONL)
        test_data_file: 原始测试数据文件路径 (Parquet)
        output_file: 可选的结果保存路径 (JSON)
        model_name: 用于输出的模型名称
    """
    print(f"加载测试数据集: {test_data_file}")
    # 读取原始测试数据
    try:
        test_df = pd.read_parquet(test_data_file)
    except Exception as e:
        print(f"错误: 无法加载测试数据文件 {test_data_file}. Error: {e}")
        return None

    # 创建索引到测试样本的映射
    test_samples = {}
    task_type_counts = {}

    # 转换为记录列表并按照与创建请求时相同的方式进行遍历
    test_records = test_df.to_dict('records')
    for idx, row in enumerate(test_records):
        # 检查 'extra_info' 是否存在且为字典
        extra_info = row.get('extra_info')
        if not isinstance(extra_info, dict):
             print(f"警告: 第 {idx} 行缺少 'extra_info' 字典或格式错误，跳过。")
             continue
        task_type = extra_info.get('task', 'unknown')

        # 统计每种任务类型的数量
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # 使用与创建请求时相同的ID构造方式
        key = f"{task_type}_{idx}"
        test_samples[key] = row

    print(f"读取完成，共加载 {len(test_samples)} 个有效测试样本")
    if not test_samples:
        print("错误: 未能从测试数据中加载任何有效样本。")
        return None

    print("测试集任务类型分布:")
    for task, count in sorted(task_type_counts.items()):
        print(f"  - {task}: {count}条")

    print(f"\n加载 {model_name} 回答: {results_file}")
    # 读取模型的回答
    model_results = []
    try:
        with open(results_file, 'r', encoding='utf-8') as f: # 指定 encoding
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        model_results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"警告: 无法解析行: {line[:50]}... Error: {e}")
    except FileNotFoundError:
         print(f"错误: 结果文件未找到 {results_file}")
         return None
    except Exception as e:
         print(f"错误: 读取结果文件时出错 {results_file}. Error: {e}")
         return None


    print(f"读取完成，共加载 {len(model_results)} 个 {model_name} 回答")
    if not model_results:
        print(f"警告: 未从 {results_file} 加载任何回答。")
        # 即使没有回答，也继续执行以生成空的或部分结果

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
    no_id_count = 0
    no_response_count = 0
    no_choices_count = 0
    no_message_count = 0
    not_found_ids = set()

    # 处理每个模型回答
    print("\n开始计算得分...")
    for result in tqdm(model_results):
        custom_id = result.get('custom_id')
        if not custom_id:
            no_id_count += 1
            skipped_count += 1
            continue

        # 检查响应结构和内容过滤
        response = result.get('response')
        if not isinstance(response, dict):
            no_response_count += 1
            skipped_count += 1
            continue

        body = response.get('body')
        if not isinstance(body, dict):
             # 尝试直接从 response 获取 choices (兼容某些格式)
             choices = response.get('choices')
             if not isinstance(choices, list):
                 no_choices_count += 1 # 假设没有 body 也没有 choices
                 skipped_count += 1
                 continue
        else:
             choices = body.get('choices')
             if not isinstance(choices, list):
                 no_choices_count += 1
                 skipped_count += 1
                 continue

        # 检查内容过滤
        if choices and choices[0].get('finish_reason') == 'content_filter':
            content_filter_count += 1
            # 不跳过，尝试评分，但记录下来
            # continue # 如果需要完全跳过被过滤的内容，取消此行注释

        # 获取回答内容和推理过程
        content = ''
        reasoning = ''
        if choices and len(choices) > 0:
            message = choices[0].get('message')
            if isinstance(message, dict):
                content = message.get('content', '')
                reasoning = message.get('reasoning_content', '') # 有些模型可能没有这个字段
            else:
                no_message_count += 1
                # 即使没有 message 也尝试继续，content 和 reasoning 会是空字符串
        else:
             # 如果 choices 为空或格式不对
             no_choices_count +=1 # 再次计数以防万一
             # skipped_count += 1 # 如果没有 choices 就无法评分，可以选择跳过
             # continue

        # 获取对应的测试样本
        test_sample = test_samples.get(custom_id)
        if test_sample is None:
            not_found_ids.add(custom_id)
            skipped_count += 1
            continue

        # 提取任务类型
        match = re.match(r'([a-zA-Z_]+)_\d+', custom_id) # 改进正则以匹配更复杂的任务名
        task_type = match.group(1) if match else 'unknown'

        # 检查 ground_truth 结构
        reward_model_info = test_sample.get('reward_model')
        if not isinstance(reward_model_info, dict):
            print(f"警告 ({custom_id}): 测试样本缺少 'reward_model' 字典。")
            skipped_count += 1
            continue
        ground_truth = reward_model_info.get('ground_truth')
        if not isinstance(ground_truth, dict):
            print(f"警告 ({custom_id}): 测试样本缺少 'ground_truth' 字典。")
            skipped_count += 1
            continue

        # 构建评分输入
        solution_str = f"<think>{reasoning or ''}</think>\n{content or ''}" # 确保不传入 None
        # 复制 ground_truth 以免修改原始数据
        current_ground_truth = ground_truth.copy()
        current_ground_truth['task'] = task_type  # 添加任务类型到ground_truth

        # 计算得分
        try:
            # 确保 compute_score 能处理 task_type
            score_results = compute_score(solution_str, current_ground_truth)
            if score_results is None or not isinstance(score_results, (list, tuple)) or len(score_results) < 2:
                 print(f"警告 ({custom_id}): compute_score 返回无效结果: {score_results}")
                 skipped_count += 1
                 continue

            total_score, accuracy_score = score_results[0], score_results[1]

            # 记录得分
            scores['all']['total'].append(total_score)
            scores['all']['accuracy'].append(accuracy_score)

            if task_type not in scores:
                 scores[task_type] = {'total': [], 'accuracy': []} # 动态添加未知任务类型
                 print(f"信息: 发现新的任务类型 '{task_type}' 并开始记录分数。")

            scores[task_type]['total'].append(total_score)
            scores[task_type]['accuracy'].append(accuracy_score)

            processed_count += 1
        except Exception as e:
            print(f"评分错误 ({custom_id}): {str(e)}")
            # print(f"  Solution: {solution_str[:100]}...") # 调试用
            # print(f"  Ground Truth: {current_ground_truth}") # 调试用
            skipped_count += 1

    # 计算平均得分
    results = {}
    all_tasks = set(scores.keys()) # 获取所有记录了分数的任务类型
    for task in sorted(list(all_tasks)): # 按字母排序输出
        task_scores = scores[task]
        if task_scores.get('total'): # 检查是否有分数记录
            avg_total = sum(task_scores['total']) / len(task_scores['total'])
            avg_accuracy = sum(task_scores['accuracy']) / len(task_scores['accuracy'])
            results[task] = {
                'avg_total_score': avg_total,
                'avg_accuracy_score': avg_accuracy,
                'count': len(task_scores['total'])
            }
        # else: # 如果需要报告没有分数的任务
        #     results[task] = {
        #         'avg_total_score': 0,
        #         'avg_accuracy_score': 0,
        #         'count': 0
        #     }


    # 输出结果
    print(f"\n===== {model_name} 在Time Reasoning任务上的表现 =====") # 使用模型名称
    print(f"处理完成 (成功评分): {processed_count}条")
    print(f"跳过 (无法评分): {skipped_count}条")
    print(f"  - 其中内容过滤: {content_filter_count}条")
    print(f"  - 其中缺少 custom_id: {no_id_count}条")
    print(f"  - 其中缺少 response 字典: {no_response_count}条")
    print(f"  - 其中缺少 choices 列表: {no_choices_count}条")
    print(f"  - 其中缺少 message 字典: {no_message_count}条")
    if not_found_ids:
        print(f"  - 其中未找到对应测试样本的ID数量: {len(not_found_ids)}")
        if len(not_found_ids) < 20: # 显示更多未找到的ID
            print(f"    未找到的ID示例: {', '.join(list(not_found_ids)[:20])}")

    if 'all' in results:
        print("\n总体平均得分:")
        print(f"  样本数量: {results['all']['count']}")
        print(f"  平均总分 (Total Score): {results['all']['avg_total_score']:.4f}")
        print(f"  平均准确度 (Accuracy Score): {results['all']['avg_accuracy_score']:.4f}")
    else:
        print("\n未能计算总体平均得分。")

    print("\n按任务类型的平均得分:")
    # 预定义任务顺序，但也包括可能发现的新任务
    task_order = ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']
    displayed_tasks = set()
    for task in task_order:
        if task in results and task != 'all':
            print(f"\n{task}:")
            print(f"  样本数量: {results[task]['count']}")
            print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
            print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")
            displayed_tasks.add(task)

    # 显示未在预定义顺序中的其他任务
    other_tasks = sorted(list(set(results.keys()) - displayed_tasks - {'all'}))
    if other_tasks:
         print("\n其他任务类型:")
         for task in other_tasks:
             print(f"\n{task}:")
             print(f"  样本数量: {results[task]['count']}")
             print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
             print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")


    # 保存结果
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f: # 指定 encoding
                json.dump(results, f, indent=2, ensure_ascii=False) # ensure_ascii=False
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"错误: 无法保存结果到 {output_file}. Error: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析模型在Time Reasoning任务上的表现")
    parser.add_argument("--results_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_32b_results.jsonl", help="模型结果文件路径 (JSONL)")
    parser.add_argument("--test_data_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet", help="原始测试数据文件路径 (Parquet)")
    parser.add_argument("--output_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_32b_scores.json", help="可选的结果保存路径 (JSON)")
    parser.add_argument("--model_name", type=str, default="DeepSeek 32B Distilled", help="用于输出的模型名称")

    args = parser.parse_args()

    # 运行分析
    analyze_reasoning_scores(
        args.results_file,
        args.test_data_file,
        args.output_file,
        args.model_name
    )