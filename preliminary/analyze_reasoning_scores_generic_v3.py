import json
import re
import pandas as pd
import os
import sys
from tqdm import tqdm
from verl.utils.reward_score.time_reasoning import compute_score
import argparse

# 正则表达式用于提取 <think> 标签内容
think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def analyze_reasoning_scores(
    results_file,
    test_data_file,
    output_file=None,
    model_name="Model"
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
    try:
        test_df = pd.read_parquet(test_data_file)
    except Exception as e:
        print(f"错误: 无法加载测试数据文件 {test_data_file}. Error: {e}")
        return None

    test_samples = {}
    task_type_counts = {}
    test_records = test_df.to_dict('records')
    for idx, row in enumerate(test_records):
        extra_info = row.get('extra_info')
        if not isinstance(extra_info, dict):
             print(f"警告: 第 {idx} 行缺少 'extra_info' 字典或格式错误，跳过。")
             continue
        task_type = extra_info.get('task', 'unknown')
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
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
    model_results = []
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
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

    print("\n开始计算得分...")
    for result in tqdm(model_results):
        custom_id = result.get('custom_id')
        if not custom_id:
            no_id_count += 1
            skipped_count += 1
            continue

        response = result.get('response')
        if not isinstance(response, dict):
            no_response_count += 1
            skipped_count += 1
            continue

        body = response.get('body')
        choices = None
        if isinstance(body, dict):
             choices = body.get('choices')
        elif isinstance(response.get('choices'), list): # 兼容直接在 response 下的 choices
             choices = response.get('choices')

        if not isinstance(choices, list):
            no_choices_count += 1
            skipped_count += 1
            continue

        if choices and choices[0].get('finish_reason') == 'content_filter':
            content_filter_count += 1
            # 仍然尝试处理，只是记录

        content = ''
        reasoning = ''
        if choices and len(choices) > 0:
            message = choices[0].get('message')
            if isinstance(message, dict):
                raw_content = message.get('content', '')

                # *** 修改开始：处理 V3 的 <think> 标签 ***
                think_match = think_pattern.search(raw_content)
                if think_match:
                    reasoning = think_match.group(1).strip() # 提取 <think> 内容
                    # 从 content 中移除 <think>...</think> 部分
                    content = think_pattern.sub('', raw_content).strip()
                    # print(f"DEBUG ({custom_id}): Found <think> in content. Reasoning extracted, content modified.") # 调试信息
                else:
                    # 如果 content 中没有 <think>，尝试获取 reasoning_content (兼容 R1)
                    content = raw_content # 原始 content
                    reasoning = message.get('reasoning_content', '') # 尝试获取独立字段
                    # print(f"DEBUG ({custom_id}): No <think> in content. Trying reasoning_content field.") # 调试信息
                # *** 修改结束 ***

            else:
                no_message_count += 1
        else:
             no_choices_count +=1

        test_sample = test_samples.get(custom_id)
        if test_sample is None:
            not_found_ids.add(custom_id)
            skipped_count += 1
            continue

        match = re.match(r'([a-zA-Z_]+)_\d+', custom_id)
        task_type = match.group(1) if match else 'unknown'

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

        # 构建评分输入，确保 reasoning 和 content 不为 None
        solution_str = f"<think>{reasoning or ''}</think>\n{content or ''}"
        current_ground_truth = ground_truth.copy()
        current_ground_truth['task'] = task_type

        try:
            score_results = compute_score(solution_str, current_ground_truth)
            if score_results is None or not isinstance(score_results, (list, tuple)) or len(score_results) < 2:
                 print(f"警告 ({custom_id}): compute_score 返回无效结果: {score_results}")
                 skipped_count += 1
                 continue

            total_score, accuracy_score = score_results[0], score_results[1]

            scores['all']['total'].append(total_score)
            scores['all']['accuracy'].append(accuracy_score)

            if task_type not in scores:
                 scores[task_type] = {'total': [], 'accuracy': []}
                 print(f"信息: 发现新的任务类型 '{task_type}' 并开始记录分数。")

            scores[task_type]['total'].append(total_score)
            scores[task_type]['accuracy'].append(accuracy_score)

            processed_count += 1
        except Exception as e:
            print(f"评分错误 ({custom_id}): {str(e)}")
            # print(f"  Solution: {solution_str[:100]}...") # 调试用
            # print(f"  Ground Truth: {current_ground_truth}") # 调试用
            skipped_count += 1

    results = {}
    all_tasks = set(scores.keys())
    for task in sorted(list(all_tasks)):
        task_scores = scores[task]
        if task_scores.get('total'):
            avg_total = sum(task_scores['total']) / len(task_scores['total'])
            avg_accuracy = sum(task_scores['accuracy']) / len(task_scores['accuracy'])
            results[task] = {
                'avg_total_score': avg_total,
                'avg_accuracy_score': avg_accuracy,
                'count': len(task_scores['total'])
            }

    print(f"\n===== {model_name} 在Time Reasoning任务上的表现 =====")
    print(f"处理完成 (成功评分): {processed_count}条")
    print(f"跳过 (无法评分): {skipped_count}条")
    print(f"  - 其中内容过滤: {content_filter_count}条")
    print(f"  - 其中缺少 custom_id: {no_id_count}条")
    print(f"  - 其中缺少 response 字典: {no_response_count}条")
    print(f"  - 其中缺少 choices 列表: {no_choices_count}条")
    print(f"  - 其中缺少 message 字典: {no_message_count}条")
    if not_found_ids:
        print(f"  - 其中未找到对应测试样本的ID数量: {len(not_found_ids)}")
        if len(not_found_ids) < 20:
            print(f"    未找到的ID示例: {', '.join(list(not_found_ids)[:20])}")

    if 'all' in results:
        print("\n总体平均得分:")
        print(f"  样本数量: {results['all']['count']}")
        print(f"  平均总分 (Total Score): {results['all']['avg_total_score']:.4f}")
        print(f"  平均准确度 (Accuracy Score): {results['all']['avg_accuracy_score']:.4f}")
    else:
        print("\n未能计算总体平均得分。")

    print("\n按任务类型的平均得分:")
    task_order = ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']
    displayed_tasks = set()
    for task in task_order:
        if task in results and task != 'all':
            print(f"\n{task}:")
            print(f"  样本数量: {results[task]['count']}")
            print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
            print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")
            displayed_tasks.add(task)

    other_tasks = sorted(list(set(results.keys()) - displayed_tasks - {'all'}))
    if other_tasks:
         print("\n其他任务类型:")
         for task in other_tasks:
             print(f"\n{task}:")
             print(f"  样本数量: {results[task]['count']}")
             print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
             print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"错误: 无法保存结果到 {output_file}. Error: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析模型在Time Reasoning任务上的表现")
    parser.add_argument("--results_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_v3_results.jsonl", help="模型结果文件路径 (JSONL)")
    parser.add_argument("--test_data_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet", help="原始测试数据文件路径 (Parquet)")
    parser.add_argument("--output_file", type=str, default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/r1_results/reasoning_v3_scores.json", help="可选的结果保存路径 (JSON)")
    parser.add_argument("--model_name", type=str, default="DeepSeek V3", help="用于输出的模型名称")

    args = parser.parse_args()

    analyze_reasoning_scores(
        args.results_file,
        args.test_data_file,
        args.output_file,
        args.model_name
    )