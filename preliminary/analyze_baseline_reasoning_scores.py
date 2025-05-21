import json
import re
import pandas as pd
import os
from tqdm import tqdm
from verl.utils.reward_score.time_reasoning import compute_score # 假设此模块可用
import argparse

# Qwen7B 的主要响应格式: REASONING_TEXT</think>\n<answer>ANSWER_TEXT</answer>
# 分组：(1) 推理过程 (2) <answer>标签内的答案文本
qwen_specific_pattern = re.compile(r"^(.*?)<\/think>\s*\n?<answer>(.*?)<\/answer>", re.DOTALL)
# 通用<answer>标签提取，用于回退
answer_tag_pattern = re.compile(r"<answer>(.*?)<\/answer>", re.DOTALL)


def analyze_model_reasoning_scores(
    results_file,
    test_data_file,
    output_file=None,
    model_name="Qwen3B" # 默认模型名称更改
):
    """
    分析模型在时间推理任务上的表现。

    参数:
        results_file: 模型的结果文件路径 (JSONL)。
        test_data_file: 原始测试数据文件路径 (Parquet)。
        output_file: 可选的结果保存路径 (JSON)。
        model_name: 用于输出的模型名称。
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
    print(f"从 Parquet 文件加载了 {len(test_records)} 条记录。")
    for idx, row_data in enumerate(test_records):
        extra_info_parquet = row_data.get('extra_info')
        if not isinstance(extra_info_parquet, dict):
             print(f"警告: 测试数据 Parquet 第 {idx} 行缺少 'extra_info' 字典或格式错误，跳过。")
             continue
        
        task_type_parquet = extra_info_parquet.get('task')
        index_parquet = extra_info_parquet.get('index') # 这是来自数据集的原始索引

        if task_type_parquet is None or index_parquet is None:
            print(f"警告: 测试数据 Parquet 第 {idx} 行 'extra_info' 中缺少 'task' 或 'index'，跳过。")
            continue
        
        task_type_counts[task_type_parquet] = task_type_counts.get(task_type_parquet, 0) + 1
        key = f"{task_type_parquet}_{index_parquet}" # 使用原始 task 和 index 构建key
        test_samples[key] = row_data

    print(f"测试数据加载完成，共 {len(test_samples)} 个有效测试样本。")
    if not test_samples:
        print("错误: 未能从测试数据中加载任何有效样本。")
        return None

    print("测试集任务类型分布:")
    for task, count in sorted(task_type_counts.items()):
        print(f"  - {task}: {count}条")

    print(f"\n加载 {model_name} 的回答: {results_file}")
    model_results = []
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        model_results.append(data)
                    except json.JSONDecodeError as e:
                        print(f"警告: 无法解析结果文件第 {line_num} 行: {line[:100]}... Error: {e}")
    except FileNotFoundError:
         print(f"错误: 结果文件未找到 {results_file}")
         return None
    except Exception as e:
         print(f"错误: 读取结果文件时出错 {results_file}. Error: {e}")
         return None

    print(f"读取完成，共加载 {len(model_results)} 个 {model_name} 的回答。")

    scores = {
        'all': {'total': [], 'accuracy': []},
        'time_inferring': {'total': [], 'accuracy': []},
        'time_difference': {'total': [], 'accuracy': []},
        'time_ordering': {'total': [], 'accuracy': []},
        'time_completion': {'total': [], 'accuracy': []}
    }

    processed_count = 0
    skipped_count = 0
    no_proper_id_count = 0 # 用于统计结果文件中缺少有效 extra_info 的情况
    no_response_str_count = 0
    not_found_in_test_data_count = 0
    not_found_keys_log = set()


    print("\n开始计算得分...")
    for result_item in tqdm(model_results, desc=f"评估 {model_name}"):
        raw_model_output_str = result_item.get('response', '')
        if not isinstance(raw_model_output_str, str) or not raw_model_output_str.strip():
            no_response_str_count += 1
            skipped_count += 1
            custom_id_for_log = result_item.get('custom_id', 'N/A')
            # print(f"警告 (custom_id: {custom_id_for_log}): response 字段为空或非字符串。")
            continue

        # 从结果文件的 extra_info 中获取 task 和 index 来定位测试样本
        result_extra_info = result_item.get('extra_info')
        lookup_key = None
        task_type_for_scoring = 'unknown'

        if isinstance(result_extra_info, dict):
            task_type_from_result = result_extra_info.get('task')
            index_from_result = result_extra_info.get('index')
            if task_type_from_result and index_from_result is not None:
                lookup_key = f"{task_type_from_result}_{index_from_result}"
                task_type_for_scoring = task_type_from_result
            else:
                no_proper_id_count += 1
        else:
            no_proper_id_count += 1
        
        if not lookup_key: # 如果无法从 result_extra_info 构建 key
            custom_id_val = result_item.get('custom_id', 'UnknownID')
            # print(f"警告: 结果条目 {custom_id_val} 的 'extra_info' 缺少 'task' 或 'index'。跳过。")
            skipped_count += 1
            continue
            
        test_sample = test_samples.get(lookup_key)
        if test_sample is None:
            not_found_in_test_data_count += 1
            not_found_keys_log.add(lookup_key)
            skipped_count += 1
            continue
        
        # 确保 task_type_for_scoring 来自测试样本的 truth，如果之前未设置
        if task_type_for_scoring == 'unknown':
             task_type_for_scoring = test_sample.get('extra_info',{}).get('task', 'unknown')


        reasoning = ''
        content_for_final_answer_tag = '' # 存储 <answer> 标签内的文本

        qwen_match = qwen_specific_pattern.search(raw_model_output_str)
        if qwen_match:
            reasoning = qwen_match.group(1).strip()
            content_for_final_answer_tag = qwen_match.group(2).strip()
        else:
            # 回退逻辑：尝试直接寻找 <answer> 标签
            answer_only_match = answer_tag_pattern.search(raw_model_output_str)
            if answer_only_match:
                content_for_final_answer_tag = answer_only_match.group(1).strip()
                # 将 <answer> 之前的内容视为推理
                reasoning_candidate = raw_model_output_str[:answer_only_match.start()].strip()
                if reasoning_candidate.endswith("</think>"): # 清理可能的悬挂 </think>
                    reasoning = reasoning_candidate[:-len("</think>")].strip()
                elif reasoning_candidate:
                    reasoning = reasoning_candidate
                else:
                    reasoning = '' # <answer> 之前无内容
            else:
                # 如果没有特定结构，将整个响应视为答案内容，无显式推理
                content_for_final_answer_tag = raw_model_output_str.strip()
                reasoning = ''
        
        solution_str = f"<think>{reasoning}</think>\n<answer>{content_for_final_answer_tag}</answer>"
        
        # 从 test_sample (Parquet 数据) 获取 ground_truth
        # 假设 ground_truth 在 parquet 文件的 'reward_model' -> 'ground_truth' 路径下
        reward_model_info = test_sample.get('reward_model') 
        if not isinstance(reward_model_info, dict):
            print(f"警告 ({lookup_key}): 测试样本缺少 'reward_model' 字典。")
            skipped_count += 1
            continue
        ground_truth = reward_model_info.get('ground_truth')
        if not isinstance(ground_truth, dict):
            print(f"警告 ({lookup_key}): 测试样本缺少 'ground_truth' 数据。")
            skipped_count += 1
            continue

        current_ground_truth = ground_truth.copy()
        current_ground_truth['task'] = task_type_for_scoring # compute_score 需要 'task' 字段

        try:
            score_results = compute_score(solution_str, current_ground_truth)
            if score_results is None or not isinstance(score_results, (list, tuple)) or len(score_results) < 2:
                 print(f"警告 ({lookup_key}): compute_score 返回无效结果: {score_results}")
                 skipped_count += 1
                 continue

            total_score, accuracy_score = score_results[0], score_results[1]

            scores['all']['total'].append(total_score)
            scores['all']['accuracy'].append(accuracy_score)

            if task_type_for_scoring not in scores: # 动态添加新的任务类型
                 scores[task_type_for_scoring] = {'total': [], 'accuracy': []}
                 # print(f"信息: 发现新的任务类型 '{task_type_for_scoring}' 并开始记录分数。")
            scores[task_type_for_scoring]['total'].append(total_score)
            scores[task_type_for_scoring]['accuracy'].append(accuracy_score)

            processed_count += 1
        except Exception as e:
            print(f"评分错误 ({lookup_key}): {str(e)}")
            # print(f"  Solution: {solution_str[:300]}...") # 调试用
            # print(f"  Ground Truth: {current_ground_truth}") # 调试用
            skipped_count += 1

    # 计算并输出结果
    results_summary = {}
    all_task_keys = sorted(list(scores.keys()))
    for task_key in all_task_keys:
        task_scores_data = scores[task_key]
        if task_scores_data.get('total'): # 确保有数据
            avg_total = sum(task_scores_data['total']) / len(task_scores_data['total'])
            avg_accuracy = sum(task_scores_data['accuracy']) / len(task_scores_data['accuracy'])
            results_summary[task_key] = {
                'avg_total_score': avg_total,
                'avg_accuracy_score': avg_accuracy,
                'count': len(task_scores_data['total'])
            }

    print(f"\n===== {model_name} 在时间推理任务上的表现 =====")
    print(f"成功评分的样本数: {processed_count}条")
    print(f"跳过的样本数 (无法评分): {skipped_count}条")
    if skipped_count > 0:
        print(f"  - 其中结果条目缺少有效 'extra_info' (task/index): {no_proper_id_count}条")
        print(f"  - 其中 response 字段为空或非字符串: {no_response_str_count}条")
        if not_found_in_test_data_count > 0:
            print(f"  - 其中在测试数据中未找到对应key的数量: {not_found_in_test_data_count}")
            if len(not_found_keys_log) < 20: # 打印少量示例
                print(f"    未找到的Key示例: {', '.join(list(not_found_keys_log)[:20])}")

    if 'all' in results_summary:
        print("\n总体平均得分:")
        print(f"  样本数量: {results_summary['all']['count']}")
        print(f"  平均总分 (Total Score): {results_summary['all']['avg_total_score']:.4f}")
        print(f"  平均准确度 (Accuracy Score): {results_summary['all']['avg_accuracy_score']:.4f}")
    else:
        print("\n未能计算总体平均得分。")

    print("\n按任务类型的平均得分:")
    defined_task_order = ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']
    displayed_tasks = set()
    for task_key in defined_task_order:
        if task_key in results_summary and task_key != 'all':
            print(f"\n任务类型: {task_key}")
            print(f"  样本数量: {results_summary[task_key]['count']}")
            print(f"  平均总分 (Total Score): {results_summary[task_key]['avg_total_score']:.4f}")
            print(f"  平均准确度 (Accuracy Score): {results_summary[task_key]['avg_accuracy_score']:.4f}")
            displayed_tasks.add(task_key)

    other_tasks = sorted(list(set(results_summary.keys()) - displayed_tasks - {'all'}))
    if other_tasks:
         print("\n其他任务类型:")
         for task_key in other_tasks:
             print(f"\n任务类型: {task_key}")
             print(f"  样本数量: {results_summary[task_key]['count']}")
             print(f"  平均总分 (Total Score): {results_summary[task_key]['avg_total_score']:.4f}")
             print(f"  平均准确度 (Accuracy Score): {results_summary[task_key]['avg_accuracy_score']:.4f}")

    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True) # 确保目录存在
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"错误: 无法保存结果到 {output_file}. Error: {e}")

    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析大语言模型在时间推理任务上的表现")
    parser.add_argument(
        "--results_file", 
        type=str, 
        default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/time_reasoning_qwen3b_results.jsonl", 
        help="模型结果文件路径 (JSONL)"
    )
    parser.add_argument(
        "--test_data_file", 
        type=str, 
        default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet", 
        help="原始测试数据文件路径 (Parquet)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen3b_results/qwen3b_time_reasoning_scores.json", 
        help="可选的结果保存路径 (JSON)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen3B", 
        help="用于输出的模型名称"
    )

    args = parser.parse_args()

    analyze_model_reasoning_scores( # 函数名已更改
        args.results_file,
        args.test_data_file,
        args.output_file,
        args.model_name
    )






# import json
# import re
# import pandas as pd
# import os
# from tqdm import tqdm
# from verl.utils.reward_score.time_reasoning import compute_score # 假设此模块可用
# import argparse

# # 正则表达式用于提取 Llama 3.1 的思考内容
# # 优先匹配 <|im_start|>think> ... <|im_end|>
# llama_think_pattern = re.compile(r"<\|im_start\|>think>(.*?)(?:<\|im_end\|>)", re.DOTALL)
# # 备用匹配 <think> ... </think>
# generic_think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# def analyze_llama31_reasoning_scores(
#     results_file,
#     test_data_file,
#     output_file=None,
#     model_name="Llama 3.1"
# ):
#     """
#     分析 Llama 3.1 模型在时间推理任务上的表现。

#     参数:
#         results_file: 模型的结果文件路径 (JSONL)。
#         test_data_file: 原始测试数据文件路径 (Parquet)。
#         output_file: 可选的结果保存路径 (JSON)。
#         model_name: 用于输出的模型名称。
#     """
#     print(f"加载测试数据集: {test_data_file}")
#     try:
#         test_df = pd.read_parquet(test_data_file)
#     except Exception as e:
#         print(f"错误: 无法加载测试数据文件 {test_data_file}. Error: {e}")
#         return None

#     test_samples = {}
#     task_type_counts = {}
#     test_records = test_df.to_dict('records')
#     for idx, row in enumerate(test_records):
#         extra_info = row.get('extra_info')
#         if not isinstance(extra_info, dict):
#              print(f"警告: 测试数据第 {idx} 行缺少 'extra_info' 字典或格式错误，跳过。")
#              continue
#         task_type = extra_info.get('task', 'unknown')
#         task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
#         # 使用 enumerate 的索引 idx 生成与 custom_id 匹配的键
#         key = f"{task_type}_{idx}"
#         test_samples[key] = row

#     print(f"测试数据加载完成，共 {len(test_samples)} 个有效测试样本。")
#     if not test_samples:
#         print("错误: 未能从测试数据中加载任何有效样本。")
#         return None

#     print("测试集任务类型分布:")
#     for task, count in sorted(task_type_counts.items()):
#         print(f"  - {task}: {count}条")

#     print(f"\n加载 {model_name} 的回答: {results_file}")
#     model_results = []
#     try:
#         with open(results_file, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(f, 1):
#                 if line.strip():
#                     try:
#                         data = json.loads(line)
#                         model_results.append(data)
#                     except json.JSONDecodeError as e:
#                         print(f"警告: 无法解析结果文件第 {line_num} 行: {line[:100]}... Error: {e}")
#     except FileNotFoundError:
#          print(f"错误: 结果文件未找到 {results_file}")
#          return None
#     except Exception as e:
#          print(f"错误: 读取结果文件时出错 {results_file}. Error: {e}")
#          return None

#     print(f"读取完成，共加载 {len(model_results)} 个 {model_name} 的回答。")

#     scores = {
#         'all': {'total': [], 'accuracy': []},
#         'time_inferring': {'total': [], 'accuracy': []},
#         'time_difference': {'total': [], 'accuracy': []},
#         'time_ordering': {'total': [], 'accuracy': []},
#         'time_completion': {'total': [], 'accuracy': []}
#     }

#     processed_count = 0
#     skipped_count = 0
#     no_id_count = 0
#     no_response_str_count = 0
#     not_found_ids = set()

#     print("\n开始计算得分...")
#     for result in tqdm(model_results, desc=f"评估 {model_name}"):
#         custom_id = result.get('custom_id')
#         if not custom_id:
#             no_id_count += 1
#             skipped_count += 1
#             continue

#         raw_model_output_str = result.get('response', '')
#         if not isinstance(raw_model_output_str, str) or not raw_model_output_str.strip():
#             no_response_str_count += 1
#             skipped_count += 1
#             # print(f"警告 ({custom_id}): response 字段为空或非字符串。")
#             continue

#         reasoning = ''
#         content_after_think = raw_model_output_str # 默认情况下，全部是回答内容

#         match_llama = llama_think_pattern.search(raw_model_output_str)
#         if match_llama:
#             reasoning = match_llama.group(1).strip()
#             content_after_think = llama_think_pattern.sub('', raw_model_output_str).strip()
#         else:
#             match_generic = generic_think_pattern.search(raw_model_output_str)
#             if match_generic:
#                 reasoning = match_generic.group(1).strip()
#                 content_after_think = generic_think_pattern.sub('', raw_model_output_str).strip()
        
#         solution_str = f"<think>{reasoning or ''}</think>\n{content_after_think or ''}"

#         test_sample = test_samples.get(custom_id)
#         if test_sample is None:
#             not_found_ids.add(custom_id)
#             skipped_count += 1
#             continue

#         # 从 custom_id 中提取 task_type，并从 test_sample 中获取 ground_truth
#         id_match = re.match(r'([a-zA-Z_]+)_\d+', custom_id)
#         task_type = id_match.group(1) if id_match else 'unknown'
        
#         # 确保从 test_sample (Parquet 数据) 获取 ground_truth
#         reward_model_info = test_sample.get('reward_model')
#         if not isinstance(reward_model_info, dict):
#             print(f"警告 ({custom_id}): 测试样本 {custom_id} 缺少 'reward_model' 字典。")
#             skipped_count += 1
#             continue
#         ground_truth = reward_model_info.get('ground_truth')
#         if not isinstance(ground_truth, dict):
#             print(f"警告 ({custom_id}): 测试样本 {custom_id} 缺少 'ground_truth' 数据。")
#             skipped_count += 1
#             continue

#         current_ground_truth = ground_truth.copy()
#         current_ground_truth['task'] = task_type # compute_score 需要 'task' 字段

#         try:
#             score_results = compute_score(solution_str, current_ground_truth)
#             if score_results is None or not isinstance(score_results, (list, tuple)) or len(score_results) < 2:
#                  print(f"警告 ({custom_id}): compute_score 返回无效结果: {score_results}")
#                  skipped_count += 1
#                  continue

#             total_score, accuracy_score = score_results[0], score_results[1]

#             scores['all']['total'].append(total_score)
#             scores['all']['accuracy'].append(accuracy_score)

#             if task_type not in scores: # 动态添加新的任务类型
#                  scores[task_type] = {'total': [], 'accuracy': []}
#                  print(f"信息: 发现新的任务类型 '{task_type}' 并开始记录分数。")
#             scores[task_type]['total'].append(total_score)
#             scores[task_type]['accuracy'].append(accuracy_score)

#             processed_count += 1
#         except Exception as e:
#             print(f"评分错误 ({custom_id}): {str(e)}")
#             # print(f"  Solution: {solution_str[:200]}...") # 调试用
#             # print(f"  Ground Truth: {current_ground_truth}") # 调试用
#             skipped_count += 1

#     # 计算并输出结果
#     results_summary = {}
#     all_task_keys = sorted(list(scores.keys()))
#     for task_key in all_task_keys:
#         task_scores_data = scores[task_key]
#         if task_scores_data.get('total'): # 确保有数据
#             avg_total = sum(task_scores_data['total']) / len(task_scores_data['total'])
#             avg_accuracy = sum(task_scores_data['accuracy']) / len(task_scores_data['accuracy'])
#             results_summary[task_key] = {
#                 'avg_total_score': avg_total,
#                 'avg_accuracy_score': avg_accuracy,
#                 'count': len(task_scores_data['total'])
#             }

#     print(f"\n===== {model_name} 在时间推理任务上的表现 =====")
#     print(f"成功评分的样本数: {processed_count}条")
#     print(f"跳过的样本数 (无法评分): {skipped_count}条")
#     if skipped_count > 0:
#         print(f"  - 其中缺少 custom_id: {no_id_count}条")
#         print(f"  - 其中 response 字段为空或非字符串: {no_response_str_count}条")
#         if not_found_ids:
#             print(f"  - 其中在测试数据中未找到 custom_id 的数量: {len(not_found_ids)}")
#             if len(not_found_ids) < 20: # 打印少量示例
#                 print(f"    未找到的ID示例: {', '.join(list(not_found_ids)[:20])}")

#     if 'all' in results_summary:
#         print("\n总体平均得分:")
#         print(f"  样本数量: {results_summary['all']['count']}")
#         print(f"  平均总分 (Total Score): {results_summary['all']['avg_total_score']:.4f}")
#         print(f"  平均准确度 (Accuracy Score): {results_summary['all']['avg_accuracy_score']:.4f}")
#     else:
#         print("\n未能计算总体平均得分。")

#     print("\n按任务类型的平均得分:")
#     # 预定义任务顺序以便更清晰地展示
#     defined_task_order = ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']
#     displayed_tasks = set()
#     for task_key in defined_task_order:
#         if task_key in results_summary and task_key != 'all':
#             print(f"\n任务类型: {task_key}")
#             print(f"  样本数量: {results_summary[task_key]['count']}")
#             print(f"  平均总分 (Total Score): {results_summary[task_key]['avg_total_score']:.4f}")
#             print(f"  平均准确度 (Accuracy Score): {results_summary[task_key]['avg_accuracy_score']:.4f}")
#             displayed_tasks.add(task_key)

#     # 处理其他未在预定义顺序中的任务类型
#     other_tasks = sorted(list(set(results_summary.keys()) - displayed_tasks - {'all'}))
#     if other_tasks:
#          print("\n其他任务类型:")
#          for task_key in other_tasks:
#              print(f"\n任务类型: {task_key}")
#              print(f"  样本数量: {results_summary[task_key]['count']}")
#              print(f"  平均总分 (Total Score): {results_summary[task_key]['avg_total_score']:.4f}")
#              print(f"  平均准确度 (Accuracy Score): {results_summary[task_key]['avg_accuracy_score']:.4f}")

#     if output_file:
#         try:
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(results_summary, f, indent=2, ensure_ascii=False)
#             print(f"\n结果已保存到: {output_file}")
#         except Exception as e:
#             print(f"错误: 无法保存结果到 {output_file}. Error: {e}")

#     return results_summary

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="分析 Llama 3.1 模型在时间推理任务上的表现")
#     parser.add_argument(
#         "--results_file", 
#         type=str, 
#         default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/time_reasoning_llama31_results.jsonl", 
#         help="Llama 3.1 模型结果文件路径 (JSONL)"
#     )
#     parser.add_argument(
#         "--test_data_file", 
#         type=str, 
#         default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet", 
#         help="原始测试数据文件路径 (Parquet)"
#     )
#     parser.add_argument(
#         "--output_file", 
#         type=str, 
#         default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/llama31_time_reasoning_scores.json", 
#         help="可选的结果保存路径 (JSON)"
#     )
#     parser.add_argument(
#         "--model_name", 
#         type=str, 
#         default="Llama 3.1", 
#         help="用于输出的模型名称"
#     )

#     args = parser.parse_args()

#     analyze_llama31_reasoning_scores(
#         args.results_file,
#         args.test_data_file,
#         args.output_file,
#         args.model_name
#     )














# import json
# import re
# import pandas as pd
# import os
# from tqdm import tqdm
# from verl.utils.reward_score.time_reasoning import compute_score
# import argparse

# # 正则表达式用于提取 Llama 3.1 的思考内容
# # 匹配 <|im_start|>think> 到 <|im_end|> 或 </think> 的内容
# think_pattern = re.compile(r"<\|im_start\|>think>(.*?)(?:<\|im_end\>|</think>)", re.DOTALL)
# # 备用模式，匹配 <think> 标签内容
# alt_think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# def analyze_llama31_reasoning_scores(
#     results_file,
#     test_data_file,
#     output_file=None,
#     model_name="Llama 3.1"
# ):
#     """
#     分析Llama 3.1模型在time reasoning任务上的表现

#     参数:
#         results_file: 模型的结果文件路径 (JSONL)
#         test_data_file: 原始测试数据文件路径 (Parquet)
#         output_file: 可选的结果保存路径 (JSON)
#         model_name: 用于输出的模型名称
#     """
#     print(f"加载测试数据集: {test_data_file}")
#     try:
#         test_df = pd.read_parquet(test_data_file)
#     except Exception as e:
#         print(f"错误: 无法加载测试数据文件 {test_data_file}. Error: {e}")
#         return None

#     test_samples = {}
#     task_type_counts = {}
#     test_records = test_df.to_dict('records')
#     for idx, row in enumerate(test_records):
#         extra_info = row.get('extra_info')
#         if not isinstance(extra_info, dict):
#              print(f"警告: 第 {idx} 行缺少 'extra_info' 字典或格式错误，跳过。")
#              continue
#         task_type = extra_info.get('task', 'unknown')
#         task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
#         key = f"{task_type}_{extra_info.get('index', idx)}"
#         test_samples[key] = row

#     print(f"读取完成，共加载 {len(test_samples)} 个有效测试样本")
#     if not test_samples:
#         print("错误: 未能从测试数据中加载任何有效样本。")
#         return None

#     print("测试集任务类型分布:")
#     for task, count in sorted(task_type_counts.items()):
#         print(f"  - {task}: {count}条")

#     print(f"\n加载 {model_name} 回答: {results_file}")
#     model_results = []
#     try:
#         with open(results_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if line.strip():
#                     try:
#                         data = json.loads(line)
#                         model_results.append(data)
#                     except json.JSONDecodeError as e:
#                         print(f"警告: 无法解析行: {line[:50]}... Error: {e}")
#     except FileNotFoundError:
#          print(f"错误: 结果文件未找到 {results_file}")
#          return None
#     except Exception as e:
#          print(f"错误: 读取结果文件时出错 {results_file}. Error: {e}")
#          return None

#     print(f"读取完成，共加载 {len(model_results)} 个 {model_name} 回答")

#     scores = {
#         'all': {'total': [], 'accuracy': []},
#         'time_inferring': {'total': [], 'accuracy': []},
#         'time_difference': {'total': [], 'accuracy': []},
#         'time_ordering': {'total': [], 'accuracy': []},
#         'time_completion': {'total': [], 'accuracy': []}
#     }

#     processed_count = 0
#     skipped_count = 0
#     content_filter_count = 0
#     no_id_count = 0
#     error_count = 0
#     not_found_ids = set()

#     print("\n开始计算得分...")
#     for result in tqdm(model_results):
#         custom_id = result.get('custom_id')
#         if not custom_id:
#             no_id_count += 1
#             skipped_count += 1
#             continue

#         # 从Llama 3.1输出中提取回答内容
#         response = result.get('response', '')
#         raw_content = response.strip() if isinstance(response, str) else ''

#         # 如果response不是字符串，尝试获取task_type和其他信息
#         if not raw_content:
#             error_count += 1
#             skipped_count += 1
#             continue

#         # 提取思考过程
#         reasoning = ''
#         content = raw_content
        
#         # 尝试匹配Llama 3.1的思考标记格式
#         think_match = think_pattern.search(raw_content)
#         if think_match:
#             reasoning = think_match.group(1).strip()
#             # 从content中移除思考部分
#             content = think_pattern.sub('', raw_content).strip()
#         else:
#             # 尝试备用格式
#             alt_match = alt_think_pattern.search(raw_content)
#             if alt_match:
#                 reasoning = alt_match.group(1).strip()
#                 content = alt_think_pattern.sub('', raw_content).strip()

#         # 获取任务类型并查找对应的测试样本
#         match = re.match(r'([a-zA-Z_]+)_\d+', custom_id)
#         task_type = match.group(1) if match else 'unknown'
        
#         test_sample = test_samples.get(custom_id)
#         if test_sample is None:
#             not_found_ids.add(custom_id)
#             skipped_count += 1
#             continue

#         # 获取地面真相数据
#         ground_truth = test_sample.get('ground_truth', {})
#         if not isinstance(ground_truth, dict):
#             print(f"警告 ({custom_id}): 测试样本缺少有效的 'ground_truth' 数据。")
#             skipped_count += 1
#             continue

#         # 构建评分输入
#         solution_str = f"<think>{reasoning or ''}</think>\n{content or ''}"
#         current_ground_truth = ground_truth.copy()
#         current_ground_truth['task'] = task_type

#         try:
#             score_results = compute_score(solution_str, current_ground_truth)
#             if score_results is None or not isinstance(score_results, (list, tuple)) or len(score_results) < 2:
#                  print(f"警告 ({custom_id}): compute_score 返回无效结果: {score_results}")
#                  skipped_count += 1
#                  continue

#             total_score, accuracy_score = score_results[0], score_results[1]

#             scores['all']['total'].append(total_score)
#             scores['all']['accuracy'].append(accuracy_score)

#             if task_type not in scores:
#                  scores[task_type] = {'total': [], 'accuracy': []}
#                  print(f"信息: 发现新的任务类型 '{task_type}' 并开始记录分数。")

#             scores[task_type]['total'].append(total_score)
#             scores[task_type]['accuracy'].append(accuracy_score)

#             processed_count += 1
#         except Exception as e:
#             print(f"评分错误 ({custom_id}): {str(e)}")
#             skipped_count += 1

#     # 计算并输出结果
#     results = {}
#     all_tasks = set(scores.keys())
#     for task in sorted(list(all_tasks)):
#         task_scores = scores[task]
#         if task_scores.get('total'):
#             avg_total = sum(task_scores['total']) / len(task_scores['total'])
#             avg_accuracy = sum(task_scores['accuracy']) / len(task_scores['accuracy'])
#             results[task] = {
#                 'avg_total_score': avg_total,
#                 'avg_accuracy_score': avg_accuracy,
#                 'count': len(task_scores['total'])
#             }

#     print(f"\n===== {model_name} 在Time Reasoning任务上的表现 =====")
#     print(f"处理完成 (成功评分): {processed_count}条")
#     print(f"跳过 (无法评分): {skipped_count}条")
#     print(f"  - 其中内容过滤: {content_filter_count}条")
#     print(f"  - 其中缺少 custom_id: {no_id_count}条")
#     print(f"  - 其中解析错误: {error_count}条")
#     if not_found_ids:
#         print(f"  - 其中未找到对应测试样本的ID数量: {len(not_found_ids)}")
#         if len(not_found_ids) < 20:
#             print(f"    未找到的ID示例: {', '.join(list(not_found_ids)[:20])}")

#     if 'all' in results:
#         print("\n总体平均得分:")
#         print(f"  样本数量: {results['all']['count']}")
#         print(f"  平均总分 (Total Score): {results['all']['avg_total_score']:.4f}")
#         print(f"  平均准确度 (Accuracy Score): {results['all']['avg_accuracy_score']:.4f}")
#     else:
#         print("\n未能计算总体平均得分。")

#     print("\n按任务类型的平均得分:")
#     task_order = ['time_inferring', 'time_difference', 'time_ordering', 'time_completion']
#     displayed_tasks = set()
#     for task in task_order:
#         if task in results and task != 'all':
#             print(f"\n{task}:")
#             print(f"  样本数量: {results[task]['count']}")
#             print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
#             print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")
#             displayed_tasks.add(task)

#     other_tasks = sorted(list(set(results.keys()) - displayed_tasks - {'all'}))
#     if other_tasks:
#          print("\n其他任务类型:")
#          for task in other_tasks:
#              print(f"\n{task}:")
#              print(f"  样本数量: {results[task]['count']}")
#              print(f"  平均总分 (Total Score): {results[task]['avg_total_score']:.4f}")
#              print(f"  平均准确度 (Accuracy Score): {results[task]['avg_accuracy_score']:.4f}")

#     if output_file:
#         try:
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(results, f, indent=2, ensure_ascii=False)
#             print(f"\n结果已保存到: {output_file}")
#         except Exception as e:
#             print(f"错误: 无法保存结果到 {output_file}. Error: {e}")

#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="分析Llama 3.1模型在Time Reasoning任务上的表现")
#     parser.add_argument("--results_file", type=str, 
#                       default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/time_reasoning_llama31_results.jsonl", 
#                       help="模型结果文件路径 (JSONL)")
#     parser.add_argument("--test_data_file", type=str, 
#                       default="/data/zliu331/temporal_reasoning/TinyZero/datasets/test_time_reasoning_combined.parquet", 
#                       help="原始测试数据文件路径 (Parquet)")
#     parser.add_argument("--output_file", type=str, 
#                       default="/data/zliu331/temporal_reasoning/TinyZero/preliminary/llama31_results/llama31_scores.json", 
#                       help="可选的结果保存路径 (JSON)")
#     parser.add_argument("--model_name", type=str, default="Llama 3.1", help="用于输出的模型名称")

#     args = parser.parse_args()

#     analyze_llama31_reasoning_scores(
#         args.results_file,
#         args.test_data_file,
#         args.output_file,
#         args.model_name
#     )







# import json
# import re
# import sys
# import math as builtin_math
# from collections import defaultdict

# # 从time_reasoning.py复制必要的函数
# def extract_answer_format(solution_str):
#     """从解答文本中提取出 <answer>...</answer> 标签中间的内容"""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if match:
#         return match.group(1).strip()
#     return None

# def is_valid_date_format(date_str):
#     """验证日期格式是否符合 YYYY-MM"""
#     pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
#     return re.match(pattern, date_str) is not None

# def date_prediction_reward(prediction, target, alpha=0.05):
#     """根据预测日期与真实日期之间的月数差距计算奖励"""
#     try:
#         pred_year, pred_month = map(int, prediction.split("-"))
#         target_year, target_month = map(int, target.split("-"))
#     except Exception:
#         return 0.0
    
#     diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
#     reward = builtin_math.exp(-alpha * diff_in_months)
#     return reward

# def format_reward_single(solution_str):
#     """检查solution_str是否严格按照要求包含标签"""
#     pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
#     return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

# def tag_count_reward_single(solution_str):
#     """根据solution_str中标签的出现次数计算分值"""
#     count = 0.0
#     if solution_str.count("</think>") == 1:
#         count += 0.33
#     if solution_str.count("<answer>") == 1:
#         count += 0.33
#     if solution_str.count("</answer>") == 1:
#         count += 0.33
#     return count

# def compute_length_repetition_penalty(solution_str):
#     """计算回答过长和内容重复的惩罚值"""
#     # 初始无惩罚
#     length_penalty = 0.0
#     repetition_penalty = 0.0
    
#     # 分词处理
#     tokens = solution_str.split()
#     token_count = len(tokens)

#     # 长度惩罚
#     # if token_count > 900:
#     #     excess_ratio = min(1.0, (token_count - 900) / 124)
#     #     length_penalty = excess_ratio * 0.3

#     if token_count > 400:
#         tokens = tokens[:400]  # 截断到400个词
    
#     # 单词级别的连续重复检测
#     if token_count > 50:
#         max_repeat_count = 1
#         current_word = None
#         current_count = 0
        
#         for word in tokens:
#             if word == current_word:
#                 current_count += 1
#                 max_repeat_count = max(max_repeat_count, current_count)
#             else:
#                 current_word = word
#                 current_count = 1
        
#         if max_repeat_count >= 5:
#             repetition_penalty = 0.1 * min(5, max_repeat_count - 4)
    
#     # 短语级别的连续重复检测
#     if token_count > 100:
#         for window_size in [3,5,7,9]:
#             for i in range(len(tokens) - window_size * 3):
#                 phrase = ' '.join(tokens[i:i+window_size])
#                 next_text = ' '.join(tokens[i+window_size:i+window_size*4])
#                 repeat_count = next_text.count(phrase)
                
#                 if repeat_count >= 2:
#                     repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)
    
#     # 全局n-gram多样性检测
#     if token_count > 200:
#         chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]
#         if chunks:
#             unique_chunks = set(chunks)
#             unique_ratio = len(unique_chunks) / len(chunks)
            
#             if unique_ratio < 0.5:
#                 repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)
    
#     # 结合长度惩罚和重复惩罚
#     total_penalty = max(length_penalty, repetition_penalty)
    
#     return total_penalty

# # 时间差异计算任务评分函数
# def extract_time_diff_answer(solution_str):
#     """提取时间差异答案"""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if not match:
#         return None, None, None
    
#     answer_text = match.group(1).strip()
    
#     date_diff_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2})\. Month difference: (\d{1,3})\.'
#     match = re.search(date_diff_pattern, answer_text)
    
#     if match:
#         event1_date = match.group(1)
#         event2_date = match.group(2)
#         month_diff = int(match.group(3))
#         return event1_date, event2_date, month_diff
    
#     return None, None, None

# def month_diff_reward(pred_diff, true_diff, alpha=0.06):
#     """计算月份差值预测的奖励"""
#     diff = abs(pred_diff - true_diff)
#     reward = builtin_math.exp(-alpha * diff)
#     return reward

# def compute_inconsistency_penalty(date1, date2, claimed_diff):
#     """计算声称的月份差与实际月份差之间的不一致性惩罚"""
#     try:
#         year1, month1 = map(int, date1.split("-"))
#         year2, month2 = map(int, date2.split("-"))
        
#         actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
#         error = abs(actual_diff - claimed_diff)
        
#         # 衰减惩罚
#         consistency_score = builtin_math.exp(-0.1 * error)
        
#         return consistency_score
#     except Exception:
#         return 0.1  # 出现异常，严重惩罚

# def compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """计算时间差异任务的总评分"""
#     event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
#     # "No event"惩罚
#     no_event_penalty = 0
#     if event1_date and event2_date:
#         if "no event" in event1_date.lower() or "no event" in event2_date.lower():
#             no_event_penalty = 0.1
#     else:
#         no_event_penalty = 0.2
    
#     # 格式奖励部分
#     format_bonus = 0.0
#     if (event1_date and event2_date and month_diff is not None and 
#         is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
#         format_bonus = bonus
    
#     # 准确性计算部分
#     event1_accuracy = 0.0
#     event2_accuracy = 0.0
#     diff_accuracy = 0.0
#     consistency_penalty = 1.0
    
#     if format_bonus > 0:  # 只有格式正确才计算准确性
#         true_event1_date = ground_truth.get("event1_pub_date")
#         true_event2_date = ground_truth.get("event2_pub_date")
#         true_month_diff = ground_truth.get("month_difference")
        
#         if true_event1_date and true_event2_date and true_month_diff is not None:
#             # 计算两个日期预测的准确性
#             if is_valid_date_format(true_event1_date):
#                 event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.25
            
#             if is_valid_date_format(true_event2_date):
#                 event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.25
            
#             # 计算月份差值预测的准确性
#             if month_diff >= 25:
#                 diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.05) * 0.5
#             else:
#                 diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.1) * 0.5
            
#             # 检查一致性并应用惩罚
#             consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
#             # 应用惩罚
#             accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
#         else:
#             accuracy_score = 0.0
#     else:
#         accuracy_score = 0.0
    
#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 应用长度和重复惩罚
#     length_repetition_penalty = compute_length_repetition_penalty(solution_str)

#     # 总分计算
#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             consistency_penalty, "time_difference")

# # 时间排序任务评分函数
# def extract_time_order_answer(solution_str):
#     """提取时间顺序答案"""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if not match:
#         return None, None, None, None
    
#     answer_text = match.group(1).strip()
    
#     date_order_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2}), Event 3: (\d{4}-\d{2})\. Event order: (\d-\d-\d)\.'
#     match = re.search(date_order_pattern, answer_text)
    
#     if match:
#         event1_date = match.group(1)
#         event2_date = match.group(2)
#         event3_date = match.group(3)
#         event_order = match.group(4)
#         return event1_date, event2_date, event3_date, event_order
    
#     return None, None, None, None

# def is_valid_order_format(order_str):
#     """验证order_str是否为以'-'分隔的包含1,2,3三个不重复数字的字符串"""
#     if not re.match(r'\d-\d-\d', order_str):
#         return False
    
#     digits = [int(d) for d in order_str.split('-')]
#     return sorted(digits) == [1, 2, 3]

# def compute_order_accuracy(predicted_order, true_order):
#     """计算顺序预测的准确性"""
#     if predicted_order == true_order:
#         return 0.999
    
#     pred_parts = predicted_order.split('-')
#     true_parts = true_order.split('-')
    
#     pairs_correct = 0
    
#     # 检查事件1和事件2的相对顺序
#     pred_1_before_2 = pred_parts.index('1') < pred_parts.index('2')
#     true_1_before_2 = true_parts.index('1') < true_parts.index('2')
#     if pred_1_before_2 == true_1_before_2:
#         pairs_correct += 1
    
#     # 检查事件1和事件3的相对顺序
#     pred_1_before_3 = pred_parts.index('1') < pred_parts.index('3')
#     true_1_before_3 = true_parts.index('1') < true_parts.index('3')
#     if pred_1_before_3 == true_1_before_3:
#         pairs_correct += 1
    
#     # 检查事件2和事件3的相对顺序
#     pred_2_before_3 = pred_parts.index('2') < pred_parts.index('3')
#     true_2_before_3 = true_parts.index('2') < true_parts.index('3')
#     if pred_2_before_3 == true_2_before_3:
#         pairs_correct += 1
    
#     return pairs_correct * 0.333

# def compute_order_consistency_penalty(event1_date, event2_date, event3_date, claimed_order):
#     """计算声称的事件排序与实际日期排序之间的不一致性惩罚"""
#     try:
#         # 解析三个日期
#         year1, month1 = map(int, event1_date.split("-"))
#         year2, month2 = map(int, event2_date.split("-"))
#         year3, month3 = map(int, event3_date.split("-"))
        
#         # 计算每个事件的总月数，用于比较先后顺序
#         event1_months = year1 * 12 + month1
#         event2_months = year2 * 12 + month2
#         event3_months = year3 * 12 + month3
        
#         # 根据日期计算实际顺序
#         events_by_time = [(1, event1_months), (2, event2_months), (3, event3_months)]
#         events_by_time.sort(key=lambda x: x[1])
#         actual_order = '-'.join(str(event[0]) for event in events_by_time)
        
#         # 比较声称的顺序与实际顺序
#         if claimed_order == actual_order:
#             return 1.0  # 完全一致，不惩罚
        
#         # 计算声称顺序与实际顺序的不一致程度
#         similarity = compute_order_accuracy(claimed_order, actual_order)
        
#         # 根据相似度设置惩罚系数
#         if similarity >= 0.666:  # 至少2组正确
#             return 0.7
#         elif similarity >= 0.333:  # 至少1组正确
#             return 0.4
#         else:  # 全部错误
#             return 0.2
#     except Exception:
#         return 0.1  # 出现异常，严重惩罚

# def compute_date_diversity_penalty(event1_date, event2_date, event3_date, event_order):
#     """检查日期是否具有多样性，以及事件顺序是否不是简单的默认顺序1-2-3"""
#     try:
#         # 解析日期
#         year1, month1 = map(int, event1_date.split("-"))
#         year2, month2 = map(int, event2_date.split("-"))
#         year3, month3 = map(int, event3_date.split("-"))
        
#         # 将日期转换为月份总数
#         total_months1 = year1 * 12 + month1
#         total_months2 = year2 * 12 + month2
#         total_months3 = year3 * 12 + month3
        
#         # 检查日期是否全部相同
#         if total_months1 == total_months2 == total_months3:
#             return 0.2  # 严重惩罚 - 所有日期都相同
        
#         # 检查是否是简单的连续月份模式
#         is_sequential_0 = False
#         is_sequential_1 = False
        
#         # 检查是否是每月递增模式
#         if (total_months2 == total_months1 + 1 and total_months3 == total_months2 + 1):
#             is_sequential_0 = True
#         # 或每月递减模式
#         elif (total_months2 == total_months1 - 1 and total_months3 == total_months2 - 1):
#             is_sequential_1 = True
        
#         # 检查事件顺序是否只是默认的1-2-3
#         is_default_order_0 = (event_order == "1-2-3")
#         is_default_order_1 = (event_order == "3-2-1")
        
#         # 组合惩罚
#         if is_sequential_0 and is_default_order_0:
#             return 0.2  # 中度惩罚 - 连续月份和默认顺序
#         elif is_sequential_1 and is_default_order_1:
#             return 0.2  # 轻度惩罚 - 只是连续月份
        
#         # 日期有多样性，且顺序不是默认的
#         return 1.0  # 不惩罚
        
#     except Exception:
#         return 0.5

# def compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """计算时间顺序任务的总评分"""
#     event1_date, event2_date, event3_date, event_order = extract_time_order_answer(solution_str)
    
#     # "No event"惩罚
#     no_event_penalty = 0
#     if event1_date and event2_date and event3_date:
#         if ("no event" in event1_date.lower() or "no event" in event2_date.lower() or 
#             "no event" in event3_date.lower()):
#             no_event_penalty = 0.1
#     else:
#         no_event_penalty = 0.2
    
#     # 格式奖励部分
#     format_bonus = 0.0
#     if (event1_date and event2_date and event3_date and event_order and
#         is_valid_date_format(event1_date) and is_valid_date_format(event2_date) and 
#         is_valid_date_format(event3_date) and is_valid_order_format(event_order)):
#         format_bonus = bonus
    
#     # 准确性计算部分
#     event1_accuracy = 0.0
#     event2_accuracy = 0.0
#     event3_accuracy = 0.0
#     order_accuracy = 0.0
#     consistency_penalty = 1.0
#     combined_penalty = 1.0
    
#     if format_bonus > 0:  # 只有格式正确才计算准确性
#         true_event1_date = ground_truth.get("event1_pub_date")
#         true_event2_date = ground_truth.get("event2_pub_date")
#         true_event3_date = ground_truth.get("event3_pub_date")
#         true_event_order = ground_truth.get("event_order")
        
#         if (true_event1_date and true_event2_date and true_event3_date and true_event_order and
#             is_valid_date_format(true_event1_date) and is_valid_date_format(true_event2_date) and 
#             is_valid_date_format(true_event3_date)):
            
#             # 计算三个日期预测的准确性
#             event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.2
#             event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.2
#             event3_accuracy = date_prediction_reward(event3_date, true_event3_date, alpha) * 0.2
            
#             # 计算顺序预测的准确性
#             order_accuracy = compute_order_accuracy(event_order, true_event_order) * 0.4
            
#             # 添加一致性惩罚
#             consistency_penalty = compute_order_consistency_penalty(
#                 event1_date, event2_date, event3_date, event_order)
            
#             # 添加日期多样性惩罚
#             diversity_penalty = compute_date_diversity_penalty(
#                 event1_date, event2_date, event3_date, event_order)
            
#             # 组合两种惩罚
#             combined_penalty = consistency_penalty * diversity_penalty
            
#             # 应用惩罚计算总准确性分数
#             accuracy_score = (event1_accuracy + event2_accuracy + event3_accuracy + order_accuracy) * combined_penalty
#         else:
#             accuracy_score = 0.0
#     else:
#         accuracy_score = 0.0
    
#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 应用长度和重复惩罚
#     length_repetition_penalty = compute_length_repetition_penalty(solution_str)

#     # 总分计算
#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             combined_penalty, "time_ordering")

# # 时间补全任务评分函数
# def extract_time_completion_answer(solution_str):
#     """提取时间补全答案"""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.search(answer_pattern, solution_str)
#     if not match:
#         return None, None
    
#     answer_text = match.group(1).strip()
    
#     completion_pattern = r'Event: (\d{4}-\d{2})\. Missing entity: (.+?)\.'
#     match = re.search(completion_pattern, answer_text)
    
#     if match:
#         event_date = match.group(1)
#         missing_entity = match.group(2).strip()
#         return event_date, missing_entity
    
#     return None, None

# def is_valid_year(year_str):
#     """验证输入是否是有效的4位数年份"""
#     try:
#         year = int(year_str)
#         return 1900 <= year <= 2100
#     except ValueError:
#         return False

# def is_valid_month(month_str):
#     """验证输入是否是有效的月份名称"""
#     month_str_lower = month_str.lower()
    
#     months_and_variants = {
#         "January": ["january", "jan", "jan."],
#         "February": ["february", "feb", "feb."],
#         "March": ["march", "mar", "mar."],
#         "April": ["april", "apr", "apr."],
#         "May": ["may"],
#         "June": ["june", "jun", "jun."],
#         "July": ["july", "jul", "jul."],
#         "August": ["august", "aug", "aug."],
#         "September": ["september", "sept", "sept.", "sep", "sep."],
#         "October": ["october", "oct", "oct."],
#         "November": ["november", "nov", "nov."],
#         "December": ["december", "dec", "dec."]
#     }
    
#     # 检查是否匹配任何月份或其变体
#     for standard_month, variants in months_and_variants.items():
#         if month_str_lower in variants or month_str_lower == standard_month.lower():
#             return True
    
#     # 如果是数字，检查是否在1-12范围内
#     try:
#         month_num = int(month_str)
#         return 1 <= month_num <= 12
#     except ValueError:
#         pass
    
#     return False

# def entity_match_score(predicted_entity, true_entity, entity_type, alpha=0.3):
#     """计算缺失实体预测的匹配得分"""
#     if predicted_entity is None or true_entity is None:
#         return 0.0
    
#     if entity_type == "year":
#         # 对于年份，使用数值差距的指数衰减
#         try:
#             pred_year = int(predicted_entity)
#             true_year = int(true_entity)
#             diff = abs(pred_year - true_year)
#             return builtin_math.exp(-alpha * diff)
#         except ValueError:
#             return 0.0
    
#     elif entity_type == "month":
#         # 对于月份，使用更宽松的匹配标准，包括所有月份变体
#         months_and_variants = {
#             "January": ["january", "jan", "jan."],
#             "February": ["february", "feb", "feb."],
#             "March": ["march", "mar", "mar."],
#             "April": ["april", "apr", "apr."],
#             "May": ["may"],
#             "June": ["june", "jun", "jun."],
#             "July": ["july", "jul", "jul."],
#             "August": ["august", "aug", "aug."],
#             "September": ["september", "sept", "sept.", "sep", "sep."],
#             "October": ["october", "oct", "oct."],
#             "November": ["november", "nov", "nov."],
#             "December": ["december", "dec", "dec."]
#         }
        
#         # 标准化预测和真实月份为小写
#         pred_lower = predicted_entity.lower()
#         true_lower = true_entity.lower()
        
#         # 尝试解析数字月份
#         try:
#             if pred_lower.isdigit():
#                 pred_month_num = int(pred_lower)
#                 if 1 <= pred_month_num <= 12:
#                     month_numbers = {1: "January", 2: "February", 3: "March", 4: "April", 
#                                     5: "May", 6: "June", 7: "July", 8: "August", 
#                                     9: "September", 10: "October", 11: "November", 12: "December"}
#                     pred_lower = month_numbers[pred_month_num].lower()
            
#             if true_lower.isdigit():
#                 true_month_num = int(true_lower)
#                 if 1 <= true_month_num <= 12:
#                     month_numbers = {1: "January", 2: "February", 3: "March", 4: "April", 
#                                     5: "May", 6: "June", 7: "July", 8: "August", 
#                                     9: "September", 10: "October", 11: "November", 12: "December"}
#                     true_lower = month_numbers[true_month_num].lower()
#         except (ValueError, KeyError):
#             pass
        
#         # 对每个标准月份，检查预测和真实值是否匹配其任一变体
#         pred_standard_month = None
#         true_standard_month = None
        
#         for standard_month, variants in months_and_variants.items():
#             if pred_lower in variants or pred_lower == standard_month.lower():
#                 pred_standard_month = standard_month
            
#             if true_lower in variants or true_lower == standard_month.lower():
#                 true_standard_month = standard_month
        
#         # 如果预测和真实月份归于同一标准月份，则匹配
#         if pred_standard_month and true_standard_month and pred_standard_month == true_standard_month:
#             return 1.0
        
#         # 如果没有精确匹配，尝试计算月份之间的距离
#         month_order = list(months_and_variants.keys())
        
#         if pred_standard_month and true_standard_month:
#             try:
#                 pred_idx = month_order.index(pred_standard_month)
#                 true_idx = month_order.index(true_standard_month)
                
#                 # 计算环形距离 - 取直接距离和绕一圈距离中的最小值
#                 direct_diff = abs(pred_idx - true_idx)
#                 circular_diff = 12 - direct_diff  # 12个月的循环
#                 month_diff = min(direct_diff, circular_diff)
                
#                 # 使用月份差距的指数衰减
#                 return builtin_math.exp(-alpha * month_diff)
#             except (ValueError, IndexError):
#                 pass
    
#     return 0.0

# def compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """计算时间补全任务的总评分"""
#     event_date, missing_entity = extract_time_completion_answer(solution_str)
    
#     # "No event"惩罚
#     no_event_penalty = 0
#     if event_date:
#         if "no event" in event_date.lower():
#             no_event_penalty = 0.1
#     else:
#         no_event_penalty = 0.2
    
#     # 获取掩码类型
#     mask_type = ground_truth.get("mask_type", "")
    
#     # 格式奖励部分
#     format_bonus = 0.0
#     if event_date and missing_entity and is_valid_date_format(event_date):
#         # 根据mask_type验证missing_entity的格式
#         if mask_type == "year" and is_valid_year(missing_entity):
#             format_bonus = bonus
#         elif mask_type == "month" and is_valid_month(missing_entity):
#             format_bonus = bonus
#         elif not mask_type:  # 如果没有mask_type，宽松处理
#             format_bonus = bonus
    
#     # 准确性计算部分
#     date_accuracy = 0.0
#     entity_accuracy = 0.0
    
#     if format_bonus > 0:  # 只有格式正确才计算准确性
#         true_event_date = ground_truth.get("event_pub_date")
#         masked_entity = ground_truth.get("masked_entity", "")
        
#         if true_event_date and mask_type and masked_entity and is_valid_date_format(true_event_date):
#             # 计算日期预测的准确性 (占50%)
#             date_accuracy = date_prediction_reward(event_date, true_event_date, alpha) * 0.5
            
#             # 计算缺失实体预测的准确性 (占50%)
#             entity_accuracy = entity_match_score(missing_entity, masked_entity, mask_type, alpha*3) * 0.5
            
#             # 总的准确性分数
#             accuracy_score = date_accuracy + entity_accuracy
#         else:
#             accuracy_score = 0.0
#     else:
#         accuracy_score = 0.0
    
#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 应用长度和重复惩罚
#     length_repetition_penalty = compute_length_repetition_penalty(solution_str)

#     # 总分计算
#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             1.0, "time_completion")

# # 时间推断任务评分函数
# def compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.05, tag_bonus_each=0.025):
#     """计算时间推断任务的总评分"""
#     answer = extract_answer_format(solution_str)

#     # "No event"惩罚
#     no_event_penalty = 0
#     if answer:
#         if "no event" in answer.lower() or "none" in answer.lower():
#             no_event_penalty = 0.1  
#     else:
#         no_event_penalty = 0.2

#     # 如果提取到了答案且符合 "YYYY-MM" 格式，则先获得格式奖励
#     format_bonus, pred_reward = 0.0, 0.0
#     if answer and is_valid_date_format(answer):
#         format_bonus = bonus
#         true_pub_date = ground_truth.get("event_pub_date")
#         # 确保 ground_truth 中的真实日期也符合 "YYYY-MM" 格式，否则不计算预测奖励
#         if true_pub_date and is_valid_date_format(true_pub_date):
#             pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
    
#     accuracy_score = pred_reward

#     # Tag 奖励部分
#     tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
#     tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
#     # 应用长度和重复惩罚
#     length_repetition_penalty = compute_length_repetition_penalty(solution_str)

#     total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
#     return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
#             1.0, "time_inferring")

# # 统一评分入口函数
# def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
#     """统一的评分入口函数，根据任务类型自动选择合适的评分函数"""
#     # 从ground_truth的extra_info里找任务类型
#     task_type = ""
    
#     # 首先尝试从非张量批次的extra_info中获取task类型
#     if isinstance(ground_truth, dict) and "task" in ground_truth:
#         task_type = ground_truth.get("task", "")
    
#     # 如果没有找到，尝试从ground_truth本身的特征来判断任务类型
#     if not task_type:
#         if "event1_pub_date" in ground_truth and "event2_pub_date" in ground_truth:
#             if "event3_pub_date" in ground_truth:
#                 task_type = "time_ordering"
#             else:
#                 task_type = "time_difference"
#         elif "mask_type" in ground_truth and "masked_entity" in ground_truth:
#             task_type = "time_completion"
#         elif "event_pub_date" in ground_truth:
#             task_type = "time_inferring"
    
#     # 根据任务类型选择合适的评分函数
#     if task_type == "time_difference":
#         return compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
#     elif task_type == "time_ordering":
#         return compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
#     elif task_type == "time_completion":
#         return compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
#     elif task_type == "time_inferring":
#         return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
#     else:
#         # 默认使用时间推断评分
#         return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)

# def calculate_scores_by_task_type(results_file):
#     """计算每种任务类型的平均分数"""
#     # 用于存储每种任务类型的分数
#     scores_by_task = defaultdict(list)
#     accuracy_by_task = defaultdict(list)
#     counts_by_task = defaultdict(int)
#     all_scores = []
#     all_accuracy = []
    
#     # 读取结果文件
#     with open(results_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 result = json.loads(line)
#                 response = result.get('response', '')
                
#                 # 获取任务类型
#                 task_type = result.get('task_type', '')
#                 if not task_type and "extra_info" in result and "task" in result["extra_info"]:
#                     task_type = result["extra_info"]["task"]
                
#                 # 获取ground_truth
#                 ground_truth = result.get('ground_truth', {})
                
#                 if not ground_truth:
#                     continue
                
#                 # 计算分数
#                 score_tuple = compute_score(response, ground_truth)
#                 total_score, accuracy_score = score_tuple[0], score_tuple[1]
                
#                 # 按任务类型保存分数
#                 scores_by_task[task_type].append(total_score)
#                 accuracy_by_task[task_type].append(accuracy_score)
#                 counts_by_task[task_type] += 1
#                 all_scores.append(total_score)
#                 all_accuracy.append(accuracy_score)
                
#             except json.JSONDecodeError as e:
#                 print(f"警告: 无法解析JSON行: {e}")
#                 continue
#             except Exception as e:
#                 print(f"处理样本时出错: {e}")
#                 continue
    
#     # 计算每种任务类型的平均分数
#     results = {}
#     for task_type in scores_by_task:
#         avg_total = sum(scores_by_task[task_type]) / len(scores_by_task[task_type]) if scores_by_task[task_type] else 0
#         avg_accuracy = sum(accuracy_by_task[task_type]) / len(accuracy_by_task[task_type]) if accuracy_by_task[task_type] else 0
#         results[task_type] = {
#             "avg_total_score": avg_total,
#             "avg_accuracy_score": avg_accuracy,
#             "count": counts_by_task[task_type]
#         }
    
#     # 添加总体平均分数
#     results["all"] = {
#         "avg_total_score": sum(all_scores) / len(all_scores) if all_scores else 0,
#         "avg_accuracy_score": sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0,
#         "count": len(all_scores)
#     }
    
#     return results

# def main():
#     # 设置结果文件路径
#     results_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen7b_results/time_reasoning_qwen7b_results.jsonl"
    
#     # 计算分数
#     results = calculate_scores_by_task_type(results_file)
    
#     # 输出结果
#     print("\n===== qwen7b 时间推理任务评分结果 =====")
#     print(f"总样本数: {results['all']['count']}")
#     print(f"总体平均 total_score: {results['all']['avg_total_score']:.5f}")
#     print(f"总体平均 accuracy_score: {results['all']['avg_accuracy_score']:.5f}")
    
#     print("\n各任务类型评分:")
#     for task_type in sorted([t for t in results if t != 'all']):
#         print(f"- {task_type}:")
#         print(f"  样本数: {results[task_type]['count']}")
#         print(f"  平均 total_score: {results[task_type]['avg_total_score']:.5f}")
#         print(f"  平均 accuracy_score: {results[task_type]['avg_accuracy_score']:.5f}")
    
#     # 保存结果到json文件
#     output_file = "/data/zliu331/temporal_reasoning/TinyZero/preliminary/qwen7b_results/reasoning_qwen7b_scores.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2)
    
#     print(f"\n详细结果已保存至: {output_file}")

# if __name__ == "__main__":
#     main()