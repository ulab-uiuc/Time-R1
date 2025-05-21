import json
import re
from collections import defaultdict
import pandas as pd

def analyze_generation_results(jsonl_file):
    # 计数字典
    monthly_counts = defaultdict(int)
    desk_monthly_counts = defaultdict(lambda: defaultdict(int))
    
    # 解析文件
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # 获取custom_id并提取月份和新闻类别
                custom_id = data.get('custom_id', '')
                if not custom_id or '_' not in custom_id:
                    continue
                    
                month, desk, _ = custom_id.split('_', 2)
                
                # 获取生成内容
                content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 通过查找"ARTICLE N:"模式来计数文章
                articles = re.findall(r'ARTICLE \d+:', content)
                article_count = len(articles)
                
                # 添加到计数中
                if article_count > 0:
                    monthly_counts[month] += article_count
                    desk_monthly_counts[month][desk] += article_count
                
            except Exception as e:
                print(f"处理行时出错: {e}")
    
    return monthly_counts, desk_monthly_counts

# 运行分析
jsonl_file = "v3_generation_4prediction_results.jsonl"
monthly_counts, desk_monthly_counts = analyze_generation_results(jsonl_file)

# 按月份显示总数
print("按月份统计的生成文章总数:")
for month in sorted(monthly_counts.keys()):
    print(f"{month}: {monthly_counts[month]}")
print()

# 创建一个表格来显示各类别在各月份的分布
months = sorted(desk_monthly_counts.keys())
desks = sorted({desk for month_data in desk_monthly_counts.values() for desk in month_data})

# 创建数据框
data = []
for month in months:
    row = {'Month': month}
    for desk in desks:
        row[desk] = desk_monthly_counts[month][desk]
    data.append(row)

df = pd.DataFrame(data)
print("\n按月份和新闻类别统计的文章数量:")
print(df)

# 生成按类别的总计
desk_totals = defaultdict(int)
for month_data in desk_monthly_counts.values():
    for desk, count in month_data.items():
        desk_totals[desk] += count

print("\n按新闻类别统计的总文章数:")
for desk in sorted(desk_totals.keys()):
    print(f"{desk}: {desk_totals[desk]}")