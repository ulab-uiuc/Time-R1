import os
import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def analyze_news_desk_distribution(data_folder, year_range=None):
    """
    分析nyt_years文件夹中每个年份的新闻类别分布情况
    
    参数:
        data_folder: 包含年份JSONL文件的文件夹路径
        year_range: 要分析的年份范围，如(2016, 2025)
        
    返回:
        yearly_stats: 每年的类别统计字典
        overall_stats: 所有年份的汇总统计字典
    """
    # 需要统计的新闻类别
    allowed_desks = {
        "Politics", "National", "Washington", "U.S.",
        "Business", "SundayBusiness", "RealEstate",
        "Foreign", "World", "Metro", "Science", "Health", "Climate",
        "Opinion", "OpEd"
    }
    
    # 存储每年的统计结果
    yearly_stats = {}
    
    # 存储所有年份的总计数
    overall_counter = Counter()
    overall_total = 0
    
    # # 获取文件夹中的所有JSONL文件
    # all_files = [f for f in os.listdir(data_folder) if f.endswith('.jsonl')]
    
    # # 如果指定了年份范围，则过滤文件
    # if year_range:
    #     start_year, end_year = year_range
    #     all_files = [f for f in all_files if start_year <= int(f.split('.')[0]) <= end_year]

    all_files = []
    if year_range:
        start_year, end_year = year_range
        for year in range(start_year, end_year + 1):
            filename = f"{year}.jsonl"
            file_path = os.path.join(data_folder, filename)
            if not os.path.isfile(file_path):
                print(f"警告: 找不到文件 {filename}，请检查数据文件夹")
                continue
            all_files.append(filename)
    
    # 按年份排序
    all_files.sort()
    
    print(f"分析{len(all_files)}个年份的数据...")
    
    # 遍历每个年份文件
    for filename in tqdm(all_files):
        year = int(filename.split('.')[0])
        file_path = os.path.join(data_folder, filename)
        
        # 本年度的类别计数器
        year_counter = Counter()
        
        # 读取JSONL文件
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                try:
                    article = json.loads(line.strip())
                    news_desk = article.get('news_desk', '')
                    
                    # 只统计允许的新闻类别
                    if news_desk in allowed_desks:
                        year_counter[news_desk] += 1
                        overall_counter[news_desk] += 1
                    
                    line_count += 1
                except json.JSONDecodeError:
                    continue
        
        # 计算年度总数
        year_total = sum(year_counter.values())
        overall_total += year_total
        
        # 计算每个类别的百分比
        year_percentages = {desk: (count / year_total * 100) if year_total > 0 else 0 
                           for desk, count in year_counter.items()}
        
        # 存储年度统计结果
        yearly_stats[year] = {
            'counts': dict(year_counter),
            'percentages': year_percentages,
            'total': year_total
        }
    
    # 计算所有年份的总体百分比
    overall_percentages = {desk: (count / overall_total * 100) if overall_total > 0 else 0 
                          for desk, count in overall_counter.items()}
    
    # 存储总体统计结果
    overall_stats = {
        'counts': dict(overall_counter),
        'percentages': overall_percentages,
        'total': overall_total
    }
    
    return yearly_stats, overall_stats

def print_distribution_report(yearly_stats, overall_stats):
    """打印分布报告"""
    # 打印每年的统计数据
    print("\n============ 年度新闻类别分布 ============")
    for year, stats in sorted(yearly_stats.items()):
        print(f"\n{year}年 (总文章数: {stats['total']})")
        print("-" * 50)
        print(f"{'新闻类别':<15} {'数量':<10} {'百分比':<10}")
        print("-" * 50)
        
        # 按数量降序排列
        sorted_desks = sorted(stats['counts'].items(), key=lambda x: x[1], reverse=True)
        for desk, count in sorted_desks:
            percentage = stats['percentages'][desk]
            print(f"{desk:<15} {count:<10} {percentage:.2f}%")
    
    # 打印所有年份的统计数据
    print("\n============ 总体新闻类别分布 (2016-2025) ============")
    print(f"总文章数: {overall_stats['total']}")
    print("-" * 50)
    print(f"{'新闻类别':<15} {'数量':<10} {'百分比':<10}")
    print("-" * 50)
    
    # 按数量降序排列
    sorted_desks = sorted(overall_stats['counts'].items(), key=lambda x: x[1], reverse=True)
    for desk, count in sorted_desks:
        percentage = overall_stats['percentages'][desk]
        print(f"{desk:<15} {count:<10} {percentage:.2f}%")

def plot_distributions(yearly_stats, overall_stats, output_folder=None):
    """绘制分布图表"""
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 准备绘图数据
    years = sorted(yearly_stats.keys())
    categories = sorted(overall_stats['counts'].keys())
    
    # 创建按年份的分布数据框
    data = []
    for year in years:
        for category in categories:
            count = yearly_stats[year]['counts'].get(category, 0)
            percentage = yearly_stats[year]['percentages'].get(category, 0)
            data.append({
                'Year': year,
                'Category': category,
                'Count': count,
                'Percentage': percentage
            })
    
    df = pd.DataFrame(data)
    
    # 1. 绘制总体分布饼图
    plt.figure(figsize=(12, 8))
    
    # 按比例降序排列
    sorted_overall = sorted(overall_stats['percentages'].items(), key=lambda x: x[1], reverse=True)
    categories_sorted = [item[0] for item in sorted_overall]
    percentages_sorted = [item[1] for item in sorted_overall]
    
    plt.pie(percentages_sorted, labels=categories_sorted, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('NYT Articles by News Desk Category (2016-2025)')
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'overall_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制年度趋势折线图
    plt.figure(figsize=(15, 10))
    
    # 只选择总体比例最高的6个类别
    top_categories = [item[0] for item in sorted_overall[:6]]
    
    for category in top_categories:
        category_data = df[df['Category'] == category]
        plt.plot(category_data['Year'], category_data['Percentage'], marker='o', linewidth=2, label=category)
    
    plt.title('Trends of Top News Categories (2016-2025)')
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'category_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制热图
    pivot_df = df.pivot(index='Category', columns='Year', values='Percentage')
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    plt.title('News Category Distribution by Year (2016-2025)')
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'category_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def export_to_excel(yearly_stats, overall_stats, output_path):
    """导出统计结果到Excel文件"""
    # 创建一个Excel writer
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    # 1. 创建总体统计表
    overall_df = pd.DataFrame({
        'Category': list(overall_stats['counts'].keys()),
        'Count': list(overall_stats['counts'].values()),
        'Percentage': [overall_stats['percentages'][cat] for cat in overall_stats['counts'].keys()]
    })
    overall_df = overall_df.sort_values('Count', ascending=False)
    overall_df.to_excel(writer, sheet_name='Overall', index=False)
    
    # 2. 创建每年统计表
    for year, stats in yearly_stats.items():
        year_df = pd.DataFrame({
            'Category': list(stats['counts'].keys()),
            'Count': list(stats['counts'].values()),
            'Percentage': [stats['percentages'][cat] for cat in stats['counts'].keys()]
        })
        year_df = year_df.sort_values('Count', ascending=False)
        year_df.to_excel(writer, sheet_name=f'Year_{year}', index=False)
    
    # 3. 创建趋势表 (所有年份的所有类别)
    trend_data = []
    years = sorted(yearly_stats.keys())
    categories = sorted(overall_stats['counts'].keys())
    
    for category in categories:
        row = {'Category': category}
        for year in years:
            row[f'{year}_Count'] = yearly_stats[year]['counts'].get(category, 0)
            row[f'{year}_Pct'] = yearly_stats[year]['percentages'].get(category, 0)
        trend_data.append(row)
    
    trend_df = pd.DataFrame(trend_data)
    trend_df.to_excel(writer, sheet_name='Trends', index=False)
    
    # 保存文件
    writer.close()
    print(f"统计结果已导出到: {output_path}")

def main():
    data_folder = "/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years"
    output_folder = "/data/zliu331/temporal_reasoning/TinyZero/future_news_generation/topic_analysis"
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 分析2016-2025年的数据
    yearly_stats, overall_stats = analyze_news_desk_distribution(data_folder, year_range=(2016, 2025))
    
    # 打印报告
    print_distribution_report(yearly_stats, overall_stats)
    
    # 导出到Excel
    export_to_excel(yearly_stats, overall_stats, 
                   os.path.join(output_folder, 'nyt_category_stats.xlsx'))
    
    # 尝试绘制图表 (如果matplotlib和seaborn可用)
    try:
        plot_distributions(yearly_stats, overall_stats, output_folder)
        print(f"可视化结果已保存到: {output_folder}")
    except NameError:
        print("注意: 缺少matplotlib或seaborn库，无法生成可视化图表")

if __name__ == "__main__":
    main()