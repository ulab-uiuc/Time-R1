import os
import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def analyze_news_desk_distribution(data_folder, year_range=None):
    """Analyze the distribution of news categories for each year in the nyt_years folder
    
    parameter:
        data_folder: The folder path containing the year JSONL file
        year_range: The range of years to be analyzed, such as (2016, 2025)
        
    return:
        yearly_stats: annual category statistics dictionary
        overall_stats: Summary statistics dictionary for all years"""
    # News categories that require statistics
    allowed_desks = {
        "Politics", "National", "Washington", "U.S.",
        "Business", "SundayBusiness", "RealEstate",
        "Foreign", "World", "Metro", "Science", "Health", "Climate",
        "Opinion", "OpEd"
    }
    
    # Store annual statistics
    yearly_stats = {}
    
    # Store the total count of all years
    overall_counter = Counter()
    overall_total = 0
    
    # # Get all JSONL files in the folder
    # all_files = [f for f in os.listdir(data_folder) if f.endswith('.jsonl')]
    
    # # If the year range is specified, filter the file
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
                print(f"Warning: File {filename} cannot be found, please check the data folder")
                continue
            all_files.append(filename)
    
    # Sort by year
    all_files.sort()
    
    print(f"Analyze data from {len(all_files)} years...")
    
    # traverse each year file
    for filename in tqdm(all_files):
        year = int(filename.split('.')[0])
        file_path = os.path.join(data_folder, filename)
        
        # Category Counter for the Year
        year_counter = Counter()
        
        # Read JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                try:
                    article = json.loads(line.strip())
                    news_desk = article.get('news_desk', '')
                    
                    # Only count the allowed news categories
                    if news_desk in allowed_desks:
                        year_counter[news_desk] += 1
                        overall_counter[news_desk] += 1
                    
                    line_count += 1
                except json.JSONDecodeError:
                    continue
        
        # Calculate the total number of years
        year_total = sum(year_counter.values())
        overall_total += year_total
        
        # Calculate the percentage of each category
        year_percentages = {desk: (count / year_total * 100) if year_total > 0 else 0 
                           for desk, count in year_counter.items()}
        
        # Store annual statistics results
        yearly_stats[year] = {
            'counts': dict(year_counter),
            'percentages': year_percentages,
            'total': year_total
        }
    
    # Calculate the overall percentage of all years
    overall_percentages = {desk: (count / overall_total * 100) if overall_total > 0 else 0 
                          for desk, count in overall_counter.items()}
    
    # Store overall statistical results
    overall_stats = {
        'counts': dict(overall_counter),
        'percentages': overall_percentages,
        'total': overall_total
    }
    
    return yearly_stats, overall_stats

def print_distribution_report(yearly_stats, overall_stats):
    """Print distribution report"""
    # Print annual statistics
    print("\n================== Distribution of annual news categories ============")
    for year, stats in sorted(yearly_stats.items()):
        print(f"\n{year}years (Total number of articles: {stats['total']})")
        print("-" * 50)
        print(f"{'News Category':<15} {'Quantity':<10} {'Percentage':<10}")
        print("-" * 50)
        
        # Arrange in descending order of quantity
        sorted_desks = sorted(stats['counts'].items(), key=lambda x: x[1], reverse=True)
        for desk, count in sorted_desks:
            percentage = stats['percentages'][desk]
            print(f"{desk:<15} {count:<10} {percentage:.2f}%")
    
    # Print statistics for all years
    print("\n================ Overall news category distribution (2016-2025) =============")
    print(f"Total number of articles: {overall_stats['total']}")
    print("-" * 50)
    print(f"{'News Category':<15} {'Quantity':<10} {'Percentage':<10}")
    print("-" * 50)
    
    # Arrange in descending order of quantity
    sorted_desks = sorted(overall_stats['counts'].items(), key=lambda x: x[1], reverse=True)
    for desk, count in sorted_desks:
        percentage = overall_stats['percentages'][desk]
        print(f"{desk:<15} {count:<10} {percentage:.2f}%")

def plot_distributions(yearly_stats, overall_stats, output_folder=None):
    """Draw a distribution chart"""
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare the drawing data
    years = sorted(yearly_stats.keys())
    categories = sorted(overall_stats['counts'].keys())
    
    # Create a distributed data frame by year
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
    
    # 1. Draw the overall distribution pie chart
    plt.figure(figsize=(12, 8))
    
    # In descending order in proportion
    sorted_overall = sorted(overall_stats['percentages'].items(), key=lambda x: x[1], reverse=True)
    categories_sorted = [item[0] for item in sorted_overall]
    percentages_sorted = [item[1] for item in sorted_overall]
    
    plt.pie(percentages_sorted, labels=categories_sorted, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('NYT Articles by News Desk Category (2016-2025)')
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'overall_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Draw an annual trend line chart
    plt.figure(figsize=(15, 10))
    
    # Only select the 6 categories with the highest overall proportion
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
    
    # 3. Draw a heat map
    pivot_df = df.pivot(index='Category', columns='Year', values='Percentage')
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    plt.title('News Category Distribution by Year (2016-2025)')
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'category_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def export_to_excel(yearly_stats, overall_stats, output_path):
    """Export statistics to Excel file"""
    # Create an Excel writer
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    # 1. Create an overall statistics table
    overall_df = pd.DataFrame({
        'Category': list(overall_stats['counts'].keys()),
        'Count': list(overall_stats['counts'].values()),
        'Percentage': [overall_stats['percentages'][cat] for cat in overall_stats['counts'].keys()]
    })
    overall_df = overall_df.sort_values('Count', ascending=False)
    overall_df.to_excel(writer, sheet_name='Overall', index=False)
    
    # 2. Create annual statistics tables
    for year, stats in yearly_stats.items():
        year_df = pd.DataFrame({
            'Category': list(stats['counts'].keys()),
            'Count': list(stats['counts'].values()),
            'Percentage': [stats['percentages'][cat] for cat in stats['counts'].keys()]
        })
        year_df = year_df.sort_values('Count', ascending=False)
        year_df.to_excel(writer, sheet_name=f'Year_{year}', index=False)
    
    # 3. Create a trend table (all categories of all years)
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
    
    # Save the file
    writer.close()
    print(f"Statistical results have been exported to: {output_path}")

def main():
    data_folder = "Time-R1/datasets/nyt_years"
    output_folder = "Time-R1/future_news_generation/topic_analysis"
    
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Analyze data from 2016 to 2025
    yearly_stats, overall_stats = analyze_news_desk_distribution(data_folder, year_range=(2016, 2025))
    
    # Print report
    print_distribution_report(yearly_stats, overall_stats)
    
    # Export to Excel
    export_to_excel(yearly_stats, overall_stats, 
                   os.path.join(output_folder, 'nyt_category_stats.xlsx'))
    
    # Try drawing the chart (if matplotlib and seaborn are available)
    try:
        plot_distributions(yearly_stats, overall_stats, output_folder)
        print(f"The visualization result has been saved to: {output_folder}")
    except NameError:
        print("Note: Missing matplotlib or seaborn library, unable to generate visual charts")

if __name__ == "__main__":
    main()