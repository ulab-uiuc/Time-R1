import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --- Plotting Style Configuration (保持原有设置) ---
try:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Prioritize Helvetica
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 18,
        'legend.fontsize': 18, # Slightly larger legend font for 2 items
        'figure.titlesize': 24
    })
except Exception as e:
    print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# --- 定义颜色和样式 ---
muted_colors = {
    'blue': '#89CFF0',   # Muted Blue - 用于 Qwen2.5-1.5B
    'red': '#F08080',    # Muted Red - 用于 Qwen2.5-0.5B
}

# --- 数据源和样式设置 ---
models_info = {
    'Qwen2.5-1.5B': {
        'column': 'royal-water-208 - train/rewards/wrapped_reward/mean',
        'color': muted_colors['blue'],
        'linestyle': '-',
        'marker': 'o',
        'markersize': 6,
        'linewidth': 2.5,
        'alpha_raw': 0.3,  # 原始数据透明度
        'alpha_smooth': 0.8  # 平滑数据透明度
    },
    'Qwen2.5-0.5B': {
        'column': 'rare-sunset-210 - train/rewards/wrapped_reward/mean',
        'color': muted_colors['red'],
        'linestyle': '--',
        'marker': 's',
        'markersize': 6,
        'linewidth': 2.5,
        'alpha_raw': 0.3,  # 原始数据透明度
        'alpha_smooth': 0.8  # 平滑数据透明度
    }
}

# --- 绘图参数 ---
max_step = 1000  # 只显示前1000步
smoothing_sigma = 10  # 平滑系数

# --- 读取数据 ---
try:
    # 读取CSV文件
    df = pd.read_csv('wandb_export_2025-05-20T00_06_55.296-05_00.csv')
    
    # 转换步骤列为数值类型
    df['train/global_step'] = pd.to_numeric(df['train/global_step'], errors='coerce')
    
    # 仅保留前1000步数据
    df = df[df['train/global_step'] <= max_step].copy()
    
    data_to_plot = {}
    
    # 处理每个模型的数据
    for model_name, info in models_info.items():
        try:
            # 转换数据列为数值
            df[info['column']] = pd.to_numeric(df[info['column']], errors='coerce')
            
            # 创建模型数据条目
            if not df.empty:
                steps = df['train/global_step'].values
                values = df[info['column']].values
                
                # 去除NaN值
                mask = ~np.isnan(values)
                steps = steps[mask]
                values = values[mask]
                
                # 平滑处理
                smooth_values = gaussian_filter1d(values, sigma=smoothing_sigma)
                
                entry = {
                    'steps': steps,
                    'raw_values': values,
                    'smooth_values': smooth_values,
                    'color': info['color'],
                    'linestyle': info['linestyle'],
                    'linewidth': info.get('linewidth', 2),
                    'marker': info.get('marker', '.'),
                    'markersize': info.get('markersize', 0.1),
                    'alpha_raw': info.get('alpha_raw', 0.1),
                    'alpha_smooth': info.get('alpha_smooth', 1)
                }
                data_to_plot[model_name] = entry
            else:
                print(f"Warning: No data to plot for {model_name}.")
        except Exception as e:
            print(f"Error processing data for {model_name}: {e}")
    
except Exception as e:
    print(f"Error reading CSV file: {e}")

# --- 创建图形 ---
if not data_to_plot:
    print("Error: No data successfully processed for plotting.")
else:
    fig, ax1 = plt.subplots(figsize=(9, 7))  # 保持原有图形大小
    
    for model_name, data in data_to_plot.items():
        # 绘制原始数据（带透明度）
        ax1.plot(data['steps'], data['raw_values'], 
                 color=data['color'], alpha=data['alpha_raw'],
                 linestyle=data['linestyle'], linewidth=1,
                 marker=data['marker'], markersize=1)
        
        # 绘制平滑处理后的数据
        ax1.plot(data['steps'], data['smooth_values'], 
                 label=model_name, color=data['color'],
                 linestyle=data['linestyle'], linewidth=3)
    
    # 设置坐标轴标签
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(r'Reward')  # 更改为奖励值
    ax1.tick_params(axis='both', which='major')
    
    # X轴设置
    ax1.set_xlim(0, max_step)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))  # 每100步显示一个刻度
    ax1.minorticks_on()
    
    # Y轴设置 - 保持与原有脚本类似的区间设置方式
    all_values = np.concatenate([data['raw_values'] for data in data_to_plot.values()])
    if len(all_values) > 0:
        min_y = np.min(all_values) * 0.98
        max_y = np.max(all_values) * 1.015
        
        # 使用固定Y轴范围或基于数据的范围
        # ax1.set_ylim(bottom=0, top=1)  # 这是固定的0-1范围
        ax1.set_ylim(bottom=min_y - 0.02, top=max_y + 0.02)  # 根据数据动态设置
    
    # 网格设置
    ax1.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)
    
    # 添加图例
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='lower right')
    else:
        print("Warning: No lines were plotted, legend will not be shown.")
    
    # 设置图形布局
    fig.tight_layout()
    
    # 保存图形
    try:
        plt.savefig("model_size_comparison.pdf", bbox_inches='tight')
        print("Plot saved as model_size_comparison.pdf")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
    
    # 显示图形
    plt.show()