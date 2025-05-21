import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting axes
import matplotlib.font_manager as fm
import os

# --- Font Setup (assuming Helvetica is available or fallback is acceptable) ---
font_dir = '/data/zliu331/fonts/' # Ensure this path is correct for your system
os.makedirs(font_dir, exist_ok=True)
font_path = os.path.join(font_dir, 'Helvetica.ttc')

if os.path.exists(font_path):
    try:
        fm._get_font.cache_clear()
    except AttributeError: # For newer matplotlib versions where _get_font might not exist
        pass
    except Exception:
        pass
    
    try:
        # For newer matplotlib, fontManager might be fm.fontManager
        if hasattr(fm, 'fontManager') and hasattr(fm.fontManager, 'addfont'):
            fm.fontManager.addfont(font_path)
        elif hasattr(fm, 'addfont'): # For older versions
             fm.addfont(font_path)
        # Rebuild font cache if possible (more robust ways might depend on exact matplotlib version)
        # from matplotlib.font_manager import _load_fontmanager
        # fm._fontmanager = _load_fontmanager(try_read_cache=False) # Force rebuild
        print(f"Attempted to register Helvetica字体: {font_path}")
    except Exception as e:
        print(f"Could not add font: {e}")
        pass
else:
    print(f"警告: 找不到字体文件 {font_path}，将使用系统默认字体")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
# --- End Font Setup ---

# 1. Data Preparation (Monthly "Avg Total Score" / "平均总分")
# Data for August 2024 - Feb 2025 (7 months)
data = {
    'Month': ["24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02"],
    # Baseline Data (July data removed)
    'Qwen2.5-3B-Instruct': [0.6006, 0.6071, 0.5864, 0.6095, 0.5763, 0.6169, 0.6285],
    'Qwen2.5-7B-Instruct': [0.7009, 0.7003, 0.6963, 0.6284, 0.6128, 0.5297, 0.4901],
    'Llama-3.1-8B-Instruct': [0.5599, 0.5611, 0.5976, 0.6654, 0.5654, 0.5964, 0.6646],
    'DeepSeek-Distill-Qwen-32B': [0.6836, 0.6390, 0.6704, 0.6349, 0.5615, 0.5821, 0.4664],
    'DeepSeek-V3-0324-671B': [0.7651, 0.7527, 0.7884, 0.6954, 0.6397, 0.6752, 0.6085],
    'DeepSeek-R1-671B': [0.8011, 0.8120, 0.7930, 0.7694, 0.6761, 0.7305, 0.6703],

    ### PLEASE PROVIDE/VERIFY ACTUAL 7-MONTH DATA for the two Time-R1 models ###
    # The data below is a placeholder (original 8-month data with July removed)
    # Replace these with the actual values from your image/logs for the 7 months (Aug to Feb)
    r'Time-R1 ($\theta_2$, 3B)': [0.88186, 0.93631, 0.8571, 0.76556, 0.71624, 0.63661, 0.59389], # Placeholder - (Yellow curve on the left)
    # 'Time-R1-S2-Direct (θ₂\', 3B)': [0.91219, 0.89192, 0.80381, 0.79036, 0.72711, 0.61085, 0.55488], # Placeholder - (Blue curve on the right)
    r"Time-R1-S2-Direct ($\theta_2^{\prime}$, 3B)": [0.88069, 0.83912, 0.76082, 0.731, 0.68431, 0.6193, 0.54829],
}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Reorder columns to put Time-R1 models first for plotting emphasis and desired legend order
# Time-R1 (θ₂, 3B) first, then Time-R1-S2-Direct (θ₂', 3B), then others
time_r1_models = [r'Time-R1 ($\theta_2$, 3B)', r"Time-R1-S2-Direct ($\theta_2^{\prime}$, 3B)"]
other_models = [col for col in df.columns if col not in time_r1_models]
cols = time_r1_models + other_models
df = df[cols]


# 2. Plotting - Enhanced Look
plt.style.use('seaborn-v0_8-ticks')
plt.figure(figsize=(9, 7)) # Slightly adjusted for potentially more legend entries
ax = plt.gca()

markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', '*'] # Added one more marker
# Defined colors - ensure enough for all lines or matplotlib cycle will be used
plot_colors = [
    '#6699CC',  # Orange/Yellow for Time-R1 (θ₂, 3B) - to match "yellow curve"  89CFF0
    '#E69F00',  # Sky Blue for Time-R1-S2-Direct (θ₂', 3B) - to match "blue curve"
    '#888888',  # Grey
    '#AABB88',  # Soft Green
    '#DEADCB',  # Gold CCBB44, DEADCB, FEDDC6
    '#EE99AA',  # Soft Pink
    '#BBCCDD',  # Pale Blue
    '#AA88BB',  # Purple
    '#0072B2'   # Darker Blue (if needed)
]


for i, column in enumerate(df.columns):
    marker_style = markers[i % len(markers)]
    color = plot_colors[i % len(plot_colors)]
    
    if "Time-R1" in column: # This will catch both your models
        ax.plot(df.index, df[column],
                 marker=marker_style,
                 linestyle='-',
                 linewidth=2.5, # Slightly thicker for emphasis
                 label=column,
                 color=color,
                 zorder=10)
    else:
        ax.plot(df.index, df[column],
                 marker=marker_style,
                 linestyle='--',
                 linewidth=2.5, # Slightly thinner dashed line
                 label=column,
                 color=color,
                 alpha=1)

# 3. Customization
plt.xlabel('Month (YY-MM)', fontsize=25, fontname='Helvetica' if os.path.exists(font_path) else 'sans-serif')
plt.ylabel(r'Average Total Score $R(x, y)$', fontsize=25, fontname='Helvetica' if os.path.exists(font_path) else 'sans-serif')

min_val = df.min().min()
max_val = df.max().max()
plt.ylim(min_val - 0.04, max_val + 0.06) # Adjusted padding slightly
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.xticks(fontsize=23, fontname='Helvetica' if os.path.exists(font_path) else 'sans-serif')
plt.yticks(fontsize=23, fontname='Helvetica' if os.path.exists(font_path) else 'sans-serif')

legend_font_prop = {'family': 'Helvetica' if os.path.exists(font_path) else 'sans-serif', 'size': 15.5}
legend = plt.legend(
    fontsize=legend_font_prop['size'],
    loc='upper right',
    frameon=True,
    facecolor='white',
    framealpha=0.8,
    prop=legend_font_prop
)

plt.grid(True, which='major', linestyle=':', linewidth=0.7, color='darkgrey') # Slightly darker grid

ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='both', which='major', length=4, width=0.5)

plt.tight_layout(pad=0.5)

# 4. Save and Show plot
output_filename = "monthly_prediction_scores_stage2_1.pdf"
plt.savefig(output_filename, bbox_inches='tight', dpi=500)
print(f"Plot saved as {output_filename}")
plt.show() # Comment out if running in a non-interactive environment or just want to save






# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker # For formatting axes
# import matplotlib.font_manager as fm
# import os

# # # 打印matplotlib查找字体的目录
# # print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
# # 创建字体目录（如果不存在）
# font_dir = '/data/zliu331/fonts/'
# os.makedirs(font_dir, exist_ok=True)

# # 注册字体
# font_path = os.path.join(font_dir, 'Helvetica.ttc')
# if os.path.exists(font_path):
#     # 清除字体缓存的几种方法，根据matplotlib版本的不同使用不同的方法
#     try:
#         # 尝试使用 cache_clear 方法
#         fm._get_font.cache_clear()
#     except:
#         pass
    
#     try:
#         # 添加字体文件
#         fm.fontManager.addfont(font_path)
#     except:
#         # 较旧版本的matplotlib可能没有addfont方法
#         pass
    
#     # 更安全的刷新字体缓存方式
#     print(f"已注册Helvetica字体: {font_path}")
# else:
#     print(f"警告: 找不到字体文件 {font_path}，将使用系统默认字体")

# # 设置全局字体为 Helvetica
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
# # 如果机器上没有Helvetica字体，会依次尝试后面的字体

# # 1. Data Preparation (Monthly "Avg Total Score" / "平均总分")
# # Updated model names in dictionary keys as requested
# data = {
#     'Month': ["24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02"],
#     # Baseline Data from text logs
#     'Qwen2.5-3B-Instruct': [0.5871, 0.6006, 0.6071, 0.5864, 0.6095, 0.5763, 0.6169, 0.6285],
#     'Qwen2.5-7B-Instruct':        [0.7214, 0.7009, 0.7003, 0.6963, 0.6284, 0.6128, 0.5297, 0.4901],
#     'Llama-3.1-8B-Instruct':      [0.5246, 0.5599, 0.5611, 0.5976, 0.6654, 0.5654, 0.5964, 0.6646],
#     'DS-Distill-Qwen-32B':    [0.6776, 0.6836, 0.6390, 0.6704, 0.6349, 0.5615, 0.5821, 0.4664],
#     'DeepSeek-V3-671B':  [0.7873, 0.7651, 0.7527, 0.7884, 0.6954, 0.6397, 0.6752, 0.6085], # Name updated
#     'DeepSeek-R1-671B':  [0.7939, 0.8011, 0.8120, 0.7930, 0.7694, 0.6761, 0.7305, 0.6703], # Name updated
#     # Your Model Data (Estimated from image - PLEASE VERIFY)
#     'Time-R1 (Ours, 3B)':    [0.9557, 0.9246, 0.8485, 0.7972, 0.8881, 0.7856, 0.6628, 0.6382] 
# }

# df = pd.DataFrame(data)
# df.set_index('Month', inplace=True)

# # Reorder columns to put Time-R1 first for plotting emphasis
# cols = ['Time-R1 (Ours, 3B)'] + [col for col in df if col != 'Time-R1 (Ours, 3B)']
# df = df[cols]


# # 2. Plotting - Enhanced Look
# # plt.style.use('seaborn-v0_8-whitegrid') 
# plt.style.use('seaborn-v0_8-ticks') # Try ticks style for cleaner axes maybe
# plt.figure(figsize=(9, 7)) # Adjusted figure size slightly
# ax = plt.gca() # Get current axes

# # Define distinct colors and markers if needed (matplotlib default cycle is usually good)
# markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p'] 
# # colors = plt.cm.tab10.colors # Example color map

# colors = [
#     '#6699CC',  # 柔和的蓝色
#     '#888888',  # 中性灰色
#     '#AABB88',  # 柔和的绿色
#     '#CCBB44',  # 金色
#     '#EE99AA',  # 柔和的粉色
#     '#BBCCDD',  # 淡蓝色
#     '#AA88BB'   # 紫色
# ]

# # colors = [
# #     '#0F6E8C',  # 孔雀蓝
# #     '#8E3B46',  # 酒红色
# #     '#30638E',  # 靛蓝色
# #     '#476A30',  # 森林绿
# #     '#7D4E57',  # 紫红褐色
# #     '#394C73',  # 海军蓝
# #     '#6B5876'   # 淡紫褐色
# # ]

# for i, column in enumerate(df.columns):
#     marker_style = markers[i % len(markers)]
    
#     color = colors[i % len(colors)] # Optional: assign colors explicitly
    
#     if "Time-R1" in column:
#         # Highlight Time-R1
#         ax.plot(df.index, df[column], 
#                  marker=marker_style, 
#                  linestyle='-',        
#                  linewidth=2.5,      # Slightly thicker
#                  label=column, 
#                  color=color, # Uncomment if using explicit colors
#                  zorder=10)      # Bring to front
#     else:
#         # Plot baselines
#         ax.plot(df.index, df[column], 
#                  marker=marker_style, 
#                  linestyle='--', 
#                  linewidth=2.5, # Slightly thicker dashed line
#                  label=column, 
#                  color=color, # Uncomment if using explicit colors
#                  alpha=0.8)      

# # 3. Customization
# # No Title
# # plt.title('Monthly Total Score $R(x, y)$ for Future Event Prediction (Stage 2)', fontsize=16, pad=20) 
    
# plt.xlabel('Month (YY-MM)', fontsize=24, fontfamily='Helvetica') # Increased axis title font size
# plt.ylabel(r'Average Total Score $R(x, y)$', fontsize=24, fontfamily='Helvetica') # Increased axis title font size

# # Adjust y-axis limits dynamically with padding
# min_val = df.min().min()
# max_val = df.max().max()
# plt.ylim(min_val - 0.03, max_val + 0.05) 
# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f')) 

# plt.xticks(fontsize=22, fontfamily='Helvetica') # Increased tick label font size
# plt.yticks(fontsize=22, fontfamily='Helvetica') # Increased tick label font size

# # Place legend inside plot, adjust font size, add frame/transparency
# # Experiment with loc: 'best', 'lower left', 'upper right', 'center left', etc.
# legend = plt.legend(fontsize=16, loc='upper right', frameon=True, facecolor='white', framealpha=0.75, prop={'family': 'Helvetica', 'size': 16}) # Increased legend font size slightly

# plt.grid(True, which='major', linestyle=':', linewidth=0.6, color='grey') # Lighter grid

# # Improve 'advanced' look: remove top/right spines
# ax.spines['top'].set_linewidth(0.5)
# ax.spines['right'].set_linewidth(0.5)
# ax.spines['left'].set_linewidth(0.5)
# ax.spines['bottom'].set_linewidth(0.5)
# ax.tick_params(axis='both', which='major', length=4, width=0.5)


# plt.tight_layout() # Adjust layout automatically

# # 4. Show plot
# plt.savefig("monthly_prediction_scores.pdf", bbox_inches='tight') # Add this line to save as PDF
# plt.show()








# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker # For formatting axes

# # 1. Data Preparation (Monthly "Avg Total Score" / "平均总分")
# # Extracted from your text logs and estimated from image_61a112.jpg for Time-R1
# data = {
#     'Month': ["24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02"],
#     # Baseline Data from text logs (using "平均总分")
#     'Qwen2.5-3B':    [0.5871, 0.6006, 0.6071, 0.5864, 0.6095, 0.5763, 0.6169, 0.6285],
#     'Qwen2.5-7B':    [0.7214, 0.7009, 0.7003, 0.6963, 0.6284, 0.6128, 0.5297, 0.4901],
#     'Llama-3.1-8B':  [0.5246, 0.5599, 0.5611, 0.5976, 0.6654, 0.5654, 0.5964, 0.6646],
#     'DS-Distill-32B':[0.6776, 0.6836, 0.6390, 0.6704, 0.6349, 0.5615, 0.5821, 0.4664],
#     'DeepSeek-V3':   [0.7873, 0.7651, 0.7527, 0.7884, 0.6954, 0.6397, 0.6752, 0.6085], # Assumed 671B
#     'DeepSeek-R1':   [0.7939, 0.8011, 0.8120, 0.7930, 0.7694, 0.6761, 0.7305, 0.6703], # Assumed 671B
#     # Time-R1 Data estimated from image plots (val/time_prediction/overall_reward_*)
#     # PLEASE VERIFY THESE VALUES FROM YOUR LOGS/DATA
#     'Time-R1 (Ours)': [0.9557, 0.9246, 0.8485, 0.7972, 0.8881, 0.7856, 0.6628, 0.6382] 
# }

# df = pd.DataFrame(data)
# df.set_index('Month', inplace=True)

# # Reorder columns to put Time-R1 first for plotting emphasis
# cols = ['Time-R1 (Ours)'] + [col for col in df if col != 'Time-R1 (Ours)']
# df = df[cols]


# # 2. Plotting
# plt.style.use('seaborn-v0_8-whitegrid') # Use a clean plot style
# plt.figure(figsize=(11, 6)) # Set figure size

# # Define markers and linestyles to cycle through
# markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p']
# linestyles = ['-', '--', '-.', ':', '--', '-.', ':'] # Different styles for baselines

# # Plot each model
# for i, column in enumerate(df.columns):
#     marker_style = markers[i % len(markers)]
#     if "Time-R1" in column:
#         # Highlight Time-R1
#         plt.plot(df.index, df[column], 
#                  marker=marker_style, 
#                  linestyle='-',        # Solid line for main model
#                  linewidth=2.5,      # Thicker line
#                  label=column, 
#                  zorder=10)          # Plot on top
#     else:
#         # Plot baselines
#         linestyle_idx = i % len(linestyles)
#         plt.plot(df.index, df[column], 
#                  marker=marker_style, 
#                  linestyle='--', # Dashed lines for baselines
#                  linewidth=1.5, 
#                  label=column, 
#                  alpha=0.8)      # Slightly transparent

# # 3. Customization
# plt.title('Monthly Total Score $R(x, y)$ for Future Event Prediction (Stage 2)', fontsize=15, pad=15)
# plt.xlabel('Month (YY-MM)', fontsize=12)
# plt.ylabel('Average Total Score $R(x, y)$', fontsize=12)

# # Set y-axis limits based on data range, add some padding
# min_val = df.min().min()
# max_val = df.max().max()
# plt.ylim(min_val - 0.05, max_val + 0.05) 
# # Or set manual limits if preferred, e.g., plt.ylim(0.4, 1.0)

# # Format y-axis ticks to 2 decimal places for better readability
# plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)

# # Place legend outside the plotting area
# plt.legend(fontsize=9.5, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) 

# plt.grid(True, which='major', linestyle='--', linewidth=0.5) # Add grid lines

# # Adjust layout to prevent legend cutoff
# plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust right margin for legend

# # 4. Show plot
# # 4. Save and Show plot
# plt.savefig("monthly_prediction_scores.pdf", bbox_inches='tight') # Add this line to save as PDF
# plt.show()