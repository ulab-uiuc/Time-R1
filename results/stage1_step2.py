import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Plotting Style Configuration ---
try:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Prioritize Helvetica
        'font.size': 15,  # Increased base font size
        'axes.labelsize': 18, # Increased
        'axes.titlesize': 22, # Increased
        'xtick.labelsize': 14, # Increased
        'ytick.labelsize': 15, # Increased
        'legend.fontsize': 13, # Increased
        'figure.titlesize': 24 # Increased
    })
except Exception as e:
    print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# --- File Information and Styling ---
# Lower saturation colors
muted_colors = {
    'blue': '#89CFF0',
    'green': '#90EE90',
    'red': '#F08080',
    'purple': '#C3B1E1',
    'orange': '#FAD7A0',
}

task_base_colors = {
    'Overall': muted_colors['blue'],
    'Completion': muted_colors['green'],
    'Difference': muted_colors['red'],
    'Inferring': muted_colors['purple'],
    'Ordering': muted_colors['orange'],
}

files_info = {
    r'Overall $R_{\mathrm{acc}}$': {
        'path': 'overall_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_overall',
        'type': 'accuracy',
        'color': task_base_colors['Overall'],
        'linestyle': '-',
        'linewidth': 3, # Thicker for Overall
        'marker': 'o',
        'markersize': 2
    },
    r'Completion $R_{\mathrm{acc}}$': {
        'path': 'completion_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_completion',
        'type': 'accuracy',
        'color': task_base_colors['Completion'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': '^',
        'markersize': 2
    },
    r'Difference $R_{\mathrm{acc}}$': {
        'path': 'difference_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_difference',
        'type': 'accuracy',
        'color': task_base_colors['Difference'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 's',
        'markersize': 2
    },
    r'Inferring $R_{\mathrm{acc}}$': {
        'path': 'inferring_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_inferring',
        'type': 'accuracy',
        'color': task_base_colors['Inferring'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'D',
        'markersize': 2
    },
    r'Ordering $R_{\mathrm{acc}}$': {
        'path': 'ordering_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_ordering',
        'type': 'accuracy',
        'color': task_base_colors['Ordering'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'P',
        'markersize': 2
    },
    r'Difference $P_{\mathrm{incon}}$': {
        'path': 'difference_consistency.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_difference/consistency_penalty_mean',
        'type': 'inconsistency',
        'color': task_base_colors['Difference'], # Same color as Difference R_acc
        'linestyle': '--',
        'linewidth': 2,
        'marker': 's', # Matching marker for consistency with R_acc
        'markersize': 2, # For scatter part
        'scatter_s': 35  # Size for scatter points
    },
    r'Ordering $P_{\mathrm{incon}}$': {
        'path': 'ordering_consistency.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_ordering/consistency_penalty_mean',
        'type': 'inconsistency',
        'color': task_base_colors['Ordering'],   # Same color as Ordering R_acc
        'linestyle': '--',
        'linewidth': 2,
        'marker': 'P', # Matching marker
        'markersize': 2, # For scatter part
        'scatter_s': 35  # Size for scatter points
    }
}

max_step = 400
step_interval = 10
min_step_to_plot = 10
rolling_window_size = 3

data_to_plot = {}

for label, info in files_info.items():
    try:
        df = pd.read_csv(info['path'])
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
        df.dropna(subset=['Step', info['value_col']], inplace=True)

        processed_df = df[
            (df['Step'] >= min_step_to_plot) &
            (df['Step'] <= max_step) &
            (df['Step'] % step_interval == 0)
        ].copy()
        
        if not processed_df.empty:
            entry = {
                'steps': processed_df['Step'],
                'original_values': processed_df[info['value_col']],
                'type': info['type'],
                'color': info['color'],
                'linestyle': info['linestyle'],
                'linewidth': info.get('linewidth', 2), # Adjusted default linewidth
                'marker': info.get('marker', '.'),
                'markersize': info.get('markersize', 6) # For line plots
            }
            if info['type'] == 'inconsistency':
                entry['smoothed_values'] = processed_df[info['value_col']].rolling(
                    window=rolling_window_size, center=True, min_periods=1).mean()
                entry['scatter_s'] = info.get('scatter_s', 35) # Size for scatter points of P_incon
            data_to_plot[label] = entry
        else:
            print(f"Warning: No data to plot for {label} after filtering.")
    except FileNotFoundError:
        print(f"Error: File not found - {info['path']}")
    except KeyError:
        print(f"Error: Column not found. Expected 'Step' and '{info['value_col']}' in {info['path']}.")
    except Exception as e:
        print(f"An error occurred processing {info['path']}: {e}")

# --- Create the Plot ---
fig, ax1 = plt.subplots(figsize=(6, 7)) # Adjusted figure size

for label, data in data_to_plot.items():
    if data['type'] == 'accuracy':
        ax1.plot(data['steps'], data['original_values'], label=label, color=data['color'],
                 linestyle=data['linestyle'], linewidth=data['linewidth'],
                 marker=data['marker'], markersize=data['markersize'])
    # P_incon plots will be handled after ax2 is created

ax1.set_xlabel('Training Step (Phase 2)')
ax1.set_ylabel(r'Accuracy Score $R_{\mathrm{acc}}$')
ax1.tick_params(axis='y')
ax1.set_xlim(0, max_step + step_interval)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax1.minorticks_on()

ax2 = ax1.twinx() # Create right y-axis sharing x-axis
for label, data in data_to_plot.items():
    if data['type'] == 'inconsistency':
        # 1. Plot original P_incon data as a low-opacity, thinner line
        ax2.plot(data['steps'], data['original_values'],
                 color=data['color'],
                 linestyle=data['linestyle'], # Retain original linestyle (e.g., dashed)
                 linewidth=data['linewidth'] * 0.75,  # Make it thinner
                 alpha=0.4,  # Low opacity
                 zorder=2,
                 label='_nolegend_') # Explicitly exclude from legend

        # 2. Plot smoothed P_incon line (this line gets the legend label)
        ax2.plot(data['steps'], data['smoothed_values'], label=label, color=data['color'],
                 linestyle=data['linestyle'], linewidth=data['linewidth'], zorder=3)

ax2.set_ylabel(r'Inconsistency Penalty Factor $P_{\mathrm{incon}}$')
ax2.tick_params(axis='y')

# --- Grid and Legend ---
ax1.grid(True, axis='x', linestyle=':', alpha=0.6, zorder=0) # Vertical grid for ax1 (x-axis)
ax2.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0) # Horizontal grid for ax2 (right y-axis)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles1 + handles2
all_labels = labels1 + labels2

if all_handles:
    ax1.legend(all_handles, all_labels, loc='lower right', ncol=1) # Legend in bottom right
else:
    print("Warning: No data was plotted, legend will not be shown.")

# plt.title('Training Performance', pad=20) # Title removed as per implicit change in provided code
fig.tight_layout()
plt.savefig("stage1_step2.pdf", bbox_inches='tight')
plt.show()





# # --- Create the Plot ---
# fig, ax1 = plt.subplots(figsize=(8, 7)) # Adjusted figure size

# for label, data in data_to_plot.items():
#     if data['type'] == 'accuracy':
#         ax1.plot(data['steps'], data['original_values'], label=label, color=data['color'],
#                  linestyle=data['linestyle'], linewidth=data['linewidth'],
#                  marker=data['marker'], markersize=data['markersize'])
#     # P_incon plots will be handled after ax2 is created

# ax1.set_xlabel('Training Step')
# ax1.set_ylabel(r'Accuracy Score $R_{\mathrm{acc}}$')
# ax1.tick_params(axis='y')
# ax1.set_xlim(0, max_step + step_interval)
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
# ax1.minorticks_on()

# ax2 = ax1.twinx() # Create right y-axis sharing x-axis
# for label, data in data_to_plot.items():
#     if data['type'] == 'inconsistency':
#         # Plot original data as scatter points for P_incon
#         ax2.scatter(data['steps'], data['original_values'], color=data['color'],
#                     s=data['scatter_s'], alpha=0.5, marker=data['marker'], zorder=2)
#         # Plot smoothed line for P_incon
#         ax2.plot(data['steps'], data['smoothed_values'], label=label, color=data['color'],
#                  linestyle=data['linestyle'], linewidth=data['linewidth'], zorder=3)

# ax2.set_ylabel(r'Inconsistency Penalty Factor $P_{\mathrm{incon}}$')
# ax2.tick_params(axis='y')

# # --- Grid and Legend ---
# ax1.grid(True, axis='x', linestyle=':', alpha=0.6, zorder=0) # Vertical grid for ax1 (x-axis)
# ax2.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0) # Horizontal grid for ax2 (right y-axis)

# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# all_handles = handles1 + handles2
# all_labels = labels1 + labels2

# if all_handles:
#     ax1.legend(all_handles, all_labels, loc='lower right', ncol=1) # Legend in bottom right
# else:
#     print("Warning: No data was plotted, legend will not be shown.")

# # plt.title('Training Performance', pad=20)
# fig.tight_layout()
# plt.savefig("stage1_step2.pdf", bbox_inches='tight') # Add this line to save as PDF
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # --- Plotting Style Configuration ---
# try:
#     # Attempt to use Helvetica, with Arial and DejaVu Sans as fallbacks
#     plt.rcParams.update({
#         'font.family': 'sans-serif',
#         'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
#         'font.size': 14,  # Default base font size
#         'axes.labelsize': 16,
#         'axes.titlesize': 20,
#         'xtick.labelsize': 14,
#         'ytick.labelsize': 14,
#         'legend.fontsize': 12,
#         'figure.titlesize': 22
#     })
# except Exception as e:
#     print(f"Could not set Helvetica font, will use Matplotlib defaults. Error: {e}")

# # --- File Information and Styling ---
# # Lower saturation colors
# muted_colors = {
#     'blue': '#89CFF0',    # Baby Blue
#     'green': '#90EE90',   # Light Green
#     'red': '#F08080',     # Light Coral
#     'purple': '#C3B1E1',  # Light Purple (Lavender)
#     'orange': '#FAD7A0',  # Light Orange (Peach)
#     'cyan': '#A0E6FF',    # Light Cyan
#     'magenta': '#FFB6C1', # Light Pink
# }

# files_info = {
#     r'Overall $R_{\mathrm{acc}}$': {
#         'path': 'overall_acc.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_overall',
#         'type': 'accuracy',
#         'color': muted_colors['blue'],
#         'linestyle': '-'
#     },
#     r'Completion $R_{\mathrm{acc}}$': {
#         'path': 'completion_acc.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_completion',
#         'type': 'accuracy',
#         'color': muted_colors['green'],
#         'linestyle': '-'
#     },
#     r'Difference $R_{\mathrm{acc}}$': {
#         'path': 'difference_acc.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_difference',
#         'type': 'accuracy',
#         'color': muted_colors['red'],
#         'linestyle': '-'
#     },
#     r'Inferring $R_{\mathrm{acc}}$': {
#         'path': 'inferring_acc.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_inferring',
#         'type': 'accuracy',
#         'color': muted_colors['purple'],
#         'linestyle': '-'
#     },
#     r'Ordering $R_{\mathrm{acc}}$': {
#         'path': 'ordering_acc.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_ordering',
#         'type': 'accuracy',
#         'color': muted_colors['orange'],
#         'linestyle': '-'
#     },
#     r'Difference $P_{\mathrm{incon}}$': {
#         'path': 'difference_consistency.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_difference/consistency_penalty_mean',
#         'type': 'inconsistency',
#         'color': muted_colors['cyan'],
#         'linestyle': '--'
#     },
#     r'Ordering $P_{\mathrm{incon}}$': {
#         'path': 'ordering_consistency.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_ordering/consistency_penalty_mean',
#         'type': 'inconsistency',
#         'color': muted_colors['magenta'],
#         'linestyle': '--'
#     }
# }

# max_step = 400
# step_interval = 10 # Plot points every 10 steps
# min_step_to_plot = 10 # All curves start from step 10

# data_to_plot = {}

# for label, info in files_info.items():
#     try:
#         df = pd.read_csv(info['path'])
#         df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
#         df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
#         df.dropna(subset=['Step', info['value_col']], inplace=True)

#         # All curves start from min_step_to_plot (e.g., 10) and are plotted every step_interval
#         processed_df = df[
#             (df['Step'] >= min_step_to_plot) &
#             (df['Step'] <= max_step) &
#             (df['Step'] % step_interval == 0)
#         ].copy()
        
#         if not processed_df.empty:
#             data_to_plot[label] = {
#                 'steps': processed_df['Step'],
#                 'values': processed_df[info['value_col']],
#                 'type': info['type'],
#                 'color': info['color'],
#                 'linestyle': info['linestyle']
#             }
#         else:
#             print(f"Warning: No data to plot for {label} after filtering. Check steps in CSV and min_step_to_plot.")

#     except FileNotFoundError:
#         print(f"Error: File not found - {info['path']}")
#         continue
#     except KeyError:
#         print(f"Error: Column not found in {info['path']}. Expected 'Step' and '{info['value_col']}'.")
#         continue
#     except Exception as e:
#         print(f"An error occurred while processing {info['path']}: {e}")
#         continue

# # --- Create the Plot ---
# fig, ax1 = plt.subplots(figsize=(9, 7)) # Adjusted figure size for better readability

# # Plot accuracy scores on the left y-axis
# for label, data in data_to_plot.items():
#     if data['type'] == 'accuracy':
#         ax1.plot(data['steps'], data['values'], label=label, color=data['color'], linestyle=data['linestyle'], linewidth=2)

# ax1.set_xlabel('Training Step')
# ax1.set_ylabel(r'Accuracy Score $R_{\mathrm{acc}}$') # Color will be default black
# ax1.tick_params(axis='y')
# ax1.set_xlim(0, max_step) # Start x-axis from 0 to see the gap if lines start at 10
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
# ax1.minorticks_on() # Show minor ticks for x-axis

# # Create a second y-axis for inconsistency penalty factors
# ax2 = ax1.twinx()
# for label, data in data_to_plot.items():
#     if data['type'] == 'inconsistency':
#         ax2.plot(data['steps'], data['values'], label=label, color=data['color'], linestyle=data['linestyle'], linewidth=2)

# ax2.set_ylabel(r'Inconsistency Penalty Factor $P_{\mathrm{incon}}$')
# ax2.tick_params(axis='y')

# # --- Grid and Legend ---
# # Add vertical grid lines based on ax1's x-axis
# ax1.grid(True, axis='x', linestyle=':', alpha=0.7)
# # Optionally, add horizontal grid lines for both axes if desired
# ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
# # ax2.grid(True, axis='y', linestyle='--', alpha=0.5) # Be cautious with overlapping grids

# # Combine legends from both axes and place inside the plot
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# all_handles = handles1 + handles2
# all_labels = labels1 + labels2

# if all_handles: # Only add legend if there are items to plot
#     # Experiment with loc and ncol for best placement inside the plot
#     ax1.legend(all_handles, all_labels, loc='lower right', ncol=2)
#     # Other options for loc: 'best', 'upper right', 'lower left', 'center left', etc.
# else:
#     print("Warning: No data was plotted, legend will not be shown.")

# # plt.title('Training Performance')
# fig.tight_layout() # Adjust layout to prevent labels from overlapping

# plt.savefig("stage1_step2.pdf", bbox_inches='tight') # Add this line to save as PDF
# plt.show()







# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.ticker as ticker
# # import matplotlib.font_manager as fm
# # import os


# # # font_dir = '/data/zliu331/fonts/'
# # # # os.makedirs(font_dir, exist_ok=True)

# # # # 注册字体
# # # font_path = os.path.join(font_dir, 'Helvetica.ttc')
# # # if os.path.exists(font_path):
# # #     # 清除字体缓存的几种方法，根据matplotlib版本的不同使用不同的方法
# # #     try:
# # #         # 尝试使用 cache_clear 方法
# # #         fm._get_font.cache_clear()
# # #     except:
# # #         pass
    
# # #     try:
# # #         # 添加字体文件
# # #         fm.fontManager.addfont(font_path)
# # #     except:
# # #         # 较旧版本的matplotlib可能没有addfont方法
# # #         pass
    
# # #     # 更安全的刷新字体缓存方式
# # #     print(f"已注册Helvetica字体: {font_path}")
# # # else:
# # #     print(f"警告: 找不到字体文件 {font_path}，将使用系统默认字体")

# # # 设置全局字体为 Helvetica
# # plt.rcParams['font.family'] = 'sans-serif'
# # plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']

# # # Define the file information
# # files_info = {
# #     'Overall Acc': {
# #         'path': 'overall_acc.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_overall',
# #         'type': 'accuracy',
# #         'color': 'blue',
# #         'linestyle': '-'
# #     },
# #     'Completion Acc': {
# #         'path': 'completion_acc.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_completion',
# #         'type': 'accuracy',
# #         'color': 'green',
# #         'linestyle': '-'
# #     },
# #     'Difference Acc': {
# #         'path': 'difference_acc.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_difference',
# #         'type': 'accuracy',
# #         'color': 'red',
# #         'linestyle': '-'
# #     },
# #     'Inferring Acc': {
# #         'path': 'inferring_acc.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_inferring',
# #         'type': 'accuracy',
# #         'color': 'purple',
# #         'linestyle': '-'
# #     },
# #     'Ordering Acc': {
# #         'path': 'ordering_acc.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - val/pred_reward_time_ordering',
# #         'type': 'accuracy',
# #         'color': 'orange',
# #         'linestyle': '-'
# #     },
# #     'Difference Incon': {
# #         'path': 'difference_consistency.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_difference/consistency_penalty_mean',
# #         'type': 'inconsistency',
# #         'color': 'cyan',
# #         'linestyle': '--'
# #     },
# #     'Ordering Incon': {
# #         'path': 'ordering_consistency.csv',
# #         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_dynamic_alpha - rewards/time_ordering/consistency_penalty_mean',
# #         'type': 'inconsistency',
# #         'color': 'magenta',
# #         'linestyle': '--'
# #     }
# # }

# # max_step = 400
# # step_interval = 10

# # data_to_plot = {}

# # for label, info in files_info.items():
# #     try:
# #         df = pd.read_csv(info['path'])
# #         # Ensure 'Step' and value columns are numeric
# #         df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
# #         df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
# #         df.dropna(subset=['Step', info['value_col']], inplace=True)

# #         if info['type'] == 'accuracy':
# #             # Accuracy files: recorded every 10 steps from step 0
# #             processed_df = df[df['Step'] <= max_step].copy()
# #             # Ensure steps are multiples of 10 (though they should be already)
# #             # and include step 0
# #             processed_df = processed_df[processed_df['Step'] % step_interval == 0]
# #         elif info['type'] == 'inconsistency':
# #             # Inconsistency files: recorded every step from step 1
# #             # We need to select steps 10, 20, ..., 400
# #             processed_df = df[(df['Step'] >= step_interval) & (df['Step'] <= max_step) & (df['Step'] % step_interval == 0)].copy()
        
# #         data_to_plot[label] = {
# #             'steps': processed_df['Step'],
# #             'values': processed_df[info['value_col']],
# #             'type': info['type'],
# #             'color': info['color'],
# #             'linestyle': info['linestyle']
# #         }
# #     except FileNotFoundError:
# #         print(f"Error: File not found - {info['path']}")
# #         continue
# #     except KeyError:
# #         print(f"Error: Column not found in {info['path']}. Check 'Step' or '{info['value_col']}'.")
# #         continue
# #     except Exception as e:
# #         print(f"An error occurred while processing {info['path']}: {e}")
# #         continue

# # # Create the plot
# # fig, ax1 = plt.subplots(figsize=(7, 7))

# # # Plot accuracy scores on the left y-axis
# # for label, data in data_to_plot.items():
# #     if data['type'] == 'accuracy':
# #         ax1.plot(data['steps'], data['values'], label=label, color=data['color'], linestyle=data['linestyle'])

# # ax1.set_xlabel('Training Step')
# # ax1.set_ylabel(r'Accuracy Score $R_{\mathrm{acc}}$', color='black')
# # ax1.tick_params(axis='y', labelcolor='black')
# # ax1.set_xlim(0, max_step)
# # ax1.xaxis.set_major_locator(ticker.MultipleLocator(50)) # Set x-axis ticks every 50 steps


# # # Create a second y-axis for inconsistency penalty factors
# # ax2 = ax1.twinx()
# # for label, data in data_to_plot.items():
# #     if data['type'] == 'inconsistency':
# #         ax2.plot(data['steps'], data['values'], label=label, color=data['color'], linestyle=data['linestyle'])

# # ax2.set_ylabel(r'Inconsistency Penalty Factor $P_{\mathrm{incon}}$', color='black')
# # ax2.tick_params(axis='y', labelcolor='black')

# # # Add legends
# # lines1, labels1 = ax1.get_legend_handles_labels()
# # lines2, labels2 = ax2.get_legend_handles_labels()
# # # Place legend below the plot or adjust as needed
# # fig.legend(lines1 + lines2, labels1 + labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))


# # plt.title('Training Performance: Accuracy Scores and Inconsistency Penalty Factors')
# # fig.tight_layout(rect=[0, 0.1, 1, 0.96]) # Adjust layout to make space for legend if it's outside
# # plt.grid(True, linestyle=':', alpha=0.7)

# # # 4. Save and Show plot
# # plt.savefig("stage1_step2.pdf", bbox_inches='tight') # Add this line to save as PDF
# # plt.show()