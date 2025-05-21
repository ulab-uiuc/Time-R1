import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np # For break marks
import sys # For sys.exit()

# --- Plotting Style Configuration (From your provided code) ---
try:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Prioritize Helvetica
        'font.size': 24,       # Base font size
        'axes.labelsize': 24,  # Axis labels
        'axes.titlesize': 22,  # Plot title (if used)
        'xtick.labelsize': 20, # X-axis tick labels
        'ytick.labelsize': 18, # Y-axis tick labels
        'legend.fontsize': 18, # Legend font size
        'figure.titlesize': 24 # Figure title (suptitle)
    })
except Exception as e:
    print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# --- File Information and Styling ---
muted_colors = {
    'blue': '#89CFF0',   # Muted Blue for "With Dynamic Reward" (typically shorter length)
    'red': '#F08080',    # Muted Red for "Without Dynamic Reward" (typically longer length)
}

files_info_length = {
    'With Dynamic Reward': {
        'path': 'overall_response_length.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - response_length/overall/mean',
        'color': muted_colors['blue'],
        'linestyle': '-',
        'marker': 'o',
        'markersize': 6,
        'linewidth': 2.5
    },
    'Without Dynamic Reward': {
        'path': 'overall_response_length.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_alpha_0.1 - response_length/overall/mean',
        'color': muted_colors['red'],
        'linestyle': '--',
        'marker': 's',
        'markersize': 6,
        'linewidth': 2.5
    }
}

# --- Plotting Parameters ---
max_step = 810
step_interval = 10
min_step_to_plot = 10

data_to_plot = {}

# --- Data Loading and Processing ---
for label, info in files_info_length.items():
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
                'values': processed_df[info['value_col']],
                'color': info['color'],
                'linestyle': info['linestyle'],
                'linewidth': info.get('linewidth', 2.5),
                'marker': info.get('marker', '.'),
                'markersize': info.get('markersize', 6)
            }
            data_to_plot[label] = entry
        else:
            print(f"Warning: No data to plot for {label} after filtering (Steps {min_step_to_plot}-{max_step}).")
    except FileNotFoundError:
        print(f"Error: File not found - {info['path']}")
        # Continue to the next file if one is not found, in case the other is present
        continue 
    except KeyError:
        cols_in_file = []
        try:
            cols_in_file = pd.read_csv(info['path'], nrows=1).columns.tolist()
        except: pass
        print(f"Error: Column not found in {info['path']}. Expected 'Step' and '{info['value_col']}'. Available cols: {cols_in_file}")
        continue # Continue to the next file
    except Exception as e:
        print(f"An error occurred processing {info['path']}: {e}")
        continue # Continue to the next file

# --- Create the Plot with Broken Y-Axis ---
if len(data_to_plot) == 2:
    all_values_list = [val for data_dict in data_to_plot.values() for val in data_dict['values']]
    if not all_values_list:
        print("Error: No numerical values to plot after processing data. Check CSV content and filtering.")
        sys.exit(1)
        
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    
    # --- KEY CHANGE: Adjust hspace for a small, visible gap ---
    # fig.subplots_adjust(hspace=0.075) # Try a value like this, e.g. 5% to 10% of axes height

    for label, data in data_to_plot.items():
        ax_top.plot(data['steps'], data['values'], label=label, color=data['color'],
                    linestyle=data['linestyle'], linewidth=data['linewidth'],
                    marker=data['marker'], markersize=data['markersize'])
        ax_bottom.plot(data['steps'], data['values'], label=label, color=data['color'],
                       linestyle=data['linestyle'], linewidth=data['linewidth'],
                       marker=data['marker'], markersize=data['markersize'])

    min_val_overall = min(all_values_list)
    max_val_overall = max(all_values_list)

    y_break_bottom_limit = 160
    y_break_top_limit = 220   

    ax_top.set_ylim(bottom=y_break_top_limit, top=max_val_overall + 15) # Added a bit more padding
    ax_bottom.set_ylim(bottom=min_val_overall - 15, top=y_break_bottom_limit) # Added a bit more padding

    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    # Add break marks - Reverted to more standard vertical span
    d = .01  # Controls the 'width' and 'height' of the slashes
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.5) # Slightly thicker slash
    ax_top.plot((-d, +d), (-d, +d), **kwargs) # Standard y-span for top-left
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs) # Standard y-span for top-right

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs) # Standard y-span for bottom-left
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # Standard y-span for bottom-right

    ax_bottom.set_xlabel('Training Step')
    fig.text(0.04, 0.5, 'Average Response Length', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize']) # Adjusted x for y-label

    ax_bottom.set_xlim(0, max_step + step_interval)
    ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_bottom.minorticks_on()
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='lower')) # Adjusted nbins
    ax_bottom.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='upper')) # Adjusted nbins

    ax_top.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)
    ax_bottom.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)

    handles_top, labels_top = ax_top.get_legend_handles_labels()
    # unique_labels = dict(zip(labels_top, handles_top)) # Get unique legend entries
    # if unique_labels:
    #      ax_top.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', ncol=1)
    # else:
    #     print("Warning: No lines were plotted for the top axis, legend will not be shown.")
    
    # plt.suptitle('Average Response Length: With vs. Without Dynamic Reward', y=0.98, fontsize=plt.rcParams['figure.titlesize'])
    fig.tight_layout(rect=[0.07, 0.05, 1, 0.93]) # Adjusted rect
    # --- Apply hspace AFTER tight_layout to override its calculation ---
    fig.subplots_adjust(hspace=0.1) # Try value like 0.1 or 0.15 for a visible gap

    try:
        plt.savefig("response_length_comparison_broken_axis.pdf", bbox_inches='tight')
        print("Plot saved as response_length_comparison_broken_axis.pdf")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
    plt.show()

else:
    print("Error: Not enough data was successfully processed (expected 2 datasets). Cannot generate plot.")
    if 'fig' in locals(): # Close the figure if it was created before erroring
        plt.close(fig)
    sys.exit(1)





# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np # For break marks

# # --- Plotting Style Configuration (From your provided code) ---
# try:
#     plt.rcParams.update({
#         'font.family': 'sans-serif',
#         'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Prioritize Helvetica
#         'font.size': 24,       # Base font size
#         'axes.labelsize': 24,  # Axis labels
#         'axes.titlesize': 22,  # Plot title (if used)
#         'xtick.labelsize': 20, # X-axis tick labels
#         'ytick.labelsize': 18, # Y-axis tick labels
#         'legend.fontsize': 18, # Legend font size
#         'figure.titlesize': 24 # Figure title (suptitle)
#     })
# except Exception as e:
#     print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# # --- File Information and Styling ---
# muted_colors = {
#     'blue': '#89CFF0',   # Muted Blue for "With Dynamic Reward" (typically shorter length)
#     'red': '#F08080',    # Muted Red for "Without Dynamic Reward" (typically longer length)
# }

# files_info_length = {
#     'With Dynamic Reward': {
#         'path': 'overall_response_length.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - response_length/overall/mean',
#         'color': muted_colors['blue'],
#         'linestyle': '-',
#         'marker': 'o',
#         'markersize': 7, # Slightly larger markers
#         'linewidth': 3 # Slightly thicker lines
#     },
#     'Without Dynamic Reward': {
#         'path': 'overall_response_length.csv',
#         'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_alpha_0.1 - response_length/overall/mean',
#         'color': muted_colors['red'],
#         'linestyle': '--',
#         'marker': 's',
#         'markersize': 7,
#         'linewidth': 3
#     }
# }

# # --- Plotting Parameters ---
# max_step = 810 # Plot up to step 840 as one series ends around there
# step_interval = 10
# min_step_to_plot = 10 # Start plotting from step 10

# data_to_plot = {}

# # --- Data Loading and Processing ---
# for label, info in files_info_length.items():
#     try:
#         df = pd.read_csv(info['path'])
#         df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
#         df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
#         df.dropna(subset=['Step', info['value_col']], inplace=True)

#         processed_df = df[
#             (df['Step'] >= min_step_to_plot) &
#             (df['Step'] <= max_step) &
#             (df['Step'] % step_interval == 0)
#         ].copy()

#         if not processed_df.empty:
#             entry = {
#                 'steps': processed_df['Step'],
#                 'values': processed_df[info['value_col']],
#                 'color': info['color'],
#                 'linestyle': info['linestyle'],
#                 'linewidth': info.get('linewidth', 2.5),
#                 'marker': info.get('marker', '.'),
#                 'markersize': info.get('markersize', 6)
#             }
#             data_to_plot[label] = entry
#         else:
#             print(f"Warning: No data to plot for {label} after filtering (Steps {min_step_to_plot}-{max_step}).")
#     except FileNotFoundError:
#         print(f"Error: File not found - {info['path']}")
#     except KeyError:
#         cols_in_file = []
#         try:
#             cols_in_file = pd.read_csv(info['path'], nrows=1).columns.tolist()
#         except: pass
#         print(f"Error: Column not found in {info['path']}. Expected 'Step' and '{info['value_col']}'. Available cols: {cols_in_file}")
#     except Exception as e:
#         print(f"An error occurred processing {info['path']}: {e}")

# # --- Create the Plot with Broken Y-Axis ---
# # --- Create the Plot with Broken Y-Axis ---
# if len(data_to_plot) == 2: # Ensure we have both datasets to plot
#     # 收集所有数据点的值用于确定y轴范围
#     all_values_list = [val for data_dict in data_to_plot.values() for val in data_dict['values']]
#     if not all_values_list:
#         print("Error: No values to plot after processing data.")
#         exit()
        
#     fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 8)) # Original figure size
    
#     # --- KEY CHANGE: Adjust hspace to be very small or zero ---
#     fig.subplots_adjust(hspace=0.01)  # Reduced hspace significantly

#     # ... (Data plotting on ax_top and ax_bottom remains the same) ...
#     for label, data in data_to_plot.items():
#         ax_top.plot(data['steps'], data['values'], label=label, color=data['color'],
#                     linestyle=data['linestyle'], linewidth=data['linewidth'],
#                     marker=data['marker'], markersize=data['markersize'])
#         ax_bottom.plot(data['steps'], data['values'], label=label, color=data['color'], 
#                        linestyle=data['linestyle'], linewidth=data['linewidth'],
#                        marker=data['marker'], markersize=data['markersize'])

#     min_val_overall = min(all_values_list) # Assuming all_values_list is defined correctly
#     max_val_overall = max(all_values_list)

#     y_break_bottom_limit = 160
#     y_break_top_limit = 220   

#     ax_top.set_ylim(bottom=y_break_top_limit, top=max_val_overall + 10) 
#     ax_bottom.set_ylim(bottom=min_val_overall - 10, top=y_break_bottom_limit)

#     # Style the broken axis - Ensure top x-axis is fully minimized
#     ax_top.spines['bottom'].set_visible(False)
#     ax_bottom.spines['top'].set_visible(False)
    
#     # Remove ax_top.xaxis.tick_top() if it adds unwanted space or ticks
#     # ax_top.xaxis.tick_top() # You had this, it might not be necessary
#     ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # More thorough hiding
#     ax_bottom.xaxis.tick_bottom()

#     # Add break marks (your current shorter version is good)
#     d = .015  # Adjusted 'd' slightly for visibility if needed, can keep as 0.01
#     kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
#     ax_top.plot((-d, +d), (-d/1.5, +d/1.5), **kwargs) # Adjusted y-span of break mark slightly
#     ax_top.plot((1 - d, 1 + d), (-d/1.5, +d/1.5), **kwargs)

#     kwargs.update(transform=ax_bottom.transAxes)
#     ax_bottom.plot((-d, +d), (1 - d/1.5, 1 + d/1.5), **kwargs) # Adjusted y-span of break mark
#     ax_bottom.plot((1 - d, 1 + d), (1 - d/1.5, 1 + d/1.5), **kwargs)

#     # ... (Rest of your axis labels, ticks, grid, legend setup) ...
#     ax_bottom.set_xlabel('Training Step')
#     fig.text(0.06, 0.5, 'Average Response Length', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

#     ax_bottom.set_xlim(0, max_step + step_interval)
#     ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(100))
#     ax_bottom.minorticks_on()
#     # ax_top.minorticks_on() # Top x-axis ticks are hidden, so this might not be needed

#     ax_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='lower')) 
#     ax_bottom.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))

#     ax_top.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)
#     ax_bottom.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)

#     handles_top, labels_top = ax_top.get_legend_handles_labels()
#     # if handles_top:
#     #     ax_top.legend(handles_top, labels_top, loc='upper right', ncol=1) 
#     # else:
#     #     print("Warning: No lines were plotted for the top axis, legend will not be shown.")

#     # fig.tight_layout(rect=[0.07, 0, 1, 0.93]) # tight_layout can sometimes override subplots_adjust
#                                                 # Call subplots_adjust AFTER tight_layout or remove rect
#     # plt.suptitle('Average Response Length: With vs. Without Dynamic Reward', y=0.96, fontsize=plt.rcParams['figure.titlesize'])
#     fig.tight_layout(rect=[0.08, 0.05, 1, 0.92]) # Adjust rect to ensure labels fit
#     fig.subplots_adjust(hspace=0.005) # Force hspace AFTER tight_layout as a final adjustment
    
#     try:
#         plt.savefig("response_length_comparison_broken_axis.pdf", bbox_inches='tight')
#         print("Plot saved as response_length_comparison_broken_axis.pdf")
#     except Exception as e:
#         print(f"Error saving plot to PDF: {e}")
#     plt.show()

# else:
#     print("Error: Not enough data was successfully processed for comparison. Cannot generate plot.")






# if len(data_to_plot) == 2: # Ensure we have both datasets to plot
#     fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 7)) # Increased figure size
#     fig.subplots_adjust(hspace=0.005)  # Adjust space between axes

#     # Determine overall min/max for y-axis settings (before break)
#     all_values_list = [val for data_dict in data_to_plot.values() for val in data_dict['values']]
#     if not all_values_list:
#         print("Error: No values to plot after processing data.")
#         plt.close(fig) # Close the empty figure
#         exit()

#     # Define y-limits for the break
#     # User specified break around 150-230
#     y_break_bottom_limit = 160 # Upper limit for bottom y-axis part
#     y_break_top_limit = 220    # Lower limit for top y-axis part

#     # Plot data on both subplots
#     for label, data in data_to_plot.items():
#         ax_top.plot(data['steps'], data['values'], label=label, color=data['color'],
#                     linestyle=data['linestyle'], linewidth=data['linewidth'],
#                     marker=data['marker'], markersize=data['markersize'])
#         ax_bottom.plot(data['steps'], data['values'], label=label, color=data['color'], # Plot again for bottom axis
#                        linestyle=data['linestyle'], linewidth=data['linewidth'],
#                        marker=data['marker'], markersize=data['markersize'])

#     # Determine robust y-limits based on data, considering the break
#     min_val_overall = min(all_values_list)
#     max_val_overall = max(all_values_list)

#     ax_top.set_ylim(bottom=y_break_top_limit, top=max_val_overall + 10) # Top part: from 230 up
#     ax_bottom.set_ylim(bottom=min_val_overall - 10, top=y_break_bottom_limit) # Bottom part: up to 150

#     # Style the broken axis
#     ax_top.spines['bottom'].set_visible(False)
#     ax_bottom.spines['top'].set_visible(False)
#     ax_top.xaxis.tick_top() # This is usually not needed for broken y-axis plots
#     ax_top.tick_params(labeltop=False)  # Hide tick labels at the top of the top plot
#     ax_bottom.xaxis.tick_bottom()

#     # Add break marks (diagonal lines)
#     d = .01  # Size of diagonal lines in axes coordinates
#     kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
#     ax_top.plot((-d, +d), (-d/2, +d/2), **kwargs)        # 减小垂直方向上的距离
#     ax_top.plot((1 - d, 1 + d), (-d/2, +d/2), **kwargs)

#     kwargs.update(transform=ax_bottom.transAxes)
#     ax_bottom.plot((-d, +d), (1 - d/2, 1 + d/2), **kwargs)
#     ax_bottom.plot((1 - d, 1 + d), (1 - d/2, 1 + d/2), **kwargs)
#     # kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
#     # ax_top.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal on top-axes
#     # ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal on top-axes
#     # kwargs.update(transform=ax_bottom.transAxes)    # Switch to bottom axes
#     # ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal on bottom-axes
#     # ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # Bottom-right diagonal on bottom-axes

#     # --- Axis and Tick Configuration ---
#     ax_bottom.set_xlabel('Training Step')
#     fig.text(0.06, 0.5, 'Average Response Length', va='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

#     ax_bottom.set_xlim(0, max_step + step_interval)
#     ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(100))
#     ax_bottom.minorticks_on()
#     ax_top.minorticks_on() # Enable for top x-axis too if desired, though labels are off

#     # Set y-ticks for clarity, especially around the break
#     ax_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='lower')) # Prune lower to avoid crowding near break
#     ax_bottom.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper')) # Prune upper

#     # --- Grid and Legend ---
#     ax_top.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)
#     ax_bottom.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)

#     handles_top, labels_top = ax_top.get_legend_handles_labels()
#     # # Since data is plotted on both, legend can be taken from one, e.g., ax_top
#     # if handles_top:
#     #     ax_top.legend(handles_top, labels_top, loc='upper right', ncol=1) # Legend in top-right of top plot
#     # else:
#     #     print("Warning: No lines were plotted for the top axis, legend will not be shown.")

#     # plt.suptitle('Average Response Length: With vs. Without Dynamic Reward', y=0.96) # Use suptitle for overall title
#     fig.tight_layout(rect=[0.07, 0, 1, 0.93]) # Adjust rect to make space for suptitle and centered y-label

#     try:
#         plt.savefig("response_length_comparison_broken_axis.pdf", bbox_inches='tight')
#         print("Plot saved as response_length_comparison_broken_axis.pdf")
#     except Exception as e:
#         print(f"Error saving plot to PDF: {e}")
#     plt.show()

# else:
#     print("Error: Not enough data was successfully processed for comparison. Cannot generate plot.")