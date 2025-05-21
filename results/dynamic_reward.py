import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Plotting Style Configuration (Reusing from your reference code) ---
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

# --- File Information and Styling ---
# Muted colors (can choose any two distinct ones)
muted_colors = {
    'blue': '#89CFF0',   # Muted Blue
    'green': '#90EE90',  # Muted Green
    'red': '#F08080',    # Muted Red
}

files_info = {
    'With Dynamic Reward': {
        'path': 'phase 3.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/test_score_time_difference',
        'color': muted_colors['blue'],
        'linestyle': '-',
        'marker': 'o',
        'markersize': 6,
        'linewidth': 2.5
    },
    'Without Dynamic Reward': {
        'path': 'no dynamic reward.csv',
        'value_col': 'time_reasoning/combined_tasks_from_inferring_easy_alpha_0.1 - val/test_score_time_difference',
        'color': muted_colors['red'],
        'linestyle': '--',
        'marker': 's',
        'markersize': 6,
        'linewidth': 2.5
    }
}

# --- Plotting Parameters ---
# Determine max_step based on the shorter dataset for fair comparison
# try:
#     df_phase3 = pd.read_csv('phase 3.csv')
#     df_nodynamic = pd.read_csv('no dynamic reward.csv')
#     max_step_phase3 = pd.to_numeric(df_phase3['Step'], errors='coerce').max()
#     max_step_nodynamic = pd.to_numeric(df_nodynamic['Step'], errors='coerce').max()
    
#     # Use the smaller of the two maximums if both are valid, otherwise adjust
#     if pd.notna(max_step_phase3) and pd.notna(max_step_nodynamic):
#         max_step = min(max_step_phase3, max_step_nodynamic, 840) # Cap at 840 as no_dynamic_reward ends there
#     elif pd.notna(max_step_nodynamic):
#         max_step = min(max_step_nodynamic, 840)
#     elif pd.notna(max_step_phase3): # Should not happen if we want to compare
#         max_step = min(max_step_phase3, 840)
#         print("Warning: 'no dynamic reward.csv' might be shorter or missing. Plotting 'phase 3.csv' up to its available steps or 840.")
#     else:
#         max_step = 840 # Default if max steps can't be determined
#         print("Warning: Could not determine max steps from CSVs. Defaulting to 840.")
    
#     # Ensure max_step is a multiple of 10 if possible, or adjust for clarity.
#     # Given data ends at 840 for one file, this is fine.
# except Exception as e:
#     print(f"Error reading CSVs to determine max_step: {e}. Defaulting max_step to 840.")
#     max_step = 840

# min_step_to_plot = 50  
max_step = 630

step_interval = 10
min_step_to_plot = 50 # Start plotting from step 0

data_to_plot = {}

# --- Data Loading and Processing ---
for label, info in files_info.items():
    try:
        df = pd.read_csv(info['path'])
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
        df.dropna(subset=['Step', info['value_col']], inplace=True)

        # Filter data based on step range and interval
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
                'linewidth': info.get('linewidth', 2),
                'marker': info.get('marker', '.'),
                'markersize': info.get('markersize', 6)
            }
            data_to_plot[label] = entry
        else:
            print(f"Warning: No data to plot for {label} after filtering (Steps {min_step_to_plot}-{max_step}).")
    except FileNotFoundError:
        print(f"Error: File not found - {info['path']}")
    except KeyError:
        cols_in_file = []
        try:
            cols_in_file = pd.read_csv(info['path'], nrows=1).columns.tolist()
        except: pass
        print(f"Error: Column not found in {info['path']}. Expected 'Step' and '{info['value_col']}'. Available columns: {cols_in_file}")
    except Exception as e:
        print(f"An error occurred processing {info['path']}: {e}")

# --- Create the Plot ---
if not data_to_plot or len(data_to_plot) < 2: # Ensure we have data for comparison
    print("Error: Not enough data successfully processed for comparison. Cannot generate plot.")
else:
    fig, ax1 = plt.subplots(figsize=(9, 5.3)) # Adjusted figure size

    for label, data in data_to_plot.items():
        ax1.plot(data['steps'], data['values'], label=label, color=data['color'],
                 linestyle=data['linestyle'], linewidth=data['linewidth'],
                 marker=data['marker'], markersize=data['markersize'])

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(r'Total Score $R(x, y)$') # Specific Y-axis label
    ax1.tick_params(axis='both', which='major') # Apply tick params to both axes if needed
    
    # Adjust x-axis to show from 0 up to max_step clearly
    # ax1.set_xlim(-step_interval/2, max_step + step_interval/2) # Start slightly before 0 for padding
    ax1.set_xlim(min_step_to_plot - step_interval/2, max_step + step_interval/2)  # min_step_to_plot
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100)) # Ticks every 100 steps
    ax1.minorticks_on()
    
    # Set Y-axis limits if needed, e.g., based on data range or fixed values
    all_values = [val for data_dict in data_to_plot.values() for val in data_dict['values']]
    if all_values:
        min_y = min(all_values) * 0.98
        max_y = max(all_values) * 1.015
        # Ensure y-axis starts from a reasonable point if all values are high
        # For scores typically between 0 and 1, this might be fine.
        # ax1.set_ylim(max(0, min_y), min(1, max_y) if max_y > 0 else 1) # Example: cap at 1 if scores are probabilities
        # ax1.set_ylim(bottom=max(0, min_y - 0.05*(max_y-min_y)), top=min_y + 1.05*(max_y-min_y) if max_y > min_y else min_y + 0.1)
        ax1.set_ylim(bottom=0.363, top=0.427)

    ax1.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='lower right') # 'best' or 'lower right' often works well
    else:
        print("Warning: No lines were plotted, legend will not be shown.")

    # plt.title('Time-Difference Estimation: Dynamic vs. No Dynamic Reward', fontsize=20) # Added a title
    fig.tight_layout()
    
    try:
        plt.savefig("dynamic_reward_comparison_timediff.pdf", bbox_inches='tight')
        print("Plot saved as dynamic_reward_comparison_timediff.pdf")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
        
    plt.show()