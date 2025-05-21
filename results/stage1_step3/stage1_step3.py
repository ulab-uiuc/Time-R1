import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Plotting Style Configuration (Reusing from reference code) ---
try:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Prioritize Helvetica
        'font.size': 15,  # Base font size
        'axes.labelsize': 18, # Axis labels
        'axes.titlesize': 22, # Plot title (if used)
        'xtick.labelsize': 14, # X-axis tick labels
        'ytick.labelsize': 15, # Y-axis tick labels
        'legend.fontsize': 13, # Legend font size
        'figure.titlesize': 24 # Figure title (if used)
    })
except Exception as e:
    print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# --- File Information and Styling (CORRECTED value_col entries) ---
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

# CORRECTED value_col names based on error messages
files_info = {
    r'Overall $R_{\mathrm{acc}}$': {
        'path': 'overall_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/pred_reward_overall', # Corrected name
        'type': 'accuracy',
        'color': task_base_colors['Overall'],
        'linestyle': '-',
        'linewidth': 3, # Thicker for Overall
        'marker': 'o',
        'markersize': 2
    },
    r'Completion $R_{\mathrm{acc}}$': {
        'path': 'completion_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/pred_reward_time_completion', # Corrected name
        'type': 'accuracy',
        'color': task_base_colors['Completion'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': '^',
        'markersize': 2
    },
    r'Difference $R_{\mathrm{acc}}$': {
        'path': 'difference_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/pred_reward_time_difference', # Corrected name
        'type': 'accuracy',
        'color': task_base_colors['Difference'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 's',
        'markersize': 2
    },
    r'Inferring $R_{\mathrm{acc}}$': {
        'path': 'inferring_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/pred_reward_time_inferring', # Corrected name
        'type': 'accuracy',
        'color': task_base_colors['Inferring'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'D',
        'markersize': 2
    },
    r'Ordering $R_{\mathrm{acc}}$': {
        'path': 'ordering_acc.csv',
        'value_col': 'time_reasoning/combined_tasks_from_dynamic_alpha_increasing - val/pred_reward_time_ordering', # Corrected name
        'type': 'accuracy',
        'color': task_base_colors['Ordering'],
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'P',
        'markersize': 2
    }
}

# --- Plotting Parameters ---
max_step = 880
step_interval = 10
min_step_to_plot = 10 # Start plotting from step 10

data_to_plot = {}

# --- Data Loading and Processing ---
for label, info in files_info.items():
    try:
        df = pd.read_csv(info['path'])
        # Ensure required columns are numeric and handle potential errors
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df[info['value_col']] = pd.to_numeric(df[info['value_col']], errors='coerce')
        # Drop rows where conversion failed or values are missing
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
                'original_values': processed_df[info['value_col']],
                'type': info['type'], # Keep type for potential future use
                'color': info['color'],
                'linestyle': info['linestyle'],
                'linewidth': info.get('linewidth', 2), # Use specified or default linewidth
                'marker': info.get('marker', '.'),
                'markersize': info.get('markersize', 6)
            }
            data_to_plot[label] = entry
        else:
            print(f"Warning: No data to plot for {label} after filtering (Steps {min_step_to_plot}-{max_step}).")

    except FileNotFoundError:
        print(f"Error: File not found - {info['path']}")
    except KeyError:
        # Try to provide more specific feedback about which column is missing
        cols_in_file = []
        try:
            cols_in_file = pd.read_csv(info['path'], nrows=1).columns.tolist()
        except: # Handle cases where reading even one row fails
            pass
        print(f"Error: Column not found in {info['path']}. Expected 'Step' and '{info['value_col']}'. Available columns: {cols_in_file}")
    except Exception as e:
        print(f"An error occurred processing {info['path']}: {e}")

# --- Create the Plot ---
if not data_to_plot:
    print("Error: No data was successfully processed. Cannot generate plot.")
else:
    fig, ax1 = plt.subplots(figsize=(5.1, 7)) # Adjusted figure size for readability

    # Plotting the accuracy data
    for label, data in data_to_plot.items():
        ax1.plot(data['steps'], data['original_values'], label=label, color=data['color'],
                 linestyle=data['linestyle'], linewidth=data['linewidth'],
                 marker=data['marker'], markersize=data['markersize'])

    # --- Axis Configuration ---
    ax1.set_xlabel('Training Step (Phase 3)')
    ax1.set_ylabel(r'Accuracy Score $R_{\mathrm{acc}}$')
    ax1.tick_params(axis='y')
    ax1.set_xlim(0, max_step + step_interval) # Set x-axis limit from 0 to 890
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100)) # Ticks every 100 steps
    ax1.minorticks_on()

    # --- Grid and Legend ---
    # Add both horizontal and vertical grid lines for the single y-axis plot
    ax1.grid(True, axis='both', linestyle=':', alpha=0.6, zorder=0)

    # handles, labels = ax1.get_legend_handles_labels()
    # if handles:
    #     # Place legend in the bottom right corner inside the plot
    #     ax1.legend(handles, labels, loc='lower right', ncol=1)
    # else:
    #     print("Warning: No lines were plotted, legend will not be shown.")

    # No title for the plot
    fig.tight_layout() # Adjust layout
    
    # --- Save and Show ---
    try:
        plt.savefig("accuracy_plot_step880.pdf", bbox_inches='tight')
        print("Plot saved as accuracy_plot_step880.pdf")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
        
    plt.show()