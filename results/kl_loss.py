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
        'legend.fontsize': 18,
        'figure.titlesize': 24
    })
except Exception as e:
    print(f"Could not set preferred fonts, will use Matplotlib defaults. Error: {e}")

# --- File Information and Styling ---
csv_file_path = '/data/zliu331/temporal_reasoning/TinyZero/results/kl_loss-response_length.csv'

muted_colors = {
    'blue': '#89CFF0',   # Muted Blue
    'red': '#FAD7A0',    # Muted Red
}

series_info = {
    'KL Coeff: 0.001': {
        'value_col': 'news_inference/train_easy_2e-6 - response_length/mean',
        'color': muted_colors['blue'],
        'linestyle': '-',
        'marker': 'o',
        'markersize': 6,
        'linewidth': 2.5
    },
    'KL Coeff: 0.0001': {
        'value_col': 'news_inference/train_easy_kl_1e-4 - response_length/mean',
        'color': muted_colors['red'],
        'linestyle': '--',
        'marker': 's',
        'markersize': 6,
        'linewidth': 2.5
    }
}

# --- Plotting Parameters ---
min_step_to_plot = 5
max_step_to_plot = 140
plot_point_interval = 5 # Plot a point every 5 steps

data_to_plot = {}

# --- Data Loading and Processing ---
try:
    df_full = pd.read_csv(csv_file_path)
    df_full['Step'] = pd.to_numeric(df_full['Step'], errors='coerce')

    for label, info in series_info.items():
        # Create a copy for each series to avoid SettingWithCopyWarning if modifying df_full directly in loop
        df_series = df_full[['Step', info['value_col']]].copy()
        df_series[info['value_col']] = pd.to_numeric(df_series[info['value_col']], errors='coerce')
        df_series.dropna(subset=['Step', info['value_col']], inplace=True)

        # Filter data based on step range and interval for plotting points
        processed_df = df_series[
            (df_series['Step'] >= min_step_to_plot) &
            (df_series['Step'] <= max_step_to_plot) &
            (df_series['Step'] % plot_point_interval == 0) # Select points every 5 steps
        ].copy()

        if not processed_df.empty:
            entry = {
                'steps': processed_df['Step'],
                'values': processed_df[info['value_col']],
                'color': info['color'],
                'linestyle': info['linestyle'],
                'linewidth': info.get('linewidth', 2.5),
                'marker': info.get('marker', 'o'),
                'markersize': info.get('markersize', 6)
            }
            data_to_plot[label] = entry
        else:
            print(f"Warning: No data to plot for {label} after filtering (Steps {min_step_to_plot}-{max_step_to_plot}, interval {plot_point_interval}).")

except FileNotFoundError:
    print(f"Error: File not found - {csv_file_path}")
except KeyError as e:
    cols_in_file = []
    try:
        cols_in_file = pd.read_csv(csv_file_path, nrows=1).columns.tolist()
    except: pass
    print(f"Error: Column not found. Expected 'Step' and relevant value columns. Error: {e}. Available columns: {cols_in_file}")
except Exception as e:
    print(f"An error occurred processing {csv_file_path}: {e}")

# --- Create the Plot ---
if not data_to_plot or len(data_to_plot) < len(series_info):
    print("Error: Not enough data successfully processed for all series. Cannot generate plot.")
else:
    fig, ax1 = plt.subplots(figsize=(9, 5.3)) # Reusing figure size

    for label, data in data_to_plot.items():
        ax1.plot(data['steps'], data['values'], label=label, color=data['color'],
                 linestyle=data['linestyle'], linewidth=data['linewidth'],
                 marker=data['marker'], markersize=data['markersize'])

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Average Response Length')
    ax1.tick_params(axis='both', which='major')
    
    ax1.set_xlim(min_step_to_plot - plot_point_interval / 2, max_step_to_plot + plot_point_interval / 2)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20)) # Ticks every 20 steps for the range 5-140
    ax1.minorticks_on()
    
    all_values = [val for data_dict in data_to_plot.values() for val in data_dict['values']]
    if all_values:
        min_y = min(all_values)
        max_y = max(all_values)
        padding = (max_y - min_y) * 0.05 # 5% padding
        ax1.set_ylim(bottom=min_y - padding, top=max_y + padding)

    ax1.grid(True, axis='both', linestyle=':', alpha=0.7, zorder=0)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='upper right') # Adjusted legend location
    else:
        print("Warning: No lines were plotted, legend will not be shown.")

    plt.title('Impact of KL Coefficient on Response Length', fontsize=20)
    fig.tight_layout()
    
    output_filename = "kl_coeff_response_length_comparison.pdf"
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved as {output_filename}")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
        
    plt.show()