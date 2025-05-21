import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 1. Data Preparation (from your provided results)
# Ensure data matches the latest results exactly
data = {
    'Month': ["24-07", "24-08", "24-09", "24-10", "24-11", "24-12", "25-01", "25-02"],
    # Baselines (Order based on previous table, V3/R1 last among baselines)
    'Qwen2.5-3B (Base)': [0.4731, 0.4727, 0.4689, 0.4739, 0.4857, 0.4877, 0.4776, 0.4694],
    'Qwen2.5-7B': [0.4775, 0.4699, 0.4978, 0.4618, 0.4853, 0.4691, 0.4888, 0.4583],
    'Llama-3.1-8B': [0.4756, 0.4899, 0.5003, 0.4742, 0.4621, 0.4706, 0.4801, 0.4803],
    'DS-Distill-Qwen-32B': [0.4801, 0.4658, 0.4678, 0.4794, 0.4704, 0.4840, 0.4730, 0.4581],
    'DeepSeek-V3-671B': [0.4991, 0.5073, 0.5177, 0.4860, 0.4846, 0.4752, 0.4771, 0.4685],
    'DeepSeek-R1-671B': [0.4743, 0.4755, 0.4964, 0.4729, 0.4529, 0.4785, 0.4730, 0.4731],
    # Your Model
    'Time-R1 (Ours, 3B)': [0.4977, 0.4779, 0.5025, 0.4991, 0.5241, 0.5081, 0.4830, 0.4995]
}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Reorder columns to match desired legend/plotting order if needed (optional)
# Example: put Time-R1 first for plotting/legend emphasis
cols = ['Time-R1 (Ours, 3B)'] + [col for col in df if col != 'Time-R1 (Ours, 3B)']
df = df[cols]


# 2. Plotting
plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
plt.figure(figsize=(11, 6)) # Adjust figure size for better readability

# Plot each model as a line
markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', '*'] # Different markers
linestyles = ['-', '--', '-.', ':', '--', '-.', ':', '-'] # Different linestyles

for i, column in enumerate(df.columns):
    style_idx = i % len(markers) # Cycle through styles
    if "Time-R1" in column:
        # Highlight Time-R1
        plt.plot(df.index, df[column], 
                 marker=markers[style_idx], 
                 linestyle='-', 
                 linewidth=2.5, # Thicker line
                 label=column, 
                 zorder=10) # Bring to front
    else:
        # Plot baselines normally
        plt.plot(df.index, df[column], 
                 marker=markers[style_idx], 
                 linestyle='--', # Dashed lines for baselines
                 linewidth=1.5, 
                 label=column, 
                 alpha=0.8) # Slight transparency

# 3. Customization
plt.title('Monthly AvgMaxSim Scores for Future Scenario Generation', fontsize=15, pad=15)
plt.xlabel('Month (YY-MM)', fontsize=12)
plt.ylabel('AvgMaxSim Score (Higher is Better)', fontsize=12)

# Improve y-axis formatting for clarity
plt.ylim(0.44, 0.54) # Adjust based on data range
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f')) # Format y-axis ticks

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=9.5, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Place legend outside plot
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust layout to make space for legend (right margin)

# 4. Save and Show plot
plt.savefig("monthly_avgmaxsim_scores.pdf", bbox_inches='tight') # Add this line to save as PDF
plt.show()