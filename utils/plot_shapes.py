import random
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

def load_and_aggregate_logs(path):
    aggregated_logs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'game_logs.json':
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    logs = json.load(f)
                    aggregated_logs.extend(logs)
    return aggregated_logs

def calculate_proportions(logs, shapes):
    proportions = {shape: [] for shape in shapes}
    counts = {shape: {'correct': 0, 'total': 0} for shape in shapes}
    for log in logs:
        correct_shape = log['correct_shape']
        chosen_shape = log['chosen_shape']
        for shape in shapes:
            if correct_shape == shape:
                counts[shape]['total'] += 1
                if correct_shape == chosen_shape:
                    counts[shape]['correct'] += 1
                    proportions[shape].append(1)
                else:
                    proportions[shape].append(0)
    return proportions, counts

def calculate_std(count, total):
    if total == 0:
        return 0
    p = count / total
    return np.sqrt(p * (1 - p) / total)

def bar_plot_shapes(base_path, models, temperatures, shapes):
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
    for model in models:
        # Setup figure for the model with subplots for each temperature, 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        fig.suptitle(f'Correct and Incorrect Answers by Shape for {model}', fontsize=16, fontweight='bold')

        axes = axes.flatten()  # Flatten the 2x2 grid to easily index it
        for idx, temp in enumerate(temperatures):
            all_counts = {shape: {'correct': 0, 'total': 0} for shape in shapes}
            
            # Construct the path for the current model and temperature
            path = f'{base_path}/{model.replace(":", "_")}/{str(temp).replace(".", "_")}'
            
            for shape in shapes:
                shape_path = os.path.join(path, shape)
                log_files = [f for f in os.listdir(shape_path) if f.endswith('game_logs.json')]
                
                for log_file in log_files:
                    # Load the log file
                    with open(os.path.join(shape_path, log_file), 'r') as file:
                        logs = json.load(file)
                        _, counts = calculate_proportions(logs, shapes)
                        
                        for shape in shapes:
                            all_counts[shape]['correct'] += counts[shape]['correct']
                            all_counts[shape]['total'] += counts[shape]['total']
            
            # Calculate proportions and standard deviations
            mean_proportions = {shape: all_counts[shape]['correct'] / all_counts[shape]['total'] if all_counts[shape]['total'] > 0 else 0 for shape in shapes}
            std_proportions = {shape: calculate_std(all_counts[shape]['correct'], all_counts[shape]['total']) for shape in shapes}

            # Prepare data for plotting
            data = []
            for shape in shapes:
                data.append({
                    'Shape': shape, 
                    'Proportion': mean_proportions[shape], 
                    'Type': 'Correct', 
                    'Std': std_proportions[shape],
                    'Count': all_counts[shape]['correct']
                })
                data.append({
                    'Shape': shape, 
                    'Proportion': 1 - mean_proportions[shape], 
                    'Type': 'Incorrect', 
                    'Std': std_proportions[shape],
                    'Count': all_counts[shape]['total'] - all_counts[shape]['correct']
                })
            
            df = pd.DataFrame(data)
            # Plotting with adjusted bar positions for slight overlap
            bar_plot = sns.barplot(x='Shape', y='Proportion', hue='Type', data=df, ax=axes[idx],
                                   palette=['green', 'red'], alpha=0.75, dodge=0.4, errorbar=None)  # Adjust dodge
            
            # Adding error bars and raw numbers
            for i, p in enumerate(bar_plot.patches):
                height = p.get_height()
                std = df.iloc[i]['Std']
                count = df.iloc[i]['Count']
                if height > 0:
                    bar_plot.errorbar(p.get_x() + p.get_width() / 2., height, yerr=std, fmt='none', c='black', capsize=5)
                    bar_plot.annotate(f'{int(count)}', 
                                      (p.get_x() + p.get_width() / 2., 0),
                                      ha='center', va='bottom', 
                                      xytext=(0, 0), textcoords='offset points', fontweight='bold')
            
            axes[idx].set_title(f'Temperature {temp}', fontweight='bold')
            axes[idx].set_xlabel('Shape' if idx // 2 == 1 else '', fontweight='bold')
            axes[idx].set_ylabel('Proportion' if idx % 2 == 0 else '', fontweight='bold')
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure y-axis ticks are integers
            axes[idx].set_ylim(0, 1)  # Proportions are between 0 and 1

            if idx == len(temperatures) - 1:
                axes[idx].legend(title='Answer Type', title_fontsize='13', loc='upper right')
            else:
                axes[idx].get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        # plt.savefig(f'{base_path}/{model.replace(":", "_")}/answers_summary_{model}.png')
        # plt.close()

def main():
    shapes = ['square', 'triangle', 'cross']
    models = ['oa:gpt-3.5-turbo-1106', 'oa:gpt-4-1106-preview']
    temperatures = [0, 0.5, 1, 1.5]

    base_path = '../experiment_shapes'
    for model in models:
        for temp in temperatures:
            all_moves = []  # Initialize here to collect all moves across shapes
            base_path_model_temp = f"{base_path}/{model.replace(':', '_')}/{str(temp).replace('.', '_')}"
            
            for shape in shapes:
                shape_path = f"{base_path_model_temp}/{shape}"
                shape_moves = load_and_aggregate_logs(shape_path)
                all_moves.extend(shape_moves)

    bar_plot_shapes(base_path, models, temperatures, shapes)

if __name__ == "__main__":
    main()
