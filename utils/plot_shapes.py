import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

def load_and_aggregate_logs(path):
    aggregated_logs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'results.json':
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    results = json.load(f)
                    logs = {'shape': os.path.basename(root), 'correct': results['P1 Wins'], 'incorrect': results['P2 Wins']}
                    aggregated_logs.append(logs)
    return aggregated_logs

def calculate_counts(logs, shapes):
    shape_data = {shape: {'correct': 0, 'incorrect': 0, 'total': 0} for shape in shapes}
    for log in logs:
        shape = log['shape']
        shape_data[shape]['correct'] += log['correct']
        shape_data[shape]['incorrect'] += log['incorrect']
        shape_data[shape]['total'] += log['correct'] + log['incorrect']
    return shape_data

def bar_plot_shapes(base_path, models, temperatures, shapes):
    plt.rcParams.update({'font.size': 20, 'font.weight': 'bold'})
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        fig.suptitle(f'', fontsize=16, fontweight='bold')

        axes = axes.flatten()
        for idx, temp in enumerate(temperatures):
            path = f'{base_path}/{model.replace(":", "_")}/{str(temp).replace(".", "_")}'
            all_logs = []

            for shape in shapes:
                shape_path = os.path.join(path, shape)
                shape_logs = load_and_aggregate_logs(shape_path)
                all_logs.extend(shape_logs)

            shape_counts = calculate_counts(all_logs, shapes)

            data = []
            for shape, counts in shape_counts.items():
                data.append({'Shape': shape, 'Count': counts['correct'], 'Type': 'Correct'})
                data.append({'Shape': shape, 'Count': counts['incorrect'], 'Type': 'Incorrect'})

            df = pd.DataFrame(data)
            bar_plot = sns.barplot(x='Shape', y='Count', hue='Type', data=df, ax=axes[idx],
                                   palette=['green', 'red'], alpha=0.75, dodge=0.4)

            for p in bar_plot.patches:
                bar_plot.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., 0),
                                  ha='center', va='bottom', textcoords='offset points', color='black', fontweight='bold', fontsize=20, xytext=(0, 10))

            axes[idx].set_title(f'Temperature {temp}', fontweight='bold')
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].set_ylim(0, max(df['Count']) + 5)  # Add some space for annotation

            if idx == len(temperatures) - 1:
                axes[idx].legend(title='Answer Type', title_fontsize='13', loc='upper right')
            else:
                axes[idx].get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def main():
    models = ['oa:gpt-3.5-turbo-1106', 'oa:gpt-4-1106-preview']
    temperatures = [0, 0.5, 1, 1.5]
    shapes = ['square', 'triangle', 'cross']

    base_path = '../experiment_shapes'
    bar_plot_shapes(base_path, models, temperatures, shapes)

if __name__ == "__main__":
    main()
