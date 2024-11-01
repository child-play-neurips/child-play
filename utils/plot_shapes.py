import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def load_results(path):
    """
    Load results from results_shapes.json and aggregate wins and losses.

    Args:
        path (str): Directory path containing results_shapes.json files.

    Returns:
        dict: Dictionary with 'Wins' and 'Losses' counts.
    """
    wins = 0
    losses = 0
    results_file = os.path.join(path, 'results_shapes.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            wins += results.get('Wins', 0)
            losses += results.get('Losses', 0)
    return {'Wins': wins, 'Losses': losses}

def bar_plot_shapes(base_path, models, temperatures, shapes):
    # Update global font size and weight
    plt.rcParams.update({'font.size': 20, 'font.weight': 'bold'})
    
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        # fig.suptitle(f'Wins and Losses by Shape for {model}', fontsize=24, fontweight='bold')

        axes = axes.flatten()
        for idx, temp in enumerate(temperatures):
            data = []
            for shape in shapes:
                path = os.path.join(base_path, model.replace(":", "_"), str(temp).replace(".", "_"), shape)
                counts = load_results(path)
                data.append({'Shape': shape, 'Count': counts['Wins'], 'Type': 'Wins'})
                data.append({'Shape': shape, 'Count': counts['Losses'], 'Type': 'Losses'})

            df = pd.DataFrame(data)
            bar_plot = sns.barplot(
                x='Shape', y='Count', hue='Type', data=df, ax=axes[idx],
                palette=['green', 'red'], alpha=0.75, dodge=0.4
            )

            # Annotate each bar at the bottom just above the x-axis
            for p in bar_plot.patches:
                height = p.get_height()
                if height > 0:  # Only annotate non-zero bars
                    bar_plot.annotate(
                        f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., 0),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        color='black', fontweight='bold', fontsize=20
                    )

            axes[idx].set_title(f'Temperature {temp}', fontweight='bold', fontsize=25)
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].set_ylim(0, df['Count'].max() + 10)  # Add some space for annotation

            if idx == len(temperatures) - 1:
                axes[idx].legend(title='Result', title_fontsize='13', loc='upper right')
            else:
                axes[idx].get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Ensure the output directory exists
        output_dir = os.path.join(base_path, model.replace(":", "_"))
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'answers_summary_{model.replace(":", "_")}.pdf'))
        plt.close()

def main():
    models = ['oa_gpt-4-1106-preview', 'oa_gpt-3.5-turbo-1106', 'oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']
    temperatures = [0, 0.5, 1, 1.5]
    shapes = ['square', 'triangle', 'cross']

    base_path = '../experiment_shapes'  # Adjust the base path as needed
    bar_plot_shapes(base_path, models, temperatures, shapes)
    print("Bar plots generated for shapes experiments.")

if __name__ == "__main__":
    main()
