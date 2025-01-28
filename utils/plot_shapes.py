import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_proportion_for_temp(model, shape, temp, base_path, group):
    """
    Returns the fraction of correct answers (0..1) for a given model, shape, and temperature.
    If the path doesn't exist or no file is found, returns 0.
    """
    total = 25  # Each shape has 25 trials per temperature
    correct = 0

    # Path to shape folder
    shape_path = os.path.join(
        base_path,
        model.replace(":", "_"),  # just in case
        str(temp).replace(".", "_"),
        shape
    )
    if not os.path.exists(shape_path):
        return 0.0

    results_file = os.path.join(shape_path, 'results.json')
    if not os.path.exists(results_file):
        return 0.0

    try:
        with open(results_file, 'r') as f:
            log = json.load(f)

            if group == 'group1':
                # group1 uses P1 Wins for correct, P2 Wins for incorrect
                correct = log.get('P1 Wins', 0)
            elif group == 'group2':
                # group2 uses Wins for correct, Losses for incorrect
                correct = log.get('Wins', 0)
    except:
        pass

    if total > 0:
        return correct / total
    return 0.0

def line_plot_shapes(base_path, models, temperatures, shapes, model_groups):
    """
    For each model and shape, pick the best temperature by correct proportion.
    Then plot one line per shape on a single figure, x-axis = model, y-axis = proportion correct.
    """
    all_rows = []

    for model in models:
        group = model_groups.get(model, 'group1')  # fallback to group1
        for shape in shapes:
            best_proportion = 0.0
            for temp in temperatures:
                prop = calculate_proportion_for_temp(model, shape, temp, base_path, group)
                if prop > best_proportion:
                    best_proportion = prop

            all_rows.append({
                'Model': model,
                'Shape': shape,
                'Proportion': best_proportion
            })

    # Create a DataFrame for plotting
    df = pd.DataFrame(all_rows)

    # Now we do a line plot:
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.title("Best Temperature Performance per Shape", fontsize=16, fontweight='bold')

    # x-axis: Model, y-axis: Proportion, color/hue: Shape
    # marker='o' draws dots at each model
    ax = sns.lineplot(
        data=df,
        x='Model', y='Proportion', hue='Shape', marker='o'
    )

    # Y-axis from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax.set_ylabel("Correct Proportion", fontsize=12, fontweight='bold')
    plt.legend(title="Shape", loc='best')
    plt.tight_layout()
    plt.show()

def main():
    shapes = ['square', 'triangle', 'cross']
    models = ["gpt3_5", "gpt4", "gpt4o", "gpt4o_mini"]
    temperatures = [0, 0.5, 1, 1.5]

    base_path = '../experiment_shapes'

    # Define model groups based on log structures
    model_groups = {
        "gpt3_5": "group1",
        "gpt4": "group1",
        "gpt4o": "group2",
        "gpt4o_mini": "group2"
    }

    # Generate line plots
    line_plot_shapes(base_path, models, temperatures, shapes, model_groups)

if __name__ == "__main__":
    main()
