import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.weight'] = 'bold'

def plot_proportions(df_validity, df_construct):
    stats_validity = df_validity.groupby(['Temperature', 'Model']).agg(
        Correct_Proportion=('Correct', 'mean'),
        Correct_Std=('Correct', 'std'),
        N=('Correct', 'count')
    ).reset_index()
    stats_validity['Correct_SE'] = stats_validity['Correct_Std'] / np.sqrt(stats_validity['N'])

    stats_construct = df_construct.groupby(['Temperature', 'Model']).agg(
        Valid_Proportion=('Valid', 'mean'),
        Valid_Std=('Valid', 'std'),
        N=('Valid', 'count')
    ).reset_index()
    stats_construct['Valid_SE'] = stats_construct['Valid_Std'] / np.sqrt(stats_construct['N'])

    plt.figure(figsize=(12, 6))

    models = stats_validity['Model'].unique()
    temperatures = stats_validity['Temperature'].unique()
    model_index = {model: i for i, model in enumerate(models)}

    bar_width = 0.35
    temp_position = np.arange(len(temperatures))

    plt.subplot(1, 2, 1)
    for model in models:
        model_data = stats_validity[stats_validity['Model'] == model]
        positions = temp_position + bar_width * model_index[model]
        plt.bar(positions, model_data['Correct_Proportion'], width=bar_width, label=model,
                yerr=model_data['Correct_SE'], capsize=5, error_kw={'capthick': 2})

    plt.xticks(temp_position + bar_width / 2, temperatures, weight='bold')
    plt.title('Proportion of Correct Responses', fontsize=20, weight='bold')
    plt.ylabel('Proportion Correct', fontsize=20, weight='bold')
    plt.xlabel('Temperature', fontsize=20, weight='bold')
    plt.legend()

    plt.subplot(1, 2, 2)
    for model in models:
        model_data = stats_construct[stats_construct['Model'] == model]
        positions = temp_position + bar_width * model_index[model]
        plt.bar(positions, model_data['Valid_Proportion'], width=bar_width, label=model,
                yerr=model_data['Valid_SE'], capsize=5, error_kw={'capthick': 2})

    plt.xticks(temp_position + bar_width / 2, temperatures, weight='bold')
    plt.title('Proportion of Valid Constructs', fontsize=20, weight='bold')
    plt.ylabel('Proportion Valid', fontsize=20, weight='bold')
    plt.xlabel('Temperature', fontsize=20, weight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    data_validity = pd.read_csv("../lcl_experiments/df_validity.csv")
    data_construct = pd.read_csv("../lcl_experiments/df_construct.csv")

    df_validity = pd.DataFrame(data_validity)
    df_construct = pd.DataFrame(data_construct)

    plot_proportions(df_validity, df_construct)

if __name__ == "__main__":
    main()
