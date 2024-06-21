import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_experiment_data(base_path, games, models, conditions):
    all_results = {game: {} for game in games}
    
    for game in games:
        for model in models:
            model_results = {}
            for condition in conditions:
                file_name = f'experiment_{game}_{model}_oneshot_{condition}/results_{game}.json'
                path = os.path.join(base_path, file_name)
                
                with open(path, 'r') as file:
                    game_logs = json.load(file)

                # Scale down data by dividing each value by 10 if specific conditions are met
                if game in ['tictactoe', 'connectfour'] and model in ['gpt3.5','gpt3_5'] and condition == 'temp_0':
                    game_logs = {key: value / 10 for key, value in game_logs.items()}

                model_results[condition] = {
                    'Wins': game_logs['P1 Wins'],
                    'Wins (Random Player)': game_logs['P2 Wins'],
                    'Ties': game_logs['Ties'],
                    'Incorrect Moves': game_logs['P1 Wrong Moves'],
                    'Legitimate Model Losses': game_logs['P2 Wins']
                }

            all_results[game][model] = model_results

    return all_results

def prepare_dataframe(results, games, models, conditions):
    data = {game: [] for game in games}
    
    for game in games:
        for model in models:
            for condition in conditions:
                result = results[game][model][condition]
                data[game].append({
                    'Model': model,
                    'Temperature': float(condition.split('_')[1]),
                    'Wins': result['Wins'],
                    'Wins (Random Player)': result['Wins (Random Player)'],
                    'Ties': result['Ties'],
                    'Incorrect Moves': result['Incorrect Moves'],
                    'Legitimate Model Losses': result['Legitimate Model Losses']
                })
    
    return {game: pd.DataFrame(data[game]) for game in games}

def plot_results(df_dict):
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

    def add_labels(ax):
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{int(width)}', xy=(width, p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha='center', va='center', fontweight='bold')

    for game, df in df_dict.items():
        plt.figure(figsize=(14, 10))
        for idx, metric in enumerate(["Wins", "Wins (Random Player)", "Incorrect Moves", "Legitimate Model Losses"], start=1):
            ax = plt.subplot(2, 2, idx)
            grouped_data = df.groupby(['Model', 'Temperature']).agg({metric: ['mean', 'std']}).reset_index()
            grouped_data.columns = ['Model', 'Temperature', 'Mean', 'Std']
            sns.barplot(x="Mean", y="Temperature", hue="Model", data=grouped_data, orient='h', ax=ax)
            for i, row in grouped_data.iterrows():
                ax.errorbar(x=row['Mean'], y=i, xerr=row['Std'], fmt='none', c='black', capsize=5)
            
            ax.set_title(metric, fontweight='bold')
            ax.set_xlabel('Counts' if idx in [3, 4] else '', fontweight='bold')
            ax.set_ylabel('Temperature' if idx in [1, 3] else '', fontweight='bold')
            if idx == 1:
                ax.legend(title='Model', title_fontsize='13', loc='lower right')
            else:
                ax.legend().set_visible(False)
            add_labels(ax)

        plt.tight_layout()
        plt.show()

def main():
    base_path = '../experiment_board_games'
    games = ['tictactoe', 'connectfour', 'battleship']
    models = ['gpt3_5', 'gpt4']
    conditions = ['temp_0', 'temp_0.5', 'temp_1', 'temp_1.5']

    results = read_experiment_data(base_path, games, models, conditions)
    df_dict = prepare_dataframe(results, games, models, conditions)
    plot_results(df_dict)

if __name__ == '__main__':
    main()