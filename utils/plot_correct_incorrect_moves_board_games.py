import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_experiment_data(base_path, games, models, conditions):
    all_results = {game: {model: {} for model in models} for game in games}
    
    for game in games:
        for model in models:
            for condition in conditions:
                file_name = f'experiment_{game}_{model}_oneshot_{condition}/results_{game}.json'
                path = os.path.join(base_path, file_name)
                
                if not os.path.exists(path):
                    print(f"Warning: File {path} does not exist.")
                    continue
                
                with open(path, 'r') as file:
                    game_logs = json.load(file)
                    
                all_results[game][model][condition] = {
                    'Wins': game_logs.get('P1 Wins', 0),
                    'Wins (Random Player)': game_logs.get('P2 Wins', 0),
                    'Ties': game_logs.get('Ties', 0),
                    'Incorrect Moves': game_logs.get('P1 Wrong Moves', 0)
                }
    
    return all_results

def prepare_dataframe(results, games, models, conditions):
    data = {game: [] for game in games}
    
    for game in games:
        for model in models:
            for condition in conditions:
                result = results[game][model].get(condition, {
                    'Wins': 0,
                    'Wins (Random Player)': 0,
                    'Ties': 0,
                    'Incorrect Moves': 0
                })
                total_plays = result['Wins'] + result['Wins (Random Player)'] + result['Ties'] + result['Incorrect Moves']
                if total_plays > 0:
                    proportions = {
                        'Wins': result['Wins'] / total_plays,
                        'Wins (Random Player)': result['Wins (Random Player)'] / total_plays,
                        'Incorrect Moves': result['Incorrect Moves'] / total_plays,
                        'Ties': result['Ties'] / total_plays
                    }
                    entry = {
                        'Model': model,
                        'Temperature': float(condition.split('_')[1]),
                        'Total Plays': total_plays
                    }
                    for key, value in proportions.items():
                        entry[key] = result[key]
    
                    data[game].append(entry)
    
    return {game: pd.DataFrame(data[game]) for game in games}

def plot_results(df_dict, plot_type='bar'):
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})

    for game, df in df_dict.items():
        print(f"Plotting results for {game}")

        if plot_type == 'bar':
            plt.figure(figsize=(14, 10))
            for idx, metric in enumerate(["Wins", "Wins (Random Player)", "Incorrect Moves", "Ties"], start=1):
                ax = plt.subplot(2, 2, idx)
                grouped_data = df.groupby(['Model', 'Temperature']).agg({
                    metric: 'mean'
                }).reset_index()
                grouped_data.columns = ['Model', 'Temperature', 'Mean']

                bars = sns.barplot(x="Mean", y="Temperature", hue="Model", data=grouped_data, orient='h', ax=ax)
                ax.set_title(f"{metric}", fontweight='bold')

                if idx > 2:
                    ax.set_xlabel('Counts')
                else:
                    ax.set_xlabel('')
                ax.set_ylabel('Temperature' if idx in [1, 3] else '')

                if idx == 1:
                    ax.legend(title='Model', title_fontsize='13', loc='lower right')
                else:
                    ax.legend().set_visible(False)

            plt.tight_layout()
            plt.show()

        elif plot_type == 'scatter':
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            handles, labels = None, None

            for idx, (ax, metric) in enumerate(zip(axes.flatten(), ["Wins", "Wins (Random Player)", "Incorrect Moves", "Ties"])):
                sns.lineplot(
                    data=df,
                    x="Temperature",
                    y=metric,
                    hue="Model",
                    style="Model",
                    markers=True,
                    dashes=False,
                    markersize=10,
                    linewidth=2,
                    palette='deep',
                    ax=ax
                )

                ax.set_title(f"{metric}", fontsize=14, fontweight='bold')
                ax.set_xlabel('Temperature')
                ax.set_ylabel(metric)
                ax.set_ylim(0, 100)  # Ensures axis scaling remains consistent

                if idx == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend_.remove()
                else:
                    ax.legend().set_visible(False)

            if handles and labels:
                fig.legend(
                    handles,
                    labels,
                    title='Model',
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1),
                    ncol=4
                )

            plt.subplots_adjust(right=0.85, hspace=0.4)
            plt.show()

def print_totals(results, games, models, conditions, games_played=1000):
    print("\n=== Total Results ===\n")
    for game in games:
        print(f"Game: {game.capitalize()}")
        for model in models:
            for condition in conditions:
                res = results[game][model].get(condition, {
                    'Wins': 0,
                    'Wins (Random Player)': 0,
                    'Ties': 0,
                    'Incorrect Moves': 0
                })
                wins = res['Wins']
                wins_random = res['Wins (Random Player)']
                ties = res['Ties']
                incorrect_moves = res['Incorrect Moves']
                total_plays = wins + wins_random + ties + incorrect_moves

                # Calculate percentages based on 1000 games per condition
                percentage_wins = (wins / games_played) * 100
                percentage_wins_random = (wins_random / games_played) * 100
                percentage_ties = (ties / games_played) * 100
                percentage_incorrect_moves = (incorrect_moves / games_played) * 100

                print(f"  Model: {model}, Temperature: {condition.split('_')[1]}")
                print(f"    Total Plays: {total_plays} / {games_played}")
                print(f"    Wins: {wins} ({percentage_wins:.2f}%)")
                print(f"    Wins (Random Player): {wins_random} ({percentage_wins_random:.2f}%)")
                print(f"    Ties: {ties} ({percentage_ties:.2f}%)")
                print(f"    Incorrect Moves: {incorrect_moves} ({percentage_incorrect_moves:.2f}%)")
        print("-" * 40)

def main():
    base_path = '../experiment_board_games'
    games = ['tictactoe', 'connectfour', 'battleship']
    models = ['gpt3_5', 'gpt4', 'gpt4o', 'gpt4o_mini']
    conditions = ['temp_0', 'temp_0.5', 'temp_1', 'temp_1.5']

    results = read_experiment_data(base_path, games, models, conditions)
    df_dict = prepare_dataframe(results, games, models, conditions)

    # Print total results to console with percentages based on 1000 plays per condition
    print_totals(results, games, models, conditions, games_played=100)

    plot_type = 'scatter'  # Change this to 'bar' for bar plots
    plot_results(df_dict, plot_type=plot_type)

if __name__ == '__main__':
    main()
