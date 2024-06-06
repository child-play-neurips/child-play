import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_experiments(base_path, models, conditions):
    all_results = {}
    np.random.seed(42)  # for reproducibility

    for model in models:
        model_results = {}
        print(f"Processing results for model: {model}")
        for condition in conditions:
            file_name = f'experiment_connectfour_{model}_oneshot_{condition}/game_logs_connectfour.json'
            path = os.path.join(base_path, file_name)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    game_logs = json.load(file)
                    if not isinstance(game_logs, list):
                        print(f"Unexpected data structure in {path}: {type(game_logs)}")
                        continue

                    # Sample 100 games for specific model and condition
                    if (model == 'gpt3_5' and condition == 'temp_0') or (model == 'random' and condition == 'temp_0'):
                        if len(game_logs) > 100:
                            game_logs = np.random.choice(game_logs, 100, replace=False).tolist()

                    game_stats = count_moves(game_logs)
                    model_results[condition] = game_stats
            else:
                print(f"File not found: {path}")
                model_results[condition] = {
                    'average_moves': {0: 0, 1: 0},
                    'missed_wins': {0: 0, 1: 0},
                    'missed_blocks': {0: 0, 1: 0}
                }
        all_results[model] = model_results

    print("=" * 80)
    print("Final Results Summary:")
    for model, conditions_results in all_results.items():
        for condition, stats in conditions_results.items():
            print(f"Model {model}, Condition {condition}:")
            print("  Model Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][0]}")
            print(f"    Missed Wins: {stats['missed_wins'][0]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][0]}")
            print("  Random Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][1]}")
            print(f"    Missed Wins: {stats['missed_wins'][1]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][1]}")
            print(f"  Total Games Played: {stats.get('total_games', 'Data not available')}")
            print("-" * 50)
    print("=" * 80)

    return all_results

def is_winning_move(board, player):
    for c in range(board.shape[1] - 3):
        for r in range(board.shape[0]):
            if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                return True
    for c in range(board.shape[1]):
        for r in range(board.shape[0] - 3):
            if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                return True
    for c in range(board.shape[1] - 3):
        for r in range(board.shape[0] - 3):
            if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                return True
    for c in range(board.shape[1] - 3):
        for r in range(3, board.shape[0]):
            if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                return True
    return False

def can_win_next(board, player):
    rows, cols = board.shape
    for col in range(cols):
        for row in range(rows-1, -1, -1):
            if board[row][col] == 0:
                board[row][col] = player
                if is_winning_move(board, player):
                    board[row][col] = 0
                    return True
                board[row][col] = 0
                break
    return False

def count_moves(game_logs):
    rows, cols = 7, 7
    board = np.zeros((rows, cols), dtype=int)
    game_stats = {0: [], 1: []}
    total_moves = {0: 0, 1: 0}
    games_played = 0
    last_player = None
    current_game_moves = 0
    win_flag = False

    for move in game_logs:
        player = move["player"]
        col = move["move"]

        if last_player is not None and last_player == player:
            continue

        for row in range(rows-1, -1, -1):
            if board[row][col] == 0:
                board[row][col] = player + 1
                break

        total_moves[player] += 1
        current_game_moves += 1
        print(f"Move made by player {player + 1} in column {col}")
        print(f"Board state:\n{board}")

        if is_winning_move(board, player + 1):
            print(f"Player {player + 1} wins. Resetting board ====================")
            games_played += 1
            board = np.zeros((rows, cols), dtype=int)
            last_player = None
            current_game_moves = 0
            win_flag = True
        else:
            win_flag = False
            missed_win = can_win_next(board, player + 1)
            missed_block = can_win_next(board, 1 - player)
            game_stats[player].append({'missed_win': missed_win, 'missed_block': missed_block})
            print(f"Turn finished for player {player + 1} ------------------------")
            last_player = player

    if not win_flag:
        games_played += 1

    average_moves = {player: total_moves[player] / games_played if games_played > 0 else 0 for player in total_moves}
    missed_wins = {player: sum(stat['missed_win'] for stat in stats) for player, stats in game_stats.items()}
    missed_blocks = {player: sum(stat['missed_block'] for stat in stats) for player, stats in game_stats.items()}

    print("*" * 50)
    print(f"Total Moves: {total_moves}")
    print(f"Total Games Played: {games_played}")
    print("Missed Wins:", missed_wins)
    print("Missed Blocks:", missed_blocks)
    print("Average Moves Per game:", average_moves)
    print("*" * 50)
    print("=" * 50)

    return {
        'average_moves': average_moves,
        'missed_wins': missed_wins,
        'missed_blocks': missed_blocks,
        'total_games': games_played
    }

def plot_results(models, results, conditions):
    data_to_plot = []
    for model in models:
        for condition in conditions:
            result = results.get(model, {}).get(condition, {})
            for player_type in (0, 1):
                average_moves = result.get('average_moves', {}).get(player_type, 0)
                missed_wins = result.get('missed_wins', {}).get(player_type, 0)
                missed_blocks = result.get('missed_blocks', {}).get(player_type, 0)

                player_label = 'Model' if player_type == 0 else 'Random'
                data_to_plot.append({
                    'Model': model,
                    'Temperature': condition,
                    'Player': f'{model} {player_label}',
                    'Average Moves': average_moves,
                    'Missed Wins': missed_wins,
                    'Missed Blocks': missed_blocks
                })

    df = pd.DataFrame(data_to_plot)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot for Average Moves
    sns.barplot(x='Temperature', y='Average Moves', hue='Player', data=df, ax=axes[0])
    axes[0].set_title('Average Moves')
    axes[0].set_ylabel('Average Moves')
    axes[0].get_legend().remove()

    # Helper function for plotting percentages
    def create_percentage_plot(ax_index, measure, title):
        ax = axes[ax_index]
        sns.barplot(x='Temperature', y=measure, hue='Player', data=df, ax=ax, estimator=lambda x: sum(x) / df[measure].sum() * 100)
        ax.set_title(title)
        ax.set_ylabel('Percentage %')
        # Remove legend except for the last plot
        if ax_index != 2:
            ax.get_legend().remove()

    # Percentage plots for Missed Wins and Missed Blocks
    create_percentage_plot(1, 'Missed Wins', 'Percentage of Missed Wins')
    create_percentage_plot(2, 'Missed Blocks', 'Percentage of Missed Blocks')

    plt.tight_layout(pad=3.0)  # Add space between plots
    plt.show()

def process_experiments(base_path, models, conditions):
    all_results = {}

    for model in models:
        model_results = {}
        print(f"Processing results for model: {model}")
        for condition in conditions:
            file_name = f'experiment_connectfour_{model}_oneshot_{condition}/game_logs_connectfour.json'
            path = os.path.join(base_path, file_name)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    game_logs = json.load(file)
                    if not isinstance(game_logs, list):
                        print(f"Unexpected data structure in {path}: {type(game_logs)}")
                        continue
                    game_stats = count_moves(game_logs)
                    model_results[condition] = game_stats
            else:
                print(f"File not found: {path}")
                model_results[condition] = {
                    'average_moves': {0: 0, 1: 0},
                    'missed_wins': {0: 0, 1: 0},
                    'missed_blocks': {0: 0, 1: 0}
                }
        all_results[model] = model_results

    print("=" * 80)
    print("Final Results Summary:")
    for model, conditions_results in all_results.items():
        for condition, stats in conditions_results.items():
            print(f"Model {model}, Condition {condition}:")
            print("  Model Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][0]}")
            print(f"    Missed Wins: {stats['missed_wins'][0]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][0]}")
            print("  Random Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][1]}")
            print(f"    Missed Wins: {stats['missed_wins'][1]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][1]}")
            print(f"  Total Games Played: {stats.get('total_games', 'Data not available')}")
            print("-" * 50)
    print("=" * 80)

    return all_results

def main():
    base_path = 'experiment_board_games'
    models = ['gpt3_5', 'gpt4']
    conditions = ['temp_0', 'temp_0.5', 'temp_1', 'temp_1.5']

    results = process_experiments(base_path, models, conditions)
    plot_results(models, results, conditions)

if __name__ == '__main__':
    main()