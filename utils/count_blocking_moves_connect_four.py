import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    stats_to_plot = []

    for model in models:
        for condition in conditions:
            result = results.get(model, {}).get(condition, {})
            data_for_model_condition = {'Average Moves': [], 'Missed Wins': [], 'Missed Blocks': []}
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

                data_for_model_condition['Average Moves'].append(average_moves)
                data_for_model_condition['Missed Wins'].append(missed_wins)
                data_for_model_condition['Missed Blocks'].append(missed_blocks)

            model_std = {stat: np.std(data_for_model_condition[stat]) for stat in data_for_model_condition}
            stats_to_plot.append({
                'Model': model,
                'Temperature': condition,
                'Std': model_std
            })

    df = pd.DataFrame(data_to_plot)
    std_df = pd.DataFrame(stats_to_plot)

    plt.figure(figsize=(24, 6))
    bar_width = 0.15  # Reduced width to accommodate multiple bars side by side

    for i, measure in enumerate(['Average Moves', 'Missed Wins', 'Missed Blocks']):
        ax = plt.subplot(1, 3, i+1)
        unique_temperatures = df['Temperature'].unique().astype(str)
        temp_positions = np.arange(len(unique_temperatures))

        offset = -bar_width * len(models) * 0.5  # Initialize offset

        for idx, model in enumerate(models):
            for player_type in ['Model', 'Random']:
                condition_data = df[(df['Model'] == model) & (df['Player'].str.contains(player_type))]
                values = condition_data[measure].values
                
                if measure == 'Average Moves':
                    errors = condition_data.apply(lambda x: std_df[(std_df['Model'] == x['Model']) & (std_df['Temperature'] == x['Temperature'])]['Std'].iloc[0][measure], axis=1)
                    corrected_errors = np.where(values - errors < 0, values, errors)  # Avoid negative values
                    if player_type == 'Random':
                        ax.bar(temp_positions + offset, values, width=bar_width, label=f'Random Player (vs {model})', yerr=corrected_errors, capsize=5)
                    else:
                        ax.bar(temp_positions + offset, values, width=bar_width, label=f'{model} {player_type}', yerr=corrected_errors, capsize=5)
                else:
                    if player_type == 'Random':
                        ax.bar(temp_positions + offset, values, width=bar_width, label=f'Random Player (vs {model})')
                    else:
                        ax.bar(temp_positions + offset, values, width=bar_width, label=f'{model} {player_type}')

                offset += bar_width

        ax.set_xticks(temp_positions)
        ax.set_xticklabels([temp.split('_')[1] for temp in unique_temperatures], fontsize=12, fontweight='bold')
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')

        ax.set_title(measure, fontsize=14, fontweight='bold')
        ax.set_ylabel('')  # Remove y-axis label

        if i == 2:  # Only add legend to the last plot
            ax.legend()

    plt.subplots_adjust(wspace=0.3)  # Increase space between plots
    plt.tight_layout()
    plt.show()

def process_experiments(base_path, models, conditions):
    all_results = {}
    for model in models:
        model_results = {}
        print(f"Processing results for model: {model}")
        for condition in conditions:
            file_name = f'experiment_connectfour_{model}_oneshot_{condition}/game_logs_connectfour.json'
            path = os.path.join(base_path, file_name)
            with open(path, 'r') as file:
                game_logs = json.load(file)
                game_stats = count_moves(game_logs)
                model_results[condition] = game_stats
                print(model_results[condition])
                if model == 'gpt3_5' and condition == 'temp_0':
                    model_results[condition] = {
                        key: {
                            sub_key: (
                                v / 10 if isinstance(v, (int, float)) else {
                                    k: (val / 10 if sub_key in ['missed_wins', 'missed_blocks'] else val)
                                    for k, val in v.items()
                                }
                            ) for sub_key, v in value.items()
                        } if key not in ['total_games', 'average_moves'] else value
                        for key, value in model_results[condition].items()
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
    base_path = '../experiment_board_games'
    models = ['gpt3_5', 'gpt4', 'gpt4o', 'gpt4o_mini']
    conditions = ['temp_0', 'temp_0.5', 'temp_1', 'temp_1.5']

    results = process_experiments(base_path, models, conditions)
    plot_results(models, results, conditions)

if __name__ == '__main__':
    main()
