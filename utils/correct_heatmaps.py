import os
import json
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from scripts_games.connectfour import ConnectFour
from scripts_games.battleship import BattleShip
from scripts_games.tictactoe import TicTacToe

def generate_heatmaps_from_logs(logs_folder, game_name, board_size, save_path):
    """
    Generate heatmaps from logs and save them as images.
    """
    print("=" * 50)
    print(logs_folder)
    # Construct the correct file path
    log_filename = f"{logs_folder}/game_logs_{game_name}.json"
    print(f"Attempting to open log file at: {log_filename}")

    try:
        with open(log_filename, 'r') as file:
            game_logs = json.load(file)
        all_game_logs = [{"player": log['player'], "move": log['move']} for log in game_logs]

        # Extract unique player indices from logs
        unique_players = sorted(set(log['player'] for log in game_logs))
        print(f"Unique player indices found: {unique_players}")

        # Map player indices to names (using "Player X" as default names)
        player_names = [f"Player {i}" for i in unique_players]

        if all_game_logs:
            plot_heatmap(all_game_logs, game_name, board_size, player_names, save_path)
        else:
            print(f"No moves to plot for {game_name} in {logs_folder}")
    except FileNotFoundError:
        print(f"Log file not found: {log_filename}")
        print("Files in directory:", os.listdir(logs_folder))

def plot_heatmap(all_moves, game_name, board_size, players, save_path):
    if game_name.lower() == "connectfour":
        heatmaps = [np.zeros((board_size,)) for _ in players]

        for move_info in all_moves:
            player_index = move_info["player"]
            column = move_info["move"]
            if player_index >= len(heatmaps):
                print(f"Warning: player_index {player_index} out of bounds for players {players}")
                continue
            heatmaps[player_index][column] += 1
    else:
        heatmaps = [np.zeros((board_size, board_size)) for _ in players]
        for move_info in all_moves:
            player_index = move_info["player"]
            row, col = move_info["move"]
            if player_index >= len(heatmaps):
                print(f"Warning: player_index {player_index} out of bounds for players {players}")
                continue
            heatmaps[player_index][row, col] += 1

    # Normalize and format percentages in each heatmap
    for i, heatmap in enumerate(heatmaps):
        total_moves = np.sum(heatmap)
        if total_moves > 0:
            heatmaps[i] = (heatmap / total_moves) * 100

    # Visualization of heatmaps
    fig, axes = plt.subplots(1, len(players), figsize=(10 * len(players), 5))
    if len(players) == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, heatmap in enumerate(heatmaps):
        if game_name.lower() == "connectfour":
            heatmap = heatmap.reshape((1, board_size))  # Reshape for Connect Four
            sns.heatmap(heatmap, ax=axes[i], annot=True, cmap='coolwarm', fmt=".2f",
                        vmin=0, vmax=100, annot_kws={"size": 9}, cbar=True, yticklabels=False)
        else:
            sns.heatmap(heatmap, ax=axes[i], annot=True, cmap='coolwarm', fmt=".2f",
                        vmin=0, vmax=100, annot_kws={"size": 9}, cbar=True)
        for _, spine in axes[i].spines.items():
            spine.set_visible(True)
        for t in axes[i].texts:
            t.set_text(t.get_text() + '%')
            t.set_fontsize(18)
        axes[i].set_ylabel('')  # Remove y-axis label

        # Set the title to the player's name
        axes[i].set_title(f'{players[i]} Moves Heatmap', fontsize=16)

    plt.suptitle(f"Distribution of Moves Across All Played Moves in {game_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    plt.savefig(save_path, format='svg')
    plt.close(fig)

def main():
    debug = True

    game_runs = [
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4_oneshot_temp_1'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4_oneshot_temp_0'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4_oneshot_temp_0.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4_oneshot_temp_1.5'},

        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt3_5_oneshot_temp_1'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt3_5_oneshot_temp_1.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt3_5_oneshot_temp_0.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt3_5_oneshot_temp_0'},

        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_1'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_1.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_0.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_0'},

        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_1'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_1.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_0.5'},
        {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'gpt-4o-mini', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_0'},

        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt3_5_oneshot_temp_1'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt3_5_oneshot_temp_1.5'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt3_5_oneshot_temp_0'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt3_5_oneshot_temp_0.5'},
        
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4_oneshot_temp_1'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4_oneshot_temp_1.5'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4_oneshot_temp_0'}, 
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4_oneshot_temp_0.5'},

        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_1'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_1.5'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_0'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_0.5'},

        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_1'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_1.5'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_0'},
        {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_0.5'},

        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4_oneshot_temp_1.5'}, 
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4_oneshot_temp_1'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4_oneshot_temp_0'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4_oneshot_temp_0.5'},
        
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt3_5_oneshot_temp_0.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt3_5_oneshot_temp_1'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt3_5_oneshot_temp_1.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-3.5-turbo-1106',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt3_5_oneshot_temp_0'},

        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_0.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_1'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_1.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_0'},

        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_0.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_1'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_1.5'},
        {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4o-mini',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_0'},
    ]

    for game_info in game_runs:
        folder_name = game_info['experiment_name']
        # The `players` list is now handled within `generate_heatmaps_from_logs`
        file_name = game_info['experiment_name'].split("/")[1]
        save_path = f"./new_heatmaps/{file_name}_heatmap.svg"
        generate_heatmaps_from_logs(folder_name, game_info['game_name'], game_info['board_size'], save_path)

if __name__ == "__main__":
    main()