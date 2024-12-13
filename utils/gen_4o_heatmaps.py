import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def load_and_aggregate_logs(path, file_name):
    """
    Load and aggregate logs from a specified JSON file.

    Args:
        path (str): Directory path where the log file is located.
        file_name (str): Name of the JSON log file.

    Returns:
        list: Aggregated list of move dictionaries.
    """
    aggregated_logs = []
    full_path = os.path.join(path, file_name)

    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        return aggregated_logs

    try:
        with open(full_path, 'r') as f:
            logs = json.load(f)
            aggregated_logs.extend(logs)
    except Exception as e:
        print(f"Error loading {full_path}: {e}")

    return aggregated_logs

def plot_heatmap(all_moves, n_games, game_name, board_size, players, save_path="heatmap.pdf"):
    """
    Plot and save a heatmap of moves for each player.

    Args:
        all_moves (list): List of move dictionaries.
        n_games (int): Number of games played.
        game_name (str): Name of the game.
        board_size (int): Size of the game board.
        players (list): List of player identifiers (e.g., ['LLM', 'Random']).
        save_path (str): File path to save the heatmap image.
    """
    if game_name == 'connectfour':
        heatmaps = [np.zeros((1, board_size)) for _ in players]
    elif game_name == "shapes":
        heatmaps = [np.zeros((1, board_size)) for _ in players]
    else:
        heatmaps = [np.zeros((board_size, board_size)) for _ in players]

    # Count moves for each player
    for move_info in all_moves:
        player_index = move_info.get("player")
        move = move_info.get("move")

        if player_index not in [0, 1]:
            print(f"Unexpected player index {player_index} in move info: {move_info}")
            continue

        if game_name == "connectfour":
            # For ConnectFour, move should be an integer representing the column
            if isinstance(move, int) and 0 <= move < board_size:
                heatmaps[player_index][0, move] += 1
            else:
                print(f"Invalid move {move} for ConnectFour by player {player_index}")
        elif game_name in ["tictactoe", "battleship"]:
            # For these games, move should be a list or tuple of two integers [row, col]
            if isinstance(move, (list, tuple)) and len(move) == 2:
                row, col = move
                if 0 <= row < board_size and 0 <= col < board_size:
                    heatmaps[player_index][row, col] += 1
                else:
                    print(f"Invalid move {move} for {game_name} by player {player_index}")
            else:
                print(f"Invalid move format {move} for {game_name} by player {player_index}")
        elif game_name == "shapes":
            # For Shapes, move is an integer index representing the chosen shape
            if isinstance(move, int) and 0 <= move < board_size:
                heatmaps[player_index][0, move] += 1
            else:
                print(f"Invalid move {move} for Shapes by player {player_index}")
        else:
            print(f"Unknown game name: {game_name}")

    # Create subplots based on the number of players
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(10 * len(heatmaps), 5))

    if len(heatmaps) == 1:
        axes = [axes]

    for i, heatmap in enumerate(heatmaps):
        sns.heatmap(
            heatmap,
            ax=axes[i],
            annot=True,
            cmap='coolwarm',
            fmt='.0f',
            cbar=True if i == 0 else False
        )
        axes[i].set_title(f'Player {players[i]} Moves Heatmap')
        if game_name == 'connectfour':
            axes[i].set_xlabel('Columns')
            axes[i].set_yticks([])
        elif game_name == "shapes":
            # Assuming shape options are indexed from 0 to board_size-1
            shape_labels = [f"Shape {i}" for i in range(board_size)]
            axes[i].set_xlabel('Chosen Shapes')
            axes[i].set_yticks([])
            axes[i].set_xticklabels(shape_labels, rotation=45)
        else:
            axes[i].set_xlabel('Columns')
            axes[i].set_ylabel('Rows')

    plt.suptitle(f"Total moves played after {n_games} games of {game_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to make space for the suptitle
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved heatmap to {save_path}")

def generate_heatmaps_for_shapes():
    """
    Generate heatmaps for Shapes experiments.
    """
    base_path = 'experiment_shapes'
    models = ['gpt4o', 'gpt4o_mini']
    temperatures = [0, 0.5, 1, 1.5]
    shapes = ['square', 'triangle', 'cross']
    players = ['LLM', 'Random']

    for model in models:
        for temp in temperatures:
            for shape in shapes:
                # Construct the experiment directory
                experiment_path = os.path.join(base_path, model, str(temp), shape)

                if not os.path.exists(experiment_path):
                    print(f"Directory {experiment_path} does not exist. Skipping.")
                    continue

                # Load the game logs
                file_name = 'game_logs.json'
                all_moves = load_and_aggregate_logs(experiment_path, file_name)

                if not all_moves:
                    print(f"No moves found in {experiment_path}. Skipping.")
                    continue

                # Define the save path for the heatmap
                save_path = os.path.join(
                    experiment_path,
                    f"{model}_{temp}_{shape}_heatmap.pdf"
                )

                # Plot and save the heatmap
                plot_heatmap(
                    all_moves,
                    n_games=100,  # Assuming 100 games per experiment
                    game_name="shapes",
                    board_size=len(shapes),  # Number of shape options
                    players=players,
                    save_path=save_path
                )
                print(f"[Heatmap] Generated heatmap for {experiment_path}")

            # After processing all shapes for a specific model and temperature, generate a combined heatmap
            combined_moves = []
            for shape in shapes:
                experiment_path = os.path.join(base_path, model, str(temp), shape)
                file_name = 'game_logs.json'
                shape_moves = load_and_aggregate_logs(experiment_path, file_name)
                if shape_moves:
                    combined_moves.extend(shape_moves)

            if combined_moves:
                combined_save_path = os.path.join(
                    base_path,
                    model,
                    str(temp),
                    f"{model}_{temp}_shapes_combined_heatmap.pdf"
                )
                plot_heatmap(
                    combined_moves,
                    n_games=100,  # Assuming 100 games per experiment
                    game_name="shapes",
                    board_size=len(shapes),
                    players=players,
                    save_path=combined_save_path
                )
                print(f"[Heatmap] Generated combined heatmap for model {model} at temperature {temp}")
            else:
                print(f"No combined moves found for model {model} at temperature {temp}")

def generate_heatmaps_for_other_games():
    """
    Generate heatmaps for Battleship, ConnectFour, and TicTacToe experiments.
    """
    base_path = 'experiment_board_games'
    games = ['battleship', 'connectfour', 'tictactoe']
    models = ['gpt4o', 'gpt4o_mini']
    temperatures = [0, 0.5, 1, 1.5]
    # Define board sizes for each game
    board_sizes = {'connectfour': 7, 'tictactoe': 3, 'battleship': 3}
    players = ['LLM', 'Random']

    for game_name in games:
        for model in models:
            for temp in temperatures:
                # Construct the experiment name based on the model, game, and temperature
                experiment_name = f'experiment_{game_name}_{model}_oneshot_temp_{temp}'
                experiment_path = os.path.join(base_path, experiment_name)

                if not os.path.exists(experiment_path):
                    print(f"Directory {experiment_path} does not exist. Skipping.")
                    continue

                # Construct the log file name
                log_file_name = f'game_logs_{game_name}.json'
                all_moves = load_and_aggregate_logs(experiment_path, log_file_name)

                if not all_moves:
                    print(f"No moves found in {experiment_path}. Skipping.")
                    continue

                # Define the save path for the heatmap
                save_path = os.path.join(
                    experiment_path,
                    f"{model}_{temp}_{game_name}_heatmap.pdf"
                )

                # Plot and save the heatmap
                plot_heatmap(
                    all_moves,
                    n_games=100,  # Assuming 100 games per experiment
                    game_name=game_name,
                    board_size=board_sizes[game_name],
                    players=players,
                    save_path=save_path
                )
                print(f"[Heatmap] Generated heatmap for {experiment_path}")

def move_heatmaps(destination_folder="all_heatmaps"):
    """
    Move all generated heatmaps to a single destination folder.

    Args:
        destination_folder (str): Path to the folder where all heatmaps will be moved.
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Define the base paths for Shapes and Other Games experiments
    shapes_path = 'experiment_shapes'
    other_games_path = 'experiment_board_games'

    # Find and move all .pdf files in shapes experiments
    for root, _, files in os.walk(shapes_path):
        for file in files:
            if file.endswith(".pdf"):
                source = os.path.join(root, file)
                destination = os.path.join(destination_folder, file)
                shutil.move(source, destination)
                print(f"Moved {source} to {destination}")

    # Find and move all .pdf files in other games experiments
    for root, _, files in os.walk(other_games_path):
        for file in files:
            if file.endswith(".pdf"):
                source = os.path.join(root, file)
                destination = os.path.join(destination_folder, file)
                shutil.move(source, destination)
                print(f"Moved {source} to {destination}")


def main():
    """
    Main function to generate heatmaps for all games.
    """
    print("Starting heatmap generation for Shapes...")
    generate_heatmaps_for_shapes()
    print("\nStarting heatmap generation for Battleship, ConnectFour, and TicTacToe...")
    generate_heatmaps_for_other_games()
    print("\nHeatmap generation completed for all games.")
    move_heatmaps()

if __name__ == '__main__':
    main()
