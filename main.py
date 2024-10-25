import random
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd

from scripts_games.connectfour import ConnectFour
from scripts_games.battleship import BattleShip
from scripts_games.tictactoe import TicTacToe
from scripts_games.shapes import Shapes, bar_plot_shapes

from wrapper import ask

class PlayerBase:
    def __init__(self, player_id, name, debug=False):
        self.name = name
        self.debug = debug
        self.player_id = player_id
        self.messages = []

    def collect_message(self, message):
        """Collects messages; prints if in debug mode, otherwise stores them."""
        if self.debug:
            print(message)
        else:
            self.messages.append(message)

class LLMPlayer(PlayerBase):
    def __init__(self, game, model_name="oa:gpt-3.5-turbo-0125", player_id=0, player_name="LLM",  temperature=0, debug=False):
        super().__init__(player_id, player_name, debug)
        self.player_id = player_id
        self.player_name = player_name
        self.model_name = model_name
        self.game = game
        self.temperature = temperature
        self.interaction_count = 0
        print(type(self.game))
        print(self.game)

    def print_details(self):
        print(f"Model Name: {self.model_name}")
        print(f"Player ID: {self.player_id}")
        print(f"Player Name: {self.player_name}")
        print(f"Game: {self.game.name}")
        print(f"Temperature: {self.temperature}")
        print(f"Interaction Count: {self.interaction_count}")

    def print_api_messages(self, api_messages):
        print("=" * 50)
        print("API Messages:")
        for i, message in enumerate(api_messages):
            print(f"Message {i + 1}:")
            print(f"  Role: {message['role']}")
            print(f"  Content: {message['content']}")
            print("-" * 60)

    def make_guess(self, game, previous_play):
        api_messages = [{"role": "system", "content": f"You are a player in a game of {self.game.name}. {self.game.prompt}."}]
        current_state = game.get_text_state(None)
        prompt = f"Player {self.player_id + 1} ({self.player_name}), it's your turn. Here's the current game state:\n{current_state}\nMy move is: "
        user_message = {"role": "user", "content": f"{prompt}"}
        
        self.collect_message(user_message)
        api_messages.append(user_message)

        if self.debug:
            print(f"\nPrompt to LLM:\n{prompt}")

        response = ask(api_messages=api_messages, temperature=self.temperature, model=self.model_name)
        self.collect_message(f"LLM Response:\n{response}")
        api_messages.append({"role": "user", "content": f"Your Response:\n{response}"})

        return self.parse_move(response, game)
    
    def parse_move(self, response, game):
        """ Parse the response from the model and validate it as a move. """
        try:
            if game.name == "shapes":
                guess = int(response)
                return guess
            elif game.name == "connectfour":
                col = int(response)
                return col if 0 <= col < game.cols and game.board[0][col] == '.' else None
            else:
                row, col = map(int, response.split())
                return (row, col) if 0 <= row < game.board_size and 0 <= col < game.board_size and game.board[row][col] in [" ", "~", "S"] else None
        except ValueError:
            self.collect_message("Failed to parse move, please provide only the required text and nothing else. Previous response: " + response)
            return None
    
class TextPlayer(PlayerBase):
    def __init__(self, player_id, callback, name, debug=False):
        super().__init__(player_id, name, debug)
        self.callback = callback

    def make_guess(self, game, previous_play=""):
        while True:  # This loop is for interactive attempts, not for re-tries in `play_one_game`
            instruction = f"\nEnter your move: "
            text_guess = self.callback(instruction)  # This calls console_callback for input

            try:
                # Attempt to parse the input according to the expected format
                if game.name == "shapes":
                    guess = int(text_guess)
                    return guess
                elif game.name in ["connectfour"]:
                    col = int(text_guess)
                    return col if 0 <= col < game.cols and game.board[0][col] == '.' else None
                else:
                    row, col = map(int, text_guess.split())
                    return (row, col) if 0 <= row < game.board_size and 0 <= col < game.board_size and game.board[row][col] == " " else None
            except (ValueError, IndexError):
                # If input parsing fails or doesn't meet the criteria, inform the player and allow another attempt
                self.collect_message("Invalid input. Please enter the column number for ConnectFour or row and column numbers separated by a space for other games.")
                return None  # Signaling `play_one_game` to handle this as an invalid move
            
class RandomPlayer(PlayerBase):
    def __init__(self, player_id, name, debug=False):
        super().__init__(player_id, name, debug)

    def make_guess(self, game, previous_play=""):
        if game.name == "shapes":
            return random.randint(0, 3)
        elif game.name == "connectfour":
            # For ConnectFour, find columns that are not full
            available_cols = [col for col in range(game.cols) if game.board[0][col] == '.']
            return random.choice(available_cols)
        elif game.name == "tictactoe":
            
            # For TicTacToe, find empty positions on the board
            available_moves = [(row, col) for row in range(game.board_size) for col in range(game.board_size) if game.board[row][col] == " "]
            return random.choice(available_moves)
        elif game.name == "battleship":
            guess_board = game.guess_board_p1 if self.player_id == 0 else game.guess_board_p2
            available_moves = [(row, col) for row in range(game.board_size) for col in range(game.board_size) if guess_board[row][col] == "~"]
            return random.choice(available_moves)

def run_game_series(game_instance, player1, player2, num_games, max_invalid_attempts, size, debug=False):
    """
    Run a series of games between an LLM player and a random player,
    tallying the results and collecting messages at the end.

    Parameters:
    - game_instance: The class instance of the game to be instantiated for each match.
    - player1, player2: Instances of PlayerBase, representing the two players.
    - num_games: Number of games to play in the series.
    - debug: If True, debug messages will be printed.

    Returns:
    - results: A dictionary with the tally of results ('P1 Wins', 'P2 Wins', 'Ties').
    - all_game_messages: A list containing the messages from each game played.
    """
    results = {'P1 Wins': 0, 'P2 Wins': 0, 'Ties': 0, 'P1 Wrong Moves': 0, 'P2 Wrong Moves': 0}
    all_game_messages = []
    all_game_logs = []  # To collect detailed move logs for each game

    for i in range(num_games):
        if debug:
            print(f"Game Iteration {i + 1}")
            print(f"="*50)
        game_messages, wrong_moves, game_log, player = play_one_game(game_instance, player1, player2, size, debug=debug)
        all_game_messages.append(game_messages)
        all_game_logs.extend(game_log)  # Append the move log from each game

        if player == 0:
            results['P1 Wins'] += 1
        elif player == 1:
            results['P2 Wins'] += 1
        elif player == 2:
            results['Ties'] += 1

        # Accumulate wrong move counts
        results['P1 Wrong Moves'] += wrong_moves[0]
        results['P2 Wrong Moves'] += wrong_moves[1]

        # Early stopping if Player 2's wins are not being counted
        if results['P2 Wins'] == 0 and results['P1 Wins'] + results['Ties'] > 10:
            print(f"Stopping early: Player 2 (random player) has not won after {i + 1} games.")
            break

    return results, all_game_messages, all_game_logs

def play_one_game(game_instance, player1, player2, size, max_invalid_attempts=1, debug=False):
    game_instance.reset_board()
    
    players = [player1, player2]
    current_player_index = 0 if game_instance.current_player == "P1" else 1
    game_messages = []
    invalid_attempts = [0, 0]  # Track invalid attempts for both players
    invalid_moves = [0, 0]
    move_log = []
    turn = 0
    
    def collect_game_message(message):
        """Collects or prints game-related messages based on debug mode."""
        game_messages.append(message)
        if debug:
            print(message)

    collect_game_message(game_instance.prompt)

    previous_play = ""

    while not game_instance.game_over:
        current_player = players[current_player_index]

        # Loop to handle repeated invalid moves without switching player
        collect_game_message(game_instance.get_text_state(current_player_index))
        collect_game_message(f"{current_player.name}'s turn to guess.")

        guess = current_player.make_guess(game_instance, previous_play)

        previous_play = guess

        if guess is not None:  # Proceed if a guess was made
            message, valid_move = game_instance.guess(current_player_index, guess, current_player)
            collect_game_message(message)

            if not valid_move:
                invalid_attempts[current_player_index] += 1
                invalid_moves[current_player_index] += 1

                if invalid_attempts[current_player_index] >= max_invalid_attempts:
                    # End game if max invalid attempts are exceeded
                    game_instance.game_over = True
                    winning_message = f"Game over. {players[1 - current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                    collect_game_message(winning_message)
                    return game_messages, invalid_moves, move_log, 1 - current_player_index  # Correctly return the other player as the winner
            else:
                invalid_attempts[current_player_index] = 0  # Reset on valid move
                if game_instance.name == "shapes":
                    move_log.append({
                        "player": current_player_index,
                        "chosen_shape": Shapes.answer_options[guess],  # the shape chosen by the player
                        "correct_shape": game_instance.shape,  # the actual shape on the board
                        "turn": turn
                    })
                else:
                    move_log.append({
                        "player": current_player_index,
                        "move": guess,
                        "turn": turn
                    })
            
        else:  # Handle case where guess is None (invalid input not leading to a guess)
            invalid_attempts[current_player_index] += 1
            invalid_moves[current_player_index] += 1
            if invalid_attempts[current_player_index] >= max_invalid_attempts:
                # End game if max invalid attempts are exceeded
                game_instance.game_over = True
                winning_message = f"{players[1 - current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                collect_game_message(winning_message)
                return game_messages, invalid_moves, move_log, 1 - current_player_index  # Correctly return the other player as the winner

        turn += 1
        if game_instance.game_over:
            final_state_message = game_instance.get_text_state(current_player_index)
            collect_game_message(final_state_message)
            if game_instance.check_win():
                outcome_message = f"Congratulations, {players[current_player_index].name} wins!"  # Change the reference here
                collect_game_message(f"Game Over. {outcome_message}")
                return game_messages, invalid_moves, move_log, current_player_index  # Return the correct player index
                
            elif game_instance.check_tie():
                outcome_message = "It's a tie!"
                collect_game_message(f"Game Over. {outcome_message}")
                return game_messages, invalid_moves, move_log, 2
                
            elif game_instance.check_loss():
                outcome_message=f"{players[1 - current_player_index].name} loses"  # Adjust reference to the correct player
                collect_game_message(f"Game Over. {outcome_message}")
                return game_messages, invalid_moves, move_log, current_player_index # count as win for other player
                
            collect_game_message("Game ended unexpectedly.")
            return game_messages, invalid_moves, move_log, -1

        current_player_index = 1 - current_player_index  # Switch turns only after valid move or game over

def play_random_moves(game, iter, debug=False):
    players = [RandomPlayer(0, "Random 1"), RandomPlayer(1, "Random 2")]
    current_player_index = 0
    game_dataset = []

    moves_played = 0
    while moves_played < iter and not game.game_over:
        current_player = players[current_player_index]

        valid_move_made = False
        while not valid_move_made and not game.game_over:
            guess = current_player.make_guess(game)
            valid_move_made = game.guess(current_player_index, guess, current_player)
            game_state = game.get_text_state(current_player_index)

            if valid_move_made:
                game_dataset.append({
                    "board_state": game_state,
                    "player": current_player.name,
                    "move": guess,
                    "winner": None  # To be updated after the game ends
                })
                moves_played += 1
                if debug:
                    print(current_player.name)
                    print(guess)
                    print(game_state)
    
        current_player_index = 1 - current_player_index

    winner = 'Tie' if game.check_tie() else game.check_win()
    for record in game_dataset:
        record["winner"] = winner

    return game_dataset, game

def load_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def generate_game_dataset(game, moves_per_game, num_games):
    all_games_dataset = []

    for _ in range(num_games):
        game_dataset = play_random_moves(game, moves_per_game)
        all_games_dataset.extend(game_dataset)

    return all_games_dataset

def save_dataset_to_json(dataset, file_name):
    with open(file_name, 'w') as f:
        json.dump(dataset, f, indent=4)

def load_and_print_board_state(file_name, record_index=0):
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for i in range(record_index):
            board_state = dataset[i]["board_state"]
            print(board_state)

def plot_shapes_heatmap(all_moves, save_path):
    shape_options = Shapes.answer_options
    heatmaps = np.zeros((len(shape_options), len(shape_options)), dtype=int)

    for move_info in all_moves:
        correct_index = shape_options.index(move_info['correct_shape'])
        chosen_index = shape_options.index(move_info['chosen_shape'])
        heatmaps[correct_index, chosen_index] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmaps, annot=True, fmt='d', cmap='coolwarm', xticklabels=shape_options, yticklabels=shape_options, annot_kws={"size": 12})
    ax.set_title('Heatmap of Predicted vs. Actual Shapes', fontsize=14)
    ax.set_xlabel('Chosen Shape', fontsize=12)
    ax.set_ylabel('Correct Shape', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(save_path)
    plt.close(fig)

def plot_heatmap(all_moves, n_games, game_name, board_size, players, save_path="heatmap.png", experiment_name=""):
    """
    Plot and save heatmaps of moves for each player.
    
    Args:
    all_moves (list): List of moves made by all players.
    n_games (int): Number of games played.
    game_name (str): Name of the game to differentiate handling (e.g., "connectfour").
    board_size (int): The dimension of the game board (n x n).
    players (list): List of player identifiers.
    save_path (str): File path to save the heatmap image.
    """
    heatmaps = [np.zeros((board_size, board_size)) if game_name != "connectfour" else np.zeros((1, board_size)) for _ in range(len(players))]

    # Count moves for each player
    for move_info in all_moves:
        player_index = move_info["player"]
        move = move_info["move"]
        # if game_name == "shapes":
        #     chosen_index, correct_shape = move_info["move"]
        #     correct_index = Shapes.answer_options.index(correct_shape)
        #     heatmaps[player_index][correct_index, chosen_index] += 1
        if game_name == "connectfour":
            # For ConnectFour, increment column based on move
            heatmaps[player_index][0, move] += 1
        else:
            # For other games, moves are tuples (row, col)
            row, col = move
            heatmaps[player_index][row, col] += 1

    # Calculate total moves for normalization
    total_moves = sum(np.sum(heatmap) for heatmap in heatmaps)

    # Normalize the heatmaps to get percentages
    if total_moves > 0:
        heatmaps = [(heatmap / total_moves) * 100 for heatmap in heatmaps]  # Convert counts to percentages

    # Create a subplot for each player's heatmap
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(10 * len(heatmaps), 5))

    for i, heatmap in enumerate(heatmaps):
        sns.heatmap(heatmap, ax=axes[i], annot=True, cmap='coolwarm', fmt='.2f')
        axes[i].set_title(f'Player {players[i]} Moves Heatmap')

    plt.suptitle(f"Total moves played after {n_games} games of {game_name}")
    plt.savefig(save_path)
    plt.close(fig)

def console_callback(message):
    print(message)
    return input()

def load_and_aggregate_logs(path):
    aggregated_logs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'game_logs.json':
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    logs = json.load(f)
                    aggregated_logs.extend(logs)
    return aggregated_logs

def main():

    debug = True
    game_runs = []

    # Shapes experiment setup
    shapes_experiments_enabled = False  # Set to False if you don't want to run shapes experiments
    if shapes_experiments_enabled:
        shapes = ['square', 'triangle', 'cross']
        # models = ['oa:gpt-3.5-turbo-1106', 'oa:gpt-4-1106-preview']
        models = ['oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']

        temperatures = [0, 0.5, 1, 1.5]
        # temperatures = [0]

        num_games = 100

        for model in models:
            for temp in temperatures:
                for shape in shapes:
                    game_runs.append({
                        'game_class': Shapes,
                        'game_name': "shapes",
                        'board_size': 15,
                        'model_name': model,
                        'num_games': num_games,
                        'experiment_name': f'experiment_shapes/{model.replace(":", "_")}/{str(temp).replace(".", "_")}/{shape}',
                        'temperature': temp,
                        'shape': shape
                    })

    # Other games setup (e.g., Battleship, ConnectFour)
    board_games_enabled = True  # Set to False if you don't want to run other games
    if board_games_enabled:
        game_runs += [
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_0', 'temperature': 0},
            
            # gpt-4o-mini-2024-07-18
            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_0', 'temperature': 0},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_0', 'temperature': 0},

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_0.5', 'temperature': 0.5},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_0.5', 'temperature': 0.5},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_0.5', 'temperature': 0.5},

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_1', 'temperature': 1},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_1', 'temperature': 1},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_1', 'temperature': 1},

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_mini_oneshot_temp_1.5', 'temperature': 1.5},
            {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_mini_oneshot_temp_1.5', 'temperature': 1.5},
            {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-mini-2024-07-18', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_mini_oneshot_temp_1.5', 'temperature': 1.5},
            
            # gpt-4o-2024-08-06
            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_0', 'temperature': 0},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_0', 'temperature': 0}

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_0.5', 'temperature': 0.5},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_0.5', 'temperature': 0.5},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_0.5', 'temperature': 0.5},

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_1', 'temperature': 1},
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_1', 'temperature': 1},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_1', 'temperature': 1},

            # {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt4o_oneshot_temp_1.5', 'temperature': 1.5},
            {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_connectfour_gpt4o_oneshot_temp_1.5', 'temperature': 1.5},
            {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-4o-2024-08-06', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4o_oneshot_temp_1.5', 'temperature': 1.5}

            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'oa:gpt-3.5-turbo-1106', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt3_5_oneshot_temp_0', 'temperature': 0}
            # Mistral
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:mistralai/Mistral-7B-Instruct-v0.2', 'num_games': 100, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_mistral_7B_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'num_games': 100, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_mistral_7B_oneshot_temp_0', 'temperature': 0},

            # distilbert/distilgpt2
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:distilbert/distilgpt2', 'num_games': 100, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_distilgpt2_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:distilbert/distilgpt2', 'num_games': 100, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_distilgpt2_oneshot_temp_0', 'temperature': 0},

            # # openai-community/gpt2
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:openai-community/gpt2', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_gpt2_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:openai-community/gpt2', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_gpt2_oneshot_temp_0', 'temperature': 0},

            # tiiuae/falcon-mamba-7b
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:tiiuae/falcon-7b-instruct', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_falcon_mamba_7B_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:tiiuae/falcon-7b-instruct', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_falcon_mamba_7B_oneshot_temp_0', 'temperature': 0},

            # # lmms-lab/llama3-llava-next-8b
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:lmms-lab/llama3-llava-next-8b', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_llama3_llava_8B_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:lmms-lab/llama3-llava-next-8b', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_llama3_llava_8B_oneshot_temp_0', 'temperature': 0},

            # # meta-llama/Meta-Llama-3-8B
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:meta-llama/Meta-Llama-3-8B', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_llama_3_8B_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:meta-llama/Meta-Llama-3-8B', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_llama_3_8B_oneshot_temp_0', 'temperature': 0},

            # # lmms-lab/llama3-llava-next-8b
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:lmms-lab/llama3-llava-next-8b', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_llama3_llava_next_8b_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:lmms-lab/llama3-llava-next-8b', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_llama3_llava_next_8b_oneshot_temp_0', 'temperature': 0},

            # # xlnet/xlnet-base-cased
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:xlnet/xlnet-base-cased', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_xlnet_base_cased_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:xlnet/xlnet-base-cased', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_xlnet_base_cased_oneshot_temp_0', 'temperature': 0},

            # # microsoft/Phi-3-mini-4k-instruct
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:microsoft/Phi-3-mini-4k-instruct', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_phi_3_mini_4k_instruct_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:microsoft/Phi-3-mini-4k-instruct', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_phi_3_mini_4k_instruct_oneshot_temp_0', 'temperature': 0},

            # # google/gemma-2-9b-it
            # {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'hf:google/gemma-2-9b-it', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_connectfour_gemma_2_9b_it_oneshot_temp_0', 'temperature': 0},
            # {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'hf:google/gemma-2-9b-it', 'num_games': 10, 'experiment_name': 'experiment_board_games/hf_experiment_tictactoe_gemma_2_9b_it_oneshot_temp_0', 'temperature': 0}
        ]

    if not game_runs:
        print("No experiments to run. Please enable either shapes experiments, other games, or both.")
        return

    aggregated_results = {'P1 Wins': 0, 'P2 Wins': 0, 'Ties': 0, 'P1 Wrong Moves': 0, 'P2 Wrong Moves': 0}
    total_time = 0
    time_log_filename = "game_time_log.txt"

    with open(time_log_filename, "a") as file:

        aggregated_logs = {}

        for game in game_runs:
            folder_name = game['experiment_name']
            os.makedirs(folder_name, exist_ok=True)

            start_time = time.time()

            # Here we are now creating an instance of the game instead of passing the class
            game_instance = game['game_class'](options={"board_size": game['board_size']})

            if game['game_name'] == 'shapes':
                game_instance.shape = game['shape']
                model_temp_key = (game['model_name'], game['temperature'])

                if model_temp_key not in aggregated_logs:
                    aggregated_logs[model_temp_key] = []

                player1 = LLMPlayer(game_instance, model_name=game['model_name'], temperature=game['temperature'], debug=debug)

            else:
                if game['model_name'] == 'random':
                    player1 = RandomPlayer(0, "Random", debug=debug)
                else:
                    player1 = LLMPlayer(game_instance, model_name=game['model_name'], temperature=game['temperature'], debug=debug)

            player2 = RandomPlayer(1, "Random", debug=debug)
            num_games = game['num_games']

            results, all_game_messages, all_game_logs = run_game_series(game_instance, player1, player2, num_games, 1, game['board_size'], debug)

            save_dataset_to_json(results, folder_name + '/results_' + game['game_name'] + '.json')
            save_dataset_to_json(all_game_messages, folder_name + '/game_messages_' + game['game_name'] + '.json')
            save_dataset_to_json(all_game_logs, folder_name + '/game_logs_' + game['game_name'] + '.json')

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            file.write(f"Experiment: {game['game_name']}, Time: {elapsed_time:.2f} seconds\n")

            for key in aggregated_results:
                aggregated_results[key] += results[key]

            if game['game_name'] == 'shapes':
                aggregated_logs[model_temp_key].extend(all_game_logs)

        # Generate heatmaps for shapes experiments
        if shapes_experiments_enabled:
            base_path = 'experiment_shapes'
            for model in models:
                for temp in temperatures:
                    all_moves = []
                    base_path_model_temp = f"{base_path}/{model.replace(':', '_')}/{str(temp).replace('.', '_')}"
                    
                    for shape in shapes:
                        shape_path = f"{base_path_model_temp}/{shape}"
                        shape_moves = load_and_aggregate_logs(shape_path)
                        all_moves.extend(shape_moves)

                        shape_heatmap_path = f"{shape_path}/{model.replace(':', '_')}_{str(temp).replace('.', '_')}_{shape}_heatmap.png"
                        plot_shapes_heatmap(shape_moves, shape_heatmap_path)

                    combined_heatmap_path = f"{base_path_model_temp}/{model.replace(':', '_')}_{str(temp).replace('.', '_')}_combined_heatmap.png"
                    plot_shapes_heatmap(all_moves, combined_heatmap_path)

            print("Heatmaps generated for all model/temperature conditions in shapes experiments.")

        print(f"Aggregated Results after {sum(game['num_games'] for game in game_runs)} games:", aggregated_results)

    # If needed, bar plots for shapes experiments can also be generated
    if shapes_experiments_enabled:
        bar_plot_shapes(base_path, models, temperatures, shapes)

if __name__ == "__main__":
    main()