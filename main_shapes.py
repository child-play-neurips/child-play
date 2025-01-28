import os
import random
import json
import re
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wrapper import ask  # Assuming you have a wrapper for the LLM API

from scripts_games.shapes import Shapes, bar_plot_shapes

class LLMPlayer:
    def __init__(self, game, model_name="oa:gpt-3.5-turbo-0125", temperature=0, debug=False):
        self.name = "LLM"
        self.model_name = model_name
        self.game = game
        self.temperature = temperature
        self.debug = debug

    def make_guess(self):
        api_messages = [
            {"role": "system", "content": f"You are playing the Shapes game. {self.game.prompt}"}
        ]
        user_prompt = (
            f"Here is the current game state:\n{self.game.get_text_state()}\n"
            f"Please provide only the number corresponding to your choice, and nothing else.\n"
            f"My move is: "
        )
        api_messages.append({"role": "user", "content": user_prompt})

        if self.debug:
            print(f"\n[LLMPlayer] Prompt to LLM:\n{user_prompt}")

        response = ask(api_messages=api_messages, temperature=self.temperature, model=self.model_name)

        if self.debug:
            print(f"\n[LLMPlayer] Received Response from Model:\n{response}\n")

        return self.parse_move(response)

    def parse_move(self, response):
        """Parse the response from the model and extract the move."""
        try:
            # Use regex to find the first integer in the response
            match = re.search(r'\b\d+\b', response)
            if match:
                guess = int(match.group())
                return guess
            else:
                if self.debug:
                    print(f"[LLMPlayer] No valid integer found in response: {response}")
                return None
        except ValueError:
            if self.debug:
                print(f"[LLMPlayer] Failed to parse move. Response was: {response}")
            return None

def run_game_series(game_instance, player, num_games, debug=False):
    results = {'Wins': 0, 'Losses': 0}
    all_game_messages = []
    all_game_logs = []

    for i in range(num_games):
        if debug:
            print(f"\n=== Starting Game {i + 1} ===")
            print(f"{'='*50}")
        game_instance.reset_board()
        game_messages, game_log, outcome = play_one_game(game_instance, player, debug=debug)
        all_game_messages.append(game_messages)
        all_game_logs.extend(game_log)

        if outcome == 0:
            results['Wins'] += 1
            if debug:
                print(f"Result: {player.name} Wins")
        elif outcome == 1:
            results['Losses'] += 1
            if debug:
                print(f"Result: {player.name} Loses")

    return results, all_game_messages, all_game_logs

def play_one_game(game_instance, player, debug=False):
    game_messages = []
    move_log = []
    turn = 0

    def collect_game_message(message):
        game_messages.append(message)
        if debug:
            print(message)

    collect_game_message(game_instance.prompt)
    if debug:
        print(f"[Game Start] {game_instance.prompt}")

    collect_game_message(game_instance.get_text_state())
    collect_game_message(f"{player.name}'s turn to guess.")
    if debug:
        print(f"\n[Turn {turn + 1}] {player.name}'s turn.")

    guess = player.make_guess()

    if guess is not None:
        message, valid_move = game_instance.guess(guess)
        collect_game_message(message)
        if debug:
            print(f"[Move] {message}")

        if valid_move:
            if message == "Win":
                move_log.append({
                    "player": 0,
                    "chosen_shape": game_instance.answer_options[guess],
                    "correct_shape": game_instance.shape,
                    "turn": turn,
                    "result": "Win"
                })
            elif message == "Loss":
                move_log.append({
                    "player": 0,
                    "chosen_shape": game_instance.answer_options[guess],
                    "correct_shape": game_instance.shape,
                    "turn": turn,
                    "result": "Loss"
                })
    else:
        # Handle invalid input
        message = "Invalid input. Failed to parse your move."
        collect_game_message(message)
        if debug:
            print(f"[Move] {message}")
        move_log.append({
            "player": 0,
            "chosen_shape": None,
            "correct_shape": game_instance.shape,
            "turn": turn,
            "result": "Invalid Input"
        })

    # Game is over after one guess in "shapes"
    final_state_message = game_instance.get_text_state()
    collect_game_message(final_state_message)
    if debug:
        print(f"[Final State]\n{final_state_message}")

    if game_instance.check_win():
        outcome_message = f"Congratulations, {player.name} wins!"
        collect_game_message(f"Game Over. {outcome_message}")
        if debug:
            print(f"[Game Over] {outcome_message}")
        return game_messages, move_log, 0
    elif game_instance.check_loss():
        outcome_message = f"{player.name} loses."
        collect_game_message(f"Game Over. {outcome_message}")
        if debug:
            print(f"[Game Over] {outcome_message}")
        return game_messages, move_log, 1  # Indicate loss
    else:
        collect_game_message("Game ended unexpectedly.")
        if debug:
            print("[Game Over] Game ended unexpectedly.")
        return game_messages, move_log, -1

def save_dataset_to_json(dataset, file_name):
    with open(file_name, 'w') as f:
        json.dump(dataset, f, indent=4)

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
    debug = True  # Set to True to see detailed outputs
    game_runs = []

    # Shapes experiment setup
    shapes = ['square', 'triangle', 'cross']
    models = ['oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']
    temperatures = [0, 0.5, 1, 1.5]
    num_games = 25  # Adjust the number of games as needed

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

    for game in game_runs:
        folder_name = game['experiment_name']
        os.makedirs(folder_name, exist_ok=True)

        start_time = time.time()

        # Create an instance of the Shapes game
        game_instance = game['game_class'](options={"board_size": game['board_size']}, shape=game['shape'])

        player = LLMPlayer(game_instance, model_name=game['model_name'], temperature=game['temperature'], debug=debug)

        num_games = game['num_games']

        results, all_game_messages, all_game_logs = run_game_series(game_instance, player, num_games, debug)

        save_dataset_to_json(results, os.path.join(folder_name, f'results.json'))
        save_dataset_to_json(all_game_messages, os.path.join(folder_name, f'game_messages_{game["game_name"]}.json'))
        save_dataset_to_json(all_game_logs, os.path.join(folder_name, f'game_logs_{game["game_name"]}.json'))

        elapsed_time = time.time() - start_time

        print(f"[Experiment] {game['game_name']} with model {game['model_name']} at temperature {game['temperature']} and shape {game['shape']} completed in {elapsed_time:.2f} seconds.")

    # Generate plots for shapes experiments
    base_path = 'experiment_shapes'
    bar_plot_shapes(base_path, models, temperatures, shapes)
    print("Bar plots generated for shapes experiments.")

if __name__ == "__main__":
    main()
