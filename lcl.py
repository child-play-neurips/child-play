import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pandas as pd
import re
import seaborn as sns
import ast
import openai
import time
import os

from wrapper import ask

class LCLVisualizer:
    def __init__(self):
        pass

    def draw_piece(self, ax, position, color='blue'):
        ax.add_patch(patches.Rectangle(position, 4, 1, edgecolor='black', facecolor=color))

    def display_construct(self, pieces, filename):
        fig, ax = plt.subplots()
        for piece in pieces:
            self.draw_piece(ax, (piece[0], piece[1]), piece[2])
        
        min_x = min(p[0] for p in pieces)
        max_x = max(p[0] for p in pieces) + 4  # Account for piece width
        min_y = min(p[1] for p in pieces)
        max_y = max(p[1] for p in pieces) + 1

        ax.set_xticks([x for x in range(int(min_x), int(max_x) + 1)])
        ax.set_xlim(left=min_x, right=max_x)
        ax.set_ylim(bottom=min_y, top=max_y)
        ax.set_xticklabels([x for x in range(int(min_x), int(max_x) + 1)])
        ax.set_yticks(range(min_y, max_y + 1))
        ax.set_aspect('equal', 'box')
        ax.axis('on')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Studs')
        ax.set_ylabel('Layers')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

class LCLGame:
    def __init__(self):
        self.pieces = []
        self.valid_colors = ['blue', 'yellow', 'red']
        self.metrics = {
            "correct_validations": 0,
            "incorrect_validations": 0,
            "valid_constructs": 0,
            "invalid_constructs": 0
        }

    def can_place_piece(self, x, y, existing_pieces):
        # Check for horizontal overlap
        for ex, ey, _ in existing_pieces:
            if ey == y and not set(range(x, x + 4)).isdisjoint(set(range(ex, ex + 4))):
                return False  # Overlap detected

        # Check for support: each piece above the first layer must be supported by at least one piece directly below it
        if y > 0:
            supported = False
            for ex, ey, _ in existing_pieces:
                if ey == y - 1 and (ex <= x < ex + 4 or ex < x + 4 <= ex + 4):
                    supported = True
                    break
            if not supported:
                return False  # No support found

        # Check for connectivity: pieces must connect via overlapping pegs
        connected = False
        for ex, ey, _ in existing_pieces:
            # Check for horizontal connectivity via overlapping pegs
            if (ey == y and (ex + 2 == x or ex - 2 == x)) or \
               (ey == y + 1 and (ex <= x + 3 and ex >= x - 3)) or \
               (ey == y - 1 and (ex <= x + 3 and ex >= x - 3)):
                connected = True
                break

        if not connected and y > 0:
            return False  # Not properly connected

        # Additional check for base layer: ensure no gaps in horizontal connection without vertical support
        if y == 0:
            left_neighbor = any(ex == x - 4 and ey == y for ex, ey, _ in existing_pieces)
            right_neighbor = any(ex == x + 4 and ey == y for ex, ey, _ in existing_pieces)
            if left_neighbor or right_neighbor:
                return False  # No connecting piece above to bridge the gap

        return True

    def find_all_valid_positions(self, existing_pieces):
        valid_positions = set()
        # Check around and directly above each existing piece for all possible peg connections
        for ex, ey, _ in existing_pieces:
            candidate_positions = [
                (ex - 4, ey), (ex + 4, ey), (ex, ey + 1),
                (ex - 2, ey), (ex + 2, ey), (ex - 2, ey + 1), (ex + 2, ey + 1)
            ]
            for x, y in candidate_positions:
                if self.can_place_piece(x, y, existing_pieces):
                    valid_positions.add((x, y))
        return list(valid_positions)

    def build_random_valid_assembly(self, num_pieces):
        if num_pieces < 1:
            return []

        # Start with the first piece at (0, 0)
        self.pieces = [(0, 0, random.choice(self.valid_colors))]

        # Try to place remaining pieces at valid positions
        for _ in range(1, num_pieces):
            valid_positions = self.find_all_valid_positions(self.pieces)
            if not valid_positions:
                print("No valid positions available, stopping early")
                break
            new_x, new_y = random.choice(valid_positions)
            new_color = random.choice(self.valid_colors)
            self.pieces.append((new_x, new_y, new_color))

        return self.pieces

    def generate_random_piece(self):
        x = random.randint(0, 10) * 4
        y = random.randint(0, 10)
        color = random.choice(self.valid_colors)
        return (x, y, color)

    def generate_random_construct(self, num_pieces):
        self.pieces = [self.generate_random_piece() for _ in range(num_pieces)]
        return self.pieces

    def is_valid_construct(self, pieces):
        occupied_positions = {}
        for x, y, color in pieces:
            # Create a set of x positions for each piece based on its width (4 studs)
            piece_positions = set(range(x, x + 4))
            if y in occupied_positions:
                # Check for any overlap in x positions with existing pieces at the same y level
                if any(pos in occupied_positions[y] for pos in piece_positions):
                    return False  # Overlap detected
                occupied_positions[y].update(piece_positions)
            else:
                # Initialize occupied positions for this y level
                occupied_positions[y] = set(piece_positions)
        return True

    def generate_valid_or_invalid_construct(self, num_pieces, valid=True):
        if valid:
            self.pieces = self.build_random_valid_assembly(num_pieces)
            print(f"Generated valid construct: {self.pieces}")
            return self.pieces
        else:
            # Generate invalid constructs randomly
            while True:
                self.pieces = self.generate_random_construct(num_pieces)
                if not self.is_valid_construct(self.pieces):
                    self.metrics["invalid_constructs"] += 1
                    print(f"Generated invalid construct: {self.pieces}")
                    return self.pieces

    def create_tower(self, height):
        self.pieces = [(0, i, random.choice(self.valid_colors)) for i in range(height)]
        return self.pieces

    def create_bridge(self, height):
        self.pieces = [(0, i, random.choice(self.valid_colors)) for i in range(height)]
        self.pieces += [(6, i, random.choice(self.valid_colors)) for i in range(height)]
        self.pieces += [(3, height, random.choice(self.valid_colors))]  # Correctly place the middle piece on top
        return self.pieces

    def create_staircase(self, steps):
        self.pieces = [(i * 2, i, random.choice(self.valid_colors)) for i in range(steps)]
        return self.pieces

    def validate_construct(self, player_answer):
        actual_validity = self.is_valid_construct(self.pieces)
        if (player_answer == "valid" and actual_validity) or (player_answer == "invalid" and not actual_validity):
            self.metrics["correct_validations"] += 1
        else:
            self.metrics["incorrect_validations"] += 1

    def save_metrics(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.metrics, f)

class RandomPlayer:
    def __init__(self):
        pass

    def generate_random_answer(self):
        return random.choice(["valid", "invalid"])

class LLMPlayer:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature

    def generate_llm_answer_validity(self, prompt):
        # This function is intended for Game 1 where a simple 'valid' or 'invalid' is expected.
        try:
            response = ask(api_messages=[{"role": "system", "content": prompt}], temperature=self.temperature, model=self.model)
            # Directly return the response if it is either 'valid' or 'invalid'.
            response = response.strip().lower()
            print(f"Model {self.model} response: {response}")
            if response in ['valid', 'invalid']:
                return response
            else:
                return "invalid"  # Default to invalid if the response isn't clear
        except openai.InternalServerError as e:
            return "invalid"

    def generate_llm_answer_list(self, prompt):
        # This function is intended for Game 2 where a list of pieces is expected.
        try:
            response = ask(api_messages=[{"role": "system", "content": prompt}], temperature=self.temperature, model=self.model)
            # Use regex to extract the list, handling multiline input
            match = re.search(r'\[\s*(.*?)\s*\]', response, re.DOTALL)
            print(f"Model {self.model} response: {response}, extracted: {match}")

            if match:
                # Use the captured group from the match, which is the content inside the brackets
                content_to_evaluate = match.group(1)
                print(f"Content to evaluate: {content_to_evaluate}")
                # Evaluate the extracted string content
                evaluated_response = ast.literal_eval(f'[{content_to_evaluate}]')
                print(f"Evaluated response: {evaluated_response}")
                return evaluated_response
            else:
                raise ValueError("Invalid format for list")
        except (openai.InternalServerError, ValueError) as e:
            print(f"Error: {str(e)}")
            return [(0, 0, 'red'), (0, 0, 'blue')]

def main():
    game = LCLGame()
    visualizer = LCLVisualizer()
    os.makedirs('./lcl_experiments/validity_experiments', exist_ok=True)
    os.makedirs('./lcl_experiments/construct_generation', exist_ok=True)

    n_experiments = 100
    models = ['oa:gpt-4-1106-preview', 'oa:gpt-3.5-turbo-1106']
    # models = ["random", "oa:gpt-3.5-turbo-1106"]

    temperatures = [0, 0.5, 1, 1.5]

    all_validity_results = []
    all_construct_results = []

    # Game 2: Construct Generation
    n_pieces = 3
    for model in models:
        for temperature in temperatures:
            for i in range(n_experiments):
                pieces = game.generate_valid_or_invalid_construct(n_pieces) if model == 'random' else LLMPlayer(model=model, temperature=temperature).generate_llm_answer_list(
                    f"A description of a Lego structure consists of a list of tuples, [(x1, y1, 'color1'), (x2, y2, 'color2')], where each tuple shows the coordinates and colors of a piece. Such a structure is valid if all Lego pieces are connected but not overlapping. A Lego piece is connected through interlocking pegs, not by merely touching sides. Two Lego pieces overlap when they share the same y-coordinate and any part of their length has the same x-coordinate. Produce a description of a valid structure using {n_pieces} Lego pieces. Reply only with the Lego structure description following the format [(x1, y1, 'color1'), (x2, y2, 'color2'), ...], write nothing else but the structure.")
                
                print(f"{model} answer: {pieces}")
            
                validity = game.is_valid_construct(pieces)
                all_construct_results.append({"Temperature": temperature, "Model": model, "Experiment": i + 1, "Valid": validity, "LLM Response": pieces})
                filename = f'./lcl_experiments/construct_generation/{model}_temp_{temperature}_experiment_{i+1}.svg'
                visualizer.display_construct(pieces, filename)

    # Game 1: Validity Testing
    for model in models:
        for temperature in temperatures:
            validity_results = []
            for i in range(n_experiments):
                is_valid = i < n_experiments/2
                pieces = game.generate_valid_or_invalid_construct(5, valid=is_valid)
                player_answer = RandomPlayer().generate_random_answer() if model == 'random' else LLMPlayer(model=model, temperature=temperature).generate_llm_answer_validity(
                    f"You will receive a description of a Lego structure, for instance, [(x1, y1, 'color1'), (x2, y2, 'color2')], which lists the coordinates and colors of two pieces. A construct is valid if all Lego pieces are connected but not overlapping. A Lego piece is connected through interlocking pegs, not by merely touching sides. Two Lego pieces overlap when they share the same y-coordinate and any part of their length has the same x-coordinate. If the following structure is valid then reply with valid, otherwise reply with invalid (do not justify your answer): {pieces}")
                actual_validity = game.is_valid_construct(pieces)
                correct = (player_answer == "valid" and actual_validity) or (player_answer == "invalid" and not actual_validity)
                validity_results.append({"Temperature": temperature, "Model": model, "Experiment": i + 1, "Player Answer": player_answer, "Actual Validity": actual_validity, "Correct": correct, "LLM Response": pieces})
                filename = f'./lcl_experiments/validity_experiments/{model}_temp_{temperature}_experiment_{i+1}.png'
                visualizer.display_construct(pieces, filename)

            all_validity_results.extend(validity_results)


    df_validity = pd.DataFrame(all_validity_results)
    df_construct = pd.DataFrame(all_construct_results)
    df_construct.to_csv("df_construct.csv", index=False)
    df_validity.to_csv("df_validity.csv", index=False)
    print(df_validity[:10])
    print(df_construct[:10])

if __name__ == "__main__":
    main()
