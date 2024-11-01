import math
import random
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

empty_character = "0"
full_character = "1"

def bar_plot_shapes(base_path, models, temperatures, shapes):
    plt.rcParams.update({'font.size': 14})
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        fig.suptitle(f'Correct and Incorrect Answers by Shape for {model}', fontsize=16, fontweight='bold')

        axes = axes.flatten()
        for idx, temp in enumerate(temperatures):
            path = os.path.join(base_path, model.replace(":", "_"), str(temp).replace(".", "_"))
            data = []

            for shape in shapes:
                counts = load_and_aggregate_logs(path, shape)
                data.append({'Shape': shape, 'Count': counts['correct'], 'Type': 'Correct'})
                data.append({'Shape': shape, 'Count': counts['incorrect'], 'Type': 'Incorrect'})

            df = pd.DataFrame(data)
            bar_plot = sns.barplot(x='Shape', y='Count', hue='Type', data=df, ax=axes[idx],
                                   palette=['green', 'red'], alpha=0.75, dodge=True)

            # Annotate each bar with its height
            for p in bar_plot.patches:
                height = p.get_height()
                if height > 0:  # Only annotate non-zero bars
                    bar_plot.annotate(f'{int(height)}',
                                      (p.get_x() + p.get_width() / 2., height),
                                      ha='center', va='bottom',
                                      xytext=(0, 5), textcoords='offset points')

            axes[idx].set_title(f'Temperature {temp}', fontweight='bold')
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].set_ylim(0, df['Count'].max() + 5)  # Add some space for annotation

            if idx == len(temperatures) - 1:
                axes[idx].legend(title='Answer Type')
            else:
                axes[idx].get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(base_path, model.replace(":", "_"), f'answers_summary_{model.replace(":", "_")}.png'))
        plt.close()

def load_and_aggregate_logs(path, shape):
    correct = 0
    incorrect = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'game_logs.json':
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    logs = json.load(f)
                    for log in logs:
                        if log['correct_shape'] == shape:
                            if log['result'] == "Win":
                                correct += 1
                            elif log['result'] == "Loss":
                                incorrect += 1
    return {'correct': correct, 'incorrect': incorrect}

def create_board(width, height):
    return [[empty_character for _ in range(width)] for _ in range(height)]

def draw_rectangle(top_left, bottom_right, width, height):
    board = create_board(width, height)
    for y in range(top_left[1], bottom_right[1] + 1):
        for x in range(top_left[0], bottom_right[0] + 1):
            if 0 <= x < width and 0 <= y < height:
                board[y][x] = full_character
    return board

def draw_triangle(top, side_length, width, height):
    board = create_board(width, height)
    tri_height = int(side_length * math.sqrt(3) / 2)
    for y in range(tri_height):
        for x in range(-y, y + 1):
            plot_x = top[0] + x
            plot_y = top[1] + y
            if 0 <= plot_x < width and 0 <= plot_y < height:
                board[plot_y][plot_x] = full_character
    return board

def draw_cross(center, arm_length, width, height):
    board = create_board(width, height)
    start_x = center[0] - arm_length
    end_x = center[0] + arm_length
    start_y = center[1] - arm_length
    end_y = center[1] + arm_length
    for x in range(start_x, end_x + 1):
        if 0 <= x < width:
            board[center[1]][x] = full_character
    for y in range(start_y, end_y + 1):
        if 0 <= y < height:
            board[y][center[0]] = full_character
    return board

class Shapes:
    possible_shapes = [
        "square",
        "triangle",
        "cross"
    ]

    answer_options = [
        *possible_shapes,
        "circle"
    ]

    def __init__(self, options=None, shape='square'):
        if options is None:
            self.board_size = 15
        else:
            self.board_size = options.get("board_size", 15)
            self.debug = options.get("debug", False)
        self.name = "shapes"
        self.shape = shape
        self.prompt = (f"Shapes is a game where you receive an {self.board_size} by {self.board_size} "
                       f"square matrix of {empty_character} and in it you will find a shape denoted by "
                       f"{full_character}. You will have multiple choices and you have to choose the correct "
                       f"option. Only output a number for the option. Output nothing else but the number.")
        self.reset_board()

    def reset_board(self):
        self.board = create_board(self.board_size, self.board_size)

        # Draw the specified shape randomly on the board
        if self.shape == "square":
            side_length = random.randint(2, self.board_size // 2)
            top_left_x = random.randint(0, self.board_size - side_length)
            top_left_y = random.randint(0, self.board_size - side_length)
            bottom_right_x = top_left_x + side_length - 1
            bottom_right_y = top_left_y + side_length - 1
            self.board = draw_rectangle((top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.board_size, self.board_size)

        elif self.shape == "triangle":
            side_length = random.randint(4, self.board_size // 2)
            top_x = random.randint(side_length, self.board_size - side_length)
            top_y = random.randint(0, self.board_size - side_length * 2)
            self.board = draw_triangle((top_x, top_y), side_length, self.board_size, self.board_size)

        elif self.shape == "cross":
            arm_length = random.randint(2, self.board_size // 4)
            center_x = random.randint(arm_length, self.board_size - arm_length - 1)
            center_y = random.randint(arm_length, self.board_size - arm_length - 1)
            self.board = draw_cross((center_x, center_y), arm_length, self.board_size, self.board_size)

        # Shuffle answer options to randomize the position of the correct shape
        answer_options = list(Shapes.answer_options)  # Make a copy of the class variable

        # Ensure the correct shape is included in the answer options
        if self.shape not in answer_options:
            answer_options.append(self.shape)

        random.shuffle(answer_options)  # Shuffle the answer options
        self.answer_options = answer_options  # Assign the shuffled list to the instance

        self.game_over = False
        self.won = False

    def get_text_state(self):
        text_board = "\n".join("".join(row) for row in self.board)
        text_answers = "Answers:\n" + "\n".join([f"{i}: {option}" for i, option in enumerate(self.answer_options)])
        prompt = text_board + "\n\n" + text_answers
        return prompt

    def guess(self, guess):
        if guess < 0 or guess >= len(self.answer_options):
            return "Invalid move. Out of range.", False

        if self.answer_options[guess] == self.shape:
            self.game_over = True
            self.won = True
            return "Win", True

        self.game_over = True
        self.won = False
        return "Loss", True

    def check_win(self) -> bool:
        return self.game_over and self.won

    def check_loss(self) -> bool:
        return self.game_over and not self.won