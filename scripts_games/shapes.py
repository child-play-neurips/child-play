import math
import random
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

empty_character="0"
full_character="1"

def bar_plot_shapes(base_path, models, temperatures, shapes):
    for model in models:
        # Setup figure for the model with subplots for each temperature, 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
        fig.suptitle(f'Correct and Incorrect Answers by Shape for {model}', fontsize=16)

        axes = axes.flatten()  # Flatten the 2x2 grid to easily index it
        for idx, temp in enumerate(temperatures):
            correct_counts = {shape: 0 for shape in shapes}
            total_counts = {shape: 0 for shape in shapes}
            
            # Construct the path for the current model and temperature
            path = f'{base_path}/{model.replace(":", "_")}/{str(temp).replace(".", "_")}'
            
            for shape in shapes:
                shape_path = os.path.join(path, shape)
                log_files = [f for f in os.listdir(shape_path) if f.endswith('game_logs.json')]
                
                for log_file in log_files:
                    # Load the log file
                    with open(os.path.join(shape_path, log_file), 'r') as file:
                        logs = json.load(file)
                        # Filter and count correct and total answers for the current shape
                        for log in logs:
                            total_counts[shape] += 1
                            if log['chosen_shape'] == log['correct_shape'] and log['correct_shape'] == shape:
                                correct_counts[shape] += 1
            
            # Prepare data for plotting
            data = []
            for shape in shapes:
                data.append({'Shape': shape, 'Count': correct_counts[shape], 'Type': 'Correct'})
                data.append({'Shape': shape, 'Count': total_counts[shape] - correct_counts[shape], 'Type': 'Incorrect'})
            
            df = pd.DataFrame(data)
            # Plotting with adjusted bar positions for slight overlap
            bar_plot = sns.barplot(x='Shape', y='Count', hue='Type', data=df, ax=axes[idx],
                                   palette=['green', 'red'], alpha=0.75, dodge=0.4)  # Adjust dodge
            axes[idx].set_title(f'Temperature {temp}')
            axes[idx].set_xlabel('Shape')
            axes[idx].set_ylabel('Count')
            axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure y-axis ticks are integers

            # Annotate each bar with its height
            for p in bar_plot.patches:
                height = p.get_height()
                if height > 0:  # Only annotate non-zero bars
                    bar_plot.annotate(f'{int(height)}', 
                                      (p.get_x() + p.get_width() / 2., height), 
                                      ha='center', va='center', 
                                      xytext=(0, 10), textcoords='offset points')

            if idx == len(temperatures) - 1:
                axes[idx].legend(title='Answer Type')
            else:
                axes[idx].get_legend().remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{base_path}/{model.replace(":", "_")}/answers_summary_{model}.png')
        plt.close()
        
def create_board(width, height):
    return [[empty_character for _ in range(width)] for _ in range(height)]

def draw_rectangle(top_left, bottom_right, width, height):
    """
    Draws a rectangle or square based on the top-left and bottom-right coordinates.

    Parameters:
    - top_left: Tuple (x, y) representing the top-left corner.
    - bottom_right: Tuple (x, y) representing the bottom-right corner.
    """
    board = create_board(width, height)
    for y in range(top_left[1], bottom_right[1] + 1):
        for x in range(top_left[0], bottom_right[0] + 1):
            if 0 <= x < width and 0 <= y < height:
                board[y][x] = full_character
    return board

def draw_circle(center, radius, width, height):
    """
    Draws an approximate circle based on the center coordinates and radius.
    This is a simplistic implementation that approximates a circle on a small grid.

    Parameters:
    - center: A tuple (x, y) representing the center of the circle.
    - radius: The radius of the circle.
    """

    board = create_board(width, height)

    for x in range(width):
        for y in range(height):
            if math.sqrt((center[0] - x)**2 + (center[1] - y)**2) <= radius:
                board[y][x] = full_character

    return board

def draw_triangle(top, side_length, width, height):
    """
    Draws an equilateral triangle given a top vertex and side length.
    """
    def draw_filled_triangle(board, p1, p2, p3):
        # Sort the vertices by y-coordinate
        vertices = sorted([p1, p2, p3], key=lambda p: p[1])

        def draw_horizontal_line(y, x1, x2):
            for x in range(x1, x2 + 1):
                if 0 <= y < len(board) and 0 <= x < len(board[0]):
                    board[y][x] = full_character

        # Get the sorted vertices
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]

        # Calculate the slopes
        slope1 = (x3 - x1) / (y3 - y1) if y3 != y1 else 0
        slope2 = (x2 - x1) / (y2 - y1) if y2 != y1 else 0
        slope3 = (x3 - x2) / (y3 - y2) if y3 != y2 else 0

        # Fill the bottom part of the triangle
        if y1 != y2:
            for y in range(y1, y2 + 1):
                xa = int(x1 + (y - y1) * slope1)
                xb = int(x1 + (y - y1) * slope2)
                draw_horizontal_line(y, min(xa, xb), max(xa, xb))

        # Fill the top part of the triangle
        if y2 != y3:
            for y in range(y2, y3 + 1):
                xa = int(x1 + (y - y1) * slope1)
                xb = int(x2 + (y - y2) * slope3)
                draw_horizontal_line(y, min(xa, xb), max(xa, xb))

    board = create_board(width, height)
    tri_height = int(side_length * math.sqrt(3) / 2)
    p1 = top
    p2 = (top[0] - side_length // 2, top[1] + tri_height)
    p3 = (top[0] + side_length // 2, top[1] + tri_height)

    # Fill the triangle
    draw_filled_triangle(board, p1, p2, p3)

    return board

def draw_cross(center, arm_length, width, height):
    """
    Draws a cross centered at a point with a specified arm length.
    """
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
            options = {}
        else:
            self.board_size = options.get("board_size", 15)
            self.debug = options.get("debug", False)
        self.name = "shapes"
        self.shape = shape
        self.prompt = f"Shapes is a game where you receive an {self.board_size} by {self.board_size} square matrix of {empty_character} and in it you will find a shape denoted by {full_character}. You will have multiple choices and you have to choose the correct option. Only output a number."
        self.reset_board()

    def reset_board(self):
        self.board = create_board(self.board_size, self.board_size)

        # Draw a square randomly
        if self.shape == "square":
            side_length = random.randint(2, int(abs(self.board_size / 2)))
            
            top_left_x = random.randint(0, self.board_size - side_length)
            top_left_y = random.randint(0, self.board_size - side_length)

            bottom_right_x = top_left_x + side_length
            bottom_right_y = top_left_y + side_length
            
            self.board = draw_rectangle((top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.board_size, self.board_size)

        # Draw an equilateral triangle randomly
        elif self.shape == "triangle":
            side_length = random.randint(4, self.board_size // 2)
            top_x = random.randint(side_length, self.board_size - side_length)
            top_y = random.randint(side_length, self.board_size - side_length)
            self.board = draw_triangle((top_x, top_y), side_length, self.board_size, self.board_size)

        # Draw a cross randomly
        elif self.shape == "cross":
            arm_length = random.randint(2, self.board_size // 4)
            center_x = random.randint(arm_length, self.board_size - arm_length)
            center_y = random.randint(arm_length, self.board_size - arm_length)
            self.board = draw_cross((center_x, center_y), arm_length, self.board_size, self.board_size)
        
        # random.shuffle(answer_options)
        self.answer_options = Shapes.answer_options
        
        self.current_player = "P1"
        self.game_over = False
        self.won = False

    def get_text_state(self, player_index=None):
        """Generates a textual representation of the game state."""
        
        text_board = "\n".join("".join(row) for row in self.board)
        text_answers = "Answers:\n" + "\n".join([f"{i}: {option}" for i, option in enumerate(self.answer_options)])
        
        prompt = text_board + "\n\n" + text_answers

        return prompt

    def guess(self, index, guess, playerobj):
        """Processes a player's guess, treating it as a move in TicTacToe."""

        if guess < 0 or guess > len(self.answer_options) - 1:
            return "Invalid move. Out of board range.", False

        if self.answer_options[guess] == self.shape:
            self.game_over = True
            self.won = True
            return "Win", True

        self.game_over = True
        self.won = False
        return "Loss", True

    def check_win(self) -> bool:
        return self.game_over and self.won

    def check_tie(self) -> bool:
        return False
    
    def check_loss(self) -> bool:
        return self.game_over and not self.won
    
    def switch_player(self):
        pass