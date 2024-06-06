import random

class CountingShapesGame:
    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.board = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        self.shapes_count = {"R-O": 0, "G-O": 0, "B-O": 0,  # Circles
                             "R-[]": 0, "G-[]": 0, "B-[]": 0,  # Squares
                             "R-Δ": 0, "G-Δ": 0, "B-Δ": 0}  # Triangles

    def reset_board(self):
        self.board = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        for key in self.shapes_count.keys():
            self.shapes_count[key] = 0

    def place_shapes(self, shape, color, number):
        placed_count = 0
        attempts = 0
        while placed_count < number and attempts < self.rows * self.cols * 2:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)
            if self.board[row][col] == " ":
                self.board[row][col] = f"{color}-{shape}"
                self.shapes_count[f"{color}-{shape}"] += 1
                placed_count += 1
            attempts += 1

        if placed_count < number:
            return f"Only placed {placed_count} out of {number} {color} {shape}s due to space limitations."
        else:
            return f"Successfully placed {number} {color} {shape}(s)."

    def print_shapes_count(self):
        for shape, count in self.shapes_count.items():
            print(f"{shape}: {count}")

    def count_shapes(self, shape=None, color=None):
        count = 0
        for key, value in self.shapes_count.items():
            key_color, key_shape = key.split('-')
            if (shape is None or key_shape == shape) and (color is None or key_color == color):
                count += value
        return count

    def compare_count(self, llm_count, shape=None, color=None):
        actual_count = self.count_shapes(shape, color)
        if actual_count == llm_count:
            return "Correct count."
        else:
            return f"Incorrect count. Actual count is {actual_count}, but LLM counted {llm_count}."

    def print_board(self):
        for row in self.board:
            print(" ".join(row))

# Example usage:
game = CountingShapesGame()
game.place_shapes('O', 'R', 5)
print(game.place_shapes('O', 'R', 5))
game.print_board()