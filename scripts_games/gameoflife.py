import time
import numpy as np

class GameOfLife:
    def __init__(self, size, rule_set, initial_population=0.1):
        self.size = size
        self.board = np.random.choice([0, 1], size=(size, size), p=[1-initial_population, initial_population])
        self.rule_set = rule_set
        self.previous_states = []

    def update_board(self):
        new_board = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                alive_neighbors = np.sum(self.board[max(i-1,0):min(i+2,self.size), max(j-1,0):min(j+2,self.size)]) - self.board[i, j]
                if self.board[i, j] == 1 and alive_neighbors in self.rule_set['survival']:
                    new_board[i, j] = 1
                elif self.board[i, j] == 0 and alive_neighbors in self.rule_set['birth']:
                    new_board[i, j] = 1
        unchanged = np.array_equal(self.board, new_board)
        self.board = new_board
        return unchanged

    def print_board(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if self.board[i, j] == 1:
                    row += "█"  # Full block for alive cells
                else:
                    row += "□"  # Light shade block for dead cells
            print(row)
        print("\n" + "=" * self.size)   # Separator between generations

    def run(self, generations):
        unchanged_count = 0
        for _ in range(generations):
            self.print_board()
            unchanged = self.update_board()
            if unchanged:
                unchanged_count += 1
            else:
                unchanged_count = 0
            
            if unchanged_count >= 5:
                print("Board unchanged for 5 iterations. Stopping simulation.")
                break

            time.sleep(0.25)

# Predefined rule sets for user selection
predefined_rule_sets = {
    'Conway': {'survival': [2, 3], 'birth': [3]},
    'HighLife': {'survival': [2, 3], 'birth': [3, 6]},
    'Day & Night': {'survival': [3, 4, 6, 7, 8], 'birth': [3, 6, 7, 8]},
    'Seeds': {'survival': [], 'birth': [2]},
    'Life Without Death': {'survival': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'birth': [3]},
    'Maze': {'survival': [1, 2, 3, 4, 5], 'birth': [3]},
    '2x2': {'survival': [1, 2, 5], 'birth': [3, 6]},
    'Blinking': {'survival': [2, 3], 'birth': [3]},
    'GGG': {'survival': [1,2,5], 'birth': [3,6]}, # Gosper Glider Gun
    'Stable': {'survival': [2, 3], 'birth': [3]},
    'Replicator': {'survival': [1, 3, 5, 7], 'birth': [1, 3, 5, 7]},
    'Coagulations': {'survival': [2, 3, 5, 6, 7, 8], 'birth': [3, 7, 8]},
    'Long life': {'survival': [5], 'birth': [3, 4, 5]}
} 

size = int(input("Enter board size: "))
print("Available rule sets and their rules:")
for name, rules in predefined_rule_sets.items():
    survival_rules = ', '.join(str(x) for x in rules['survival'])
    birth_rules = ', '.join(str(x) for x in rules['birth'])
    print(f"- {name}: Survival [{survival_rules}], Birth [{birth_rules}]")
rule_set_name = input("Enter rule set (Conway, HighLife, etc.): ")
gen = int(input("Enter number of iterations: "))

rule_set = predefined_rule_sets.get(rule_set_name, predefined_rule_sets['Conway'])

game = GameOfLife(size, rule_set, initial_population=0.5)
game.run(generations=gen)

"""TODO
1. Add prediction question - the model is asked to produce the next state of the board based on the rules and current board, and we check it against the actual board
2. Ask it to predict the rules in the proposed language given 10 states of the board
3. Ask it to predict the initial conditions/inital state given the rules or 10 states of the board and the final state.
"""