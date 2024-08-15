import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import matplotlib.ticker as mticker
import torch.multiprocessing as mp

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self

    def step(self, action):
        row, col = action
        if self.board[row, col] != 0:
            return self, -10, True  # Invalid move
        self.board[row, col] = self.current_player
        if self.check_winner(self.current_player):
            return self, 1 if self.current_player == 1 else -1, True
        if not (self.board == 0).any():
            return self, 0, True  # Draw
        self.current_player = -self.current_player
        return self, 0, False

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def available_actions(self):
        return list(zip(*np.where(self.board == 0)))

    def minimax(self, depth, player):
        if self.check_winner(1):
            return 1
        if self.check_winner(-1):
            return -1
        if not self.available_actions():
            return 0

        if player == 1:
            best_value = -np.inf
            for action in self.available_actions():
                self.board[action] = player
                value = self.minimax(depth + 1, -player)
                self.board[action] = 0
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = np.inf
            for action in self.available_actions():
                self.board[action] = player
                value = self.minimax(depth + 1, -player)
                self.board[action] = 0
                best_value = min(best_value, value)
            return best_value

    def best_move(self):
        best_value = -np.inf
        best_move = None
        for action in self.available_actions():
            self.board[action] = 1
            move_value = self.minimax(0, -1)
            self.board[action] = 0
            if move_value > best_value:
                best_value = move_value
                best_move = action
        return best_move

def play_game(_):
    env = TicTacToe()
    move_heatmap = np.zeros((3, 3), dtype=int)
    env.reset()
    done = False

    while not done:
        if env.current_player == 1:
            action = env.best_move()
        else:
            action = random.choice(env.available_actions())

        row, col = action
        state, reward, done = env.step(action)
        if env.current_player == -1:  # Last move was by the Minimax
            move_heatmap[row, col] += 1

    return reward, move_heatmap

def parallel_play_games(games=1000):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(play_game, range(games))

    wins = {"Minimax": 0, "Random": 0, "Draw": 0}
    move_heatmap = np.zeros((3, 3), dtype=int)

    for result in results:
        reward, game_heatmap = result
        move_heatmap += game_heatmap
        if reward == 1:
            wins["Minimax"] += 1
        elif reward == -1:
            wins["Random"] += 1
        else:
            wins["Draw"] += 1

    return wins, move_heatmap

if __name__ == "__main__":
    print("Starting game simulation...")
    wins, move_heatmap = parallel_play_games(games=1000)
    print("Game simulation completed")

    # Print results
    print("Wins: ", wins)

    # Prepare data for bar plots
    data = {
        'Outcome': ['Minimax Wins', 'Random Wins', 'Draws'],
        'Count': [wins['Minimax'], wins['Random'], wins['Draw']]
    }

    df = pd.DataFrame(data)

    # Plotting the results
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})  # Increased font size and made it bold
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))

    def add_labels(ax):
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{int(width)}', xy=(width, p.get_y() + p.get_height() / 2),
                        xytext=(10, 0), textcoords='offset points', ha='center', va='center',
                        fontsize=14, fontweight='bold')  # Increased annotation font size and made it bold

    # Bar plot of wins
    sns.barplot(x='Count', y='Outcome', data=df, orient='h', ax=axs[0])
    axs[0].set_xlabel("Number of Games", fontsize=16, fontweight='bold')
    axs[0].set_ylabel("Outcome", fontsize=16, fontweight='bold')
    axs[0].set_title("Game Outcomes after 1000 Games", fontsize=18, fontweight='bold')
    axs[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
    axs[0].xaxis.get_major_formatter().set_useOffset(False)
    axs[0].xaxis.get_major_formatter().set_scientific(False)
    add_labels(axs[0])
    plt.savefig('game_outcomes.svg', format='svg')

    # Heatmap of Minimax's moves with percentages
    total_moves = np.sum(move_heatmap)
    move_heatmap_percentages = (move_heatmap / total_moves) * 100

    sns.heatmap(move_heatmap_percentages, annot=True, fmt=".2f", cmap='coolwarm', ax=axs[1],
                annot_kws={"size": 14, "weight": "bold"})  # Larger and bolder annotations on heatmap
    axs[1].set_title('Minimax Move Heatmap (Percentages)', fontsize=18, fontweight='bold')
    axs[1].set_xlabel('Columns', fontsize=16, fontweight='bold')
    axs[1].set_ylabel('Rows', fontsize=16, fontweight='bold')
    plt.savefig('move_heatmap.svg', format='svg')

    plt.tight_layout()
    plt.show()