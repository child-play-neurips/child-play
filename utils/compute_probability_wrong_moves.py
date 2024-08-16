import numpy as np

# Temperature values
temperature = np.array([0.0, 0.5, 1.0, 1.5])

# Incorrect Moves for each game (from extracted data)
incorrect_moves_tictactoe = np.array([47, 76, 76, 81])
incorrect_moves_connect_four = np.array([36, 20, 13, 12])
incorrect_moves_battleship = np.array([86, 89, 97, 93])

# Average Moves for each game (from extracted data)
average_moves_tictactoe = np.array([1, 2, 3, 4])
average_moves_connect_four = np.array([2, 4, 6, 8])
average_moves_battleship = np.array([10, 8, 3, 3])

# Total number of moves (sum of average moves as an estimate for simplicity)
total_moves_tictactoe = np.sum(average_moves_tictactoe)
total_moves_connect_four = np.sum(average_moves_connect_four)
total_moves_battleship = np.sum(average_moves_battleship)

# Calculate the probability of incorrect moves
prob_incorrect_tictactoe = incorrect_moves_tictactoe / total_moves_tictactoe
prob_incorrect_connect_four = incorrect_moves_connect_four / total_moves_connect_four
prob_incorrect_battleship = incorrect_moves_battleship / total_moves_battleship

# Calculate the average number of steps
avg_steps_tictactoe = np.mean(average_moves_tictactoe)
avg_steps_connect_four = np.mean(average_moves_connect_four)
avg_steps_battleship = np.mean(average_moves_battleship)

# Display results
print("Tic-Tac-Toe Probability of Incorrect Moves:", prob_incorrect_tictactoe)
print("Tic-Tac-Toe Average Steps:", avg_steps_tictactoe)

print("Connect Four Probability of Incorrect Moves:", prob_incorrect_connect_four)
print("Connect Four Average Steps:", avg_steps_connect_four)

print("Battleship Probability of Incorrect Moves:", prob_incorrect_battleship)
print("Battleship Average Steps:", avg_steps_battleship)
