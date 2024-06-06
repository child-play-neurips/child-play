# ./experiment_shapes
Is organized in two subfolders, one per model. These are further organized in 4 subfolders, one per temperature setting and an barplot detailing the total correct and incorrect answers. Each temperature subfolder is devided into 3 other subfolders, one per shape, namely cross, square, and triangle. There is also a heatmap detailing the model answers for all shapes. Inside each shape folder can be found three jsons, namely results showing the number of correct moves (P1) and the number of incorrect moves (P2) or shape comparisons. "Ties", "p1 wrong moves", and "p2 wrong moves" should be ignored. Game messages are the literal back and forth between the game and the models recorded as is. The game_logs.json contains what shape was shown and what shape the model chose.

oa_gpt-3.5-turbo-1106<br />
│   ├── 0<br />
│   │   ├── cross<br />
│   │   │   ├── game_logs.json<br />
│   │   │   ├── game_messages.json<br />
│   │   │   ├── oa_gpt-3.5-turbo-1106_0_cross_heatmap.png<br />
│   │   │   └── results.json<br />
│   │   ├── oa_gpt-3.5-turbo-1106_0_combined_heatmap.png<br />
│   │   ├── square<br />
│   │   │   ├── game_logs.json<br />
│   │   │   ├── game_messages.json<br />
│   │   │   ├── oa_gpt-3.5-turbo-1106_0_square_heatmap.png<br />
│   │   │   └── results.json<br />
│   │   └── triangle<br />
│   │       ├── game_logs.json<br />
│   │       ├── game_messages.json<br />
│   │       ├── oa_gpt-3.5-turbo-1106_0_triangle_heatmap.png<br />
│   │       └── results.json<br />
...

## Shapes Recognition Game

Data was generated through simulated plays of the Shapes Recognition game, involving various types of shape recognitions like squares, triangles, and crosses. The game is played on a matrix grid which can be configured in size but typically uses a 15x15 board.

### Game Mechanics
- **Board Setup**: A grid where shapes are randomly drawn, represented by '1' for filled spaces and '0' for empty spaces.
- **Shape Placement**: The game randomly places one of several predefined shapes such as squares, triangles, or crosses on the grid.
- **Player Turns**: Players are presented with the grid and a list of possible shapes. They must identify the correct shape present on the board.

### Game Flow
- **Initialization**: The board is initialized with a randomly chosen shape placed on it. The size of the board and the shapes can be customized through game options.
- **Shape Identification Phase**: Players are given a choice of shapes to identify from the board. They submit their guess by selecting the corresponding number.
- **Feedback on Guess**: The game immediately provides feedback if the guess was correct, leading to a win, or incorrect, resulting in a loss.
- **Endgame Scenarios**: The game ends after one guess, with either a win if the correct shape is identified or a loss otherwise.

### Simulation Details
- **Shape Complexity**: The complexity of the shapes can vary based on the size of the board and the configuration settings.
- **Outcome Recording**: Each guess and its correctness are logged for analysis.
- **Game End Conditions**: The game does not have a tie condition; it ends with either a win or a loss based on the player's guess.
