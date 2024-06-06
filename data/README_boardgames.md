# ./experiment_board_games

Organized in experiments per board game following the format experiment_{game}_{model}_oneshot_temp_{temperature}. Each subfolder has 4 files, one heatmap detailing the moves recorded over all games played, a game_logs_{game}.json detailing the move choice per player, game_messages_{game}.json which is a literal recording of the prompts sent back and forth between the game and the models, and finally, results_{game}.json detailing the number of wins per player, where P1 is always the model and P2 always the random player, the number of ties, and the wrong moves per player.

├── experiment_board_games<br />
│   ├── experiment_battleship_gpt3_5_oneshot_temp_0<br />
│   │   ├── battleship_heatmap.png<br />
│   │   ├── battleship_heatmap_temp0.png<br />
│   │   ├── game_logs_battleship.json<br />
│   │   ├── game_messages_battleship.json<br />
│   │   └── results_battleship.json<br />
│   ├── experiment_battleship_gpt3_5_oneshot_temp_0.5<br />
│   │   ├── battleship_heatmap_0_5.png<br />
│   │   ├── battleship_heatmap.png<br />
│   │   ├── game_logs_battleship.json<br />
│   │   ├── game_messages_battleship.json<br />
│   │   └── results_battleship.json<br />
...

## Tic-Tac-Toe
Data was generated through simulated games of Tic-Tac-Toe, involving two types of players—algorithmically controlled bots (`RandomPlayer`) and players based on predefined strategies (`StrategyPlayer`). The game is played on a standard 3x3 grid.

### Game Mechanics
- **Board Setup**: A 3x3 grid where players place their marks (X or O).
- **Player Turns**: Two players take turns, one marked as "X" and the other as "O".
- **Game End Condition**: The game ends when one player aligns three of their marks horizontally, vertically, or diagonally or all grid spaces are filled without a winner (tie).

### Game Flow
- **Initialization**: The game starts with an empty board and players alternate placing their marks.
- **Turn Mechanics**: Players input their moves by specifying the row and column numbers where they wish to place their mark.
- **Winning Conditions**: The game checks for a win after every move by assessing if there are three consecutive marks in a line.
- **Tie Conditions**: If all spaces are filled and no player has won, the game is declared a tie.

### Simulation Details
- **Player Strategies**: The `RandomPlayer` makes moves at random, while the `StrategyPlayer` uses a set of predetermined strategies that aim to maximize winning opportunities.
- **Outcome Recording**: Each game's moves and outcomes are recorded for analysis, focusing on the strategies employed and their effectiveness.
- **Feedback Mechanisms**: The simulation includes real-time feedback on the validity of moves and the game state, informing players immediately if a move is invalid or if it results in a win or a tie.

## Battleship
Data was generated through simulated games of Battleship between two types of players—algorithmically controlled bots (`RandomPlayer`) and players based on predefined strategies (`StrategyPlayer`). Battleship is played on a square grid.

### Game Mechanics
- **Board Setup**: A grid (typically 5x5) where players place their ships. The board size can be customized.
- **Ship Placement**: Each player places a set number of ships of varying sizes on the grid in secret. Ship placement can be horizontal or vertical and must not overlap.
- **Player Turns**: Players take turns guessing the locations of the opponent's ships on the grid.
- **Game End Condition**: The game ends when one player has successfully guessed the locations of all the opponent's ships, effectively sinking them. The game also has checks for tie conditions and invalid moves.

### Game Flow
- **Initialization**: Players initialize their boards with ships placed in predetermined or random positions.
- **Guessing Phase**: Each player guesses a coordinate on the opponent’s board, aiming to hit their ships.
- **Hit or Miss Feedback**: Players receive feedback if the guess was a hit (marked with 'X') or a miss (marked with 'O').
- **Game Progression**: The game alternates turns between players. Players adjust their strategies based on previous hits and misses.
- **Winning Condition**: A player wins by sinking all ships of the opponent. The game records the sequence of moves for analysis.

### Simulation Details
- **Player Strategies**: The `RandomPlayer` makes random guesses, while the `StrategyPlayer` uses a predefined set of strategies based on past successful games.
- **Outcome Recording**: The game logs each move's coordinates and the result (hit/miss) for further analysis.
- **Endgame Scenarios**: Besides winning by sinking all ships, the game also handles scenarios like board fills without a winner (tie).

## Connect Four
Data was generated through simulated games of Connect Four between two types of players—algorithmically controlled bots (`RandomPlayer`) and players based on predefined strategies (`StrategyPlayer`). Connect Four is played on a vertically suspended 7x7 grid.

### Game Mechanics
- **Board Setup**: A vertical grid (commonly 7 rows by 7 columns) where players drop discs.
- **Disc Placement**: Players alternate turns dropping colored discs into the columns of the grid. A disc falls straight down, occupying the lowest available space within the column.
- **Player Turns**: Players take turns dropping discs, starting with player one who uses the 'X' symbol.
- **Game End Condition**: The game ends when a player forms a horizontal, vertical, or diagonal line of four discs, or when the board is filled completely without any winning line (tie).

### Game Flow
- **Initialization**: The board is initialized with empty slots represented by periods ('.').
- **Disc Dropping Phase**: Each player selects a column (0-6) to drop their disc into the grid.
- **Victory Check**: After each move, the game checks for a sequence of four consecutive discs from the same player.
- **Feedback on Moves**: Players receive feedback on whether the move resulted in a win, a tie, or was a valid move allowing the game to continue.
- **Endgame Scenarios**: The game can end with a win for one of the players or a tie if the board fills without a line of four.

### Simulation Details
- **Player Strategies**: The `RandomPlayer` selects columns at random, while the `StrategyPlayer` uses more complex logic based on the current state of the board.
- **Outcome Recording**: Each move's column and result (win/tie/continue) are logged for analysis.
- **Tie and Win Conditions**: The game includes checks for tie conditions when the top row is completely filled and no more moves are possible, along with checks for winning lines of four discs.