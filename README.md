# child-play

## Overview
The ChildPlay Repository hosts the ChildPlay benchmark suite, introduced at the NeurIPS 2024 Datasets and Benchmarks Track. This benchmark is designed to evaluate the strategic reasoning and cognitive capabilities of large language models (LLMs) through non-linguistic games like Tic-Tac-Toe, Connect Four, and Battleship.

## The repository
.<br />
├── experiment_board_games<br />
├── experiment_shapes<br />
├── imgs<br />
├── lcl_experiments<br />
├── molecule_app<br />
├── new_heatmaps<br />
├── main_shapes<br />
├── lcl.py<br />
├── scripts_games<br />
├── utils<br />
├── main.py<br />
└── wrapper.py<br />

## Getting Started

To use the ChildPlay benchmarks, clone this repository and install the required dependencies as follows:

```bash
git clone https://github.com/yourrepository/child-play.git
cd child-play
conda env create -f environment.yml
conda activate myenv
```

## Dataset Description

The ChildPlay benchmark includes a series of games encoded in various formats, including ASCII for strategic board games. Each game is structured to test the models on their ability to plan, reason, and make decisions, thus we are attempting to go beyond their training on conventional linguistic datasets.

### Games Included:
- **Tic-Tac-Toe**
- **Connect Four**
- **Battleship**
- **Shapes**
- **LCL**
- **GTS**

Each game is accompanied by its initial configuration, and rules to provide a comprehensive testing framework.

Most games can be found in ./scripts_games, to the exception of LCL which exists in ./lcl.py
To run experiments involving all games except lcl, one can run main.py.
For example, to run a shapes experiment:
```bash
game_runs = []

shapes = ['square', 'triangle', 'cross']
models = ['oa:gpt-3.5-turbo-1106', 'oa:gpt-4-1106-preview', 'oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']
temperatures = [0, 0.5, 1, 1.5]
num_games = 25

for model in models:
    for temp in temperatures:
        for shape in shapes:
            game_runs.append({'game_class': Shapes, 'game_name': "shapes", 'board_size': 15, 'model_name': model,  'num_games': num_games, 'experiment_name': f'experiment_shapes/{model.replace(":", "_")}/{str(temp).replace(".", "_")}/{shape}', 'temperature':temp, 'shape': shape})
```

Or, to run a board game experiment (should be pretty self-explanatory):
```bash
game_runs = [
    {'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 5, 'model_name': 'gpt-3.5-turbo-1106', 'num_games': 1, 'experiment_name': 'experiment_to_plot'}, 
    {'game_class': ConnectFour, 'game_name': 'connectfour', 'board_size': 7, 'model_name': 'gpt-3.5-turbo-1106', 'num_games': 1000,  'experiment_name': 'experiment_board_games/experiment_connectfour_gpt3_5_oneshot_temp_0'},{'game_class': BattleShip, 'game_name': 'battleship', 'board_size': 5, 'model_name': 'gpt-3.5-turbo-1106', 'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_battleship_gpt3_5_oneshot_temp_0'},
    {'game_class': TicTacToe, 'game_name': 'tictactoe', 'board_size': 3, 'model_name': 'gpt-4-1106-preview',  'num_games': 100, 'experiment_name': 'experiment_board_games/experiment_tictactoe_gpt4_oneshot_temp_0'}
    ]
```

## Utils
In utils one can find a registry of previous runs or experiments, some scripts that were used to produce plots currently in the paper, and the lcl visualizer (even though this is also found in lcl.py)

## Experiments
The results of the experiments can be found unaltered in experiment_board_games, experiment_shapes, and lcl_experiments.

# Data
## Data Collection
Data can be found in the following folders:
- **./experiment_board_games**: Organized in experiments per board game following the format experiment_{game}_{model}_oneshot_temp_{temperature}. Each subfolder has 4 files, one heatmap detailing the moves recorded over all games played, a game_logs_{game}.json detailing the move choice per player, game_messages_{game}.json which is a literal recording of the prompts sent back and forth between the game and the models, and finally, results_{game}.json detailing the number of wins per player, where P1 is always the model and P2 always the random player, the number of ties, and the wrong moves per player.

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

- **./lcl_experiments**: Has two subfolders, construct_generation and validity_experiments. The former consists of the svg renderings of the model's Lego constructions, and the latter consists of pngs automatically generated of valid and invalid images used to test the models. Two dataframes are also provided, df_validity and df_construct, the results from both experiments, validity test and construct generation respectively. These dataframes list the temperature, the model, the model's answer, if the answer was correct and the Lego construct written in LCL.

├── construct_generation<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_100.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_10.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_11.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_12.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_13.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_14.svg<br />
│   ├── oa:gpt-3.5-turbo-1106_temp_0.5_experiment_15.svg<br />
...<br />
├── df_construct.csv<br />
├── df_validity.csv<br />
├── test.py<br />


- **./experiment_shapes**: Is organized in two subfolders, one per model. These are further organized in 4 subfolders, one per temperature setting and an barplot detailing the total correct and incorrect answers. Each temperature subfolder is devided into 3 other subfolders, one per shape, namely cross, square, and triangle. There is also a heatmap detailing the model answers for all shapes. Inside each shape folder can be found three jsons, namely results showing the number of correct moves (P1) and the number of incorrect moves (P2) or shape comparisons. "Ties", "p1 wrong moves", and "p2 wrong moves" should be ignored. Game messages are the literal back and forth between the game and the models recorded as is. The game_logs.json contains what shape was shown and what shape the model chose.

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

## LCL (Lego Connect Language) Game Simulation

Data was generated through a simulated environment where artificial intelligence models and random players build and validate Lego constructs. The simulation focuses on determining the validity of constructions based on predefined rules of connectivity and overlap.

### Game Mechanics
- **Piece Placement**: Pieces are represented as rectangles and placed on a grid. Each piece has a specified length and is placed at specific coordinates with a color.
- **Connectivity and Overlap Rules**: Pieces must connect through interlocking pegs and must not overlap on the same layer. A piece is considered connected if it overlaps by at least one unit with another piece directly adjacent or on an adjacent layer.

### Simulation Setup
- **Board Configuration**: The board does not have a fixed size but is dynamically adjusted based on the pieces' placements.
- **Valid and Invalid Constructs**: The simulation generates both valid and invalid constructs to test the models' and players' ability to correctly identify construct validity.

### Data Collection
- **Model Interactions**: Various AI models, including versions of OpenAI's GPT models, are tested at different "temperatures" to simulate different randomness levels in response generation.
- **Player Answers**: Random players generate answers based on a 50/50 chance to provide a baseline for model performance.
- **Visualizations**: Constructs are visualized using matplotlib, with each piece drawn as a rectangle on a grid, and saved as images for further analysis.

### Analysis and Visualization
- **Validity Testing (Game 1)**: The game randomly generates Lego constructs, and the models must determine if the construct is valid based on the game's rules. The results are visualized and logged for analysis.
- **Construct Generation (Game 2)**: Models are prompted to generate descriptions of valid Lego structures. These structures are then built and visualized to assess the models' understanding and application of the rules.

# Data Organization
Data from each game session is structured in JSON format, where logs of moves and game outcomes are recorded, which facilitates ease of analysis and replayability.

### Game logs
- **game_logs_{game}**: Data on game logic and player strategy is openly documented, ensuring transparency in data generation.

## Ethical and Responsible Use
- **Transparency**: Data on game logic and player strategy is openly documented, ensuring transparency in data generation.
- **Fairness**: Not applicable - the data consists of model answers to board games, questions about shapes, or questions about an invented Lego construction language.
- **Data Privacy**: No personal data is collected in the simulation, adhering to privacy norms.

## Data Accessibility and Maintenance
- **Storage**: Data is stored in JSON format, PNGs, CSVs, or SVGs, making it platform-independent and easy to access for further processing.
- **Maintenance**: Updates to the game logic or player algorithms will be versioned to maintain consistency in comparative studies.

## Usage Recommendations
- **Research**: Ideal for research in game theory, AI behavior analysis, and strategic decision-making processes.
- **Education**: Can be used in educational settings for demonstrating basic AI concepts and programming practices.
- **AI Testing**: Provides a controlled environment for testing AI algorithms in decision-making scenarios.

## GTS: Guess-the-SMILES

The **Guess-the-SMILES (GTS)** game is a molecule identification game designed to test large language models' ability to interpret structural data and predict chemical representations. This game challenges the model to convert a graphical or ASCII depiction of a molecule into its corresponding SMILES (Simplified Molecular Input Line Entry System) notation.

### Game Mechanics

- **Molecule Generation**: Random molecular structures are generated using the SELFIES (Self-Referencing Embedded Strings) encoding, with constraints on the minimum and maximum number of atoms to ensure valid and complex molecules.
- **Display Format**: Molecules are presented either as ASCII art or PNG images, displaying the molecular structure using atomic symbols and bond representations.
- **Player Interaction**: Models are tasked with interpreting the ASCII or image representation of the molecule and providing the correct SMILES string.
- **Evaluation**: The provided SMILES string is evaluated for correctness, chemical similarity, and string distance to the original SMILES.

You can play the game here: [GTS website](https://child-play.onrender.com/).
Note that the code is not accessible as this is supposed to be a blind experiment.


## Guess-the-SMILES Public API Documentation

**Base URL:** `https://child-play.onrender.com/`

The **Guess-the-SMILES API** allows users to generate molecular representations and evaluate their SMILES predictions based on ASCII drawings of molecules.

### Endpoints

---

### 1. Generate Molecule

- **URL:** `/generate_molecule`
- **Method:** `POST`
- **Description:** Generates a random molecule and returns its ASCII representation or PNG image along with a unique `molecule_id`.

#### Request Parameters:

| Parameter | Type    | Description                                  | Default   |
|-----------|---------|----------------------------------------------|-----------|
| length    | Integer | Number of SELFIES characters in the string   | 30        |
| min_atoms | Integer | Minimum number of atoms in the molecule      | 10        |
| max_atoms | Integer | Maximum number of atoms in the molecule      | 15        |
| format    | String  | Desired output format: `"ascii"` or `"png"`  | `"ascii"` |

#### Request Body Example:

```json
{
  "length": 30,
  "min_atoms": 10,
  "max_atoms": 15,
  "format": "ascii"
}
```

#### Responses:

- **Success (ASCII Format):**

  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "ascii": "  C - C - O
   |   |
   N - C",
      "molecule_id": 1
    }
    ```

- **Success (PNG Format):**

  - **Status Code:** `200 OK`
  - **Headers:**
    ```
    Content-Type: image/png
    ```
  - **Body:** Binary PNG image data.

- **Error:**

  - **Status Code:** `400 Bad Request`
  - **Body:**
    ```json
    {
      "error": "Failed to generate a molecule"
    }
    ```

---

### 2. Evaluate Prediction

- **URL:** `/evaluate_prediction`
- **Method:** `POST`
- **Description:** Evaluates a predicted SMILES string against the original molecule using `molecule_id`.

#### Request Parameters:

| Parameter       | Type   | Description                      | Required |
|-----------------|--------|----------------------------------|----------|
| molecule_id     | Integer| ID of the generated molecule     | Yes      |
| predicted_smile | String | User's predicted SMILES string   | Yes      |

#### Request Body Example:

```json
{
  "molecule_id": 1,
  "predicted_smile": "C1=CC=CC=C1"
}
```

#### Responses:

- **Success:**

  - **Status Code:** `200 OK`
  - **Body:**
    ```json
    {
      "correct": true,
      "chemical_similarity": 1.0,
      "string_distance": 0
    }
    ```

  - **Fields:**
    - `correct`: Indicates if the prediction matches the original SMILES.
    - `chemical_similarity`: Dice similarity score between original and predicted SMILES.
    - `string_distance`: Levenshtein distance between original and predicted SMILES.

- **Error:**

  - **Status Code:** `400 Bad Request`
  - **Body:**
    ```json
    {
      "error": "Invalid molecule ID"
    }
    ```

---

### Usage Examples

#### 1. Generate an ASCII Molecule

**Request:**

```bash
curl -X POST https://child-play.onrender.com/generate_molecule      -H "Content-Type: application/json"      -d '{"format": "ascii"}'
```

**Response:**

```json
{
  "ascii": "  C - C - O
   |   |
   N - C",
  "molecule_id": 1
}
```

#### 2. Generate a PNG Image

**Request:**

```bash
curl -X POST https://child-play.onrender.com/generate_molecule      -H "Content-Type: application/json"      -d '{"format": "png"}' --output molecule.png
```

**Response:**

- Saves the PNG image as `molecule.png`.

#### 3. Evaluate a SMILES Prediction

**Request:**

```bash
curl -X POST https://child-play.onrender.com/evaluate_prediction      -H "Content-Type: application/json"      -d '{"molecule_id": 1, "predicted_smile": "C1=CC=CC=C1"}'
```

**Response:**

```json
{
  "correct": true,
  "chemical_similarity": 1.0,
  "string_distance": 0
}
```
