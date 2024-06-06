# child-play

## Overview
The ChildPlay Repository hosts the ChildPlay benchmark suite, introduced at the NeurIPS 2024 Datasets and Benchmarks Track. This benchmark is designed to evaluate the strategic reasoning and cognitive capabilities of large language models (LLMs) through non-linguistic games like Tic-Tac-Toe, Connect Four, and Battleship.

## The repository
.<br />
├── experiment_board_games<br />
├── experiment_shapes<br />
├── imgs<br />
├── lcl_experiments<br />
├── lcl.py<br />
├── logs<br />
├── new_heatmaps<br />
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

Each game is accompanied by its initial configuration, and rules to provide a comprehensive testing framework.

Most games can be foun in ./scripts_games, to the exception of LCL which exists in ./lcl.py
To run experiments involving all games except lcl, one can run main.py.
For example, to run a shapes experiment:
```bash
game_runs = []

shapes = ['square', 'triangle', 'cross']
models = ['oa:gpt-3.5-turbo-1106', 'oa:gpt-4-1106-preview']
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
