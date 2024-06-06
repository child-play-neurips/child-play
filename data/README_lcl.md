# ./lcl_experiments
Has two subfolders, construct_generation and validity_experiments. The former consists of the svg renderings of the model's Lego constructions, and the latter consists of pngs automatically generated of valid and invalid images used to test the models. Two dataframes are also provided, df_validity and df_construct, the results from both experiments, validity test and construct generation respectively. These dataframes list the temperature, the model, the model's answer, if the answer was correct and the Lego construct written in LCL.

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