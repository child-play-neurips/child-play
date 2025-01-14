import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ---------------------- Configuration ----------------------

# Define the order and labels of models
MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o', 'gpt4o_mini']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o': 'GPT-4o',
    'gpt4o_mini': 'GPT-4o-mini',
    'oa_gpt-4-1106-preview': 'GPT-4 Preview',
    'oa_gpt-3.5-turbo-1106': 'GPT-3.5 Turbo',
    'oa:gpt-4o-2024-08-06': 'GPT-4o',
    'oa:gpt-4o-mini-2024-07-18': 'GPT-4o-mini'
}

# Define the games
GAMES = ['shapes', 'lcl', 'tictactoe', 'connectfour', 'battleship', 'molecule_app']

# Paths to data directories
BASE_PATH_BOARDGAMES = '../experiment_board_games'   # Path to general board games data
BASE_PATH_SHAPES = '../experiment_shapes'            # Path to Shapes data
BASE_PATH_LCL = '../lcl_experiments'                 # Path to LCL data
BASE_PATH_MOLECULE_APP = '../molecule_app'           # Path to Molecule App CSVs

# Temperature conditions
TEMPERATURES = [0, 0.5, 1, 1.5]

# ---------------------- Model Mapping ----------------------

# Mapping from actual model names to standardized identifiers
MODEL_NAME_MAPPING = {
    'oa_gpt-4-1106-preview': 'gpt4',
    'oa_gpt-3.5-turbo-1106': 'gpt3_5',
    'oa:gpt-4o-2024-08-06': 'gpt4o',
    'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
}

# ---------------------- Data Extraction Functions ----------------------

def load_shapes_results(path):
    """
    Load results from results.json for the Shapes game.
    Extracts 'Wins' and 'Losses' counts.

    Parameters:
    - path (str): Path to the experiment directory.

    Returns:
    - dict: Dictionary with 'Wins' and 'Losses' counts.
    """
    results_file = os.path.join(path, 'results.json')
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} does not exist. Skipping.")
        return None
    with open(results_file, 'r') as f:
        try:
            results = json.load(f)
            # Extract 'Wins' and 'Losses'
            wins = results.get('Wins', 0)
            losses = results.get('Losses', 0)
            return {'Wins': wins, 'Losses': losses}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {results_file}. Skipping.")
            return None

def load_lcl_results(validity_path, construct_path):
    """
    Load 'Valid Proportion' from df_validity and df_construct CSV files for the LCL game.

    Parameters:
    - validity_path (str): Path to df_validity.csv.
    - construct_path (str): Path to df_construct.csv.

    Returns:
    - float: Valid proportion in percentage.
    """
    if not os.path.exists(validity_path) or not os.path.exists(construct_path):
        print(f"Warning: One of the LCL data files does not exist ({validity_path}, {construct_path}). Skipping.")
        return None
    try:
        df_validity = pd.read_csv(validity_path)
        df_construct = pd.read_csv(construct_path)
        # Calculate Valid Proportion as mean of 'Valid' column times 100
        valid_proportion = df_construct['Valid'].mean() * 100
        return valid_proportion
    except Exception as e:
        print(f"Error processing LCL data in {validity_path} and {construct_path}: {e}. Skipping.")
        return None

def load_boardgame_wins(path, game):
    """
    Load 'P1 Wins' from game_logs_{game}.json for games like Tic-Tac-Toe, Connect Four, Battleship.

    Parameters:
    - path (str): Path to the experiment directory.
    - game (str): Game name.

    Returns:
    - int: Number of P1 wins.
    """
    results_file = os.path.join(path, f'game_logs_{game}.json')
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} does not exist. Skipping.")
        return None
    with open(results_file, 'r') as f:
        try:
            game_logs = json.load(f)
            if isinstance(game_logs, list):
                # Assuming game_logs is a list of game results, each with a 'winner' key
                p1_wins = sum(1 for game_result in game_logs if game_result.get('winner') == 'P1')
                return p1_wins
            elif isinstance(game_logs, dict):
                # If game_logs is a dict, try to get 'P1 Wins' directly
                p1_wins = game_logs.get('P1 Wins', 0)
                return p1_wins
            else:
                print(f"Error: Unexpected data structure in {results_file}. Expected a list or dict.")
                return None
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {results_file}. Skipping.")
            return None

def load_molecule_app_results_detailed(molecule_app_path):
    """
    Load detailed accuracy per model and temperature from Molecule App CSVs.

    Parameters:
    - molecule_app_path (str): Path to Molecule App data directory.

    Returns:
    - dict: Nested dictionary {model_id: {temperature: accuracy}}
    """
    detailed_accuracy = {}
    for file in os.listdir(molecule_app_path):
        if file.startswith('benchmark_resultsoa:gpt') and file.endswith('.csv'):
            file_path = os.path.join(molecule_app_path, file)
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    model_full = row['model']  # e.g., 'oa:gpt-3.5-turbo-0125'
                    temperature = row['temperature']
                    correct = row['correct']
                    
                    # Map model_full to model_id
                    model_id = MODEL_NAME_MAPPING.get(model_full, model_full)
                    
                    # Initialize nested dictionaries
                    if model_id not in detailed_accuracy:
                        detailed_accuracy[model_id] = {}
                    if temperature not in detailed_accuracy[model_id]:
                        detailed_accuracy[model_id][temperature] = []
                    
                    # Append correct value (True/False)
                    detailed_accuracy[model_id][temperature].append(correct)
            except Exception as e:
                print(f"Error processing Molecule App CSV {file_path}: {e}. Skipping this file.")
                continue
    
    # Compute average accuracy per model and temperature
    for model_id in detailed_accuracy:
        for temp in detailed_accuracy[model_id]:
            correct_list = detailed_accuracy[model_id][temp]
            if len(correct_list) == 0:
                accuracy = np.nan
            else:
                # Calculate accuracy as percentage of correct predictions
                accuracy = (sum(correct_list) / len(correct_list)) * 100
            detailed_accuracy[model_id][temp] = accuracy
    return detailed_accuracy

# ---------------------- Data Loading Function ----------------------

def load_all_game_metrics():
    """
    Load all performance metrics for all games.

    Returns:
    - dict: Nested dictionary {game: {model: {temperature: metric}}}
    """
    all_data = {game: {model: {} for model in MODEL_ORDER} for game in GAMES}

    # -------------------- Process Shapes --------------------
    print("Processing Shapes data...")
    shapes_models = ['oa_gpt-4-1106-preview', 'oa_gpt-3.5-turbo-1106', 'oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']
    shapes_temperatures = TEMPERATURES
    shapes_shapes = ['square', 'triangle', 'cross']

    for model in shapes_models:
        standard_model = MODEL_NAME_MAPPING.get(model, model)
        if standard_model not in MODEL_ORDER:
            print(f"Warning: Standard model '{standard_model}' derived from '{model}' is not in MODEL_ORDER. Skipping.")
            continue
        for temp in shapes_temperatures:
            for shape in shapes_shapes:
                experiment_dir = os.path.join(BASE_PATH_SHAPES, model.replace(":", "_"), str(temp).replace(".", "_"), shape)
                metric = load_shapes_results(experiment_dir)
                if metric is not None:
                    # Using 'Wins' as the performance metric
                    wins = metric.get('Wins', 0)
                    # Accumulate wins across shapes
                    if temp in all_data['shapes'][standard_model]:
                        all_data['shapes'][standard_model][temp] += wins
                    else:
                        all_data['shapes'][standard_model][temp] = wins
                else:
                    # If any shape's data is missing, consider the total as NaN
                    all_data['shapes'][standard_model][temp] = np.nan

    # -------------------- Process LCL --------------------
    print("Processing LCL data...")
    lcl_validity_path = os.path.join(BASE_PATH_LCL, 'df_validity.csv')
    lcl_construct_path = os.path.join(BASE_PATH_LCL, 'df_construct.csv')
    lcl_metric = load_lcl_results(lcl_validity_path, lcl_construct_path)
    if lcl_metric is not None:
        for model in MODEL_ORDER:
            for temp in TEMPERATURES:
                # Assign the same validity proportion to all models and temperatures
                all_data['lcl'][model][temp] = lcl_metric
    else:
        print("Warning: LCL metric not available for any model and temperature.")

    # -------------------- Process Board Games --------------------
    print("Processing Board Games data...")
    board_games = ['tictactoe', 'connectfour', 'battleship']
    for game in board_games:
        for model in MODEL_ORDER:
            for temp in TEMPERATURES:
                experiment_dir = os.path.join(BASE_PATH_BOARDGAMES, f"experiment_{game}_{model}_oneshot_temp_{temp}")
                p1_wins = load_boardgame_wins(experiment_dir, game)
                if p1_wins is not None:
                    all_data[game][model][temp] = p1_wins
                else:
                    all_data[game][model][temp] = np.nan  # Assign NaN if data is missing

    # -------------------- Process Molecule App --------------------
    print("Processing Molecule App data...")
    molecule_app_data = load_molecule_app_results_detailed(BASE_PATH_MOLECULE_APP)
    for model_id in molecule_app_data:
        if model_id not in MODEL_ORDER:
            print(f"Warning: Molecule App model '{model_id}' is not in MODEL_ORDER. Skipping.")
            continue
        for temp, accuracy in molecule_app_data[model_id].items():
            all_data['molecule_app'][model_id][temp] = accuracy

    return all_data

# ---------------------- Best Temperature Selection ----------------------

def select_best_temperature(all_data, games, models):
    """
    For each game and model, select the temperature with the best performance metric.

    Parameters:
    - all_data (dict): Nested dictionary {game: {model: {temperature: metric}}}
    - games (list): List of game names.
    - models (list): List of model identifiers.

    Returns:
    - dict: Nested dictionary {game: {model: (best_temperature, best_metric)}}
    """
    best_temps = {game: {} for game in games}
    for game in games:
        for model in models:
            temps_metrics = all_data[game][model]
            if not temps_metrics:
                print(f"Warning: No data for game '{game}', model '{model}'.")
                continue
            # Filter out NaN values
            valid_temps = {temp: metric for temp, metric in temps_metrics.items() if not np.isnan(metric)}
            if not valid_temps:
                print(f"Warning: All data for game '{game}', model '{model}' are NaN.")
                continue
            # Determine if higher metric is better or lower is better based on game
            if game == 'shapes':
                # For 'shapes', higher 'Wins' is better
                best_temp = max(valid_temps, key=lambda temp: valid_temps[temp])
            else:
                # For other games, higher metrics are better
                best_temp = max(valid_temps, key=lambda temp: valid_temps[temp])
            best_metric = valid_temps[best_temp]
            best_temps[game][model] = (best_temp, best_metric)
    return best_temps

# ---------------------- Plotting Function ----------------------

def plot_all_games(best_temps, games, models, model_labels):
    """
    Plot performance progression for all specified games using line plots.

    Parameters:
    - best_temps (dict): Nested dictionary {game: {model: (best_temperature, best_metric)}}
    - games (list): List of game names.
    - models (list): List of model identifiers.
    - model_labels (dict): Mapping of model identifiers to labels.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    
    for game in games:
        plt.figure(figsize=(10, 6))
        plt.title(f'Performance in {game.capitalize()}', fontsize=18, fontweight='bold')
        plt.xlabel('Temperature', fontsize=14, fontweight='bold')
        plt.ylabel('Performance Metric', fontsize=14, fontweight='bold')
        
        for model in models:
            temps = []
            metrics = []
            for temp in TEMPERATURES:
                metric = best_temps[game].get(model, (None, None))[1]
                if metric is not None and not np.isnan(metric):
                    temps.append(temp)
                    metrics.append(metric)
            if temps and metrics:
                plt.plot(temps, metrics, marker='o', label=model_labels.get(model, model))
            else:
                print(f"Warning: No valid data to plot for game '{game}', model '{model}'.")
        
        plt.legend(title='Model')
        plt.xticks(TEMPERATURES)
        plt.tight_layout()
        plt.show()

# ---------------------- Main Execution ----------------------

def main():
    # Load all game metrics
    all_data = load_all_game_metrics()
    
    # Select best temperatures
    best_temps = select_best_temperature(all_data, GAMES, MODEL_ORDER)
    
    # Print best temperatures for verification
    print("\nBest Temperatures and Metrics per Game and Model:")
    for game in GAMES:
        print(f"\nGame: {game.capitalize()}")
        for model in MODEL_ORDER:
            if model in best_temps[game]:
                temp, metric = best_temps[game][model]
                if game in ['tictactoe', 'connectfour', 'battleship']:
                    print(f"  {MODEL_LABELS[model]}: T={temp} with {metric} P1 Wins")
                elif game == 'lcl':
                    print(f"  {MODEL_LABELS[model]}: T={temp} with {metric:.2f}% Validity Proportion")
                elif game == 'shapes':
                    print(f"  {MODEL_LABELS[model]}: T={temp} with {metric} Wins")
                elif game == 'molecule_app':
                    print(f"  {MODEL_LABELS[model]}: T={temp} with {metric:.2f}% Accuracy")
            else:
                print(f"  {MODEL_LABELS[model]}: No data available.")
    
    # Plot performance progression
    plot_all_games(best_temps, GAMES, MODEL_ORDER, MODEL_LABELS)
    
    print("\nPlotting completed.")

if __name__ == "__main__":
    main()
