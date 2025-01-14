import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Configuration ----------------------

MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o_mini', 'gpt4o']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt4o': 'GPT-4o',
    # Possibly other name mappings:
    'oa_gpt-3.5-turbo-1106': 'GPT-3.5 Turbo',
    'oa_gpt-4-1106-preview': 'GPT-4 Preview',
    'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini',
    'oa:gpt-4o-2024-08-06': 'gpt4o'
}

GAMES = ['shapes', 'lcl', 'tictactoe', 'connectfour', 'battleship', 'molecule_app']
TEMPERATURES = [0, 0.5, 1, 1.5]

BASE_PATH_SHAPES        = '../experiment_shapes'
BASE_PATH_LCL           = '../lcl_experiments'
BASE_PATH_BOARDGAMES    = '../experiment_board_games'
BASE_PATH_MOLECULE_APP  = '../molecule_app'

# -------------------------- 1) Shapes --------------------------
def load_shapes_metrics():
    """
    For shapes, keep ONLY the 'Wins' metric.
    """
    data = {m: {} for m in MODEL_ORDER}
    shape_folders = ['square', 'triangle', 'cross']

    for rm in MODEL_ORDER:
        total_wins = 0
        for t in TEMPERATURES:
            for shape in shape_folders:
                path = os.path.join(
                    BASE_PATH_SHAPES,
                    rm.replace(":", "_"),
                    str(t).replace(".", "_"),
                    shape,
                    'results.json'
                )
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r') as f:
                        d = json.load(f)
                        if 'P1 Wins' in d and isinstance(d['P1 Wins'], (int, float)):
                            total_wins += d['P1 Wins']
                        elif 'Wins' in d and isinstance(d['Wins'], (int, float)):
                            total_wins += d['Wins']
                except Exception as e:
                    print(f"Error reading {path}: {e}")
        
        data[rm] = {'Wins': total_wins}
    return data

# -------------------------- 2) LCL --------------------------
def load_lcl_metrics():
    """
    Produces two lines: "Correct Proportion (%)" and "Valid Proportion (%)",
    ignoring temperature (group by Model).
    """
    data = {m: {} for m in MODEL_ORDER}
    # If your LCL CSV 'Model' uses different strings, map them here:
    model_map_lcl = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-3.5-turbo-1106': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }

    try:
        dfv_main = pd.read_csv(os.path.join(BASE_PATH_LCL, 'df_validity.csv'))
        dfv_4o   = pd.read_csv(os.path.join(BASE_PATH_LCL, 'df_validity_4o_experiments.csv'))
        df_valid = pd.concat([dfv_main, dfv_4o], ignore_index=True)
    except Exception as e:
        print(f"Warning: Can't load df_validity CSVs. Error: {e}")
        df_valid = pd.DataFrame(columns=['Model','Correct'])

    try:
        dfc_main = pd.read_csv(os.path.join(BASE_PATH_LCL, 'df_construct.csv'))
        dfc_4o   = pd.read_csv(os.path.join(BASE_PATH_LCL, 'df_construct_4o_experiments.csv'))
        df_construct = pd.concat([dfc_main, dfc_4o], ignore_index=True)
    except Exception as e:
        print(f"Warning: Can't load df_construct CSVs. Error: {e}")
        df_construct = pd.DataFrame(columns=['Model','Valid'])

    # Fix the 'Model' column so it matches MODEL_ORDER
    if 'Model' in df_valid.columns:
        df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    if 'Model' in df_construct.columns:
        df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])

    grouped_correct = df_valid.groupby('Model')['Correct'].mean() * 100
    grouped_valid   = df_construct.groupby('Model')['Valid'].mean()  * 100

    for m in MODEL_ORDER:
        data[m] = {}
    for m in grouped_correct.index:
        if m in MODEL_ORDER:
            data[m]['Correct Proportion (%)'] = grouped_correct.loc[m]
    for m in grouped_valid.index:
        if m in MODEL_ORDER:
            data[m]['Valid Proportion (%)']   = grouped_valid.loc[m]
    return data

# -------------------------- 3) Board Games --------------------------
def load_boardgame_metrics(game):
    """
    Summation of numeric keys across all temps, skipping 'P2 Wins' and 'P2 Wrong Moves'.
    We rename 'P1' -> 'LLM' in the key name.
    """
    data = {m: {} for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        aggregator = {}
        for t in TEMPERATURES:
            path = os.path.join(
                BASE_PATH_BOARDGAMES,
                f"experiment_{game}_{m}_oneshot_temp_{t}",
                f"results_{game}.json"
            )
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r') as f:
                    d = json.load(f)
                    for mk, mv in d.items():
                        if isinstance(mv, (int, float)) and mk not in ('P2 Wins', 'P2 Wrong Moves'):
                            key = mk.replace('P1', 'LLM')
                            aggregator[key] = aggregator.get(key, 0) + mv
            except Exception as e:
                print(f"Error reading {path}: {e}")
        data[m] = aggregator
    return data

# -------------------------- 4) Molecule App --------------------------
def load_molecule_metrics():
    """
    Produces multiple lines ignoring temperature:
      'Accuracy (%)', 'Avg Chem. Similarity', 'Avg String Distance', 'Total Incorrect SMILES (%)'.
    """
    data = {m: {} for m in MODEL_ORDER}
    
    # Mapping from CSV 'model' strings to standardized IDs
    model_map_mol = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }

    # Initialize accumulators for metrics
    correct_total = {m: 0 for m in MODEL_ORDER}
    chem_similarity_total = {m: 0.0 for m in MODEL_ORDER}
    string_distance_total = {m: 0.0 for m in MODEL_ORDER}
    incorrect_trials_total = {m: 0 for m in MODEL_ORDER}
    count_total = {m: 0 for m in MODEL_ORDER}

    files = [f for f in os.listdir(BASE_PATH_MOLECULE_APP)
             if f.startswith('benchmark_resultsoa:gpt') and f.endswith('.csv')]
    for fname in files:
        path = os.path.join(BASE_PATH_MOLECULE_APP, fname)
        try:
            df = pd.read_csv(path)
            # Convert 'model' to standardized IDs
            if 'model' in df.columns:
                df['model'] = df['model'].map(model_map_mol).fillna(df['model'])

            # Debug print to verify non-zero values
            print(f"Reading {path}, shape={df.shape}")
            print(df[['model','correct','chemical_similarity','string_distance','incorrect_smiles_count']].head(10))

            for _, row in df.iterrows():
                rm = row.get('model', '')
                if rm not in MODEL_ORDER:
                    continue
                # Accumulate correct counts
                correct_total[rm] += 1 if row.get('correct', False) else 0
                # Accumulate chemical similarity
                chem_similarity_total[rm] += row.get('chemical_similarity', 0.0)
                # Accumulate string distance
                string_distance_total[rm] += row.get('string_distance', 0.0)
                # Accumulate incorrect trials (count if incorrect_smiles_count > 0)
                if row.get('incorrect_smiles_count', 0) > 0:
                    incorrect_trials_total[rm] += 1
                # Accumulate total trials
                count_total[rm] += 1

        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Now calculate metrics
    for mm in MODEL_ORDER:
        c_total = count_total[mm]
        if c_total > 0:
            accuracy = (correct_total[mm] / c_total) * 100
            avg_chem_similarity = chem_similarity_total[mm] / c_total
            avg_string_distance = string_distance_total[mm] / c_total
            total_incorrect_smiles_pct = (incorrect_trials_total[mm] / c_total) * 100

            data[mm]['Accuracy (%)'] = accuracy
            data[mm]['Avg Chem. Similarity'] = avg_chem_similarity
            data[mm]['Avg String Distance'] = avg_string_distance
            data[mm]['Total Incorrect SMILES (%)'] = total_incorrect_smiles_pct

    return data

# -------------------------- Main Aggregation & Plotting --------------------------
def main():
    shapes_data       = load_shapes_metrics()
    lcl_data          = load_lcl_metrics()
    tictactoe_data    = load_boardgame_metrics('tictactoe')
    connectfour_data  = load_boardgame_metrics('connectfour')
    battleship_data   = load_boardgame_metrics('battleship')
    molecule_data     = load_molecule_metrics()

    all_data = {
        'shapes':       shapes_data,
        'lcl':          lcl_data,
        'tictactoe':    tictactoe_data,
        'connectfour':  connectfour_data,
        'battleship':   battleship_data,
        'molecule_app': molecule_data
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    for game_name, game_dict in all_data.items():
        all_metrics = set()
        for m in MODEL_ORDER:
            all_metrics.update(game_dict[m].keys())
        if not all_metrics:
            print(f"No metrics found for '{game_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        plt.title(f"{game_name.capitalize()} - Multiple Metrics", fontsize=18, fontweight='bold')
        x_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

        for metric_name in sorted(all_metrics):
            y_vals = []
            for m in MODEL_ORDER:
                val = game_dict[m].get(metric_name, np.nan)
                y_vals.append(val)
            plt.plot(x_labels, y_vals, marker='o', label=metric_name)

        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Value (%)", fontsize=14, fontweight='bold')
        plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
