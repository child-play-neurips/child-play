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

TEMPERATURES = [0, 0.5, 1, 1.5]

BASE_PATH_SHAPES        = '../experiment_shapes'
BASE_PATH_LCL           = '../lcl_experiments'
BASE_PATH_BOARDGAMES    = '../experiment_board_games'
BASE_PATH_MOLECULE_APP  = '../molecule_app'

# -------------------------- 1) Shapes --------------------------
def load_shapes_metrics():
    """
    For each model, find the single 'best performing temperature' in terms of
    highest overall 'Win Probability' across the 3 shape tasks (square/triangle/cross).
    
    Each shape had 25 trials => total 75 per temp (3 shapes × 25).
    We'll store 'Win Probability' in [0,1].
    """
    data = {m: {} for m in MODEL_ORDER}
    shape_folders = ['square', 'triangle', 'cross']

    for rm in MODEL_ORDER:
        best_ratio = 0.0
        for t in TEMPERATURES:
            total_wins = 0
            total_losses = 0
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
                        # If 'P1 Wins' or 'Wins' is present, add them
                        if 'P1 Wins' in d and isinstance(d['P1 Wins'], (int, float)):
                            total_wins += d['P1 Wins']
                        elif 'Wins' in d and isinstance(d['Wins'], (int, float)):
                            total_wins += d['Wins']
                        
                        # Losses
                        if 'Losses' in d and isinstance(d['Losses'], (int, float)):
                            total_losses += d['Losses']
                except:
                    pass
            
            attempts = total_wins + total_losses
            if attempts > 0:
                ratio = total_wins / attempts  # fraction in [0,1]
                if ratio > best_ratio:
                    best_ratio = ratio
        
        data[rm]['Win Probability'] = best_ratio

    return data

# -------------------------- 2) LCL --------------------------
def load_lcl_metrics():
    """
    LCL produces two lines: "Correct Proportion" and "Valid Proportion".
    Both are stored as fractions in [0,1].
    """
    data = {m: {} for m in MODEL_ORDER}
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

    if 'Model' in df_valid.columns:
        df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    if 'Model' in df_construct.columns:
        df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])

    grouped_correct = df_valid.groupby('Model')['Correct'].mean()  # fraction [0,1]
    grouped_valid   = df_construct.groupby('Model')['Valid'].mean() # fraction [0,1]

    for m in MODEL_ORDER:
        data[m] = {}
    for m in grouped_correct.index:
        if m in MODEL_ORDER:
            data[m]['Correct Proportion'] = grouped_correct.loc[m]
    for m in grouped_valid.index:
        if m in MODEL_ORDER:
            data[m]['Valid Proportion']   = grouped_valid.loc[m]
    return data

# -------------------------- 3) Board Games --------------------------
def load_boardgame_metrics(game):
    """
    For each model, pick the single best temperature (max 'P1 Wins').
    Then store:
      - Win Probability     = (P1 Wins / 100)
      - Ties Probability    = (Ties / 100)
      - Wrong Moves Probability = (P1 Wrong Moves / 100), if available
    """
    data = {m: {} for m in MODEL_ORDER}

    for m in MODEL_ORDER:
        best_wins = -1
        best_temp = None
        # 1) Find the best temp by P1 (LLM) wins
        for t in TEMPERATURES:
            folder = f"experiment_{game}_{m}_oneshot_temp_{t}"
            path   = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r') as f:
                    d = json.load(f)
                    llm_wins = d.get('P1 Wins', 0)  # or d.get('LLM Wins', 0)
                    if llm_wins > best_wins:
                        best_wins = llm_wins
                        best_temp = t
            except:
                pass

        # If no valid data found, skip
        if best_wins < 0:
            best_wins = 0

        # 2) Load that best temp's JSON again to read Ties, Wrong Moves, etc.
        ties = 0
        wrong_moves = 0
        if best_temp is not None:
            folder = f"experiment_{game}_{m}_oneshot_temp_{best_temp}"
            path   = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        d = json.load(f)
                        ties         = d.get('Ties', 0)
                        wrong_moves  = d.get('P1 Wrong Moves', 0)
                except:
                    pass

        data[m]['Win Probability']         = best_wins / 100.0
        data[m]['Tie Probability']         = ties / 100.0
        data[m]['Wrong Moves Probability'] = wrong_moves / 100.0

    return data

# -------------------------- 4) Molecule App --------------------------
def load_molecule_metrics():
    """
    For each model, read the CSVs. We'll store everything in [0,1].
    - 'Accuracy' = correct / total
    - 'Avg Chem. Similarity' (mean)
    - 'Avg String Distance' (mean)
    - 'Incorrect SMILES Fraction' = # with invalid SMILES / total
    """
    data = {m: {} for m in MODEL_ORDER}
    model_map_mol = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }

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
            if 'model' in df.columns:
                df['model'] = df['model'].map(model_map_mol).fillna(df['model'])

            for _, row in df.iterrows():
                rm = row.get('model', '')
                if rm not in MODEL_ORDER:
                    continue
                correct_total[rm] += 1 if row.get('correct', False) else 0
                chem_similarity_total[rm] += row.get('chemical_similarity', 0.0)
                string_distance_total[rm] += row.get('string_distance', 0.0)
                if row.get('incorrect_smiles_count', 0) > 0:
                    incorrect_trials_total[rm] += 1
                count_total[rm] += 1
        except:
            pass

    for mm in MODEL_ORDER:
        c_total = count_total[mm]
        if c_total > 0:
            accuracy = correct_total[mm] / c_total
            avg_chem_similarity = chem_similarity_total[mm] / c_total
            avg_string_distance = string_distance_total[mm] / c_total
            incorrect_smiles_fraction = incorrect_trials_total[mm] / c_total

            data[mm]['Accuracy'] = accuracy
            data[mm]['Avg Chem. Similarity'] = avg_chem_similarity
            data[mm]['Avg String Distance'] = avg_string_distance
            data[mm]['Incorrect SMILES Fraction'] = incorrect_smiles_fraction

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
        'Shapes':       shapes_data,
        'LCL':          lcl_data,
        'Tic-Tac-Toe':  tictactoe_data,
        'Connect-Four': connectfour_data,
        'Battleship':   battleship_data,
        'GtS':          molecule_data
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    # 1) Plot each game separately, as in your existing code
    for game_name, game_dict in all_data.items():
        all_metrics = set()
        for m in MODEL_ORDER:
            all_metrics.update(game_dict[m].keys())
        if not all_metrics:
            print(f"No metrics found for '{game_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        plt.title(f"Best Temperature per Model for the {game_name} Game", fontsize=18, fontweight='bold')
        
        x_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

        for metric_name in sorted(all_metrics):
            y_vals = []
            for m in MODEL_ORDER:
                val = game_dict[m].get(metric_name, np.nan)
                y_vals.append(val)
            
            plt.plot(x_labels, y_vals, marker='o', label=metric_name)

        plt.ylim(0, 1)
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Probability or Fraction", fontsize=14, fontweight='bold')
        plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show()

    # 2) Create a single "Combined Score" line plot
    #    We'll pick one metric from each domain and average them for each model.
    #    - Shapes -> Win Probability
    #    - Tic-Tac-Toe -> Win Probability
    #    - Connect-Four -> Win Probability
    #    - Battleship -> Win Probability
    #    - LCL -> Correct Proportion
    #    - GtS (Molecule) -> Accuracy
    combined_scores = []
    for m in MODEL_ORDER:
        metrics = []

        # from shapes:
        sh_val = shapes_data[m].get('Win Probability', None)
        if sh_val is not None:
            metrics.append(sh_val)
        
        # from tictactoe:
        ttt_val = tictactoe_data[m].get('Win Probability', None)
        if ttt_val is not None:
            metrics.append(ttt_val)

        # from connectfour:
        cf_val = connectfour_data[m].get('Win Probability', None)
        if cf_val is not None:
            metrics.append(cf_val)

        # from battleship:
        bs_val = battleship_data[m].get('Win Probability', None)
        if bs_val is not None:
            metrics.append(bs_val)

        # from LCL -> "Correct Proportion"
        lcl_val = lcl_data[m].get('Correct Proportion', None)
        if lcl_val is not None:
            metrics.append(lcl_val)

        # from molecules -> "Accuracy"
        mol_val = molecule_data[m].get('Accuracy', None)
        if mol_val is not None:
            metrics.append(mol_val)

        if metrics:
            combined = np.mean(metrics)
        else:
            combined = 0.0  # If no data, you can choose 0 or np.nan

        combined_scores.append({'model': m, 'combined_score': combined})

    # 3) Plot the combined scores in a single line
    df_combined = pd.DataFrame(combined_scores)
    df_combined['model_label'] = df_combined['model'].map(MODEL_LABELS)

    plt.figure(figsize=(10, 6))
    plt.title("Final Combined Score (Average of Key Metrics)", fontsize=18, fontweight='bold')

    # Prepare x and y
    x_labels = df_combined['model_label'].tolist()
    y_vals   = df_combined['combined_score'].tolist()

    plt.plot(x_labels, y_vals, marker='o', label='Combined Score')
    plt.ylim(0, 1)
    plt.xlabel("Model", fontsize=14, fontweight='bold')
    plt.ylabel("Probability (Averaged)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
