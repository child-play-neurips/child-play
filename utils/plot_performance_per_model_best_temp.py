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
    For each model, we:
      1) Print the (wins, losses) per shape, per temperature, plus the derived rate = wins/25, losses/25.
      2) Compute the single 'best performing temperature' in terms of highest overall 'Win Probability'
         across the 3 shape tasks (square, triangle, cross).
      3) Return that best probability in [0,1].
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
                shape_wins = 0
                shape_losses = 0
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            d = json.load(f)
                            # If 'P1 Wins' or 'Wins' is present, add them
                            if 'P1 Wins' in d and isinstance(d['P1 Wins'], (int, float)):
                                shape_wins = d['P1 Wins']
                            elif 'Wins' in d and isinstance(d['Wins'], (int, float)):
                                shape_wins = d['Wins']
                            
                            if 'Losses' in d and isinstance(d['Losses'], (int, float)):
                                shape_losses = d['Losses']
                    except:
                        pass

                # Print for THIS shape & temp
                # Each shape has exactly 25 attempts
                print(f"[SHAPES] Model={rm}, Temp={t}, Shape={shape}, "
                      f"Wins={shape_wins}, Losses={shape_losses}, "
                      f"WinRate={shape_wins/25:.3f}, LossRate={shape_losses/25:.3f}")
                
                total_wins += shape_wins
                total_losses += shape_losses
            
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
    LCL:
      1) Print for each model: total # of attempts in df_valid, # correct, # incorrect, plus fraction.
      2) Print for each model: total # in df_construct, # valid, # invalid, fraction.
      3) Return the aggregated fractions in [0,1].
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

    # For 'correct'
    # We'll do a simple group to get the sum and count
    if not df_valid.empty:
        for m in MODEL_ORDER:
            sub = df_valid[df_valid['Model'] == m]
            total_count = len(sub)
            correct_count = sub['Correct'].sum() if 'Correct' in sub.columns else 0
            incorrect_count = total_count - correct_count
            if total_count > 0:
                print(f"[LCL Validity] Model={m}, total={total_count}, correct={correct_count}, "
                      f"incorrect={incorrect_count}, correctness={correct_count/total_count:.3f}")

    # For 'valid'
    if not df_construct.empty:
        for m in MODEL_ORDER:
            sub = df_construct[df_construct['Model'] == m]
            total_count = len(sub)
            valid_count = sub['Valid'].sum() if 'Valid' in sub.columns else 0
            invalid_count = total_count - valid_count
            if total_count > 0:
                print(f"[LCL Construct] Model={m}, total={total_count}, valid={valid_count}, "
                      f"invalid={invalid_count}, validity={valid_count/total_count:.3f}")

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
    For each model and temperature, print:
      - P1 Wins, Ties, P2 Wins, P1 Wrong Moves, etc. => out of 100
    Then pick the single best temperature (max 'P1 Wins') for final data.

    We store in 'data' only the best fraction, but we do print everything.
    """
    data = {m: {} for m in MODEL_ORDER}

    for m in MODEL_ORDER:
        best_wins = -1
        best_temp = None

        # 1) Print for every temperature first
        for t in TEMPERATURES:
            folder = f"experiment_{game}_{m}_oneshot_temp_{t}"
            path   = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
            if not os.path.exists(path):
                print(f"[{game.upper()}] Missing path for Model={m}, Temp={t}")
                continue

            try:
                with open(path, 'r') as f:
                    d = json.load(f)
                    p1_wins      = d.get('P1 Wins', 0)
                    p2_wins      = d.get('P2 Wins', 0)
                    ties         = d.get('Ties', 0)
                    p1_wrong     = d.get('P1 Wrong Moves', 0)
                    p2_wrong     = d.get('P2 Wrong Moves', 0)

                    print(f"[{game.upper()}] Model={m}, Temp={t}, "
                          f"P1Wins={p1_wins}, P2Wins={p2_wins}, Ties={ties}, "
                          f"P1Wrong={p1_wrong}, P2Wrong={p2_wrong}, "
                          f"WinRate={p1_wins/100:.3f}, TieRate={ties/100:.3f}, LossRate={p2_wins/100:.3f}")

                    # Check if it's the best so far
                    if p1_wins > best_wins:
                        best_wins = p1_wins
                        best_temp = t
            except:
                pass

        # 2) Re-load that best temp to store final fraction
        if best_wins < 0:
            best_wins = 0
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
    For each model, we print total # rows, how many correct, fraction correct,
    how many had invalid SMILES, etc. Then store final [0,1] metrics in `data`.
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
        if not os.path.exists(path):
            continue
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

    # Print final tallies for each model
    for mm in MODEL_ORDER:
        c_total = count_total[mm]
        corr = correct_total[mm]
        inc_smiles = incorrect_trials_total[mm]
        if c_total > 0:
            print(f"[MOLECULES] Model={mm}, total={c_total}, correct={corr}, "
                  f"correctRate={corr/c_total:.3f}, invalidSMILES={inc_smiles}, "
                  f"invalidRate={inc_smiles/c_total:.3f}")
            
            accuracy = corr / c_total
            avg_chem_similarity = chem_similarity_total[mm] / c_total
            avg_string_distance = string_distance_total[mm] / c_total
            incorrect_smiles_fraction = inc_smiles / c_total

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

    for game_name, game_dict in all_data.items():
        # Gather all metrics from this dictionary
        all_metrics = set()
        for m in MODEL_ORDER:
            all_metrics.update(game_dict[m].keys())
        if not all_metrics:
            print(f"No metrics found for '{game_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        plt.title(f"Best Temperature per Model for the {game_name} Game", fontsize=18, fontweight='bold')
        
        # The x-axis is the 4 models (best temp for each)
        x_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

        # Plot each metric as a separate line
        for metric_name in sorted(all_metrics):
            y_vals = []
            for m in MODEL_ORDER:
                val = game_dict[m].get(metric_name, np.nan)
                y_vals.append(val)
            
            plt.plot(x_labels, y_vals, marker='o', label=metric_name)

        # Probability scale => y in [0,1]
        plt.ylim(0, 1)
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Probability or Fraction", fontsize=14, fontweight='bold')
        plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
