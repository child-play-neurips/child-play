import pandas as pd
import requests
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from wrapper import ask

# Define the models and temperatures to test
# models = [
#     "oa:gpt-3.5-turbo-0125",
#     "oa:gpt-4-1106-preview"
# ]
# temperatures = [0, 0.5, 1, 1.5]

models = ['oa:gpt-4o-2024-08-06', 'oa:gpt-4o-mini-2024-07-18']
temperatures = [0, 0.5, 1, 1.5]

# URL of the local server running the API
SERVER_URL = "http://127.0.0.1:5000"

def extract_smiles_from_response(response_text):
    """
    Extracts the SMILES string from the model's response text.
    """
    prefix = "The SMILES string for the given molecule is:"
    if prefix in response_text:
        return response_text.split(prefix)[-1].strip()
    else:
        return response_text.strip()

def generate_molecule(format="ascii"):
    """
    Generates a molecule using the API and returns the representation and molecule ID.
    """
    response = requests.post(f"{SERVER_URL}/generate_molecule", json={
        "length": 30,
        "min_atoms": 10,
        "max_atoms": 15,
        "format": format
    })
    
    if response.status_code == 200:
        return response.json()  # Returns both the ASCII/PNG and the molecule ID
    else:
        raise Exception("Failed to generate molecule")

def evaluate_prediction(molecule_id, predicted_smile):
    """
    Evaluates the predicted SMILES using the API.
    """
    response = requests.post(f"{SERVER_URL}/evaluate_prediction", json={
        "molecule_id": molecule_id,
        "predicted_smile": predicted_smile
    })
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to evaluate prediction")

def run_benchmark():
    results = []
    evaluation_results = []
    temp_c = 100  # Number of trials per model-temperature combination
    
    for model in models:
        print(f"Starting benchmark for model: {model}")
        for temp in temperatures:
            print(f"Testing with temperature: {temp}")
            trial_results = []
            correct_count = 0
            incorrect_count = 0
            incorrect_smiles_count = 0  # Count of incorrect SMILES predictions
            
            for trial in range(temp_c):
                print(f"Running trial {trial + 1} for model {model} at temperature {temp}...")
                
                # Step 1: Generate a molecule and get its representation and ID
                molecule_data = generate_molecule(format="ascii")
                ascii_representation = molecule_data["ascii"]
                molecule_id = molecule_data["molecule_id"]
                print(f"Generated ASCII:\n{ascii_representation}\n")
                
                # Step 2: Prepare the API messages
                api_messages = [
                    {"role": "system", "content": "You are a SMILES interpreter."},
                    {"role": "user", "content": f"The following is a depiction of a molecule that uses ASCII characters. The depiction uses element symbols for the corresponding atoms and * characters for single bonds between the atoms. Hence, the * characters are placed between two atoms that are connected by a single bond. Based on this depiction, write the SMILES string of the depicted molecule. Make sure that the SMILES string is correct and does not include the * characters:\n{ascii_representation}\nPlease provide the corresponding SMILES string, write nothing else but the SMILES string."}
                ]
                
                # Step 3: Ask the model to predict the SMILES
                print(f"Asking model {model} to predict the SMILES at temperature {temp}...\n")
                response_text = ask(api_messages, temperature=temp, model=model)
                predicted_smile = extract_smiles_from_response(response_text)
                print(f"Model predicted SMILES: {predicted_smile}\n")
                
                # Step 4: Evaluate the prediction
                eval_result = evaluate_prediction(molecule_id, predicted_smile)
                trial_results.append(eval_result)
                
                # Update correct/incorrect counts
                if eval_result["correct"]:
                    correct_count += 1
                else:
                    incorrect_count += 1
                    if eval_result["chemical_similarity"] == -1:
                        incorrect_smiles_count += 1
                
                # Step 5: Store the results
                results.append({
                    "model": model,
                    "temperature": temp,
                    "trial": trial + 1,
                    "predicted_smile": predicted_smile,
                    "correct": eval_result["correct"],
                    "chemical_similarity": eval_result["chemical_similarity"],
                    "string_distance": eval_result["string_distance"],
                    "incorrect_smiles_count": incorrect_smiles_count
                })

            # Append evaluation metrics per temperature/model combination
            chemical_similarities = [r["chemical_similarity"] for r in trial_results]
            string_distances = [r["string_distance"] for r in trial_results]

            accuracy = accuracy_score([True] * len(trial_results), [r["correct"] for r in trial_results])
            f1 = f1_score([True] * len(trial_results), [r["correct"] for r in trial_results], zero_division=1)
            avg_chemical_similarity = np.mean(chemical_similarities)
            avg_string_distance = np.mean(string_distances)

            evaluation_results.append({
                "model": model,
                "temperature": temp,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "accuracy": accuracy,
                "f1_score": f1,
                "avg_chemical_similarity": avg_chemical_similarity,
                "avg_string_distance": avg_string_distance,
                "incorrect_smiles_count": incorrect_smiles_count
            })

        # Step 6: Save results to a CSV file
        print("Saving benchmark results to 'benchmark_results.csv'...")
        df = pd.DataFrame(results)
        df.to_csv("benchmark_results" + model + ".csv", index=False, escapechar='\\')
        print("Benchmarking completed. Results saved to 'benchmark_results.csv'.")

        # Print the evaluation summary
        summary_df = pd.DataFrame(evaluation_results)
        print("\nEvaluation Summary:\n")
        print(summary_df.to_string(index=False))
        summary_df.to_csv("evaluation_summary" +  model + ".csv", index=False, escapechar='\\')

if __name__ == "__main__":
    run_benchmark()
