import openai
import pandas as pd
from wrapper import ask
import requests
import json

# Define the models and temperatures to test
models = [
    "oa:gpt-3.5-turbo-0125",
    "oa:gpt-4-1106-preview"
]
temperatures = [0, 0.5, 1, 1.5]

# URL of the local server running the API
SERVER_URL = "http://127.0.0.1:5000"

def generate_molecule(format="ascii"):
    """
    Generates a molecule using the API and returns the representation and encrypted SMILES.
    """
    response = requests.post(f"{SERVER_URL}/generate_molecule", json={
        "length": 30,
        "min_atoms": 10,
        "max_atoms": 15,
        "format": format
    })
    
    if response.status_code == 200:
        return response.json()  # Returns both the ASCII/PNG and the encrypted SMILES
    else:
        raise Exception("Failed to generate molecule")

def evaluate_prediction(encrypted_smile, predicted_smile):
    """
    Evaluates the predicted SMILES against the encrypted original using the API.
    """
    response = requests.post(f"{SERVER_URL}/evaluate_prediction", json={
        "encrypted_smile": encrypted_smile,
        "predicted_smile": predicted_smile
    })
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to evaluate prediction")

def run_benchmark():
    results = []

    for model in models:
        for temp in temperatures:
            for trial in range(10):  # Run each model-temp combination 10 times
                print(f"Running trial {trial + 1} for model {model} at temperature {temp}...")
                
                # Step 1: Generate a molecule and get its representation
                molecule_data = generate_molecule(format="ascii")
                ascii_representation = molecule_data["ascii"]
                encrypted_smile = molecule_data["encrypted_smile"]
                
                # Step 2: Prepare the API messages
                api_messages = [
                    {"role": "system", "content": "You are a molecular structure interpreter."},
                    {"role": "user", "content": f"Here is the ASCII representation of a molecule:\n{ascii_representation}\nPlease provide the corresponding SMILES string."}
                ]
                
                # Step 3: Ask the model to predict the SMILES
                predicted_smile = ask(api_messages, temperature=temp, model=model)
                
                # Step 4: Evaluate the prediction
                evaluation_result = evaluate_prediction(encrypted_smile, predicted_smile)
                
                # Step 5: Store the results
                results.append({
                    "model": model,
                    "temperature": temp,
                    "trial": trial + 1,
                    "predicted_smile": predicted_smile,
                    "correct": evaluation_result["correct"],
                    "similarity": evaluation_result["similarity"]
                })

    # Step 6: Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("Benchmarking completed. Results saved to 'benchmark_results.csv'.")

if __name__ == "__main__":
    run_benchmark()
