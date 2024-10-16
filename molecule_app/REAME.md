# Molecule Generation and Evaluation API

## Overview

This repository provides a Flask-based API for generating molecular structures and evaluating SMILES string predictions. The primary goal is to benchmark zero-shot learning models by testing their ability to interpret molecular ASCII representations and predict the corresponding SMILES string without explicit training.

## Features

- **Random Molecule Generation**: Generate molecules using the SELFIES representation with control over the number of atoms, molecule length, and output format (ASCII or PNG).
- **Chemical Similarity Calculation**: Compare predicted SMILES strings to the original using chemical fingerprints and the Dice similarity metric.
- **String Distance Calculation**: Compute the Levenshtein distance between the original and predicted SMILES strings to assess string-based similarity.

## Example

Here is an example of a molecule generated in ASCII format:

                    O                                        
                                                             
                *       *                                    
                                                             
            P               N                                
                                                             
            *               *                                
                                                             
            B               C                                
                                                             
      *         *       *                                    
                                                             
O                   N                                        


The stars (`*`) represent bonds between the atoms in the ASCII. The corresponding SMILES string for this molecule is: **OB1NCNOP1**.

## Usage

### Prerequisites (requirements)

- transformers
- rdkit
- openai
- selfies
- pandas
- numpy
- requests
- flask
- pillow
- levenshtein

### Setup

You can simply use the main.py to base your own experiment on. The API is accessible via the [website](https://child-play.onrender.com). 
                                
## API Endpoints

### `POST /generate_molecule`

Generates a molecule with specified parameters.

- **Request Body**:
  ```json
  {
      "length": 30,
      "min_atoms": 10,
      "max_atoms": 15,
      "format": "ascii"  // or "png"
  }                            
                                                             
- Response: Returns the ASCII representation or PNG image and a molecule ID.

### `POST /evaluate_prediction`

Evaluates a predicted SMILES string.

- **Request Body**:
  ```json
    {
        "molecule_id": 1,
        "predicted_smile": "OB1NCNOP1"
    }                         
                                                             
- Response: Returns whether the prediction is correct, the chemical similarity, and the string distance.

### Chemical Similarity Calculation

The `calculate_similarity` function is used to compute the chemical similarity between two SMILES strings. It uses the following steps:

1. **Convert SMILES to Molecule**: The function first converts the SMILES strings into RDKit molecule objects using `rdc.MolFromSmiles`.
2. **Generate Fingerprints**: It then generates molecular fingerprints using the Morgan fingerprint method (`rdca.GetMorganGenerator(radius=2)`), which encodes the structural information of the molecules.
3. **Calculate Similarity**: The Dice similarity score is computed between the two fingerprints using `rdd.DiceSimilarity(fp1, fp2)`. This score ranges from 0 to 1, where 1 indicates identical molecules, and 0 indicates no similarity.

If any errors occur during the calculation, the function returns `-1`.

### String Distance Calculation

The `calculate_string_distance` function calculates the Levenshtein distance between two SMILES strings. This metric measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into the other.

1. **Canonicalize SMILES**: The function attempts to canonicalize the SMILES strings using `rdc.CanonSmiles` to ensure a standardized form before comparison.
2. **Calculate Distance**: The Levenshtein distance is calculated using the `Levenshtein.distance` function. A lower distance indicates higher similarity.

These calculations are used to evaluate the correctness and similarity of predicted SMILES strings in the `/evaluate_prediction` API endpoint.

### Wrapper: `ask`

The `ask` function is a wrapper that interfaces with different models (e.g., OpenAI, Hugging Face) to generate SMILES predictions based on the provided prompts.

**Parameters:**

- `api_messages`: The prompt messages.
- `temperature`: The temperature setting for the model.
- `model`: The model identifier (e.g., `"oa:gpt-3.5-turbo-0125"`).

**Response Handling:**

The function selects the appropriate model based on the prefix (`oa`, `hf`, `ans`) and returns the model's output.                         
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             
                                                             