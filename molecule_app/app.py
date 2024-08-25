import logging
from flask import Flask, render_template, jsonify, request
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os
import random as rd
import rdkit.Chem as rdc
import rdkit.Chem.AllChem as rdca
from io import BytesIO
from PIL import Image
import selfies as sf
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdCoordGen
from rdkit import Geometry
import numpy as np
import Levenshtein
import rdkit.DataStructs as rdd

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("app.log", mode='w')
                    ])
logger = logging.getLogger(__name__)

generated_molecules = {}

MAX_SMILES_LENGTH = 200  # Set a maximum reasonable length for a SMILES string

def random_selfies(samples, length, min_atoms, max_atoms, alphabet):
    smiles_list = []
    batch_size = 100  # Generate a batch of 100 SELFIES at once
    while len(smiles_list) < samples:
        selfie_batch = ["".join(rd.choices(alphabet, k=length)) for _ in range(batch_size)]
        for random_selfie in selfie_batch:
            decoded_smiles = sf.decoder(random_selfie)
            mol = rdc.MolFromSmiles(decoded_smiles)
            if mol:
                num_atoms = mol.GetNumAtoms(onlyExplicit=False)
                if min_atoms <= num_atoms <= max_atoms:
                    # Additional validity checks
                    if Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                        smiles = rdc.MolToSmiles(mol, kekuleSmiles=True)
                        cleaned_smiles = smiles.replace('#', '').replace('=', '')
                        if 'H' not in cleaned_smiles:
                            smiles_list.append(cleaned_smiles)
                            if len(smiles_list) >= samples:
                                break
        logger.info(f"Generated {len(smiles_list)} valid SMILES so far.")
    return smiles_list

def draw_mol_coordgen(mol, save_path):
    rdCoordGen.AddCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    drawer.drawOptions().fixedBondLength = 40
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    bio = BytesIO(drawer.GetDrawingText())
    img = Image.open(bio)
    img.save(save_path)

def print_mol_ascii(mol):
    rdCoordGen.AddCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.drawOptions().fixedBondLength = 40
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    atom_symbols = []
    atom_xpos = []
    atom_ypos = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        atom_symbols.append(mol.GetAtomWithIdx(i).GetSymbol())
        pos_A = conf.GetAtomPosition(i)
        pos_point = Geometry.Point2D(pos_A.x, pos_A.y)
        dpos = drawer.GetDrawCoords(pos_point)
        atom_xpos.append(dpos.x)
        atom_ypos.append(dpos.y)

    atom_coords = list(zip(atom_xpos, atom_ypos))
    scale_atom_coords = [(p[0] / 14, p[1] / 14) for p in atom_coords]
    round_atom_coords = [(round(p[0]), round(p[1])) for p in scale_atom_coords]

    xmin = min(c[0] for c in round_atom_coords)
    xmax = max(c[0] for c in round_atom_coords)
    ymin = min(c[1] for c in round_atom_coords)
    ymax = max(c[1] for c in round_atom_coords)

    xymin = min(xmin, ymin)
    norm_atom_coords = [(p[0] - xymin, p[1] - xymin) for p in round_atom_coords]

    atom_begin = []
    atom_end = []
    bond_type = []

    for bond in mol.GetBonds():
        atom_begin.append(bond.GetBeginAtomIdx())
        atom_end.append(bond.GetEndAtomIdx())
        bond_type.append(bond.GetBondTypeAsDouble())

    begin_connection_xpos = []
    begin_connection_ypos = []
    for i in atom_begin:
        begin_connection_xpos.append(norm_atom_coords[i][0])
        begin_connection_ypos.append(norm_atom_coords[i][1])

    end_connection_xpos = []
    end_connection_ypos = []
    for i in atom_end:
        end_connection_xpos.append(norm_atom_coords[i][0])
        end_connection_ypos.append(norm_atom_coords[i][1])

    midx_bond = []
    for x1, x2 in zip(begin_connection_xpos, end_connection_xpos):
        middle = (x1 + x2) / 2
        midx_bond.append(middle)

    midy_bond = []
    for y1, y2 in zip(begin_connection_ypos, end_connection_ypos):
        middle = (y1 + y2) / 2
        midy_bond.append(middle)

    bond_center_points = list(zip(midx_bond, midy_bond))

    xymax = max(xmax, ymax)
    start = 0
    stop = round(xymax + 1)
    samples = round((stop * 2) + 1)
    print_range = np.linspace(start, stop, samples)

    ascii_representation = []

    bond_symbol = '*'
    fill_symbol = ' '
    for y in print_range:
        chars = []
        for x in print_range:
            if (x, y) in norm_atom_coords:
                index_value = norm_atom_coords.index((x, y))
                chars.append(atom_symbols[index_value])
            elif (x, y) in bond_center_points:
                chars.append(bond_symbol)
            else:
                chars.append(fill_symbol)
        ascii_representation.append(' '.join(chars))

    return "\n".join(ascii_representation)

def calculate_similarity(smile1, smile2):
    try:
        mol1 = rdc.MolFromSmiles(smile1)
        mol2 = rdc.MolFromSmiles(smile2)
        fpgen = rdca.GetMorganGenerator(radius=2)
        fp1 = fpgen.GetSparseCountFingerprint(mol1)
        fp2 = fpgen.GetSparseCountFingerprint(mol2)
        similarity = rdd.DiceSimilarity(fp1, fp2)
        return similarity
    except Exception as e:
        logger.warning(f"Chemical similarity calculation failed: {e}")
        return -1  # Return -1 for invalid SMILES

def calculate_string_distance(smile1, smile2):
    """Calculate the Levenshtein distance between two SMILES strings."""
    #try:
    #    canon1 = rdc.CanonSmiles(smile1)
    #    canon2 = rdc.CanonSmiles(smile2)
    #except Exception as e:
    #    logger.warning(f"SMILES canonicalization failed: {e}")
    #    return -1  # Return -1 to indicate an error
    return Levenshtein.distance(smile1, smile2)

@app.route('/generate_molecule', methods=['POST'])
def generate_molecule():
    data = request.json
    length = data.get('length', 30)
    min_atoms = data.get('min_atoms', 10)
    max_atoms = data.get('max_atoms', 15)
    format_type = data.get('format', 'ascii')

    alphabet = list(sf.get_semantic_robust_alphabet())
    alphabet = [ai for ai in alphabet if ("+" not in ai) and ("-" not in ai)]
    alphabet = [ai for ai in alphabet if ai not in ['[=B]', '[#B]', '[=P]', '[#P]', '[#S]', '[Cl]', '[Br]', '[I]']]

    smiles_list = random_selfies(1, length, min_atoms, max_atoms, alphabet=alphabet)
    smile = smiles_list[0] if smiles_list else None

    if smile:
        molecule_id = len(generated_molecules) + 1
        generated_molecules[molecule_id] = smile
        
        logger.info(f"Generated SMILES (ID: {molecule_id})")

        mol = rdc.MolFromSmiles(smile)
        if format_type == 'png':
            png_path = f'generated_molecule_{molecule_id}.png'
            draw_mol_coordgen(mol, png_path)
            with open(png_path, 'rb') as f:
                img_data = f.read()
            return img_data, 200, {'Content-Type': 'image/png'}
        else:
            ascii_representation = print_mol_ascii(mol)
            return jsonify({'ascii': ascii_representation, 'molecule_id': molecule_id}), 200
    else:
        logger.error("Failed to generate a valid molecule.")
        return jsonify({'error': 'Failed to generate a molecule'}), 400

@app.route('/evaluate_prediction', methods=['POST'])
def evaluate_prediction():
    data = request.json
    molecule_id = data.get('molecule_id')
    predicted_smile = data.get('predicted_smile')

    # Validate the SMILES string length and content
    if len(predicted_smile) > MAX_SMILES_LENGTH or not predicted_smile:
        return jsonify({
            "correct": False,
            "chemical_similarity": 0.0,
            "string_distance": -1
        }), 200
    
    original_smile = generated_molecules.get(molecule_id)
    if not original_smile:
        return jsonify({'error': 'Invalid molecule ID'}), 400
    
    try:
        canon_original_smile = rdc.CanonSmiles(original_smile)
        canon_predicted_smile = rdc.CanonSmiles(predicted_smile)
    except Exception as e:
        logger.warning(f"Canonicalization failed: {e}")
        return jsonify({'error': 'SMILES canonicalization failed'}), 400
    
    string_distance = calculate_string_distance(canon_original_smile, canon_predicted_smile)
    chemical_similarity = calculate_similarity(canon_original_smile, canon_predicted_smile)

    is_correct = canon_original_smile == canon_predicted_smile
    
    return jsonify({
        "correct": is_correct,
        "chemical_similarity": chemical_similarity,
        "string_distance": string_distance,
    }), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)