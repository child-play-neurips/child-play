from flask import Flask, request, jsonify
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
from encryption_utils import encrypt_smiles, load_key, decrypt_smiles

app = Flask(__name__)

key = load_key()

def random_selfies(samples, length, min_atoms, max_atoms, alphabet):
    random_selfies = ["".join(rd.choices(alphabet, k=length)) for _ in range(samples)]
    smiles_list = []
    for si in random_selfies:
        mol = rdc.MolFromSmiles(sf.decoder(si))
        if mol:
            num_atoms = mol.GetNumAtoms(onlyExplicit=False)
            if min_atoms <= num_atoms <= max_atoms:
                smiles = rdc.MolToSmiles(mol, kekuleSmiles=True)
                cleaned_smiles = smiles.replace('#', '').replace('=', '')
                if 'H' not in cleaned_smiles:
                    smiles_list.append(cleaned_smiles)
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
    mol1 = rdc.MolFromSmiles(smile1)
    mol2 = rdc.MolFromSmiles(smile2)
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
    return Chem.DataStructs.FingerprintSimilarity(fp1, fp2)

@app.route('/generate_molecule', methods=['POST'])
def generate_molecule():
    data = request.json
    length = data.get('length', 30)
    min_atoms = data.get('min_atoms', 10)
    max_atoms = data.get('max_atoms', 15)
    format_type = data.get('format', 'ascii')  # 'ascii' or 'png'

    alphabet = list(sf.get_semantic_robust_alphabet())
    alphabet = [ai for ai in alphabet if ("+" not in ai) and ("-" not in ai)]  # Remove charged atoms
    alphabet = [ai for ai in alphabet if ai not in ['[=B]', '[#B]', '[=P]', '[#P]', '[#S]', '[Cl]', '[Br]', '[I]']]  # Remove unusual atom types

    smiles_list = random_selfies(1, length, min_atoms, max_atoms, alphabet=alphabet)
    smile = smiles_list[0] if smiles_list else None

    if smile:
        mol = rdc.MolFromSmiles(smile)
        if format_type == 'png':
            png_path = 'generated_molecule.png'
            draw_mol_coordgen(mol, png_path)
            with open(png_path, 'rb') as f:
                img_data = f.read()
            encrypted_smile = encrypt_smiles([smile], key)[0]  # Encrypting SMILES before sending it to storage
            return jsonify({'image_data': img_data.decode('latin1'), 'encrypted_smile': encrypted_smile}), 200
        else:
            ascii_representation = print_mol_ascii(mol)
            encrypted_smile = encrypt_smiles([smile], key)[0]  # Encrypting SMILES before sending it to storage
            return jsonify({'ascii': ascii_representation, 'encrypted_smile': encrypted_smile}), 200
    else:
        return jsonify({'error': 'Failed to generate a molecule'}), 400

@app.route('/evaluate_prediction', methods=['POST'])
def evaluate_prediction():
    data = request.json
    original_smile = decrypt_smiles([data.get('encrypted_smile')], key)[0]
    predicted_smile = data.get('predicted_smile')

    if not original_smile or not predicted_smile:
        return jsonify({'error': 'Missing SMILES data'}), 400

    similarity = calculate_similarity(original_smile, predicted_smile)
    is_correct = original_smile == predicted_smile

    return jsonify({
        'correct': is_correct,
        'similarity': similarity
    })

if __name__ == '__main__':
    app.run(debug=True)
