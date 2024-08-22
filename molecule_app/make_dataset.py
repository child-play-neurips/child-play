import os
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import random as rd
import rdkit.Chem as rdc
import rdkit.Chem.AllChem as rdca
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdCoordGen
from rdkit import Geometry
import selfies as sf
from encryption_utils import encrypt_smiles, generate_key, load_key

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

def create_dataset(num_smiles, length, min_atoms, max_atoms, output_dir):
    key = load_key()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alphabet = list(sf.get_semantic_robust_alphabet())
    alphabet = [ai for ai in alphabet if ("+" not in ai) and ("-" not in ai)]  # Remove charged atoms
    alphabet = [ai for ai in alphabet if ai not in ['[=B]', '[#B]', '[=P]', '[#P]', '[#S]', '[Cl]', '[Br]', '[I]']]  # Remove unusual atom types

    df = pd.DataFrame(columns=['SMILES', 'Encrypted_SMILES', 'PNG_Path', 'ASCII_Path'])

    index = 0
    while index < num_smiles:
        smiles_list = random_selfies(1, length, min_atoms, max_atoms, alphabet=alphabet)
        for smile in smiles_list:
            mol = rdc.MolFromSmiles(smile)
            if mol:
                try:
                    encrypted_smile = encrypt_smiles([smile], key)[0]

                    png_path = os.path.join(output_dir, f'mol_{index}.png')
                    draw_mol_coordgen(mol, png_path)

                    ascii_representation = print_mol_ascii(mol)
                    ascii_path = os.path.join(output_dir, f'mol_{index}.txt')
                    with open(ascii_path, 'w') as ascii_file:
                        ascii_file.write(ascii_representation)

                    df.loc[index] = [smile, encrypted_smile, png_path, ascii_path]
                    index += 1
                except Exception as e:
                    print(f"Skipping invalid SMILES: {smile}. Error: {e}")

    df.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    generate_key()  # Generate and save the encryption key
    num_smiles = 100  # Target number of valid SMILES to generate
    length = 30  # Number of SELFIES characters in each random string
    min_atoms = 10  # Minimum number of atoms in generated molecules
    max_atoms = 15  # Maximum number of atoms in generated molecules
    output_dir = 'molecule_dataset'  # Directory to save the dataset
    create_dataset(num_smiles, length, min_atoms, max_atoms, output_dir)
