from cryptography.fernet import Fernet
import pandas as pd

def generate_key():
    """Generate a key for encryption and save it to a file."""
    key = Fernet.generate_key()
    with open("key.key", "wb") as key_file:
        key_file.write(key)

def load_key():
    """Load the encryption key from the current directory."""
    return open("key.key", "rb").read()

def encrypt_smiles(smiles_list, key):
    """Encrypt a list of SMILES strings using the provided key."""
    f = Fernet(key)
    encrypted_smiles = [f.encrypt(smile.encode()).decode() for smile in smiles_list]
    return encrypted_smiles

def decrypt_smiles(encrypted_smiles_list, key):
    """Decrypt a list of encrypted SMILES strings using the provided key."""
    f = Fernet(key)
    decrypted_smiles = [f.decrypt(smile.encode()).decode() for smile in encrypted_smiles_list]
    return decrypted_smiles
