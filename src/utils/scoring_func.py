import numpy as np
from collections import Counter
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed

from src.utils.sascore import compute_sa_score


def get_logp(mol):
    return Crippen.MolLogP(mol)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

def general_properties(smiles):
    """
    Calculate the topological structure characteristics of molecules and save the results to an output file.

    :param file_name: File name of the SDF file
    :param smiles: SMILES
    :param output_folder: Output file path
    """
    
    qed_score = None
    sa_score = None
    logp_score = None
    lipinski_score = None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Failed to load molecule from {file_name}")
            return None

        validity = '.' not in smiles
        if validity:
            qed_score = qed(mol)
            sa_score = compute_sa_score(mol)
            logp_score = get_logp(mol)
            lipinski_score = obey_lipinski(mol)

    except Exception as e:
        print(f"Error processing molecule from {smiles}: {e}")

    return {
        'smiles': smiles,
        'validity': validity,
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        # 'ring_size': ring_size
    }