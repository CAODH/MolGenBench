import os
import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from itertools import combinations
from multiprocessing import Pool

#Calculate Morgan fingerprint
def calculate_morgan_fingerprint(molecule, radius=2, nBits=2048):
    if molecule is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
    return fingerprint

def calculate_Morgan_fingerprint_div_single_target(molecule_list: List[Chem.Mol]):
    mols = [mol for mol in molecule_list if mol is not None]
    if len(mols) == 0:
        return 0.0
    if len(mols) == 1:
        return 1.0

    morgan_fingerprints = [calculate_morgan_fingerprint(mol) for mol in mols]

    tanimoto_similarities = [
        TanimotoSimilarity(f1, f2)
        for f1, f2 in combinations(morgan_fingerprints, 2)
    ]
    return 1 - np.mean(tanimoto_similarities)