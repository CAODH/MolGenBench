import os
from collections import Counter
from copy import deepcopy
import numpy as np
from typing import List
from itertools import combinations
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.DataStructs import TanimotoSimilarity
from joblib import Parallel, delayed
from tqdm import tqdm


from src.utils.sascore import compute_sa_score


def calculate_morgan_fingerprint(molecule, radius=2, nBits=2048):
    if molecule is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
    return fingerprint


def calculate_Morgan_fingerprint_Tdiv(molecule_list: List[Chem.Mol]):
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

def calculate_uniqueness(molecule_list: List[Chem.Mol]):
    mols = [mol for mol in molecule_list if mol is not None]
    if len(mols) == 0:
        return 0.0
    if len(mols) == 1:
        return 1.0
    
    unique_molecules = set()
    for mol in mols:
        unique_molecules.add(Chem.MolToSmiles(mol))
    uniqueness = len(unique_molecules) / len(mols)
    
    return uniqueness
            

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

        # Calculate various topological structural features
        validity = '.' in smiles
        if validity:
            qed_score = qed(mol)
            sa_score = compute_sa_score(mol)
            logp_score = get_logp(mol)
            lipinski_score = obey_lipinski(mol)

    except Exception as e:
        print(f"Error processing molecule from {smiles}: {e}")

    return {
        'validity': validity,
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        # 'ring_size': ring_size
    }


def structural_properties(smiles):
    """
    Calculate the topological structure characteristics of molecules and save the results to an output file.

    :param file_name: File name of the SDF file
    :param smiles: SMILES
    :param output_folder: Output file path
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Failed to load molecule from {file_name}")
            return None

        # Calculate various topological structural features
        num_atoms = mol.GetNumAtoms()
        num_non_hydrogen_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        num_chiral_carbons = sum(1 for idx, _ in chiral_centers if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        # Format the properties string
        properties_str = (f"smiles:\t{smiles}\theavy-atom:\t{num_non_hydrogen_atoms}\tchiral-center:\t{num_chiral_carbons}\t"
                          f"Rings:\t{num_rings}\tAromatic-Rings:\t{num_aromatic_rings}\t"
                          f"Rotatable-Bonds:\t{num_rotatable_bonds}\tFsp3:\t{fraction_csp3:.3f}")
        
        return properties_str
        # Write the properties string to the output file

    except Exception as e:
        print(f"Error processing molecule from {smiles}: {e}")
def parallelStructuralProperties(file_name, njobs,output_file):
    """
    Parallel Calculate the topological structure characteristics of molecules and save the results to an output file.

    :param file_name: File name of the SDF file
    :param njobs: number of process to use
    :param output_file: Output file path
    
    
    """
    if os.path.exists(output_file):
        os.remove(output_file)
    if file_name.endswith('.sdf'):
        smiles_list = [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(file_name)]
    elif file_name.endswith('.smi') or file_name.endswith('.txt'):
        smiles_list = [line.strip() for line in open(file_name)]
    results = Parallel(n_jobs=njobs)(delayed(structural_properties)(smiles) for smiles in tqdm(smiles_list,total = len(smiles_list)))
    for idx, result in enumerate(results):
        if result is not None:
            with open(output_file, 'a') as result_file:
                result_file.write(result + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate 2D structural properties of molecules')
    
    parser.add_argument('--file_name', type=str,default='/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_result_from_PGMG.txt', help='File name of the SDF or .smi or .txt file')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    args = parser.parse_args()
    import time 
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.file_name)}/{start}_2D_properties_cal_result.txt'
    parallelStructuralProperties(args.file_name, args.njobs,args.output_file)
    

    
    
    
    