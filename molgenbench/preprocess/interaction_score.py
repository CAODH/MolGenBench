# this python script have been used to get interactions in ref ligands with protein 
import rdkit
import prolif as plf
from Bio import PDB
from glob import glob
from rdkit import Chem
from rdkit.Chem import Lipinski
import tqdm
import os
import pandas as pd
import numpy as np
import json
import multiprocessing
# get pocket and save to file
from joblib import Parallel, delayed
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
def filterAtomTypes(mol_list,filter_list = ['As']):
    """
    Filter atoms in a molecule based on their atomic symbol.
    """
    # Create a new molecule to store the filtered atoms
    filtered_atoms = []
    # Iterate through the atoms in the original molecule
    for mol in mol_list:
        keep = True
        for atom in mol.GetAtoms():
            # Check if the atom's symbol is 'As'
            if atom.GetSymbol() == 'As':
                keep = False
                break
        if keep:
            # If the atom is not 'As', add it to the filtered list
            filtered_atoms.append(mol)
    return filtered_atoms
def nonBondInteractions(protein_path,ligand_path,interaction_norm_map=None):
    if interaction_norm_map is None:
        all_interactions = plf.Fingerprint.list_available()
    else:
        
        all_interactions = list(interaction_norm_map.keys())

    fp = plf.Fingerprint(interactions = all_interactions)

    mols = rdkit.Chem.SDMolSupplier(ligand_path,removeHs = False)
    # Ahh if not have Hs
    if 'glide_sp' not in ligand_path:
        mols = [Chem.AddHs(mol,addCoords = True) for mol in mols if mol is not None]

    
    m2 = rdkit.Chem.MolFromPDBFile(protein_path,removeHs = False)

    prot = plf.Molecule(m2)
    ligands = [plf.Molecule.from_rdkit(m1) for m1 in mols if m1 is not None and len(m1.GetAtoms())> 0 and not np.isnan(m1.GetConformer().GetPositions()).any()]
    ligands = [ligand for ligand in ligands if ligand is not None and len(ligand.GetAtoms())> 0 and not np.isnan(ligand.GetConformer().GetPositions()).any()]
    # filter rotatable bonds >15
    
    ligands = [ligand for ligand in ligands if Lipinski.NumRotatableBonds(ligand) < 15]
    # filter rotatable bonds >2
    
    ligands = filterAtomTypes(ligands)
    fp.run_from_iterable(ligands,prot,progress=False,n_jobs = 1,)
    df = fp.to_dataframe()
    df['NumRotatableBonds'] = [Lipinski.NumRotatableBonds(ligand) for ligand in ligands]

    return df
##  这些interaction 只要出现了一样的就行，而不是要供体受体要一样
interaction_norm_map = {'HBAcceptor':'Hydrogen',
                        'HBDonor':'Hydrogen',
                        
                        'CationPi':'CationPi',
                        'PiCation':'CationPi',
                        
                        'PiStacking':'PiStacking',
                        'EdgeToFace':'PiStacking',
                        'FaceToFace':'PiStacking',
                        
                        'XBAcceptor':'X-Hydrogen',
                        'XBDonor':'X-Hydrogen',
                        
                        'MetalAcceptor':'Metal',
                        'MetalDonor':'Metal',
                        
                        'Anionic':'Ion',
                        'Cationic':'Ion',
                        
                        'VdWContact':'VdWContact',
                        'Hydrophobic':'Hydrophobic',
                        
                        }

def UniprotInteractions(uniprot_id,generated_dir,root_save_dir):
    protein_path = os.path.join(generated_dir,uniprot_id,f'{uniprot_id}_prep.pdb')
    ligand_dir_list = os.listdir(os.path.join(generated_dir,uniprot_id))
    for ligand_path_dir in ligand_dir_list:
        # if ligand_path_dir in ligand_path_list:
        if not os.path.isdir(os.path.join(generated_dir,uniprot_id,ligand_path_dir)):
            continue
        if ligand_path_dir == 'reference_active_molecules':

            ligand_path_list = glob(
                            os.path.join(generated_dir,uniprot_id,ligand_path_dir,f'{uniprot_id}_*.sdf'))
        
            for ligand_path in ligand_path_list:
                try:
                    save_dir = os.path.join(root_save_dir,uniprot_id,ligand_path_dir)
                    os.makedirs(save_dir,exist_ok=True)
                    save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_interactions.csv'))
                    if os.path.exists(save_path):
                        continue
                    pandas_result = nonBondInteractions(protein_path,ligand_path,interaction_norm_map=None)
                    pandas_result.to_csv(save_path)
                except Exception as e:
                    print(e)
                    print(f'error in {ligand_path}')
                    
        else:
            # len(glob(os.path.join('/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_repeat_output_250218','O14757','Round1','*','*','*.sdf')))
            ligand_path_list = glob(os.path.join(generated_dir,uniprot_id,ligand_path_dir,'*','*','*','*.sdf')) + \
                                glob(os.path.join(generated_dir,uniprot_id,ligand_path_dir,'*','*','*.sdf'))
            for ligand_path in ligand_path_list:
                try:
                    save_dir = os.path.dirname(ligand_path).replace(generated_dir,root_save_dir)
                    os.makedirs(save_dir,exist_ok=True)
                    save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_interactions.csv'))
                    if os.path.exists(save_path):
                        continue
                    pandas_result = nonBondInteractions(protein_path,ligand_path,interaction_norm_map=None)
                    pandas_result.to_csv(save_path)
                except Exception as e:
                    print(e)
                    print(f'error in {ligand_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PreProcess the hit information of generated molecules')
    parser.add_argument('--generated_dir', type=str, help='The directory of results', required=True)
    parser.add_argument('--root_save_dir', type=str, default=None, help='The root directory to save results')
    parser.add_argument('--n_jobs', type=int, default=None, help='The number of parallel jobs')
    
    args = parser.parse_args()
    generated_dir = args.generated_dir
    root_save_dir = args.root_save_dir if args.root_save_dir is not None else args.generated_dir
    cpu_counts = multiprocessing.cpu_count() - 15 if args.n_jobs is None else args.n_jobs

    _ = Parallel(n_jobs=cpu_counts,backend='loky',verbose = 20)(delayed(UniprotInteractions)(uniprot_id,generated_dir,root_save_dir) for uniprot_id in tqdm.tqdm(os.listdir(generated_dir),total = len(os.listdir(generated_dir))))