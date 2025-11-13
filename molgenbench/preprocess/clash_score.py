import pandas as pd
import os
from tqdm import tqdm
from glob import glob 
import numpy as np
from posecheck import PoseCheck
from posecheck.utils.chem import remove_radicals
# get pocket and save to file
from joblib import Parallel, delayed
from rdkit.Chem import Lipinski
import json
import multiprocessing
from rdkit import Chem
import rdkit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
def loadMols(ligand_path):
    mols = rdkit.Chem.SDMolSupplier(ligand_path,removeHs = False)
    # Ahh if not have Hs
    if 'glide_sp' not in ligand_path:
        mols = [Chem.AddHs(mol,addCoords = True) for mol in mols if mol is not None]
    ligands = [m1 for m1 in mols if m1 is not None and len(m1.GetAtoms())> 0 and not np.isnan(m1.GetConformer().GetPositions()).any()]
    # remove radical 
    mol_filtered = []
    for ligand in ligands:
        try:
            ligand = remove_radicals(ligand)
        except:
            continue
        if ligand is not None and Lipinski.NumRotatableBonds(ligand) < 15:
            mol_filtered.append(ligand)
    return mol_filtered
    
def UniprotClash(uniprot_id,generated_dir,root_save_dir):
    pc = PoseCheck()
    protein_path = os.path.join(generated_dir,uniprot_id,f'{uniprot_id}_pocket10.pdb')
    pc.load_protein_from_pdb(protein_path)
    
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
                    save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_clash_score.csv'))
                    # save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_interactions.csv'))
                    if os.path.exists(save_path):
                        continue
                    pc.load_ligands_from_mols(loadMols(ligand_path))
                    # pc.load_protein_from_pdb(protein_path)
                    ligand_names = [lig.GetProp('_Name') for lig in pc.ligands]
                    pandas_result = {}
                    pandas_result['ligand_name'] = ligand_names
                    pandas_result['clash_score'] = pc.calculate_clashes()
                    pd.DataFrame(pandas_result).to_csv(save_path)
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
                    save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_clash_score.csv'))
                    if os.path.exists(save_path):
                        continue
                    # ligand_path
                    pc.load_ligands_from_mols(loadMols(ligand_path))

                    ligand_names = [lig.GetProp('_Name') for lig in pc.ligands]
                    pandas_result = {}
                    pandas_result['ligand_name'] = ligand_names
                    pandas_result['clash_score'] = pc.calculate_clashes()
                    pd.DataFrame(pandas_result).to_csv(save_path)
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
    
    _ = Parallel(n_jobs=cpu_counts,backend='loky',verbose = 20)(delayed(UniprotClash)(uniprot_id,generated_dir,root_save_dir) for uniprot_id in tqdm(os.listdir(generated_dir),total = len(os.listdir(generated_dir))))