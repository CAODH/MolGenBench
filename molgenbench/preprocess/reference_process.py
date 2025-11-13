
import rdkit
from rdkit import Chem,DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem,Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

import os

import pickle


def enumerate_tautomer_and_partial_chirality(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(50)
        canon_mols = enumerator.Enumerate(mol)
        # 设定遍历选项，固定已知手性中心
        opts = StereoEnumerationOptions(onlyUnassigned=True) # 仅遍历未指定的手性中心
        canon_isomers = [
            isomer 
            for canon_mol in canon_mols 
            for isomer in EnumerateStereoisomers(canon_mol, options=opts)
        ]
        isomers = canon_isomers + [mol]
        # 生成 inchi 结果
        enumerated_inchi = set([Chem.MolToInchi(iso) for iso in isomers if iso is not None])
        # print(len(enumerated_inchi))
        return enumerated_inchi
    except Exception as e:
        print('Unknown Error:',str(e))
        return set()
        

def GetScaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # 计算分子骨架
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # 输出分子骨架的 SMILES
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception as e:
        return 'None'



def compute_smiles_scaffold_inchi_map(uniprot_id,active_dir):
    """
        Short description:
        Read reference active molecules for a given UniProt ID from an SDF file,
        generate canonical SMILES for the largest fragment of each molecule,
        enumerate tautomers and partial-chirality variants for each SMILES and
        for each molecule scaffold, build mappings from each enumerated InChI-like
        key to the original SMILES or scaffold, and persist those mappings as
        pickle files (if they do not already exist).
    """
    try:#active_dir/{uniprot_id}/reference_active_molecules
        save_smiles_path = f'{active_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl'
        save_scaffold_path = f'{active_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl'
        os.makedirs(os.path.dirname(save_smiles_path),exist_ok=True)
        os.makedirs(os.path.dirname(save_scaffold_path),exist_ok=True)
        ref_actives_path = os.path.join(active_dir,uniprot_id,f'reference_active_molecules/{uniprot_id}_reference_active_molecules.sdf')
        ref_mols = Chem.SDMolSupplier(ref_actives_path)
        
        ref_inchi_map_ref_smiles = {}
        ref_smiles_list = []
        
        for ref_mol in ref_mols:
            if ref_mol is None:
                print('error mol in : ',ref_actives_path)
                continue
            frags = Chem.GetMolFrags(ref_mol, asMols=True)
            ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
            ref_temp = Chem.MolToSmiles(ref_mol)
            ref_smiles_list.append(ref_temp)
            if not os.path.exists(save_smiles_path):
                ref_inchi_set  = enumerate_tautomer_and_partial_chirality(ref_temp)
                ref_inchi_map_ref_smiles.update({key:ref_temp for key in ref_inchi_set})
        if not os.path.exists(save_smiles_path):
            with open(save_smiles_path,'wb') as f :
                pickle.dump(ref_inchi_map_ref_smiles,f)
        
        ref_inchi_map_ref_scaffold = {}
        ref_scaffold_list = list(set([GetScaffold(smiles) for smiles in ref_smiles_list]))
        for ref_temp_scaffold in ref_scaffold_list:
            if ref_temp_scaffold != 'None':
                if not os.path.exists(save_scaffold_path):
                    ref_scaffold_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp_scaffold)
                    ref_inchi_map_ref_scaffold.update({key:ref_temp_scaffold for key in ref_scaffold_inchi_set})    
        if not os.path.exists(save_scaffold_path):       
            with open(save_scaffold_path,'wb') as f :
                pickle.dump(ref_inchi_map_ref_scaffold,f)
    except Exception as e:
        # print(f'Error in {model_name}')
        print('Error:',str(e))
        # continue

def main(argv=None):
    parser = argparse.ArgumentParser(description='PreProcess Reference Data to Enumerate Stereo and Tautomer')
    parser.add_argument('--reference_dir', type=str, help='The directory of active compounds', required=True)
    parser.add_argument('--n_jobs', type=int, default=10, help='The number of parallel jobs')
    parser.add_argument('--uniprot_ids', type=str, nargs='*', default=None, help='The list of uniprot ids to process; if not given, process all')
    args = parser.parse_args(argv)
    n_jobs = args.n_jobs
    reference_dir = args.reference_dir
    uniprot_ids = args.uniprot_ids
    if uniprot_ids is None:
        uniprot_ids = [d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))]
    print(f'Processing Num Protein: {len(uniprot_ids)}')

    Parallel(n_jobs=args.n_jobs)(
        delayed(compute_smiles_scaffold_inchi_map)(uniprot_id, reference_dir)
        for uniprot_id in tqdm(uniprot_ids, total=len(uniprot_ids))
    )

if __name__ == "__main__":
    main()
       
