# We implement the strain_energy calculation in this file.
# PoseCheck is used in this file to calculate the strain energy of a pose.
# https://github.com/cch1999/posecheck

# Install PoseCheck
# git clone https://github.com/cch1999/posecheck.git
# cd posecheck
# git checkout 57a1938  # the calculation of strain energy used in our paper
# pip install -e .
# pip install -r requirements.txt
# conda install -c mx reduce

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from rdkit import Chem
from posecheck import PoseCheck


def get_strain_energy(idx, mol):
    """
    Calculate the strain energy of a molecule.

    :param mol: RDKit Mol object
    """
    strain = np.nan
    try:
        pc = PoseCheck()
        # idx = mol.GetProp('_Name')
        pc.load_ligands_from_mols([mol])
        strain = pc.calculate_strain_energy()[0]
        
    except Exception as e:
        print(f"[Strain Energy] Error processing molecule: {e}")
    
    return {
        'idx': idx,
        'strain_energy': strain
    }

def parallelStrainEnergy(file_name, njobs, output_file):
    """
    Calculate the strain energy of molecules in parallel.

    :param file_name: File name of the SDF
    :param njobs: Number of processes to use
    :param output_file: Output file path
    """
    mols = [(mol.GetProp('_Name'), mol) for mol in Chem.SDMolSupplier(file_name) if mol is not None]
    
    results = Parallel(n_jobs=njobs)(delayed(get_strain_energy)(idx, mol) for idx, mol in tqdm(mols, total=len(mols)))
    
    results_dict = {
        'idx': [],
        'strain_energy': []
    }

    for result in results:
        if result is not None:
            results_dict['idx'].append(result['idx'])
            results_dict['strain_energy'].append(result['strain_energy'])

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_file, index=False)
    
    return results_df


def get_summary_stats(results_df):
    
    strain_energy_list = results_df["strain_energy"]
    isnan = np.isnan(strain_energy_list)
    n_isnan = isnan.sum()
    n_total = len(strain_energy_list)
    
    perc = np.percentile(strain_energy_list[~isnan], [25, 50, 75])
    
    summary_dict = {
        'strain_fail': n_isnan / n_total,
        'strain_25': perc[0],
        'strain_50': perc[1],
        'strain_75': perc[2]
    }
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['value']).transpose()
    
    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate 3D structural properties of molecules')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    parser.add_argument('--file_name', type=str,default = '/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules.sdf', help='File name of the SDF that contains 3D molecules')
    
    args = parser.parse_args()
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.file_name)}/{start}_strain.csv'
    results_df = parallelStrainEnergy(args.file_name, args.njobs, args.output_file)
    summary_df = get_summary_stats(results_df)
    
    summary_output_file =f'{os.path.dirname(args.file_name)}/{start}_strain_summary.csv'
    summary_df.to_csv(summary_output_file, index=False)