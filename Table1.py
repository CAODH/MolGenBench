import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
from joblib import Parallel, delayed

from rdkit import Chem

from src.utils.scoring_func import general_properties
from src.utils.Morgan_Tdiv import calculate_Morgan_fingerprint_div_single_target

def readMols(file_name):
    if file_name.endswith('.sdf'):
        return [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(file_name)]
    elif file_name.endswith('.smi') or file_name.endswith('.txt'):
        return [line.strip() for line in open(file_name)]
    else:
        raise ValueError('Invalid file format')


def parallelGeneralProperties(file_name, njobs, output_file):
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    smiles_list = readMols(file_name)
    results = Parallel(n_jobs=njobs)(delayed(general_properties)(smiles) for smiles in tqdm(smiles_list,total = len(smiles_list)))
    
    results_dict = {
        'smiles': [],
        'validity': [],
        'qed': [],
        'sa': [],
        'logp': [],
        'lipinski': [],
    }
    
    for result in results:
        if result is not None:
            for key in results_dict.keys():
                results_dict[key].append(result[key])

    results = pd.DataFrame(results_dict)
    results.to_csv(output_file, index = False)
    
    return results
    
def get_summary_stats(results_df):
    summary_dict = {
        'validity': results_df["validity"].sum() / 1000,
        'qed': results_df["qed"].mean(),
        'sa': results_df["sa"].mean(),
        'logp': results_df["logp"].mean(),
        'lipinski': results_df["lipinski"].mean(),
    }
    valid_smiles = results_df[results_df["validity"]==True]["smiles"]
    summary_dict["uniqueness"] = len(valid_smiles.unique()) / len(valid_smiles)
    summary_dict["diversity"] = calculate_Morgan_fingerprint_div_single_target(valid_smiles)
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['value']).transpose()
    
    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate 2D structural properties of molecules')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    parser.add_argument('--file_name', type=str,default = '/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_result_from_PGMG.txt', help='File name of the SDF or .smi or .txt file')
    
    args = parser.parse_args()
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.file_name)}/{start}_2D_general_properties_Table-1.csv'
    results_df = parallelGeneralProperties(args.file_name, args.njobs, args.output_file)
    summary_df = get_summary_stats(results_df)
    
    summary_output_file =f'{os.path.dirname(args.file_name)}/{start}_2D_general_properties_Table-1_summary.csv'
    summary_df.to_csv(summary_output_file, index=False)