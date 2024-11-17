# We implement the rmsd calculation in this file.
# spyrmsd is used for the calculation of RMSD.
# conda install -c conda-forge spyrmsd


import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdMolAlign
from spyrmsd import molecule
from spyrmsd import rmsd as spy_rmsd

from src.utils.cacl_SC_RDKit import calc_SC_RDKit_score

def calc_sc_rdkit_full_mol(idx, gen_mol, ref_mol):
    sc_score = np.nan
    try:
        gen_mol = Chem.AddHs(gen_mol)
        ref_mol = Chem.AddHs(ref_mol)
        _ = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        sc_score = calc_SC_RDKit_score(gen_mol, ref_mol)
    except:
        sc_score = np.nan
        
    return {
        'idx': idx,
        'sc_rdkit': sc_score
    }


def get_symmetry_rmsd(idx, ref, mol):
    # with time_limit(10):
    rmsd_val = np.nan
    try:
        mol = molecule.Molecule.from_rdkit(mol)
        ref = molecule.Molecule.from_rdkit(ref)
        coords_ref = ref.coordinates
        anum_ref = ref.atomicnums
        adj_ref = ref.adjacency_matrix
        coords = mol.coordinates
        anum = mol.atomicnums
        adj = mol.adjacency_matrix
        rmsd_val = spy_rmsd.symmrmsd(
            coords_ref,
            coords,
            anum_ref,
            anum,
            adj_ref,
            adj,
        )
    except Exception as e:
        print(f"[RMSD] Error processing molecule: {e}")
    
    return {
        'idx': idx,
        'spyrmsd': rmsd_val
    }
    

def pair_molecules(generated_file, docked_file):
    """
    Pair molecules from generated and docked files based on their '_Name' property.

    :param generated_file: Path to the SDF file containing generated molecules.
    :param docked_file: Path to the SDF file containing docked molecules.
    :return: A dictionary where keys are molecule indices (idx) and values are tuples (gen_mol, docked_mol).
    """
    
    gen_suppl = Chem.SDMolSupplier(generated_file)
    docked_suppl = Chem.SDMolSupplier(docked_file)
    
    gen_dict = {mol.GetProp("_Name"): mol for mol in gen_suppl if mol is not None}
    
    paired_dict = {idx: (gen_mol, None) for idx, gen_mol in gen_dict.items()}


    for mol in docked_suppl:
        if mol is not None:
            idx = mol.GetProp("_Name")
            if idx in paired_dict:
                paired_dict[idx] = (paired_dict[idx][0], mol)
    
    return paired_dict
    
    
def parallelRMSD(generated_file, docked_file, njobs, output_file):
    """
    Calculate the strain energy of molecules in parallel.

    :param generated_file: File name of the model generated SDF
    :param docked_file: File name of the docked SDF
    :param njobs: Number of processes to use
    :param output_file: Output file path
    """
    
    paired_dict = pair_molecules(generated_file, docked_file)
    
    results = Parallel(n_jobs=njobs)(delayed(get_symmetry_rmsd)(k, v[0], v[1]) for k, v in tqdm(paired_dict.items(), total=len(paired_dict)))
    
    results_dict = {
        'idx': [],
        'spyrmsd': []
    }

    for result in results:
        if result is not None:
            results_dict['idx'].append(result['idx'])
            results_dict['spyrmsd'].append(result['spyrmsd'])

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_file, index=False)
    
    return results_df


def parallelSC_RDkit(input_file, reference_file, njobs, output_file):
    
    ref_mol = Chem.SDMolSupplier(reference_file)[0]    
    input_suppl = Chem.SDMolSupplier(input_file)
    paired_list = [(mol.GetProp('_Name'), mol, ref_mol) for mol in input_suppl if mol is not None]
    
    results = Parallel(n_jobs=njobs)(delayed(calc_sc_rdkit_full_mol)(item[0], item[1], item[2]) for item in tqdm(paired_list, total=len(paired_list)))
    
    results_dict = {
        'idx': [],
        'sc_rdkit': []
    }
    
    for result in results:
        if results is not None:
            results_dict['idx'].append(result['idx'])
            results_dict['sc_rdkit'].append(result['sc_rdkit'])
                                     
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_file, index=False)
    
    return results_df
    
def get_rmsd_summary_stats(results_df):
    
    rmsd_list = results_df["spyrmsd"]
    isnan = np.isnan(rmsd_list)
    n_isnan = isnan.sum()
    n_total = len(rmsd_list)
    
    summary_dict = {
        'rmsd_fail': n_isnan / n_total,
        'rmsd_mean': np.mean(rmsd_list[~isnan]),
        'rmsd_<2A': np.mean(rmsd_list[~isnan] < 2), # calculate the percentage of RMSD < 2A
    }
    
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['value']).transpose()
    
    return summary_df


def get_sc_rdkit_summary_stats(results_df):

    sc_rdkit_list = results_df["sc_rdkit"]
    isnan = np.isnan(sc_rdkit_list)
    n_isnan = isnan.sum()
    n_total = len(sc_rdkit_list)

    summary_dict = {
        'sc_rdkit_fail': n_isnan / n_total, # calculate the percentage of failed cases
        'sc_rdkit_0.7': np.mean(sc_rdkit_list[~isnan] > 0.7), # calculate the percentage of SC_RDKIT > 0.7
        'sc_rdkit_0.8': np.mean(sc_rdkit_list[~isnan] > 0.8), # calculate the percentage of SC_RDKIT > 0.8
        'sc_rdkit_0.9': np.mean(sc_rdkit_list[~isnan] > 0.9), # calculate the percentage of SC_RDKIT > 0.9
    }
    
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['value']).transpose()
    
    return summary_df
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RMSD')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    parser.add_argument('--generated_file', type=str,default = '/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules.sdf', help='File name of the SDF that contains generated 3D molecules')
    parser.add_argument('--docked_file', type=str, default = '/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules_ligprep_glide_sp_pv.sdf', help='File name of the SDF that contains docked 3D molecules')
    parser.add_argument('--ref_file', type=str, default='/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_lig.sdf', help='File name of the reference positive 3D molecules')
    
    args = parser.parse_args()
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.generated_file)}/{start}_rmsd.csv'
    rmsd_results_df = parallelRMSD(args.generated_file, args.docked_file, args.njobs, args.output_file)
    rmsd_summary_df = get_rmsd_summary_stats(rmsd_results_df)
    rmsd_summary_output_file =f'{os.path.dirname(args.generated_file)}/{start}_rmsd_summary.csv'
    rmsd_summary_df.to_csv(rmsd_summary_output_file, index=False)
    
    gen_sc_rdkit_df = parallelSC_RDkit(args.generated_file, args.ref_file, args.njobs, f'{os.path.dirname(args.generated_file)}/{start}_sc_rdkit_gen.csv')
    gen_sc_rdkit_summary_df = get_sc_rdkit_summary_stats(gen_sc_rdkit_df)
    gen_sc_rdkit_summary_output_file = f'{os.path.dirname(args.generated_file)}/{start}_sc_rdkit_summary_gen.csv'
    gen_sc_rdkit_summary_df.to_csv(gen_sc_rdkit_summary_output_file, index=False)
    
    docked_sc_rdkit_df = parallelSC_RDkit(args.docked_file, args.ref_file, args.njobs, f'{os.path.dirname(args.generated_file)}/{start}_sc_rdkit_docked.csv')
    docked_sc_rdkit_summary_df = get_sc_rdkit_summary_stats(docked_sc_rdkit_df)
    docked_sc_rdkit_summary_output_file = f'{os.path.dirname(args.generated_file)}/{start}_sc_rdkit_summary_docked.csv'
    docked_sc_rdkit_summary_df.to_csv(docked_sc_rdkit_summary_output_file, index=False)
