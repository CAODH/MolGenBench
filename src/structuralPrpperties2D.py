import os
from typing import List
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed
from tqdm import tqdm
<<<<<<< HEAD




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
            


=======
import collections
from scipy import spatial as sci_spatial
import numpy as np
from EFGs import mol2frag
from rdkit import Chem
from collections import Counter
import pandas as pd
import json
def evalSubTypeDist(ref_groups,generate_group,ref_num_mols,generated_num_mols):
    """
    Calculate the JSD and MAE of subtypes of molecules.

    :param ref_groups: reference molecule subtypes distribution
    :param generate_group: generated molecule subtypes distribution
    :param ref_num_mols: number of reference molecules
    :param generated_num_mols: number of generated molecules
    
    
    """

    generated_type_ratio = {}
    ref_type_ratio = {}
    
    generated_type_dist = {}
    ref_type_dist = {}
    generated_total_num_types = sum(generate_group.values())
    ref_total_num_types = sum(ref_groups.values())
    for k in ref_groups:
        # ratio calculation
        if k in generate_group.keys():
            generated_type_ratio[k] = generate_group[k] / generated_num_mols
            generated_type_dist[k] = generate_group[k] / generated_total_num_types
        else:
            generated_type_ratio[k] = 0
            generated_type_dist[k] = 0
            
        ref_type_ratio[k] = ref_groups[k] / ref_num_mols
        # distributions calculation
        # generated_type_dist[k] = generate_func_group[k] / generated_total_num_types
        ref_type_dist[k] = ref_groups[k] / ref_total_num_types
        
    js = sci_spatial.distance.jensenshannon(np.array(list(ref_type_dist.values())),
                                            np.array(list(generated_type_dist.values())))
        
    mae = np.abs((np.array(list(ref_type_ratio.values())) - 
                    np.array(list(generated_type_ratio.values())))).mean()
    return mae, generated_type_ratio,js,generated_type_dist,ref_type_ratio,ref_type_dist


def getFuncGroup(mol):
    """"
    mol: rdkit mol object
    
    """
    try:
        fgs, _ = mol2frag(mol)
    except:
        return []
    return fgs
def getAtomType(mol):

    return [ atom.GetSymbol() for atom in mol.GetAtoms()] 
def getRingType(mol):
    ring_info = mol.GetRingInfo()
    ring_type = [len(r) for r in ring_info.AtomRings()]
    return ring_type
def getAllTypes(mol):
    return getFuncGroup(mol) , getAtomType(mol) , getRingType(mol)
def readMols(file_name):
    if file_name.endswith('.sdf'):
        return [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(file_name)]
    elif file_name.endswith('.smi') or file_name.endswith('.txt'):
        return [line.strip() for line in open(file_name)]
    else:
        raise ValueError('Invalid file format')

def parallelEvalSubTypeDist(ref_file,generated_file,output_file,njobs):
    # load actives and generated molecules      
    # actives_path = '/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_all_active_molecules.sdf'
    mols_actives = readMols(ref_file)
    # generated_path = '/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules.sdf'
    mols_generated = readMols(generated_file)
    actives_fg_stat =[]
    actives_atom_stat = []
    actives_ring_stat = []
    generated_fg_stat =[]
    generated_atom_stat = []
    generated_ring_stat = []
    actives_result_list = Parallel(n_jobs=njobs)(delayed(getAllTypes)(Chem.MolFromSmiles(mol)) for mol in mols_actives)

    generated_result_list = Parallel(n_jobs=njobs)(delayed(getAllTypes)(Chem.MolFromSmiles(mol)) for mol in mols_generated)
    for actives_result in actives_result_list:
        actives_fg_stat.extend(actives_result[0])
        actives_atom_stat.extend(actives_result[1])
        actives_ring_stat.extend(actives_result[2])
    for generated_result in generated_result_list:
        generated_fg_stat.extend(generated_result[0])
        generated_atom_stat.extend(generated_result[1])
        generated_ring_stat.extend(generated_result[2])
    # get the statistics of the functional groups
    actives_fg_stat = Counter(actives_fg_stat)
    generated_fg_stat = Counter(generated_fg_stat)
    # only keep the functional groups frequency more than 5
    actives_fg_stat = {k:v for k,v in actives_fg_stat.items() if v >= 5}
    generated_fg_stat = {k:v for k,v in generated_fg_stat.items() if v >= 5}
    # get the statistics of the atom types
    actives_atom_stat = Counter(actives_atom_stat)
    generated_atom_stat = Counter(generated_atom_stat)
    # only keep the functional groups frequency more than 5
    actives_atom_stat = {k: v for k, v in actives_atom_stat.items() if v >= 5}
    generated_atom_stat = {k: v for k, v in generated_atom_stat.items() if v >= 5}
    # get the statistics of the ring types
    actives_ring_stat = Counter(actives_ring_stat)
    generated_ring_stat = Counter(generated_ring_stat)
    ## # only keep the functional groups frequency more than 5
    actives_ring_stat = {k: v for k, v in actives_ring_stat.items() if v >= 5}
    generated_ring_stat = {k: v for k, v in generated_ring_stat.items() if v >= 5}
    # calculate the MAE, JS and ratio of the functional groups
    mae_fg, generated_fg_ratio, js_fg, generated_fg_dist ,ref_fg_ratio,ref_fg_dist= evalSubTypeDist(actives_fg_stat,generated_fg_stat,len(actives_result_list),len(generated_result_list))
    # calculate the MAE, JS and ratio of the atom types
    mae_atom, generated_atom_ratio, js_atom, generated_atom_dist ,ref_atom_ratio,ref_atom_dist= evalSubTypeDist(actives_atom_stat,generated_atom_stat,len(actives_result_list),len(generated_result_list))
    # calculate the MAE, JS and ratio of the ring types
    mae_ring, generated_ring_ratio, js_ring, generated_ring_dist,ref_ring_ratio,ref_ring_dist = evalSubTypeDist(actives_ring_stat,generated_ring_stat,len(actives_result_list),len(generated_result_list))
    return_result = {}
    # write the results to the output file
    return_result['Functional Group MAE'] = [mae_fg]
    return_result['Functional Group JS'] = [js_fg]
    return_result['Functional Group Ratio'] = [json.dumps(generated_fg_ratio)]
    return_result['Functional Group Distribution'] = [json.dumps(generated_fg_dist)]
    return_result['Functional Group Reference Ratio'] = [json.dumps(ref_fg_ratio)]
    return_result['Functional Group Reference Distribution'] =[ json.dumps(ref_fg_dist)]
    return_result['Atom Type MAE'] =[ mae_atom]
    return_result['Atom Type JS'] = [js_atom]
    return_result['Atom Type Ratio'] = [json.dumps(generated_atom_ratio)]
    return_result['Atom Type Distribution'] = [json.dumps(generated_atom_dist)]
    return_result['Atom Type Reference Ratio'] = [json.dumps(ref_atom_ratio)]
    return_result['Atom Type Reference Distribution'] = [json.dumps(ref_atom_dist)]
    return_result['Ring Type MAE'] = [mae_ring]
    return_result['Ring Type JS'] = [js_ring]
    return_result['Ring Type Ratio'] = [json.dumps(generated_ring_ratio)]
    return_result['Ring Type Distribution'] = [json.dumps(generated_ring_dist)]
    return_result['Ring Type Reference Ratio'] = [json.dumps(ref_ring_ratio )]
    return_result['Ring Type Reference Distribution'] =[ json.dumps(ref_ring_dist)]
    pd.DataFrame(return_result).to_csv(output_file,index = False)

>>>>>>> 2fb3ed6 (format output to csv)
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
        properties_str = {"smiles":smiles,"theavy-atom":num_non_hydrogen_atoms,"chiral-center":num_chiral_carbons,
                          "Rings":num_rings,"Aromatic-Rings":num_aromatic_rings,
                          "Rotatable-Bonds":num_rotatable_bonds,"Fsp3":fraction_csp3}
        
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

    smiles_list = readMols(file_name)

    results = Parallel(n_jobs=njobs)(delayed(structural_properties)(smiles) for smiles in tqdm(smiles_list,total = len(smiles_list)))
    # results = [result for result in results if result is not None]
    # print(results)
    results_dict = {"smiles":[],"theavy-atom":[],"chiral-center":[],
                          "Rings":[],"Aromatic-Rings":[],
                          "Rotatable-Bonds":[],"Fsp3":[]}
    for result in results:
        if result is not None:
            for key in results_dict.keys():
                results_dict[key].append(result[key])
    results = pd.DataFrame(results_dict)
    results.to_csv(output_file,index = False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate 2D structural properties of molecules')
    
    parser.add_argument('--ref_file_name', type=str,default='/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_result_from_PGMG.txt', help='File name of the SDF or .smi or .txt file')
    parser.add_argument('--generated_file_name', type=str,default='/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules.sdf', help='File name of the SDF or .smi or .txt file')
    
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    parser.add_argument('--file_name', type=str,default = '/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_result_from_PGMG.txt', help='File name of the SDF or .smi or .txt file')
    
    args = parser.parse_args()
    import time 
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.file_name)}/{start}_2D_properties_base.csv'
    parallelStructuralProperties(args.ref_file_name, args.njobs,args.output_file)
    args.output_file =f'{os.path.dirname(args.file_name)}/{start}_2D_properties_JSD_MAE.csv'
    parallelEvalSubTypeDist(args.ref_file_name,args.generated_file_name,args.output_file,args.njobs)
    
    
    
    

    
    
    
    