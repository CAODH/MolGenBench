import rdkit
from rdkit.Chem import AllChem
import pandas as pd
import os
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import Chem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
import pickle

import swifter

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
def enumerate_tautomer_and_partial_chirality(smiles):
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
from rdkit.Chem.Scaffolds import MurckoScaffold
def GetScaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # 计算分子骨架
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # 输出分子骨架的 SMILES
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except:
        return 'None'
def enumerate_scaffold_partial_chirality_rdkit(smiles):
    # Generate scaffold from the input SMILES
    scaffold = GetScaffold(smiles)
    
    # Enumerate tautomers and partial chirality for the scaffold
    enumerated_inchi = enumerate_tautomer_and_partial_chirality(scaffold)
    
    return enumerated_inchi  
def find_smiles_and_inchi(ref_smiles,gen_smiles,uniprot_id,generated_dir):
    query_inchi_set = set([Chem.MolToInchi(Chem.MolFromSmiles(temp)) for temp in gen_smiles if Chem.MolFromSmiles(temp) is not None])
    ref_inchi_map_ref_smiles = {}
    # {generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl
    if not os.path.exists(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl'):
        os.makedirs(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/',exist_ok=True)
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_smiles.update({key:ref_temp for key in ref_inchi_set})
        with open(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_smiles,f)
    else:
        with open(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','rb') as f :
            ref_inchi_map_ref_smiles = pickle.load(f)
    intersections_inchi = set(ref_inchi_map_ref_smiles.keys()).intersection(query_inchi_set)
    interactions_smiles = set([ref_inchi_map_ref_smiles[inchi] for inchi in intersections_inchi])
    return list(interactions_smiles),list(intersections_inchi)
def find_scaffold_and_inchi(ref_smiles,gen_smiles,uniprot_id,generated_dir):

    query_inchi_set = set([Chem.MolToInchi(Chem.MolFromSmiles(temp)) for temp in gen_smiles if Chem.MolFromSmiles(temp) is not None])
    ref_inchi_map_ref_scaffold = {}

    if not os.path.exists(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl'):
        os.makedirs(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/',exist_ok=True)
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_scaffold.update({key:ref_temp for key in ref_inchi_set})
            
        with open(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_scaffold,f)
    else:
        with open(f'{generated_dir}/{uniprot_id.split("_")[0]}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','rb') as f :
            ref_inchi_map_ref_scaffold = pickle.load(f)
    intersections_inchi = set(ref_inchi_map_ref_scaffold.keys()).intersection(query_inchi_set)
    interactions_smiles = set([ref_inchi_map_ref_scaffold[inchi] for inchi in intersections_inchi])
    return list(interactions_smiles),list(intersections_inchi)
def FixStereoFrom3D(mol):
    Chem.RemoveStereochemistry(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol
# get reference smiles (actives compounds)
from tqdm import tqdm
import glob

from collections import defaultdict
def compute_hit_info_h2l(generated_dir,round_id_list,model_name_list):
    """
    Compute hit information for hit-to-lead generated molecules and save per-model CSV reports.
    This function scans a directory of generated and reference molecules organized by Uniprot ID,
    collects reference and generated SMILES, identifies exact SMILES and scaffold "hits" between
    reference and generated sets, and saves the results to CSV files.
    Args:
        generated_dir (str): Path to the directory containing generated and reference molecules.
        round_id_list (list): List of round IDs to process.
        model_name_list (list): List of model names to process.
    Saves:
        Per-model CSV files summarizing hit information, including:
        - Number of reference SMILES.
        - Number of generated SMILES.   
        - Number of exact SMILES hits and their InChI representations.
        - Number of scaffold hits and their InChI representations.
    Handles exceptions such as:
        - Missing reference or generated molecule files.
        - Malformed molecules encountered while reading SDFs.
        - General exceptions encountered while processing a model (processing continues to next model/round).
    """

    for round_id in round_id_list:
        
        save_dir = os.path.join(os.path.dirname(generated_dir),'TestSample_H2L',f'Round{round_id}','Hit_to_Lead_Results','Hit_Info_Results')
        for model_name in model_name_list:
            # try:
                if os.path.exists(os.path.join(save_dir,f'{model_name}.csv')):
                    continue
                    
                uniprot_id_list = []
                uniprot_ref_smiles_list = []
                uniprot_gen_smiles_list = []
                
                for uniprot_id in tqdm(os.listdir(generated_dir),total = len(os.listdir(generated_dir))):
                    
                    ref_actives_path = os.path.join(generated_dir,uniprot_id,f'reference_active_molecules/{uniprot_id}_reference_active_molecules.sdf')
                    
                    serise_ids = os.listdir(os.path.join(generated_dir,uniprot_id,f'Round{round_id}','Hit_to_Lead_Results'))
                    if len(serise_ids) == 0:
                        print(f'No generated molecules for {uniprot_id} in {model_name}')
                        continue
                    uniprot_ref_smiles_map =  defaultdict(list)
                    ref_mols = Chem.SDMolSupplier(ref_actives_path)
                    for ref_mol in ref_mols:
                        if ref_mol is None:
                            print('erro mol in : ',ref_actives_path)
                            continue
                        mol_name = ref_mol.GetProp('_Name')
                        mol_serise_id = '_'.join(mol_name.split('_')[:2])
                        frags = Chem.GetMolFrags(ref_mol, asMols=True)
                        ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
                        uniprot_ref_smiles_map[mol_serise_id].append(Chem.MolToSmiles(ref_mol))
                        
                    
                    for serise_id in serise_ids:
                        
                        uniprot_gen_smiles = []
                        generate_mol_path = os.path.join(generated_dir,uniprot_id,f'Round{round_id}','Hit_to_Lead_Results',serise_id,f'{model_name}_Hit_to_Lead',f'{uniprot_id}_{serise_id}_{model_name}_Hit_to_Lead.sdf')
            

                        if not os.path.exists(generate_mol_path):
                            print(f'No generated molecules for {uniprot_id} in {model_name},{generate_mol_path}')
                            continue
                        ############################# generated molecules ########################################################
                        
                        try:
                            generate_mols = Chem.SDMolSupplier(generate_mol_path)
                        except:
                            print('Error in ',generate_mol_path)
                            continue
                        for generate_mol in generate_mols:
                            if generate_mol is None:
                                print('erro mol in : ',generate_mol_path)
                                continue
                            frags = Chem.GetMolFrags(generate_mol, asMols=True)
                            generate_mol = max(frags, key=lambda x: x.GetNumAtoms())
                            if model_name not in ['TamGen','PGMG']:
                                generate_mol = FixStereoFrom3D(generate_mol)
                            uniprot_gen_smiles.append(Chem.MolToSmiles(generate_mol))
                            
                        if len(uniprot_gen_smiles) == 0:
                            print(f'No generated molecules for {uniprot_id} in {model_name}')
                            continue
                        uniprot_gen_smiles_list.append(list(set(uniprot_gen_smiles)))
                        uniprot_ref_smiles_list.append(list(set(uniprot_ref_smiles_map[uniprot_id+'_'+serise_id])))
                        
                        uniprot_id_list.append(uniprot_id+'_'+serise_id)
                    
                result = pd.DataFrame({'UniprotID':uniprot_id_list ,
                        'Reference_Smiles':uniprot_ref_smiles_list ,
                        'Generated_Smiles':uniprot_gen_smiles_list})
                
                result['Reference_Smiles_num']=result['Reference_Smiles'].apply(len)
                
                # result['Finded_Smiles'] = result.apply(lambda x:list(set(x.Reference_Smiles).intersection(set(x.Generated_Smiles))),axis = 1)
                result['Finded_Smiles_and_Inchi'] = result.swifter.apply(lambda x:find_smiles_and_inchi(set(x.Reference_Smiles),set(x.Generated_Smiles),x.UniprotID,generated_dir),axis = 1)
                result['Finded_Smiles'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: x[0])
                result['Finded_Smiles_Num']=result['Finded_Smiles'].apply(len)
                result['Finded_Inchi'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: x[1])
                result['Finded_Inchi_Num'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: len(x))

                result['Reference_Scaffolds'] = result['Reference_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
                result['Reference_Scaffolds'] = result['Reference_Scaffolds'].apply(lambda x: [_ for _ in x if _ != 'None'])
                result['Reference_Scaffolds_Num'] = result['Reference_Scaffolds'].apply(len)
                
                result['Generated_Scaffolds'] = result['Generated_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
                result['Generated_Scaffolds'] = result['Generated_Scaffolds'].apply(lambda x: [_ for _ in x if _ != 'None'])
                result['Generated_Scaffolds_Num'] = result['Generated_Scaffolds'].apply(len)
                
                
                result['Finded_Scaffolds_and_Inchi'] = result.swifter.apply(lambda x:find_scaffold_and_inchi(set(x.Reference_Scaffolds),set(x.Generated_Scaffolds),x.UniprotID,generated_dir),axis = 1)
                result['Finded_Scaffolds'] = result['Finded_Scaffolds_and_Inchi'].apply(lambda x: x[0])
                # result['Finded_Scaffolds'] = result.apply(lambda x:list(set(x.Reference_Scaffolds).intersection(set(x.Generated_Scaffolds))),axis = 1)
                result['Finded_Scaffolds_Num'] = result['Finded_Scaffolds'].apply(len)
                result['Finded_Scaffolds_Inchi'] = result['Finded_Scaffolds_and_Inchi'].apply(lambda x: x[1])
                result['Finded_Scaffolds_Inchi_Num'] = result['Finded_Scaffolds_Inchi'].apply(len)
                # 计算hit 到的骨架在所有分子里面占到的比例
                result['Finded_Scaffolds_Frequency_Rate'] =result.apply(lambda x:x.Finded_Scaffolds_Inchi_Num/len(x.Generated_Smiles),axis=1)
                result['Finded_Scaffolds_Rate'] = result['Finded_Scaffolds_Num']/result['Reference_Scaffolds_Num']
                
                # print('Sucessed Find at least one Scaffold',len(result[result['Finded_Scaffolds_Num'] >0]))

                # save result csv file
                os.makedirs(save_dir,exist_ok=True)
                result.to_csv(os.path.join(save_dir,f'{model_name}.csv'))
            # except Exception as e:
            #     print(f'Error in {model_name}')
            #     print('Error:',str(e))
            #     continue
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='PreProcess the hit information of generated molecules')
    parser.add_argument('--generated_dir', type=str, help='The directory of active compounds', required=True)
    parser.add_argument('--round_id_list', type=str, nargs='*', default=None, help='The list of uniprot ids to process; if not given, process all')
    parser.add_argument('--model_name_list', type=str, nargs='*', default=['DecompDiff','diffSBDD_cond_crossdocked','diffSBDD_cond_moad','FLAG','MolCraft','PocketFlow','TamGen','Pocket2Mol','SurfGen','TargetDiff'], help='The list of model names to process; if not given, process all')

    args = parser.parse_args(argv)
    generated_dir = args.generated_dir
    model_name_list = args.model_name_list
    round_id_list = args.round_id_list
    

    compute_hit_info_h2l(generated_dir,round_id_list, model_name_list)
    



if __name__ == "__main__":
    main()