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
## ignore all rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
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
    except:
        return set()
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
    if not os.path.exists(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl'):
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_smiles.update({key:ref_temp for key in ref_inchi_set})
            
        with open(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_smiles,f)
    else:
        with open(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','rb') as f :
            ref_inchi_map_ref_smiles = pickle.load(f)
    intersections_inchi = set(ref_inchi_map_ref_smiles.keys()).intersection(query_inchi_set)
    interactions_smiles = set([ref_inchi_map_ref_smiles[inchi] for inchi in intersections_inchi])
    return list(interactions_smiles),list(intersections_inchi)
def find_scaffold_and_inchi(ref_smiles,gen_smiles,uniprot_id,generated_dir):

    query_inchi_set = set([Chem.MolToInchi(Chem.MolFromSmiles(temp)) for temp in gen_smiles if Chem.MolFromSmiles(temp) is not None])
    ref_inchi_map_ref_scaffold = {}

    if not os.path.exists(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl'):
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_scaffold.update({key:ref_temp for key in ref_inchi_set})
            
        with open(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_scaffold,f)
    else:
        with open(f'{generated_dir}/{uniprot_id}/reference_active_molecules/inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','rb') as f :
            ref_inchi_map_ref_scaffold = pickle.load(f)
    intersections_inchi = set(ref_inchi_map_ref_scaffold.keys()).intersection(query_inchi_set)
    interactions_smiles = set([ref_inchi_map_ref_scaffold[inchi] for inchi in intersections_inchi])
    return list(interactions_smiles),list(intersections_inchi)

# get reference smiles (actives compounds)
def FixStereoFrom3D(mol):
    Chem.RemoveStereochemistry(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol
from tqdm import tqdm
import glob
################################################################################################################################
def compute_hit_info(generated_dir,round_id_list,model_name_list,save_dir):
    """
    Compute hit information for de novo generated molecules and save per-model CSV reports.
    This function scans a directory of generated and reference molecules organized by Uniprot ID,
    collects reference and generated SMILES, identifies exact SMILES and scaffold "hits" between
    reference and generated sets, computes various counts and rates, and writes a CSV file for each
    (model, round) processed.
    Parameters
    ----------
    generated_dir : str
        Root directory containing per-Uniprot subdirectories. Expected layout (examples):
          {generated_dir}/{uniprot_id}/reference_active_molecules/{uniprot_id}_reference_active_molecules.sdf
          {generated_dir}/{uniprot_id}/Round{round_id}/De_novo_Results/{model_name}_generated_molecules/{uniprot_id}_{model_name}_generated_molecules.sdf
    round_id_list : Iterable
        Iterable of round identifiers (e.g., ints or strings). For each round, a subfolder Round{round_id}
        is inspected and a Hit_Info_Results folder is created to store results.
    model_name_list : Iterable[str]
        Iterable of model names to analyze. For each model a CSV named "{model_name}.csv" is written
        into the round's Hit_Info_Results directory. If that CSV already exists, processing for that
        model and round is skipped.
    Behavior and outputs
    --------------------
    - For each round and model, iterates over every Uniprot ID directory found directly under generated_dir.
    - Reads the reference SDF and the model-generated SDF for that Uniprot ID (if present).
    - For each molecule read from SDFs:
      - Skips None entries (malformed entries are reported via printed messages).
      - Keeps the largest fragment (by atom count) when multi-fragment molecules are present.
      - Converts molecules to SMILES using RDKit's MolToSmiles.
      - For generated molecules (except for models 'TamGen' and 'PGMG') attempts to fix stereochemistry via FixStereoFrom3D.
    - Deduplicates SMILES lists (uses set) and aggregates per-Uniprot data into a pandas DataFrame with columns including:
      - UniprotID, Reference_Smiles, Generated_Smiles, Reference_Smiles_num
      - Finded_Smiles_and_Inchi, Finded_Smiles, Finded_Smiles_Num, Finded_Inchi, Finded_Inchi_Num
      - Reference_Scaffolds, Reference_Scaffolds_Num, Generated_Scaffolds, Generated_Scaffolds_Num
      - Finded_Scaffolds_and_Inchi, Finded_Scaffolds, Finded_Scaffolds_Num, Finded_Scaffolds_Inchi, Finded_Scaffolds_Inchi_Num
      - Finded_Scaffolds_Frequency_Rate, Finded_Scaffolds_Rate
    - Uses helper functions find_smiles_and_inchi and find_scaffold_and_inchi to identify matching SMILES/scaffolds
      and to obtain associated InChI information.
    - Saves the resulting DataFrame as CSV to:
      {generated_dir}/Round{round_id}/De_novo_Results/Hit_Info_Results/{model_name}.csv
      The directory is created if it does not exist.
    - Prints progress messages and warnings to stdout. If an unexpected exception occurs while processing a model,
      the error is printed and processing continues with the next model.
    Return
    ------
    None
        Results are written to disk; the function does not return a value.
    Dependencies and assumptions
    ----------------------------
    - Expects the following to be available in the module/global scope:
      RDKit (Chem), pandas, swifter, tqdm, os, and helper functions:
      FixStereoFrom3D, GetScaffold, find_smiles_and_inchi, find_scaffold_and_inchi.
    - Input SDF filenames and directory layout must follow the conventions described above.
    - Malformed or missing SDF files are skipped and reported; such cases will reduce the number of rows for that model/round.
    - For large datasets this function can be I/O-bound and memory intensive. swifter.apply is used for some row-wise
      operations and may attempt parallel execution depending on environment.
    Error handling and logging
    --------------------------
    - Printing is used to report:
      - Missing generated molecule files for a given Uniprot ID and model.
      - Malformed molecules encountered while reading SDFs.
      - General exceptions encountered while processing a model (processing continues to next model/round).
    """

    for round_id in round_id_list:
        save_dir = os.path.join(save_dir,f'Round{round_id}')

        for model_name in model_name_list:
            try:
                if os.path.exists(os.path.join(save_dir,f'{model_name}.csv')):
                    print(f'{model_name} has been analyzed')
                    continue
                    
                uniprot_id_list = []
                uniprot_ref_smiles_list = []
                uniprot_gen_smiles_list = []
                
                for uniprot_id in tqdm(os.listdir(generated_dir),total = len(os.listdir(generated_dir))):
                    
                    ref_actives_path = os.path.join(generated_dir,uniprot_id,f'reference_active_molecules/{uniprot_id}_reference_active_molecules.sdf')
                    

                    generate_mol_path = f"{generated_dir}/{uniprot_id}/Round{round_id}/De_novo_Results/{model_name}_generated_molecules/{uniprot_id}_{model_name}_generated_molecules.sdf"
                    if not os.path.exists(generate_mol_path):
                        print(f'No generated molecules for {uniprot_id} in {model_name}')
                        continue

                    
                    uniprot_ref_smiles = []
                    ref_mols = Chem.SDMolSupplier(ref_actives_path)
                    for ref_mol in ref_mols:
                        if ref_mol is None:
                            print('erro mol in : ',ref_actives_path)
                            continue
                        frags = Chem.GetMolFrags(ref_mol, asMols=True)
                        ref_mol = max(frags, key=lambda x: x.GetNumAtoms())
                        uniprot_ref_smiles.append(Chem.MolToSmiles(ref_mol))
                    
                    ############################# generated molecules ########################################################
                    uniprot_gen_smiles = []
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
                    uniprot_ref_smiles_list.append(list(set(uniprot_ref_smiles)))
                    uniprot_id_list.append(uniprot_id)
                    
                result = pd.DataFrame({'UniprotID':uniprot_id_list ,
                        'Reference_Smiles':uniprot_ref_smiles_list ,
                        'Generated_Smiles':uniprot_gen_smiles_list})
                result['Reference_Smiles_num']=result['Reference_Smiles'].apply(len)
                

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

                result['Finded_Scaffolds_Num'] = result['Finded_Scaffolds'].apply(len)
                result['Finded_Scaffolds_Inchi'] = result['Finded_Scaffolds_and_Inchi'].apply(lambda x: x[1])
                result['Finded_Scaffolds_Inchi_Num'] = result['Finded_Scaffolds_Inchi'].apply(len)

                result['Finded_Scaffolds_Frequency_Rate'] =result.apply(lambda x:x.Finded_Scaffolds_Inchi_Num/len(x.Generated_Smiles),axis=1)
                result['Finded_Scaffolds_Rate'] = result['Finded_Scaffolds_Num']/result['Reference_Scaffolds_Num']

                # save result csv file
                os.makedirs(save_dir,exist_ok=True)
                result.to_csv(os.path.join(save_dir,f'{model_name}.csv'))
                
            except Exception as e:
                print(f'Error in {model_name}')
                print('Error:',str(e))
                continue

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='PreProcess the hit information of generated molecules')
    parser.add_argument('--generated_dir', type=str, help='The directory of active compounds', required=True)
    parser.add_argument('--save_dir', type=str, help='The directory of results', required=True)
    parser.add_argument('--round_id_list', type=str, nargs='*', default=[1], help='The list of uniprot ids to process; if not given, process all')
    parser.add_argument('--model_name_list', type=str, nargs='*', default=['PocketFlow'], help='The list of model names to process; if not given, process all')

    args = parser.parse_args(argv)
    generated_dir = args.generated_dir
    model_name_list = args.model_name_list
    round_id_list = args.round_id_list
    save_dir = args.save_dir
    
    # 

    compute_hit_info(generated_dir,round_id_list, model_name_list,save_dir )
    



if __name__ == "__main__":
    main()
       