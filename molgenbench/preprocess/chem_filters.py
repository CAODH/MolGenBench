# from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import PandasTools
import datamol as dm
import pandas as pd
import medchem as mc
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import swifter
from medchem.structural.lilly_demerits import LillyDemeritsFilters
dfilter = LillyDemeritsFilters()
base_alerts = mc.structural.CommonAlertsFilters()
nibr_filters = mc.structural.NIBRFilters()

def lilly_medchem_rules_old(smiles):
    """
    Apply Lilly MedChem rules to a SMILES string.
    Returns True if the SMILES passes all rules, otherwise returns False.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return False

    else:
        try:
            value = mc.functional.lilly_demerit_filter(
                mols=[smiles],
                n_jobs=1,
                progress=True,
                return_idx=False
            )
            # print(f"Processing {smiles}: {value}")
            return value[0]
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return False
def get_smiles_name_from_sdf(sdf_file):
    """
    Read molecules from an SDF file and return a DataFrame with SMILES and names.
    """
    try:
        supplier = Chem.SDMolSupplier(sdf_file)
    except:
        print(f"Error reading SDF file: {sdf_file}")
        return pd.DataFrame({"smiles": [], "Name": []})
    mols = [mol for mol in supplier if mol is not None]
    smiles_list = [Chem.MolToSmiles(mol) for mol in mols if Chem.MolToSmiles(mol) is not None]
    mol_names = [mol.GetProp('_Name') for mol in mols if mol is not None and Chem.MolToSmiles(mol) is not None]
    return pd.DataFrame({"smiles": smiles_list, "Name": mol_names})
def lilly_medchem_rules(smiles):
    """
    Apply Lilly MedChem rules to a SMILES string.
    Returns True if the SMILES passes all rules, otherwise returns False.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return 'Error',  False

    else:
        try:
            result = dfilter(
            mols=[smiles],
            n_jobs=1,
            progress=True )
            return result['reasons'].values[0], result['pass_filter'].values[0]
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return 'Error',  False
def base_alters_rules(smiles):
    """
    Apply Lilly MedChem rules to a SMILES string.
    Returns True if the SMILES passes all rules, otherwise returns False.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return 'Error',  False

    else:
        try:
            result = base_alerts(
            mols=[smiles],
            n_jobs=1,
            progress=True )
            return result['reasons'].values[0], result['pass_filter'].values[0]
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return 'Error',  False
def nibr_filters_rules(smiles):
    """
    Apply Lilly MedChem rules to a SMILES string.
    Returns True if the SMILES passes all rules, otherwise returns False.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return 'Error',  False

    else:
        try:
            result = nibr_filters(
            mols=[smiles],
            n_jobs=1,
            progress=True )
            return result['reasons'].values[0], result['pass_filter'].values[0]
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return 'Error',  False
def get_medchem_filter_info(generate_sdf):
    """
    Generate a CSV file with SMILES and names from an SDF file.
    """
    # data = get_smiles_name_from_sdf(generate_sdf)
    if not os.path.exists(generate_sdf):
        print(f"The file {generate_sdf} does not exist.")
        return 
    save_dir = os.path.dirname(generate_sdf)
    base_name = os.path.basename(generate_sdf).split('.sdf')[0]
    if os.path.exists(os.path.join(save_dir,f'{base_name}_medchem_filter_results.csv')):
        print(f"The file {os.path.join(save_dir,f'{base_name}_medchem_filter_results.csv')} already exists.")
        print("Skiping this file.")
        # os.remove(os.path.join(save_dir,f'{base_name}_medchem_filter_results.csv'))
        return
    
    data = get_smiles_name_from_sdf(generate_sdf)
    if len(data) == 0:
        print(f"No valid SMILES found in {generate_sdf}.")
        return
    data["rule_of_five_beyond"] = data["smiles"].apply(mc.rules.basic_rules.rule_of_five_beyond)
    final_result = data.copy()

    for alerts_name in [ 'dfilter','base_alerts', 'nibr_filters']:

        if alerts_name =='dfilter':
            final_result[['reasons_'+ alerts_name,'pass_filter_'+ alerts_name]] = final_result['smiles'].apply(lilly_medchem_rules).apply(pd.Series)
        elif alerts_name == 'base_alerts':
            final_result[['reasons_'+ alerts_name,'pass_filter_'+ alerts_name]] = final_result['smiles'].apply(base_alters_rules).apply(pd.Series)
        elif alerts_name == 'nibr_filters':
            final_result[['reasons_'+ alerts_name,'pass_filter_'+ alerts_name]] = final_result['smiles'].apply(nibr_filters_rules).apply(pd.Series)
    final_result.to_csv(os.path.join(save_dir,f'{base_name}_medchem_filter_results.csv'), index=False)

def main():
    import random
    random.seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Process SDF files for MedChem filter analysis.")
    # add a num to control the number of SDF files to process
    parser.add_argument('--father_dir', type=int, default=1, help='Number of SDF files to process.')
    parser.add_argument('--num_start', type=int, default=0, help='Number of SDF files to process.')
    parser.add_argument('--num_end', type=int, default=-1, help='Number of SDF files to process.')
    
    args = parser.parse_args()
    father_dir = args.father_dir

    all_generated_sdf = []
    for uniprot_id in os.listdir(father_dir):
        if os.path.isdir(os.path.join(father_dir, uniprot_id)):
            
            for sub_dir_name in os.listdir(os.path.join(father_dir, uniprot_id)):
                if sub_dir_name == 'reference_active_molecules':
                    all_generated_sdf.append(os.path.join(father_dir,uniprot_id,sub_dir_name,f'{uniprot_id}_reference_active_molecules.sdf'))
                if 'Round' in sub_dir_name:
                    for method in os.listdir(os.path.join(father_dir, uniprot_id, sub_dir_name, 'De_novo_Results')):
                            all_generated_sdf.append(os.path.join(father_dir, uniprot_id, sub_dir_name, 'De_novo_Results', method, f"{uniprot_id}_{method}.sdf"))
                    for serise_id in os.listdir(os.path.join(father_dir, uniprot_id, sub_dir_name, 'Hit_to_Lead_Results')):
                        for method in os.listdir(os.path.join(father_dir, uniprot_id, sub_dir_name, 'Hit_to_Lead_Results', serise_id)):
                            all_generated_sdf.append(os.path.join(father_dir, uniprot_id, sub_dir_name, 'Hit_to_Lead_Results',serise_id, method, f"{uniprot_id}_{serise_id}_{method}.sdf"))


    # sort all_generated_sdf by name
    all_generated_sdf = [sdf for sdf in all_generated_sdf if os.path.exists(sdf) and not os.path.exists(sdf.replace('.sdf', '_medchem_filter_results.csv'))]
    all_generated_sdf.sort(key=lambda x: x)
    
    for sdf in tqdm(all_generated_sdf[args.num_start:args.num_end], desc="Processing SDF files"):
        try:
            get_medchem_filter_info(sdf)
        except Exception as e:
            print(f"Exception occurred for {sdf}: {e}")

            continue
    print("All done!")
if __name__ == "__main__":
    main()