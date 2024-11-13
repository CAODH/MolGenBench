import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from itertools import combinations
from multiprocessing import Pool

#Calculate Morgan fingerprint
def calculate_morgan_fingerprint(molecule, radius=2, nBits=2048):
    if molecule is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
    return fingerprint

#Process folders and calculate Tanimoto similarity
def calculate_Morgan_fingerprint_Tdiv(config):
    config_output_path = config['output']['output_path']
    input_path = os.path.join(config_output_path, 'General-molecular-quality-metrics')
    uniqueness_base_folder = os.path.join(input_path, 'Uniqueness')
    diversity_base_folder = os.path.join(input_path, 'Morgan-Interal-diversity')
    os.makedirs(diversity_base_folder, exist_ok=True)
    
    for numeric_subfolder in range(32):
        numeric_subfolder_path = os.path.join(uniqueness_base_folder, str(numeric_subfolder))
        if not os.path.isdir(numeric_subfolder_path):
            continue
        
        molecules = []  #Initialize the molecules variable
        diversity_folder_path = os.path.join(diversity_base_folder, str(numeric_subfolder))
        os.makedirs(diversity_folder_path, exist_ok=True)
        
        for file_name in os.listdir(numeric_subfolder_path):
            if file_name.endswith('.txt') and not file_name.startswith('Tanimoto_'):
                file_path = os.path.join(numeric_subfolder_path, file_name)
                smiles_list = []
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.split('\t')
                        if len(parts) < 2:
                            continue
                        smiles = parts[1].strip()
                        smiles_list.append(smiles)
                molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
                output_path = os.path.join(diversity_folder_path, f'Tanimoto_{numeric_subfolder}_{file_name}')
                with open(output_path, 'w') as output_file:
                    for i, j in combinations(range(len(molecules)), 2):
                        mol1, mol2 = molecules[i], molecules[j]
                        fp1 = calculate_morgan_fingerprint(mol1)
                        fp2 = calculate_morgan_fingerprint(mol2)
                        if fp1 is None or fp2 is None:
                            continue
                        tanimoto_similarity = TanimotoSimilarity(fp1, fp2)
                        output_file.write(f"Molecule {i+1} vs Molecule {j+1}: Tanimoto similarity:\t{tanimoto_similarity}\n")
                #print(f"Successfully processed file: {file_path}")

        if molecules:  
            tanimoto_sum = 0.0
            total_molecules = len(molecules)
            for file_name in os.listdir(diversity_folder_path):
                if file_name.startswith(f'Tanimoto_{numeric_subfolder}_'):
                    file_path = os.path.join(diversity_folder_path, file_name)
                    with open(file_path, 'r') as file:
                        for line in file:
                            tanimoto_similarity = line.split('\t')[-1].strip()
                            tanimoto_sum += float(tanimoto_similarity)
            total_combinations = total_molecules**2
            overall_diversity = 1 - (tanimoto_sum / total_combinations)
        else:
            overall_diversity = None

        overall_output_path = os.path.join(diversity_folder_path, 'morgan-interal-diversity.txt')
        with open(overall_output_path, 'a') as overall_output_file:  
            if overall_diversity is not None:
                overall_output_file.write(f"Overall diversity for folder\t{numeric_subfolder}:\t{overall_diversity}\n")
            else:
                overall_output_file.write(f"Overall diversity for folder\t{numeric_subfolder}\tcould not be calculated due to lack of valid molecules.\n")

def process_folders_in_parallel(base_folder_path, folders_to_process, diversity_base_folder):
    with Pool() as pool:
        pool.starmap(calculate_Morgan_fingerprint_Tdiv, [(os.path.join(base_folder_path, str(folder_num)), diversity_base_folder) for folder_num in folders_to_process])