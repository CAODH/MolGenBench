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
# 定义一个分子，例如苯甲酸
smiles = "C1=CC=C(C=C1)C(=O)O"
def GetScaffold(smiles):
        
    mol = Chem.MolFromSmiles(smiles)

    # 计算分子骨架
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # 输出分子骨架的 SMILES
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles
# get reference smiles (actives compounds)
from tqdm import tqdm
import glob
# ref_actives_path = os.path.join('/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_archive_241203',uniprot_id,'all_active_molecules_new_20241120',f'{uniprot_id}_all_active_molecules_new_20241120_duplicate.sdf')
            
            
            # generate_mol_path = glob.glob(f"{generate_dir}/{uniprot_id}/{uniprot_id}_{model_name}_*.sdf")
generate_dir = '/home/datahouse1/caoduanhua/MolGens/TestResults/MolCraft_Plus/sampling_results'
save_dir = '/home/datahouse1/caoduanhua/MolGens/TestResults/MolCraft_Plus/analysisResults/molecule_scaffold_recovery'
# model_name = 'PocketFlow'
model_name_list = []
for name in glob.glob(f"{generate_dir}/O75874/*.sdf"):
    # print(name)
    model_name = '_'.join(name.split('/')[-1].split('.sdf')[0].split('_')[1:])
    model_name_list.append(model_name)
# model_name_list.append('TamGen')
print(model_name_list)
for model_name in model_name_list:
    try:
        if os.path.exists(os.path.join(save_dir,f'{model_name}.csv')):
            # print(f'{model_name} has been analyzed')
            os.remove(os.path.join(save_dir,f'{model_name}.csv'))
            # continue
            
        uniprot_id_list = []
        uniprot_ref_smiles_list = []
        uniprot_gen_smiles_list = []
        
        for uniprot_id in tqdm(os.listdir(generate_dir),total = len(os.listdir(generate_dir))):
            
            ref_actives_path = os.path.join('/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_archive_241203',uniprot_id,'all_active_molecules_new_20241120',f'{uniprot_id}_all_active_molecules_new_20241120_duplicate.sdf')
            
            
            if model_name == 'TamGen':
                generate_mol_path = os.path.join('/home/datahouse1/caoduanhua/MolGens/Test_Results/TamGenOutput/Smiles_min_50_max_inf_Scaffold_50_Serise_20_pocket',uniprot_id,f'{uniprot_id}_{model_name}_generated_molecules.sdf')
            else:
                generate_mol_path = glob.glob(f"{generate_dir}/{uniprot_id}/{uniprot_id}_{model_name}.sdf")
            
                if len(generate_mol_path) ==0:
                    print(f'No generated molecules for {uniprot_id} in {model_name}')
                    continue
                else:
                    generate_mol_path = generate_mol_path[0]
            if not os.path.exists(ref_actives_path) or not os.path.exists(generate_mol_path):
                print(f'No generated molecules for {uniprot_id} in {model_name},{generate_mol_path}')
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
        result['Reference_Smiles_num']=result['Reference_Smiles'].apply(lambda x:len(x))
        
        result['Finded_Smiles'] = result.apply(lambda x:list(set(x.Reference_Smiles).intersection(set(x.Generated_Smiles))),axis = 1)
        
        
        result['Finded_Smiles_Num']=result['Finded_Smiles'].apply(lambda x:len(x))
        
        
        result['Reference_Scaffolds'] = result['Reference_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
        result['Reference_Scaffolds_Num'] = result['Reference_Scaffolds'].apply(lambda x:len(x))
        result['Generated_Scaffolds'] = result['Generated_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
        result['Generated_Scaffolds_Num'] = result['Generated_Scaffolds'].apply(lambda x:len(x))
        
        result['Finded_Scaffolds'] = result.apply(lambda x:list(set(x.Reference_Scaffolds).intersection(set(x.Generated_Scaffolds))),axis = 1)
        
        result['Finded_Scaffolds_Frequency'] = result.apply(lambda x:len([ GetScaffold(smiles) for smiles in x.Generated_Smiles if  GetScaffold(smiles) in x.Finded_Scaffolds]),axis=1)
        
        
        # 计算hit 到的骨架在所有分子里面占到的比例
        result['Finded_Scaffolds_Frequency_Rate'] =result.apply(lambda x:x.Finded_Scaffolds_Frequency/len(x.Generated_Smiles),axis=1)
        result['Finded_Scaffolds_Num'] = result['Finded_Scaffolds'].apply(lambda x:len(x))
        result['Finded_Scaffolds_Rate'] = result['Finded_Scaffolds_Num']/result['Reference_Scaffolds_Num']
        
        # print('Sucessed Find at least one Scaffold',len(result[result['Finded_Scaffolds_Num'] >0]))

        # save result csv file
        result.to_csv(os.path.join(save_dir,f'{model_name}.csv'))
    except Exception as e:
        print(f'Error in {model_name}')
        print('Error:',str(e))
        continue