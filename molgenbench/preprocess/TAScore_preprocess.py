
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import json
with open('../../sup_info/crossdock2020_duplicated_uniprotId_map_smiles_in_trainset.json','r') as f:
    crossdock2020_duplicated_uniprotId_map_smiles_in_trainset = json.load(f)
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
def get_trainset_scaffold(x,ref_map):
    uniprot_id  = x['UniprotID']
    if uniprot_id not in ref_map:
        return []
    else:
        all_find_scaffold = [i[1:-1] for i in x.Finded_Scaffolds[1:-1].split(', ')]
        all_find_scaffold = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_find_scaffold if i != '']
        ref_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in ref_map[uniprot_id]['scaffold'] if i != '']
        list_dup_scaffolds = list(set(all_find_scaffold).intersection(set(ref_scaffolds)))
        return list_dup_scaffolds
def get_trainset_smiles(x,ref_map):
    uniprot_id  = x['UniprotID']
    if uniprot_id not in ref_map:
        return []
    else:
        all_find_smiles = [i[1:-1] for i in x.Finded_Smiles[1:-1].split(', ')]
        all_find_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_find_smiles if i != '']
        ref_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in ref_map[uniprot_id]['smiles'] if i != '']
        list_dup_smiles = list(set(all_find_smiles).intersection(set(ref_smiles)))
    
        return list_dup_smiles
def merge_smiles_string(smiles_string_list):
    all_smiles = []
    for smiles_string in smiles_string_list:
        for smiles in smiles_string[1:-1].split(', ') :
            if smiles != '':
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles[1:-1],sanitize = False))
                    all_smiles.append(smiles)
                except:
                    print('Error smiles:',smiles)
    return all_smiles
def Scaffold2smiles_interaction_all(x,all_generated_info_map):
    all_generated_scaffolds = all_generated_info_map[x.ModelName]['all_generated_scaffolds']
    all_generated_smiles = all_generated_info_map[x.ModelName]['all_generated_smiles']
    
    all_ref_scaffolds = [i[1:-1] for i in x.Reference_Scaffolds[1:-1].split(', ')]
    all_ref_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_ref_scaffolds if i != '']
    all_interaction_scaffolds = set(all_ref_scaffolds).intersection(all_generated_scaffolds)
    # 找到所有interaction scaffold 对应了那些分子
    find_Scaffold_to_smiles = []
    for scaffold in all_interaction_scaffolds:
        if scaffold not in all_generated_info_map[x.ModelName]:
            continue
        else:
            find_Scaffold_to_smiles_tmp = all_generated_info_map[x.ModelName][scaffold]
            if len(find_Scaffold_to_smiles_tmp) == 0:
                continue
            else:
                find_Scaffold_to_smiles.extend(find_Scaffold_to_smiles_tmp )

    return len(set(find_Scaffold_to_smiles))/len(set(all_generated_smiles))
def Scaffold2smiles_interaction_specific(x,all_generated_info_map):
    all_gen_scaffolds = [i[1:-1] for i in x.Generated_Scaffolds[1:-1].split(', ')]
    all_gen_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_gen_scaffolds if i != '']
    
    all_ref_scaffolds = [i[1:-1] for i in x.Reference_Scaffolds[1:-1].split(', ')]
    all_ref_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_ref_scaffolds if i != '']
    all_interaction_scaffolds = set(all_ref_scaffolds).intersection(all_gen_scaffolds)
    # 找到所有interaction scaffold 对应了那些分子
    # all_interaction_smiles = []
    all_generated_smiles = [i[1:-1] for i in x.Generated_Smiles[1:-1].split(', ')]
    all_generated_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_generated_smiles if i != '']
    # all_generated_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_generated_smiles if i != '']
    find_Scaffold_to_smiles = []
    for generated_smiles in all_generated_smiles:
        generated_scaffold = GetScaffold(generated_smiles)
        if generated_scaffold in all_interaction_scaffolds and generated_scaffold != 'None':
            find_Scaffold_to_smiles.append(generated_smiles)

    return len(set(find_Scaffold_to_smiles))/len(set(all_generated_smiles))

def Smiles_interaction_all(x,all_generated_info_map):
    all_generated_scaffolds = all_generated_info_map[x.ModelName]['all_generated_smiles']
    all_ref_scaffolds = [i[1:-1] for i in x.Reference_Smiles[1:-1].split(', ')]
    all_ref_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_ref_scaffolds if i != '']
    
    return len(set(all_ref_scaffolds).intersection(all_generated_scaffolds))/len(set(all_generated_scaffolds))
def Smiles_interaction_specific(x,all_generated_info_map):
    all_gen_scaffolds = [i[1:-1] for i in x.Generated_Smiles[1:-1].split(', ')]
    all_gen_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_gen_scaffolds if i != '']
    
    all_ref_scaffolds = [i[1:-1] for i in x.Reference_Smiles[1:-1].split(', ')]
    all_ref_scaffolds = [Chem.MolToSmiles(Chem.MolFromSmiles(i,sanitize = False)) for i in all_ref_scaffolds if i != '']
    
    return len(set(all_ref_scaffolds).intersection(all_gen_scaffolds ))/len(set(all_gen_scaffolds))
def main(argv=None):

    import argparse
    parser = argparse.ArgumentParser(description='PreProcess the hit information of generated molecules')
    parser.add_argument('--denovo_hit_info_dir', type=str, help='The directory of the result from hit_info_preprocess_denovo.py', required=True)
    parser.add_argument('--round_id_list', type=str, nargs='*', default=[1,2,3], help='The list of uniprot ids to process; if not given, process all')
    parser.add_argument('--save_dir ', type=str, default=None, help='save directory')
   
    args = parser.parse_args(argv)
    denovo_hit_info_dir = args.generated_dir
    save_dir = args.save_dir
    round_id_list = args.round_id_list
    for round_num in round_id_list:
        all_result_path = f'{denovo_hit_info_dir}/Round{round_num}'
        all_results = os.listdir(all_result_path)
        all_results_pd = []
        # 读取所有model 的输出
        for temp_path in all_results:
            model_name = os.path.splitext(temp_path)[0]
            temp_path = os.path.join(all_result_path, temp_path)
            temp_pd = pd.read_csv(temp_path)
            temp_pd['ModelName'] = [model_name]*len(temp_pd)
            all_results_pd.append(temp_pd)
            
            
        merged_result_pd = pd.concat(all_results_pd,axis = 0)#['ModelName'].value_counts()
        merged_result_pd['Dupliceted_UniprotID'] = merged_result_pd['UniprotID'].swifter.apply(lambda x: x in crossdock2020_duplicated_uniprotId_map_smiles_in_trainset)
        # 把一个模型的所有结果收集起来
        merged_result_pd_grouped = merged_result_pd.groupby('ModelName')[['Generated_Smiles','Generated_Scaffolds']].agg(list).reset_index()

        merged_result_pd_grouped['Generated_Smiles'] = merged_result_pd_grouped['Generated_Smiles'].swifter.apply(lambda x: merge_smiles_string(x))
        merged_result_pd_grouped['Generated_Scaffolds'] = merged_result_pd_grouped['Generated_Scaffolds'].swifter.apply(lambda x: merge_smiles_string(x))

        from collections import defaultdict
        all_generated_info_map = defaultdict(dict)
        for index, row in merged_result_pd_grouped.iterrows():
            all_generated_info_map[row['ModelName']] = {'all_generated_smiles':set(row['Generated_Smiles']),'all_generated_scaffolds':set(row['Generated_Scaffolds'])}
            ## 每个骨架对应了多少smiles
            for smiles in row['Generated_Smiles']:
                scaffold = GetScaffold(smiles)
                if scaffold == 'None':
                    continue
                else:
                    if scaffold not in all_generated_info_map[row['ModelName']]:
                        all_generated_info_map[row['ModelName']][scaffold] = []
                        all_generated_info_map[row['ModelName']][scaffold].append(smiles)
                    else:
                        all_generated_info_map[row['ModelName']][scaffold].append(smiles)

        merged_result_pd['Scaffold2smiles_interaction_specific'] = merged_result_pd.swifter.apply(lambda x: Scaffold2smiles_interaction_specific(x,all_generated_info_map),axis = 1)
        merged_result_pd['Scaffold2smiles_interaction_all'] = merged_result_pd.swifter.apply(lambda x: Scaffold2smiles_interaction_all(x,all_generated_info_map),axis = 1)
        ## smiles
        merged_result_pd['smiles_interaction_specific'] = merged_result_pd.swifter.apply(lambda x: Smiles_interaction_specific(x,all_generated_info_map),axis = 1)
        merged_result_pd['smiles_interaction_all'] = merged_result_pd.swifter.apply(lambda x:Smiles_interaction_all(x,all_generated_info_map),axis = 1)

        merged_result_pd.to_csv(f'{save_dir}/Round{round_num}.csv')
if __name__ == '__main__':
    main(None)