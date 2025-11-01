
## hit rediscovery metric for molecular generation
from typing import Dict, Type, Any, List, Optional
from molgenbench.metrics.base import MetricBase
from molgenbench.types import MoleculeRecord
############## raw code ##############
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
# 定义一个分子，例如苯甲酸
smiles = "C1=CC=C(C=C1)C(=O)O"
import swifter
# import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30) 
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
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
def find_smiles_and_inchi(ref_smiles,gen_smiles,uniprot_id):
    query_inchi_set = set([Chem.MolToInchi(Chem.MolFromSmiles(temp)) for temp in gen_smiles if Chem.MolFromSmiles(temp) is not None])
    ref_inchi_map_ref_smiles = {}
    if not os.path.exists(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl'):
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_smiles.update({key:ref_temp for key in ref_inchi_set})
            
        with open(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_smiles,f)
    else:
        with open(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_smiles.pkl','rb') as f :
            ref_inchi_map_ref_smiles = pickle.load(f)
    intersections_inchi = set(ref_inchi_map_ref_smiles.keys()).intersection(query_inchi_set)
    interactions_smiles = set([ref_inchi_map_ref_smiles[inchi] for inchi in intersections_inchi])
    return list(interactions_smiles),list(intersections_inchi)
def find_scaffold_and_inchi(ref_smiles,gen_smiles,uniprot_id):

    query_inchi_set = set([Chem.MolToInchi(Chem.MolFromSmiles(temp)) for temp in gen_smiles if Chem.MolFromSmiles(temp) is not None])
    ref_inchi_map_ref_scaffold = {}

    if not os.path.exists(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl'):
        for ref_temp in ref_smiles:
            ref_inchi_set = enumerate_tautomer_and_partial_chirality(ref_temp)
            ref_inchi_map_ref_scaffold.update({key:ref_temp for key in ref_inchi_set})
            
        with open(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','wb') as f :
            pickle.dump(ref_inchi_map_ref_scaffold,f)
    else:
        with open(f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/scripts/temp_inchi_map_smiles_pkls/{uniprot_id}_scaffold.pkl','rb') as f :
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
round = 3
generate_dir = '/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_repeat_output_250211'
# ~/O14757/Round2/Hit_to_Lead_Results/Sries14139/DeleteHit2Lead(CrossDock)_Hit_to_Lead/O14757_Sries14139_DeleteHit2Lead(CrossDock)_Hit_to_Lead.sdf
# model_name = 'PocketFlow'
save_dir = f'/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/AnalysisResults/hit_to_lead_scaffold_recovery/withFixedSteroAndTautomer/results/Round{round}'
model_name_list = ['DeleteHit2Lead(CrossDock)','ShapeMol','shepherd_x1x3x4_mosesaq_submission','DiffDec','diffSBDD_cond_crossdocked','diffSBDD_cond_moad','PGMG']

print(model_name_list)
for model_name in model_name_list:
    try:
        if os.path.exists(os.path.join(save_dir,f'{model_name}.csv')):
            print(f'{model_name} has been analyzed')
            continue
            
        uniprot_id_list = []
        uniprot_ref_smiles_list = []
        uniprot_gen_smiles_list = []
        
        for uniprot_id in tqdm(os.listdir(generate_dir),total = len(os.listdir(generate_dir))):
            
            ref_actives_path = os.path.join('/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_archive_241203',uniprot_id,'all_active_molecules_new_20241120',f'{uniprot_id}_all_active_molecules_new_20241120_duplicate.sdf')
            
            serise_ids = os.listdir(os.path.join(generate_dir,uniprot_id,f'Round{round}','Hit_to_Lead_Results'))
            if len(serise_ids) == 0:
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
            uniprot_gen_smiles = []
            for serise_id in serise_ids:
                generate_mol_path = os.path.join(generate_dir,uniprot_id,f'Round{round}','Hit_to_Lead_Results',serise_id,f'{model_name}_Hit_to_Lead',f'{uniprot_id}_{serise_id}_{model_name}_Hit_to_Lead.sdf')
    

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
            uniprot_ref_smiles_list.append(list(set(uniprot_ref_smiles)))
            uniprot_id_list.append(uniprot_id)
            
        result = pd.DataFrame({'UniprotID':uniprot_id_list ,
                'Reference_Smiles':uniprot_ref_smiles_list ,
                'Generated_Smiles':uniprot_gen_smiles_list})
        result['Reference_Smiles_num']=result['Reference_Smiles'].apply(lambda x:len(x))
        
        # result['Finded_Smiles'] = result.apply(lambda x:list(set(x.Reference_Smiles).intersection(set(x.Generated_Smiles))),axis = 1)
        result['Finded_Smiles_and_Inchi'] = result.swifter.apply(lambda x:find_smiles_and_inchi(set(x.Reference_Smiles),set(x.Generated_Smiles),x.UniprotID),axis = 1)
        result['Finded_Smiles'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: x[0])
        result['Finded_Smiles_Num']=result['Finded_Smiles'].apply(lambda x:len(x))
        result['Finded_Inchi'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: x[1])
        result['Finded_Inchi_Num'] = result['Finded_Smiles_and_Inchi'].apply(lambda x: len(x))

        result['Reference_Scaffolds'] = result['Reference_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
        result['Reference_Scaffolds'] = result['Reference_Scaffolds'].apply(lambda x: [_ for _ in x if _ != 'None'])
        result['Reference_Scaffolds_Num'] = result['Reference_Scaffolds'].apply(lambda x:len(x))
        
        result['Generated_Scaffolds'] = result['Generated_Smiles'].apply(lambda x:list(set([GetScaffold(smiles) for smiles in x])))
        result['Generated_Scaffolds'] = result['Generated_Scaffolds'].apply(lambda x: [_ for _ in x if _ != 'None'])
        result['Generated_Scaffolds_Num'] = result['Generated_Scaffolds'].apply(lambda x:len(x))
        
        
        result['Finded_Scaffolds_and_Inchi'] = result.swifter.apply(lambda x:find_scaffold_and_inchi(set(x.Reference_Scaffolds),set(x.Generated_Scaffolds),x.UniprotID),axis = 1)
        result['Finded_Scaffolds'] = result['Finded_Scaffolds_and_Inchi'].apply(lambda x: x[0])
        # result['Finded_Scaffolds'] = result.apply(lambda x:list(set(x.Reference_Scaffolds).intersection(set(x.Generated_Scaffolds))),axis = 1)
        result['Finded_Scaffolds_Num'] = result['Finded_Scaffolds'].apply(lambda x:len(x))
        result['Finded_Scaffolds_Inchi'] = result['Finded_Scaffolds_and_Inchi'].apply(lambda x: x[1])
        result['Finded_Scaffolds_Inchi_Num'] = result['Finded_Scaffolds_Inchi'].apply(lambda x:len(x))
        # 计算hit 到的骨架在所有分子里面占到的比例
        result['Finded_Scaffolds_Frequency_Rate'] =result.apply(lambda x:x.Finded_Scaffolds_Inchi_Num/len(x.Generated_Smiles),axis=1)
        result['Finded_Scaffolds_Rate'] = result['Finded_Scaffolds_Num']/result['Reference_Scaffolds_Num']
        
        # print('Sucessed Find at least one Scaffold',len(result[result['Finded_Scaffolds_Num'] >0]))

        # save result csv file
        os.makedirs(save_dir,exist_ok=True)
        result.to_csv(os.path.join(save_dir,f'{model_name}.csv'))
    except Exception as e:
        print(f'Error in {model_name}')
        print('Error:',str(e))
        continue
########## raw  end ##############
class hitRediscoveryMetric(MetricBase):
    name = "HitRediscovery"

    def compute_one(self, record: MoleculeRecord, ref_record: Optional[MoleculeRecord] = None) -> Dict[str, Any]:
        # 计算命中重发现指标的逻辑
        
        return {"hit_rediscovery_score": 0.8}  # 示例返回值
