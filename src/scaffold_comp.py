from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import sys
from rdkit import Chem
from rdkit.Chem import rdFMCS
from joblib import Parallel, delayed
import pandas as pd
import os
import time 
def find_scaffold_with_coverage_filter(mols, min_coverage=0.33,threshold=0.8):
    """
    提取符合最小覆盖率要求的最大公共骨架（MCS）。
    
    参数:
    - mols: 输入的分子列表 (list of RDKit Mol objects)
    - min_coverage: 骨架覆盖率阈值，范围为 0~1 (float)
    
    返回:
    - scaffold: 提取的骨架 (RDKit Mol object)
    - filtered_mols: 满足覆盖率要求的分子列表 (list of RDKit Mol objects)
    """
    current_mols = list(set(mols))
    numbers = len(current_mols)
    scaffold, filtered_mols = None, []
    while True:
        # 计算 MCS
        if len(current_mols) <= 2:
            break
        mcs_result = rdFMCS.FindMCS(
            current_mols,
            threshold=threshold,  # 要求所有当前分子包含此骨架
            atomCompare=rdFMCS.AtomCompare.CompareElements,  # 原子比较方式
            bondCompare=rdFMCS.BondCompare.CompareOrder,    # 键比较方式
            ringMatchesRingOnly=True,                       # 保证环完整性
            completeRingsOnly=True,                         # 仅匹配完整环
            timeout=30,                                      # 超时时间
        )
        scaffold = Chem.MolFromSmarts(mcs_result.smartsString)
        # 找到满足覆盖率要求的分子
        filtered_mols = []
        for mol in current_mols:
            match = mol.GetSubstructMatch(scaffold)
            if match:
                coverage = len(match) / mol.GetNumAtoms()  # 计算骨架覆盖率
                if coverage >= min_coverage:
                    filtered_mols.append(mol)
        
        # 如果没有分子被剔除，停止迭代
        if len(filtered_mols) == len(current_mols):
            break
        # 更新分子列表
        current_mols = filtered_mols
        
    
    return scaffold, filtered_mols
def adjustThreshold(serise_id,mols,min_coverage=0.33):
    scafold,filtered_mols = None,None
    for i in range(10,3,-1):
        threshold = i*0.1
        
        scaffold, filtered_mols = find_scaffold_with_coverage_filter(mols, min_coverage,threshold)
        if len(filtered_mols) <= int(0.8*len(mols)):
            continue
        else:
            break   
        
    return serise_id,scaffold, filtered_mols,threshold

def resultToCsv(result,uniprot_id,father_path):

    pd_result = {"SeriseID":[],"Scaffold":[],"FilteredMols":[],"Threshold":[],"NumFilteredMols":[]}
    for serise_id,scaffold,filtered_mols,threshold in result:
        if scaffold is not None and len(filtered_mols)>0:
            # continue
            pd_result["SeriseID"].append(serise_id)
            pd_result["Scaffold"].append(Chem.MolToSmarts(scaffold))
            pd_result["FilteredMols"].append(','.join([Chem.MolToSmiles(mol) for mol in filtered_mols]))
            pd_result["NumFilteredMols"].append(len(filtered_mols))
            pd_result["Threshold"].append(threshold)

    pd_result = pd.DataFrame(pd_result).sort_values(by = "NumFilteredMols",ascending = False)
    
    # start =  time.strftime("%Y-%m-%d", time.localtime())
    pd_result.to_csv(f"{father_path}/{uniprot_id}/{uniprot_id}_scaffold_info_20241120.csv",index = False)



if __name__ == "__main__":
    
    
    
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='Calculate 2D scaffold for series of molecules')
    
    parser.add_argument('--father_path', type=str,default='/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/Smiles_min_50_max_inf_Scaffold_50_Serise_20_pocket', help='dir name of the benchmark')

    parser.add_argument('--njobs', type=int,default = 40, help='num of process to use')

    args = parser.parse_args()
    father_path = args.father_path
    uniprot_ids = os.listdir(father_path )
    for uniprot_id in uniprot_ids:
        actives_path = os.path.join(father_path,uniprot_id,f'{uniprot_id}_all_active_molecules_new_20241120.sdf')
        # save_path = f"{father_path}/{uniprot_id}/{start}_{uniprot_id}_scaffold_info_20241120.csv"
        
        actives = Chem.SDMolSupplier(actives_path)
        seriseID_map_mols = {}
        for mol in actives:
            if mol is not None:
                if mol.GetProp('_Name').split('_')[1] == 'NoSeriseID':
                    continue
                serise = '_'.join(mol.GetProp('_Name').split('_')[:2])
                if serise not in seriseID_map_mols:
                    seriseID_map_mols[serise] = []
                name = mol.GetProp('_Name')
                frags = Chem.GetMolFrags(mol, asMols=True)
                mol = max(frags, key=lambda x: x.GetNumAtoms())
                mol.SetProp('_Name', name)
                seriseID_map_mols[serise].append(mol)
                
        result = Parallel(n_jobs=args.njobs,verbose = 40)(delayed(adjustThreshold)(serise_id,molecules) for serise_id,molecules in tqdm(seriseID_map_mols.items(),total = len(seriseID_map_mols)))
        
        resultToCsv(result,uniprot_id,father_path)

