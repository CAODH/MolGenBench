# this python script have been used to get interactions in ref ligands with protein 
import rdkit
import prolif as plf
from Bio import PDB
import glob
from rdkit import Chem
import tqdm
import os
import pandas as pd
# get pocket and save to file
from joblib import Parallel, delayed
def splitComplex(pdb_file,save_dir):
    # 输入 PDB 文件
    # input_pdb_file
    # 输出文件名
    protein_output_file = f"{save_dir}/{os.path.basename(pdb_file)}"
    ligand_output_file = f"{save_dir}/{os.path.basename(pdb_file).replace('.pdb','lig.pdb')}"
    if os.path.exists(ligand_output_file):
        print('split complex exists! skip this process')
        return protein_output_file,ligand_output_file.replace('.pdb','.sdf')
    # 初始化 PDB 解析器
    parser = PDB.PDBParser(QUIET=True)
    # 解析结构
    structure = parser.get_structure("complex", pdb_file)
    # 初始化 PDB IO 对
    protein_io = PDB.PDBIO()
    ligand_io = PDB.PDBIO()

    # 定义选择类，用于选择蛋白质和配体
    class ProteinSelect(PDB.Select):
        def accept_residue(self, residue):
            # 排除水分子 (HOH) 和异源分子 (HETATM)
            if residue.id[0] == " ":
                return True
            else:
                return False

    class LigandSelect(PDB.Select):
        def accept_residue(self, residue):
            # 仅保留异源分子 (HETATM)，排除水分子
            if residue.id[0] != " " and residue.id[0] != "W":
                return True
            else:
                return False

    # 保存蛋白质
    protein_io.set_structure(structure)
    protein_io.save(protein_output_file, select=ProteinSelect())

    # 保存配体
    ligand_io.set_structure(structure)
    ligand_io.save(ligand_output_file, select=LigandSelect())
    ligand_sdf_file = ligand_output_file.replace('.pdb','.sdf')
    writer = Chem.SDWriter(ligand_sdf_file)
    mol = Chem.MolFromPDBFile(ligand_output_file,removeHs = False)
    writer.write(mol)
    writer.close()
    
    
    # print(f"Protein saved to {protein_output_file}")
    # print(f"Ligand saved to {ligand_sdf_file}")
    return protein_output_file,ligand_sdf_file
def nonBondInteractions(protein_path,ligand_path,interaction_norm_map=None):
    if interaction_norm_map is None:
        all_interactions = plf.Fingerprint.list_available()
    else:
        
        all_interactions = list(interaction_norm_map.keys())

    fp = plf.Fingerprint(interactions = all_interactions)

    mols = rdkit.Chem.SDMolSupplier(ligand_path,removeHs = False)

    
    m2 = rdkit.Chem.MolFromPDBFile(protein_path,removeHs = False)

    prot = plf.Molecule(m2)
    ligands = [plf.Molecule.from_rdkit(m1) for m1 in mols]

    fp.run_from_iterable(ligands,prot,progress=False,n_jobs = 1,)
    df = fp.to_dataframe()

    return df
##  这些interaction 只要出现了一样的就行，而不是要供体受体要一样
interaction_norm_map = {'HBAcceptor':'Hydrogen',
                        'HBDonor':'Hydrogen',
                        
                        'CationPi':'CationPi',
                        'PiCation':'CationPi',
                        
                        'PiStacking':'PiStacking',
                        'EdgeToFace':'PiStacking',
                        'FaceToFace':'PiStacking',
                        
                        'XBAcceptor':'X-Hydrogen',
                        'XBDonor':'X-Hydrogen',
                        
                        'MetalAcceptor':'Metal',
                        'MetalDonor':'Metal',
                        
                        'Anionic':'Ion',
                        'Cationic':'Ion',
                        
                        'VdWContact':'VdWContact',
                        'Hydrophobic':'Hydrophobic',
                        
                        }
benchmark_dir='/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_archive_241203'
root_save_dir = '/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/AnalysisResults/interactions_recovery/interaction_results'
# /home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/selfGenBench_archive_241203/O14757/all_active_molecules_new_20241120/O14757_all_active_molecules_new_20241120_ligprep_glide_sp_pv_duplicate.sdf
# split_protein_ligand_dir = '/home/datahouse1/caoduanhua/MolGens/SelfConstructedBenchmark/Smiles_min_50_max_inf_Scaffold_50_Serise_20_interactions'

import json
import multiprocessing
cpu_counts = multiprocessing.cpu_count() - 5
print('cpu counts:',cpu_counts)
def UniprotInteractions(uniprot_id):
    ligand_path = os.path.join(benchmark_dir,uniprot_id,f'all_active_molecules_new_20241120/{uniprot_id}_all_active_molecules_new_20241120_ligprep_glide_sp_pv_duplicate.sdf')
    protein_path = os.path.join(benchmark_dir,uniprot_id,f'{uniprot_id}_prep.pdb')
    save_dir = os.path.join(root_save_dir,uniprot_id)
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,ligand_path.split('/')[-1].replace('.sdf','_interactions.json'))
    if os.path.exists(save_path):
        return
    pandas_result = nonBondInteractions(protein_path,ligand_path,interaction_norm_map=None)
    pandas_result.to_csv(save_path)

_ = Parallel(n_jobs=cpu_counts,backend='loky',verbose = 20)(delayed(UniprotInteractions)(uniprot_id) for uniprot_id in tqdm.tqdm(os.listdir(benchmark_dir),total = len(os.listdir(benchmark_dir))))