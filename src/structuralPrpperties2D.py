import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed
from tqdm import tqdm
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
        properties_str = (f"smiles:\t{smiles}\theavy-atom:\t{num_non_hydrogen_atoms}\tchiral-center:\t{num_chiral_carbons}\t"
                          f"Rings:\t{num_rings}\tAromatic-Rings:\t{num_aromatic_rings}\t"
                          f"Rotatable-Bonds:\t{num_rotatable_bonds}\tFsp3:\t{fraction_csp3:.3f}")
        
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
    if file_name.endswith('.sdf'):
        smiles_list = [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(file_name)]
    elif file_name.endswith('.smi') or file_name.endswith('.txt'):
        smiles_list = [line.strip() for line in open(file_name)]
    results = Parallel(n_jobs=njobs)(delayed(structural_properties)(smiles) for smiles in tqdm(smiles_list,total = len(smiles_list)))
    for idx, result in enumerate(results):
        if result is not None:
            with open(output_file, 'a') as result_file:
                result_file.write(result + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate 2D structural properties of molecules')
    
    parser.add_argument('--file_name', type=str,default='/home/datahouse1/caoduanhua/MolGens/Evalutions/SBDDBench/TestSamples/O14757/O14757_result_from_PGMG.txt', help='File name of the SDF or .smi or .txt file')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--njobs', type=int,default = 30, help='num of process to use')
    args = parser.parse_args()
    import time 
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.file_name)}/{start}_2D_properties_cal_result.txt'
    parallelStructuralProperties(args.file_name, args.njobs,args.output_file)
    

    
    
    
    