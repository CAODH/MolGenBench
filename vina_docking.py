import os
import logging
import argparse
import pandas as pd
import multiprocessing as mp

from rdkit import Chem
from joblib import Parallel, delayed

from molgenbench.vina.vina_utils import prepare_output_df, process_row, choose_processable_path


def get_logger(filename, level=logging.INFO):
    logging.basicConfig(
        filename=filename,
        filemode='w',
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    return logger


def process_sdf_group(
    protein_path: str,
    ref_mol,
    input_sdf_path: str,
    output_root: str,
    model_name: str,
    logger
):
    """
    Shared routine for both De_novo and Hit_to_Lead processing.
    Splits an SDF file, prepares ligand files, and returns a DataFrame ready for docking.

    Args:
        protein_path: path to receptor pdb file
        ref_mol: RDKit molecule of reference ligand
        input_sdf_path: path to the sdf file to be processed
        output_root: target directory to save temp and docked files
        model_name: name of the model being processed
        logger: logging object
    Returns:
        pd.DataFrame: dataframe containing ligand file metadata
    """
    if not os.path.exists(input_sdf_path):
        logger.warning(f"{input_sdf_path} not found, skipping...")
        return None

    tmp_ligand_root = os.path.join(output_root, f'{model_name}_tmp_split')
    docked_root = os.path.join(output_root, f'{model_name}_vinadocked_output')
    os.makedirs(docked_root, exist_ok=True)
    os.makedirs(tmp_ligand_root, exist_ok=True)

    # Split the sdf into single molecules
    logger.info(f"Splitting {input_sdf_path} into individual SDFs...")
    for i, mol in enumerate(Chem.SDMolSupplier(input_sdf_path)):
        try:
            Chem.MolToMolFile(mol, os.path.join(tmp_ligand_root, f"{i}.sdf"))
        except Exception as e:
            logger.warning(f"Failed to write molecule {i}: {e}")
            continue

    # Prepare dataframe for docking
    logger.info("Preparing output dataframe...")
    df = prepare_output_df(protein_path, tmp_ligand_root, ref_mol, docked_root)
    return df


def get_prepared_df(
    data_path, 
    output_path, 
    model_name, 
    prepared_df_path, 
    logger, 
    round_name,
    mode='De_novo_Results'
):
    """
    Unified function to aggregate docking preparation DataFrames
    for both De_novo and Hit_to_Lead modes.
    """
    dfs = []

    for uniprot in os.listdir(data_path):
        base_uniprot_root = os.path.join(data_path, uniprot)
        source_root = os.path.join(data_path, uniprot, round_name, mode)
        target_root = os.path.join(output_path, f"{model_name}_workdir", uniprot, round_name, mode)
        os.makedirs(target_root, exist_ok=True)
        logger.critical(f"Processing {uniprot}...")

        # reference files
        ref_lig_path = os.path.join(base_uniprot_root, f"{uniprot}_lig.sdf")
        protein_path = os.path.join(base_uniprot_root, f"{uniprot}_prep.pdb")
        pocket_path = os.path.join(base_uniprot_root, f"{uniprot}_pocket10.pdb")

        protein_path = choose_processable_path(protein_path, pocket_path)
        ref_mol = Chem.SDMolSupplier(ref_lig_path)[0]

        if mode == 'De_novo_Results':
            # path: <data>/<uniprot>/<round>/De_novo_Results/<model>/<uniprot>_<model>.sdf
            task_sdf_path = os.path.join(source_root, model_name, f"{uniprot}_{model_name}.sdf")
            logger.info(f"Processing De_novo task {model_name}...")
            df = process_sdf_group(protein_path, ref_mol, task_sdf_path, target_root, model_name, logger)
            if df is not None:
                dfs.append(df)

        elif mode == 'Hit_to_Lead_Results':
            # path: <data>/<uniprot>/<round>/Hit_to_Lead_Results/<series_id>/<model>/<uniprot>_<series_id>_<model>.sdf
            for series_id in os.listdir(source_root):
                series_src = os.path.join(source_root, series_id)
                series_tgt = os.path.join(target_root, series_id)
                os.makedirs(series_tgt, exist_ok=True)

                task_sdf_path = os.path.join(series_src, model_name, f"{uniprot}_{series_id}_{model_name}.sdf")
                logger.info(f"Processing Hit_to_Lead series {series_id} for {model_name}...")
                df = process_sdf_group(protein_path, ref_mol, task_sdf_path, series_tgt, model_name, logger)
                if df is not None:
                    dfs.append(df)

    if not dfs:
        logger.warning("No valid SDFs found, prepared_df will be empty.")
        return pd.DataFrame()

    prepared_df = pd.concat(dfs, ignore_index=True)
    prepared_df.to_pickle(prepared_df_path)
    return prepared_df
        

def merge_sdf_folder(input_dir: str, output_sdf_path: str):
    """
    Merge all SDF files under `input_dir` into a single output file.
    Skips invalid or unreadable molecules.
    """
    writer = Chem.SDWriter(output_sdf_path)
    for sdf_file in os.listdir(input_dir):
        if not sdf_file.endswith('.sdf'):
            continue
        sdf_path = os.path.join(input_dir, sdf_file)
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
            if mol is not None:
                writer.write(mol)
    writer.close()


def merge_docked_sdfs(
    data_path, 
    output_path, 
    model_name, 
    round_name,
    mode='De_novo_Results'
):
    """
    Merge all docked molecules into per-Uniprot (and per-series) SDF files.
    """
    for uniprot in os.listdir(data_path):
        source_root = os.path.join(data_path, uniprot, round_name, mode)
        target_root = os.path.join(output_path, f"{model_name}_workdir", uniprot, round_name, mode)
        os.makedirs(target_root, exist_ok=True)

        if mode == 'De_novo_Results':
            # example: <data>/<uniprot>/<round>/De_novo_Results/<model>/<uniprot>_<model>_vina_docked.sdf
            docked_root = os.path.join(target_root, f"{model_name}_vinadocked_output")
            output_sdf = os.path.join(source_root, model_name, f"{uniprot}_{model_name}_vina_docked.sdf")

            merge_sdf_folder(docked_root, output_sdf)

        elif mode == 'Hit_to_Lead_Results':
            # example: <data>/<uniprot>/<round>/Hit_to_Lead_Results/<series_id>/<model>/<uniprot>_<series_id>_<model>_vina_docked.sdf
            for series_id in os.listdir(source_root):
                series_src = os.path.join(source_root, series_id)
                series_tgt = os.path.join(target_root, series_id)
                docked_root = os.path.join(series_tgt, f"{model_name}_vinadocked_output")
                output_sdf = os.path.join(series_src, model_name, f"{uniprot}_{series_id}_{model_name}_vina_docked.sdf")

                merge_sdf_folder(docked_root, output_sdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Path/to/data/', help='Path to the data directory contain uniprot folders')
    parser.add_argument('--output_path', type=str, default='Path/to/output/', help='Path to the output directory')
    parser.add_argument('--round_name', type=str, default='Round1', help='Name of the current round')
    parser.add_argument('--mode', choices=['De_novo_Results', 'Hit_to_Lead_Results'], default='De_novo_Results', help='Task mode')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the generative model')
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output_path
    model_name = args.model_name
    round_name = args.round_name
    mode = args.mode
    
    logger_path = os.path.join(output_path, f'{model_name}_{mode}_vina_docking_{round_name}.log')
    prepared_df_path = os.path.join(output_path, f'{model_name}_{mode}_prepared_df_{round_name}.pkl')
    results_df_path = os.path.join(output_path, f'{model_name}_{mode}_vina_results_{round_name}.csv')
    
    logger = get_logger(logger_path)
    
    if not os.path.exists(prepared_df_path):
        logger.info(f"Preparing combined_df...")
        prepared_df = get_prepared_df(
            data_path=data_path,
            output_path=output_path, 
            model_name=model_name, 
            prepared_df_path=prepared_df_path, 
            logger=logger, 
            round_name=round_name,
            mode=mode
        )
        
    else:
        logger.info(f"Reading combined_df from {prepared_df_path}...")
        prepared_df = pd.read_pickle(prepared_df_path)
    
    logger.info("Docking...")
    cores = max(1, (mp.cpu_count()-5)//8)
    results = Parallel(n_jobs=cores, verbose = 100)(delayed(process_row)(row) for _, row in prepared_df.iterrows())
    for idx, affinity, docked_path in results:
        prepared_df.at[idx, 'affinity'] = affinity
        prepared_df.at[idx, 'docked_path'] = docked_path
    
    prepared_df.to_csv(results_df_path, index=False)
    
    logger.info("Merging docked SDFs...")
    
    merge_docked_sdfs(
        data_path=data_path, 
        output_path=output_path, 
        model_name=model_name, 
        round_name=round_name,
        mode=mode
    )

