import importlib
import shutil
from pathlib import Path
import pytest
import os
ROOT = Path(__file__).resolve().parents[2]  # MolGenBench
TESTSAMPLES_DIR = ROOT / "TestSamples"
SAMPLE = "O14757"
Round_id = 1

@pytest.mark.integration
def test_hit_info_preprocess_denovo_main_writes_output(tmp_path):
    
    """
    Copy TestSamples/SAMPLE to a temporary location and run the real
    compute_smiles_scaffold_inchi_map(uniprot_id, reference_dir) against it.
    Assert that some new files are created in the temporary sample directory.

    This test performs real IO but keeps everything inside tmp_path to avoid
    modifying the repository/testdata.
    """
    ## test for compute_smiles_scaffold_inchi_map for reference preprocess
    # verify sample exists
    src_sample_dir = TESTSAMPLES_DIR / SAMPLE
    if not src_sample_dir.exists():
        pytest.skip(f"Sample directory not found: {src_sample_dir}")

    mod = importlib.import_module("molgenbench.preprocess.reference_process")
    if not hasattr(mod, "compute_smiles_scaffold_inchi_map"):
        pytest.skip("compute_smiles_scaffold_inchi_map not present in module")

    # prepare temporary working copy
    dest_root = tmp_path / "TestSamples"
    dest_sample = dest_root / SAMPLE
    shutil.copytree(src_sample_dir, dest_sample)

    # record files before
    before = {p.relative_to(dest_root) for p in dest_sample.rglob("*") if p.is_file()}

    # run real function (may create files under dest_root)
    # compute_smiles_scaffold_inchi_map(uniprot_id, active_dir)
    mod.compute_smiles_scaffold_inchi_map(SAMPLE, str(dest_root))

    # record files after
    after = {p.relative_to(dest_root) for p in dest_sample.rglob("*") if p.is_file()}

    new_files = after - before
    assert new_files, f"No new files created by compute_smiles_scaffold_inchi_map; before={len(before)} after={len(after)}"

    # Prefer that at least one .pkl or .txt or .json was created (adjust as needed)
    assert any(str(f).endswith((".pkl", ".pickle", ".txt", ".json")) for f in new_files), (
        f"Expected some result files (.pkl/.txt/.json) among new files, got: {new_files}"
    )
    
    ## test for compute_hit_info for denovo preprocess

    """
    Copy TestSamples/SAMPLE to a temporary location and run the real
    hit_info_preprocess.main(...) against it. Assert that some new files
    are created in the temporary sample directory.
    """
    # verify sample exists
    src_sample_dir = TESTSAMPLES_DIR
    if not src_sample_dir.exists():
        pytest.skip(f"Sample directory not found: {src_sample_dir}")

    mod = importlib.import_module("molgenbench.preprocess.hit_info_preprocess_denovo")
    if not hasattr(mod, "main"):
        pytest.skip("main not present in module")

    # prepare temporary working copy
    dest_root = tmp_path / "TestSamples"
   
    # shutil.copytree(src_sample_dir, dest_root, dirs_exist_ok=True )
    

    # record files before
    # before = {p.relative_to(dest_root) for p in dest_root.rglob("*") if p.is_file()}

    mod.compute_hit_info(generated_dir=str(dest_root), round_id_list=[Round_id], model_name_list=['PocketFlow', 'TamGen'],save_dir = str(tmp_path / "TestSample_Denovo_Results"))

    # record files after
    result_path = tmp_path / "TestSample_Denovo_Results"
    after = {p.relative_to(tmp_path) for p in result_path.rglob("*") if p.is_file()}

    new_files = after - set()
    assert new_files, f"No new files created by hit_info_preprocess_denovo.main; before={len(before)} after={len(after)}"

    # Prefer that at least one .csv was created (adjust as needed)
    assert any(str(f).endswith((".csv")) for f in new_files), (
        f"Expected some result files (.csv) among new files, got: {new_files}"
    )