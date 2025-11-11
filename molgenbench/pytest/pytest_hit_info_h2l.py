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
def test_hit_info_preprocess_h2l_main_writes_output(tmp_path):
    


    """
    Copy TestSamples/SAMPLE to a temporary location and run the real
    hit_info_preprocess.main(...) against it. Assert that some new files
    are created in the temporary sample directory.
    """
    # verify sample exists
    src_sample_dir = TESTSAMPLES_DIR / SAMPLE
    if not src_sample_dir.exists():
        pytest.skip(f"Sample directory not found: {src_sample_dir}")

    mod = importlib.import_module("molgenbench.preprocess.hit_info_preprocess_h2l")
    if not hasattr(mod, "main"):
        pytest.skip("main not present in module")
    # prepare temporary working copy
    dest_root = tmp_path / "TestSamples"
    dest_sample = dest_root / SAMPLE
    shutil.copytree(src_sample_dir, dest_sample)
   

    # record files before
    # before = {p.relative_to(dest_root) for p in dest_sample.rglob("*") if p.is_file()}

    mod.compute_hit_info_h2l(generated_dir=str(dest_root), round_id_list=[Round_id], model_name_list=['DeleteHit2Lead(CrossDock)'])

    # record files after
    result_path = tmp_path / "TestSample_H2L" 
    after = {p.relative_to(tmp_path) for p in result_path.rglob("*") if p.is_file()}

    new_files = after - set()
    assert new_files, f"No new files created by hit_info_preprocess_h2l.main; before={len(before)} after={len(after)}"

    # Prefer that at least one .csv was created (adjust as needed)
    assert any(str(f).endswith((".csv")) for f in new_files), (
        f"Expected some result files (.csv) among new files, got: {new_files}"
    )