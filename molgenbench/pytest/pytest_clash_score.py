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
def test_clash_score_main_writes_output(tmp_path):
    


    """
    Copy TestSamples/SAMPLE to a temporary location and run the real
    hit_info_preprocess.main(...) against it. Assert that some new files
    are created in the temporary sample directory.
    """
    # verify sample exists
    src_sample_dir = TESTSAMPLES_DIR / SAMPLE
    if not src_sample_dir.exists():
        pytest.skip(f"Sample directory not found: {src_sample_dir}")

    mod = importlib.import_module("molgenbench.preprocess.clash_score")

    # prepare temporary working copy
    dest_root = tmp_path / "TestSamples"
    dest_sample = dest_root / SAMPLE
    shutil.copytree(src_sample_dir, dest_sample)
    
   

    # record files before
    before = {p.relative_to(dest_root) for p in dest_sample.rglob("*") if p.is_file()}

    mod.UniprotClash(uniprot_id=SAMPLE,generated_dir=str(dest_root),root_save_dir=str(dest_root))

    # record files after

    after = {p.relative_to(dest_root) for p in dest_sample.rglob("*") if p.is_file()}

    new_files = after - before
    assert new_files, f"No new files created by hit_info_preprocess_h2l.main; before={len(before)} after={len(after)}"

    # Prefer that at least one .csv was created (adjust as needed)
    assert any(str(f).endswith((".csv")) for f in new_files), (
        f"Expected some result files (.csv) among new files, got: {new_files}"
    )