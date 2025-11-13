import importlib
import shutil
from pathlib import Path
import pytest
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # MolGenBench
TESTSAMPLES_DIR = ROOT / "TestSamples"

@pytest.mark.integration
def test_get_medchem_filter_info_writes_csv(tmp_path):
    """
    Find a sample .sdf in TestSamples, copy it to tmp_path and run
    chem_filters.get_medchem_filter_info on the copied file.
    Assert that a *_medchem_filter_results.csv file is created and readable.
    """
    # locate a .sdf sample
    sdf_path = next(TESTSAMPLES_DIR.rglob("*.sdf"), None)
    if sdf_path is None:
        pytest.skip(f"No .sdf files found under {TESTSAMPLES_DIR}")

    mod = importlib.import_module("molgenbench.preprocess.chem_filters")
    if not hasattr(mod, "get_medchem_filter_info"):
        pytest.skip("chem_filters.get_medchem_filter_info not found")

    # prepare destination and copy file
    dest_dir = tmp_path / "sdf_sample"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_sdf = dest_dir / sdf_path.name
    shutil.copy(str(sdf_path), str(dest_sdf))

    # record before files
    before = {p.relative_to(dest_dir) for p in dest_dir.rglob("*") if p.is_file()}

    # run the real function
    mod.get_medchem_filter_info(str(dest_sdf))

    # record after files
    after = {p.relative_to(dest_dir) for p in dest_dir.rglob("*") if p.is_file()}
    new_files = after - before
    assert new_files, f"No new files created; before={len(before)} after={len(after)}"

    # expected CSV name
    base_name = dest_sdf.stem
    expected_csv = dest_dir / f"{base_name}_medchem_filter_results.csv"
    assert expected_csv.exists(), f"Expected CSV not found: {expected_csv}"

    # try reading CSV with pandas and basic sanity checks
    df = pd.read_csv(expected_csv)
    assert not df.empty, "Result CSV is empty"
    assert "smiles" in df.columns, "Expected 'smiles' column in result CSV"