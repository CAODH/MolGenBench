import importlib
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]  # MolGenBench
TESTSAMPLES_DIR = ROOT / "TestSamples"
SAMPLE = "O14757"

@pytest.mark.integration
def test_compute_smiles_scaffold_writes_output(tmp_path):
    """
    Copy TestSamples/SAMPLE to a temporary location and run the real
    compute_smiles_scaffold_inchi_map(uniprot_id, reference_dir) against it.
    Assert that some new files are created in the temporary sample directory.

    This test performs real IO but keeps everything inside tmp_path to avoid
    modifying the repository/testdata.
    """
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