import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]  # MolGenBench folder
print("ROOT:", ROOT)
TESTSAMPLES_DIR = ROOT / "TestSamples"

def find_any_match(sample_dir: Path, patterns):
    for p in patterns:
        if list(sample_dir.glob(p)):
            return True
    return False

def sample_expected_patterns(sample_name: str):
    # 针对当前仓库示例，允许的文件名模式（更宽松以容错）
    ligand_patterns = [f"{sample_name}_lig*.sdf", "*lig*.sdf", "*.sdf"]
    pocket_patterns = [f"{sample_name}_pocket*.pdb", "*pocket*.pdb"]
    prep_patterns = [f"{sample_name}_prep*.pdb", "*prep*.pdb"]
    return ligand_patterns, pocket_patterns, prep_patterns

def collect_samples():
    assert TESTSAMPLES_DIR.exists(), f"TestSamples directory not found at {TESTSAMPLES_DIR}"
    samples = [p for p in TESTSAMPLES_DIR.iterdir() if p.is_dir()]
    assert samples, f"No sample directories found under {TESTSAMPLES_DIR}"
    return samples

def test_testsamples_directory_exists_and_has_samples():
    samples = collect_samples()
    # 如果至少找到一个样本就通过（更具体的结构由下个测试校验）
    assert len(samples) >= 1

@pytest.mark.parametrize("sample_dir", collect_samples())
def test_sample_structure_matches_example(sample_dir: Path):
    sample_name = sample_dir.name

    ligand_patterns, pocket_patterns, prep_patterns = sample_expected_patterns(sample_name)

    has_lig = find_any_match(sample_dir, ligand_patterns)
    assert has_lig, (
        f"Sample '{sample_name}' missing ligand .sdf file. "
        f"Expected patterns: {ligand_patterns} inside {sample_dir}"
    )

    has_pocket = find_any_match(sample_dir, pocket_patterns)
    assert has_pocket, (
        f"Sample '{sample_name}' missing pocket .pdb file. "
        f"Expected patterns: {pocket_patterns} inside {sample_dir}"
    )

    has_prep = find_any_match(sample_dir, prep_patterns)
    assert has_prep, (
        f"Sample '{sample_name}' missing prep .pdb file. "
        f"Expected patterns: {prep_patterns} inside {sample_dir}"
    )

    # 至少包含 reference_active_molecules 或 Round1 子目录（按当前示例定义）
    ref_dir = sample_dir / "reference_active_molecules"
    round1_dir = sample_dir / "Round1"
    assert ref_dir.exists() and round1_dir.exists(), (
        f"Sample '{sample_name}' should contain 'reference_active_molecules' or 'Round1' subdir. "
        f"Found: {list(sample_dir.iterdir())}"
    )