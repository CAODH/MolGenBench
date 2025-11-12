import importlib
import sys
import shutil
from pathlib import Path
import pytest
from rdkit import Chem
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]  # MolGenBench 根目录
sys.path.insert(0, str(ROOT))
TESTSAMPLES_DIR = ROOT / "TestSamples"
SAMPLE = "O14757"  # 示例 UniProt ID


@pytest.mark.integration
def test_vina_docking_hit2lead_process_row(tmp_path):
    """
    Integration test for vina_docking.py in Hit-to-Lead mode.
    It copies TestSamples/O14757 to a temporary directory,
    sets up a minimal Hit-to-Lead structure, runs get_prepared_df(),
    executes process_row() in parallel, and checks that docked
    SDFs and result CSV are generated successfully.
    """

    # === Step 1️⃣: 检查样例存在 ===
    src_sample_dir = TESTSAMPLES_DIR / SAMPLE
    if not src_sample_dir.exists():
        pytest.skip(f"Sample directory not found: {src_sample_dir}")

    # === Step 2️⃣: 导入模块 ===
    mod = importlib.import_module("vina_docking")
    utils = importlib.import_module("molgenbench.vina.vina_utils")

    required_funcs = ["get_prepared_df", "merge_docked_sdfs", "get_logger"]
    for f in required_funcs:
        if not hasattr(mod, f):
            pytest.skip(f"{f} not found in vina_docking module")

    if not hasattr(utils, "process_row"):
        pytest.skip("process_row() not found in vina_utils")

    # === Step 3️⃣: 准备临时目录 ===
    dest_root = tmp_path / "TestSamples"
    dest_sample = dest_root / SAMPLE
    shutil.copytree(src_sample_dir, dest_sample)

    # === 清理旧的 vina_docked 输出，防止干扰本次测试 ===
    old_docked_files = list(dest_sample.rglob("*_vina_docked.sdf"))
    for f in old_docked_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete old docked file: {f}, error={e}")

    # 路径配置
    data_path = str(dest_root)
    output_path = str(tmp_path / "output")
    model_name = "DeleteHit2Lead(CrossDock)_Hit_to_Lead"
    round_name = "Round1"
    mode = "Hit_to_Lead_Results"
    series_id = "Sries14139"

    # === Step 4️⃣: 创建 Hit-to-Lead 模拟目录结构 ===
    # <data>/<uniprot>/<round>/Hit_to_Lead_Results/<series_id>/<model>/<uniprot>_<series_id>_<model>.sdf
    model_dir = dest_sample / round_name / mode / series_id / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # === Step 5️⃣: 执行 get_prepared_df() ===
    logger = mod.get_logger(str(Path(output_path) / "vina_hit2lead.log"))
    prepared_df_path = Path(output_path) / f"{model_name}_{mode}_prepared_df_{round_name}.pkl"

    df = mod.get_prepared_df(
        data_path=data_path,
        output_path=output_path,
        model_name=model_name,
        prepared_df_path=str(prepared_df_path),
        logger=logger,
        round_name=round_name,
        mode=mode,
    )

    assert not df.empty, "Prepared DataFrame is empty"
    assert prepared_df_path.exists(), "Prepared DataFrame .pkl file missing"

    # === Step 6️⃣: 调用 process_row() 执行 docking ===
    results = Parallel(n_jobs=1)(
        delayed(utils.process_row)(row) for _, row in df.iterrows()
    )

    for idx, affinity, docked_path in results:
        df.at[idx, "affinity"] = affinity
        df.at[idx, "docked_path"] = docked_path

    results_df_path = Path(output_path) / f"{model_name}_{mode}_vina_results_{round_name}.csv"
    df.to_csv(results_df_path, index=False)
    assert results_df_path.exists(), "Vina result CSV not created"

    # === Step 7️⃣: 合并对接结果 ===
    mod.merge_docked_sdfs(
        data_path=data_path,
        output_path=output_path,
        model_name=model_name,
        round_name=round_name,
        mode=mode,
    )

    merged_sdf_path = (
        dest_sample
        / round_name
        / mode
        / series_id
        / model_name
        / f"{SAMPLE}_{series_id}_{model_name}_vina_docked.sdf"
    )

    # === Step 8️⃣: 验证输出文件 ===
    assert merged_sdf_path.exists(), f"Merged SDF not found: {merged_sdf_path}"

    mols = [m for m in Chem.SDMolSupplier(str(merged_sdf_path)) if m is not None]
    assert mols, "Merged SDF contains no valid molecules"
