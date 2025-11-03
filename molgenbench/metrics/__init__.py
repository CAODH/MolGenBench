# molgenbench/metrics/__init__.py
import importlib
import os
import pkgutil
from pathlib import Path

# 导出注册表
from .base import METRIC_REGISTRY

# 当前目录路径
package_dir = Path(__file__).parent

def auto_import_metrics():
    """
    自动扫描并导入 molgenbench.metrics 下所有模块，
    除 base.py 自身外，以触发 MetricBase 自动注册。
    """
    package_name = __name__  # 'molgenbench.metrics'

    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if is_pkg:
            continue
        if module_name == "base":
            continue

        full_module_name = f"{package_name}.{module_name}"

        try:
            importlib.import_module(full_module_name)
            print(f"[Metrics] Loaded: {full_module_name}")
        except Exception as e:
            print(f"[Metrics] Failed to load {full_module_name}: {e}")

# 自动导入（模块导入时即执行）
auto_import_metrics()

__all__ = ["METRIC_REGISTRY"]

