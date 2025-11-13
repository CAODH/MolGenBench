# molgenbench/metrics/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from molgenbench.io.types import MoleculeRecord
import logging
import traceback

METRIC_REGISTRY: Dict[str, Type["MetricBase"]] = {}


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, records, **kwargs):
        pass

class MetricBase(ABC):
    """所有指标的抽象基类，带自动注册、异常捕获、并行计算能力。"""

    name: str = "BaseMetric"
    requires_3d: bool = False
    parallel: bool = False     # 是否允许并行

    # ---------- 自动注册 ----------
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name != "BaseMetric":
            METRIC_REGISTRY[cls.name.lower()] = cls
            logging.getLogger(cls.__name__).info(f"Registered metric: {cls.name}")

    # ---------- 初始化 ----------
    def __init__(self, requires_3d: Optional[bool] = None, parallel: Optional[bool] = None):
        if requires_3d is not None:
            self.requires_3d = requires_3d
        if parallel is not None:
            self.parallel = parallel
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------- 抽象核心方法 ----------
    ## 强制继承基类的方法提供这个方法实现
    @abstractmethod
    def compute_one(self, record: MoleculeRecord, ref_record: Optional[MoleculeRecord] = None) -> Dict[str, Any]:
        """计算单个分子的指标（子类必须实现）"""
        pass

    # ---------- 批量计算逻辑 ----------
    # 批量计算逻辑需要根据计算类别可能需要重写，但是不做@abstractmethod 硬性要求
    def compute(self, records: List[MoleculeRecord], ref_records: Optional[List[MoleculeRecord]] = None) -> Dict[str, Any]:
        try:
            valid_records = self.filter_valid(records)
            if not valid_records:
                self.logger.warning(f"No valid molecules for {self.name}")
                return {self.name: None}

            if self.parallel:
                results = self._parallel_compute(valid_records, ref_records)
            else:
                results = [self._safe_compute_one(r, ref_records) for r in valid_records]

            return self.aggregate_results(results)

        except Exception as e:
            self.logger.error(f"Error computing {self.name}: {e}")
            self.logger.debug(traceback.format_exc())
            return {self.name: None}

    # ---------- 并行计算 ----------
    def _parallel_compute(self, records: List[MoleculeRecord], ref_records: Optional[List[MoleculeRecord]]):
        executor_cls = ProcessPoolExecutor if self.requires_3d else ThreadPoolExecutor
        results = []
        with executor_cls() as executor:
            futures = {
                executor.submit(self._safe_compute_one, r, ref_records): r.id for r in records
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    self.logger.warning(f"Parallel task failed for {futures[fut]}: {e}")
        return results

    # ---------- 安全计算单个分子 ----------
    def _safe_compute_one(self, record: MoleculeRecord, ref_records: Optional[List[MoleculeRecord]] = None) -> Dict[str, Any]:
        try:
            result = self.compute_one(record, ref_records)
            if isinstance(result, dict):
                record.metadata.update(result)
            return result
        except Exception as e:
            self.logger.debug(f"Failed to compute {self.name} for {record.id}: {e}")
            return {}

    # ---------- 结果聚合 ----------
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """默认行为：对每个键求平均，可在子类重写"""
        summary = {}
        if not results:
            return {self.name: None}

        # 收集每个字段的值
        keys = {k for r in results for k in r.keys()}
        for k in keys:
            vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
            summary[k] = self.safe_mean(vals)
        return summary

    # ---------- 通用静态工具 ----------
    @staticmethod
    def filter_valid(records: List[MoleculeRecord]) -> List[MoleculeRecord]:
        return [r for r in records if r.rdkit_mol is not None]

    @staticmethod
    def safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def describe_metric(result: Dict[str, Any]) -> str:
        return ", ".join(f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in result.items())

    @classmethod
    def info(cls) -> str:
        return f"{cls.__name__}(requires_3d={cls.requires_3d}, parallel={cls.parallel})"

    # ---------- 统一调用入口 ----------
    def __call__(self, records: List[MoleculeRecord], ref_records: Optional[List[MoleculeRecord]] = None) -> Dict[str, Any]:
        self.logger.info(f"Running metric: {self.name}")
        return self.compute(records, ref_records)
