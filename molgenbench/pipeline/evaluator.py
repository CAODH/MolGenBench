import os
import numpy as np
import pandas as pd
from rdkit import Chem
from typing import List, Dict, Any
from molgenbench.io.types import MoleculeRecord
from molgenbench.io.reader import read_sdf_to_records, attach_docked_molecules
from molgenbench.metrics.basic import ValidMetric, QEDMetric, SAMetric
from molgenbench.metrics.conformer import PoseBusterMetric, StrainEnergyMetrics, RMSDMetric
from molgenbench.metrics.distribution import DiversityMetric, UniquenessMetric


class Evaluator:
    """
    42
    Unified evaluator that handles molecule-level and dataset-level evaluation,
    grouped by Uniprot or Series depending on task type.
    """

    def __init__(self, metric_names: List[str] = None):
        self.metric_names = metric_names or ["Validity", "QED", "SA", "Uniqueness", "Diversity", "PoseBuster", "StrainEnergy", "RMSD"]
        self.metric_map = {
            "Validity": ValidMetric(),
            "QED": QEDMetric(),
            "SA": SAMetric(),
            "Uniqueness": UniquenessMetric(),
            "Diversity": DiversityMetric(),
            
            "PoseBuster": PoseBusterMetric(),
            "StrainEnergy": StrainEnergyMetrics(),
            "RMSD": RMSDMetric(),
        }

        # 按类型分类 metric（单分子级 vs 数据集级）
        self.molecule_metrics = [
            self.metric_map[n] for n in self.metric_names if n in ["Validity", "QED", "SA", "PoseBuster", "StrainEnergy", "RMSD"]
        ]
        self.dataset_metrics = [
            self.metric_map[n] for n in self.metric_names if n in ["Diversity", "Uniqueness"]
        ]

    
    def _save_metrics_to_csv(
        self,
        molecule_records: List[MoleculeRecord],
        dataset_metrics: List[Dict[str, Any]],
        uniprot: str,
        series: str,
        save_dir: str,
        model_name: str,
        mode: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        molecule_metric_data = {}
        dataset_metric_data = {}

        # 1️⃣ 分子级指标
        for r in molecule_records:
            smiles = getattr(r, "smiles", None)
            mol_id = getattr(r, "id", None)
            num_rotatable_bonds = getattr(r, "num_rotatable_bonds", None)

            for metric_name, value in r.metadata.items():
                if value is None:
                    continue
                if metric_name not in molecule_metric_data:
                    molecule_metric_data[metric_name] = []

                base_info = {
                    "id": mol_id, 
                    "smiles": smiles, 
                    "num_rotatable_bonds": num_rotatable_bonds, 
                    "uniprot": uniprot, 
                    "series": series,
                }

                # 如果是 dict，展开
                if isinstance(value, dict):
                    base_info.update(value)
                else:
                    base_info[metric_name] = value

                molecule_metric_data[metric_name].append(base_info)

        # 2️⃣ 数据集级指标
        for k, v in dataset_metrics.items():
            if k not in dataset_metric_data:
                dataset_metric_data[k] = []
            dataset_metric_data[k].append({"uniprot": uniprot, "series": series, k: v})

        # 3️⃣ 保存 molecule-level
        for metric_name, rows in molecule_metric_data.items():
            if metric_name not in self.metric_names:
                continue
            df = pd.DataFrame(rows)
            out_path = os.path.join(save_dir, f"{metric_name}.csv")
            df.to_csv(out_path, index=False)

        # 4️⃣ 保存 dataset-level
        for metric_name, rows in dataset_metric_data.items():
            if metric_name not in self.metric_names:
                continue
            df = pd.DataFrame(rows)
            out_path = os.path.join(save_dir, f"{metric_name}.csv")
            df.to_csv(out_path, index=False)
            
    
    # ---------------------------- #
    #   Molecule-level evaluation
    # ---------------------------- #
    def evaluate_molecule_metrics(self, records: List[MoleculeRecord]) -> List[MoleculeRecord]:
        """Compute metrics like Valid, QED, SA for each molecule."""
        for record in records:
            for metric in self.molecule_metrics:
                metric.compute(record)
        return records

    # ---------------------------- #
    #   Dataset-level evaluation
    # ---------------------------- #
    def evaluate_dataset_metrics(self, records: List[MoleculeRecord]) -> Dict[str, Any]:
        """Compute dataset-level metrics like Diversity, Uniqueness."""
        results = {}
        for metric in self.dataset_metrics:
            results.update(metric.compute(records))
        return results

    # ---------------------------- #
    #   顶层 pipeline
    # ---------------------------- #
    def run(
        self, 
        root_dir: str, 
        model_name: str, 
        round: str,
        mode: str = "De_novo_Results"
    ) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Args:
            root_dir: directory containing Uniprot folders
            mode: 'denovo' or 'hit2lead'
        Returns:
            pd.DataFrame of aggregated results
        """

        for uniprot in os.listdir(root_dir):
            uniprot_path = os.path.join(root_dir, uniprot, round, mode)
            prot_path = os.path.join(root_dir, uniprot, f"{uniprot}_prep.pdb")
            ref_active_path = os.path.join(uniprot_path, "reference_active_molecules", f"{uniprot}_reference_active_molecules.sdf")

            if mode == "De_novo_Results":
                sdf_path = os.path.join(uniprot_path, model_name, f"{uniprot}_{model_name}.sdf")
                if not os.path.exists(sdf_path):
                    continue
                
                docked_path = sdf_path.replace(".sdf", "_vina_docked.sdf")
                records = read_sdf_to_records(sdf_path, prot_path, ref_active_path)
                records = attach_docked_molecules(records, docked_path)
                
                records = self.evaluate_molecule_metrics(records)
                dataset_results = self.evaluate_dataset_metrics(records)
                self._save_metrics_to_csv(
                    records,
                    dataset_results,
                    save_dir=os.path.join(uniprot_path, model_name, "results"),
                    uniprot=uniprot,
                    series=None,
                    model_name=model_name,
                    mode=mode,
                )

            elif mode == "Hit_to_Lead_Results":
                for series_id in os.listdir(uniprot_path):
                    series_path = os.path.join(uniprot_path, series_id)
                    
                    sdf_path = os.path.join(series_path, model_name, f"{uniprot}_{series_id}_{model_name}.sdf")
                    if not os.path.exists(sdf_path):
                        continue
                    
                    docked_path = sdf_path.replace(".sdf", "_vina_docked.sdf")
                    records = read_sdf_to_records(sdf_path, prot_path, ref_active_path)
                    records = attach_docked_molecules(records, docked_path)
                    
                    records = self.evaluate_molecule_metrics(records)
                    dataset_results = self.evaluate_dataset_metrics(records)
                    self._save_metrics_to_csv(
                        records,
                        dataset_results,
                        save_dir=os.path.join(series_path, model_name, "results"),
                        uniprot=uniprot,
                        series=series_id,
                        model_name=model_name,
                        mode=mode,
                    )
