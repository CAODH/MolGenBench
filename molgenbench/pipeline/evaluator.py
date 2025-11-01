from typing import List, Dict, Any
from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.basic import ValidMetric, QEDMetric, SAMetric
from molgenbench.metrics.distribution import DiversityMetric, UniquenessMetric


class Evaluator:
    """
    Evaluator runs a set of registered metrics over a list of MoleculeRecord objects.

    It supports both:
      - per-molecule metrics (e.g., Valid, QED, SA)
      - dataset-level metrics (e.g., Diversity, Uniqueness)
    """

    def __init__(self, metrics=None, filter_invalid: bool = True, use_3d: bool = False):
        # Register default metrics if none provided
        self.metrics = metrics or [
            ValidMetric(),
            QEDMetric(),
            SAMetric(),
            UniquenessMetric(),
            DiversityMetric(),
        ]
        self.filter_invalid = filter_invalid
        self.use_3d = use_3d

    def evaluate(self, records: List[MoleculeRecord]):
        """
        Compute all registered metrics for a given set of MoleculeRecords.

        Args:
            records: list of MoleculeRecord objects

        Returns:
            results: dict of aggregated metric values (means, ratios, etc.)
            valid_records: list of valid molecules (after filtering)
        """
        if not records:
            return {"Valid_ratio": 0.0}, []

        # --- Step 1: Compute per-molecule metrics ---
        for record in records:
            for metric in self.metrics:
                # Only apply molecule-level metrics (skip distribution-level ones here)
                if metric.name.lower() in ["diversity", "uniqueness"]:
                    continue
                metric.compute(record)

        # --- Step 2: Filter valid molecules (optional) ---
        valid_records = (
            [r for r in records if getattr(r, "valid", False)]
            if self.filter_invalid
            else records
        )

        # --- Step 3: Compute dataset-level metrics (Diversity, Uniqueness) ---
        results: Dict[str, Any] = {}
        for metric in self.metrics:
            name = metric.name.lower()
            if name in ["diversity", "uniqueness"]:
                results.update(metric.compute(valid_records))

        # --- Step 4: Compute Valid ratio and means of other properties ---
        if len(records) > 0:
            # TODO: 下次改成denovo 1000，hit2lead 200
            results["Valid_ratio"] = len(valid_records) / 1000
        else:
            results["Valid_ratio"] = 0.0

        # Compute global means for per-molecule metrics
        qed_values = [r.metadata.get("QED") for r in valid_records if r.metadata.get("QED") is not None]
        sa_values = [r.metadata.get("SA") for r in valid_records if r.metadata.get("SA") is not None]

        results["QED_mean"] = sum(qed_values) / len(qed_values) if qed_values else 0.0
        results["SA_mean"] = sum(sa_values) / len(sa_values) if sa_values else 0.0

        return results, valid_records