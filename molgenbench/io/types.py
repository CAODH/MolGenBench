from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from rdkit import Chem

@dataclass
class MoleculeRecord:
    """
    Unified data container for a molecule used throughout the benchmark pipeline.
    It stores identifiers, molecular representations, optional 3D conformers,
    and computed properties in a metadata dictionary.
    """
    id: str
    smiles: str
    rdkit_mol: Optional[Chem.Mol] = None
    num_rotatable_bonds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional validity flag (set after ValidMetric)
    valid: Optional[bool] = None

    def set_metric(self, name: str, value: Any):
        self.metadata[name] = value

    def get_metric(self, name: str, default=None):
        return self.metadata.get(name, default)