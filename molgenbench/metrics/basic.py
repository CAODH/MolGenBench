from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import MetricBase
from molgenbench.utils.sascore import compute_sa_score


def is_valid(mol):
    """
    Check if a molecule is valid, i.e., it has no disconnected fragments.
    """
    if mol is None:
        return False
    return '.' not in Chem.MolToSmiles(mol)


class ValidMetric(MetricBase):
    name = "Valid"
    def compute(self, record: MoleculeRecord):
        # Check both RDKit parsing and multi-fragment structure
        record.valid = is_valid(record.rdkit_mol)
        return record.valid

class QEDMetric(MetricBase):
    name = "QED"
    def compute(self, record: MoleculeRecord):
        if not record.valid:
            record.metadata[self.name] = None
            return None
        val = QED.qed(record.rdkit_mol)
        record.metadata[self.name] = val
        return val

class SAMetric(MetricBase):
    name = "SA"
    def compute(self, record: MoleculeRecord):
        if not record.valid:
            record.metadata[self.name] = None
            return None
        sa_score = compute_sa_score(record.rdkit_mol)
        record.metadata[self.name] = sa_score
        return sa_score