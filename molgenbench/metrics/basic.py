from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import MetricBase
from molgenbench.utils.sascore import compute_sa_score
class ValidMetric(MetricBase):
    name = "Valid"
    def compute(self, record: MoleculeRecord):
        # Check both RDKit parsing and multi-fragment structure
        if record.rdkit_mol is None:
            record.valid = False
        else:
            record.valid = '.' not in record.smiles
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