from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import Metric
from molgenbench.utils.sascore import compute_sa_score
class ValidMetric(Metric):
    name = "Valid"
    def compute(self, record: MoleculeRecord):
        # Check both RDKit parsing and multi-fragment structure
        if record.rdkit_mol is None:
            record.valid = False
        else:
            record.valid = '.' not in record.smiles
        return record.valid

class QEDMetric(Metric):
    name = "QED"
    def compute(self, record: MoleculeRecord):
        if not record.valid:
            record.metadata[self.name] = None
            return None
        val = QED.qed(record.rdkit_mol)
        record.metadata[self.name] = val
        return val

class SAMetric(Metric):
    name = "SA"
    def compute(self, record: MoleculeRecord):
        if not record.valid:
            record.metadata[self.name] = None
            return None
        sa_score = compute_sa_score(record.rdkit_mol)
        record.metadata[self.name] = sa_score
        return sa_score