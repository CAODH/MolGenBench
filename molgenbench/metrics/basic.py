from rdkit import Chem
from rdkit.Chem import QED
from medchem.structural import CommonAlertsFilters, NIBRFilters
from medchem.structural.lilly_demerits import LillyDemeritsFilters

from molgenbench.io.types import MoleculeRecord
from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.utils.sascore import compute_sa_score


def is_valid(mol):
    """
    Check if a molecule is valid, i.e., it has no disconnected fragments.
    """
    if mol is None:
        return False
    return '.' not in Chem.MolToSmiles(mol)


class ValidMetric(Metric):
    name = "Validity"
    def compute(self, record: MoleculeRecord):
        # Check both RDKit parsing and multi-fragment structure
        record.valid = is_valid(record.rdkit_mol)
        record.metadata[self.name] = record.valid
        return record.valid

class QEDMetric(Metric):
    name = "QED"
    def compute(self, record: MoleculeRecord):
        if not is_valid(record.rdkit_mol):
            record.metadata[self.name] = None
            return None
        val = QED.qed(record.rdkit_mol)
        record.metadata[self.name] = val
        return val

class SAMetric(Metric):
    name = "SA"
    def compute(self, record: MoleculeRecord):
        if not is_valid(record.rdkit_mol):
            record.metadata[self.name] = None
            return None
        sa_score = compute_sa_score(record.rdkit_mol)
        record.metadata[self.name] = sa_score
        return sa_score
    
class ChemFilterMetric(Metric):
    """
    Compute chemical filter metric for a given molecule.
    Outputs True if the molecule passes the filters, False otherwise.
    """
    name = "ChemFilter"

    def passes_chem_filters(self, mol):
        common_filter = CommonAlertsFilters()
        nibr_filter = NIBRFilters()
        lilly_filter = LillyDemeritsFilters()

        results_common = common_filter([Chem.MolToSmiles(mol)])
        results_nibr = nibr_filter([Chem.MolToSmiles(mol)])
        results_lilly = lilly_filter([Chem.MolToSmiles(mol)])

        # A molecule passes the filters only if it passes all individual filters
        return results_common["pass_filter"][0] and results_nibr["pass_filter"][0] and results_lilly["pass_filter"][0]
    

    def compute(self, record: MoleculeRecord):
        """
        Compute chemical filter metric for the molecule.

        Args:
            record: MoleculeRecord object
        """

        passes_filters = False
        mol = record.rdkit_mol
        
        if not is_valid(mol):
            record.metadata[self.name] = passes_filters
            return record.metadata[self.name]
        
        try:
            passes_filters = self.passes_chem_filters(mol)
        except:
            passes_filters = False

        record.metadata[self.name] = passes_filters
        return passes_filters