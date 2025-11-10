import numpy as np
from itertools import combinations
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.metrics.basic import is_valid
from molgenbench.io.types import MoleculeRecord

class UniquenessMetric(Metric):
    """
    Compute the fraction of unique SMILES strings among all valid molecules.
    """
    name = "Uniqueness"

    def compute(self, records: List[MoleculeRecord]) -> Dict[str, float]:
        smiles_list = [r.smiles for r in records if is_valid(r.rdkit_mol)]
        if not smiles_list:
            return {self.name: None}
        unique_smiles = set(smiles_list)
        uniqueness = len(unique_smiles) / len(smiles_list)
        return {self.name: uniqueness}


class DiversityMetric(Metric):
    """
    Compute molecular diversity for a single target set using
    average pairwise Tanimoto distance over Morgan fingerprints.
    (1 - mean similarity)
    
    De novo results is averaged by Uniprot.
    Hit2Lead results is averaged by Series.
    """
    name = "Diversity"

    def __init__(self, radius: int = 2, nBits: int = 2048):
        self.radius = radius
        self.nBits = nBits

    def _calculate_morgan_fingerprint(self, mol):
        """Return the Morgan fingerprint bit vector for a molecule."""
        if mol is None:
            return None
        try:
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
        except Exception:
            return None

    def compute(self, records: List[MoleculeRecord]) -> Dict[str, float]:
        """Compute the diversity value for all valid molecules."""
        # Extract valid SMILES
        smiles_list = [r.smiles for r in records if is_valid(r.rdkit_mol)]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        mols = [mol for mol in mols if mol is not None]

        if len(mols) == 0:
            return {self.name: 0.0}
        if len(mols) == 1:
            return {self.name: 1.0}

        fps = [self._calculate_morgan_fingerprint(mol) for mol in mols]

        tanimoto_similarities = [
            TanimotoSimilarity(f1, f2)
            for f1, f2 in combinations(fps, 2)
        ]
        diversity = 1 - np.mean(tanimoto_similarities)
        return {self.name: diversity}