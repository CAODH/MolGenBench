import numpy as np
from itertools import combinations
from typing import List, Dict
from collections import Counter

from EFGs import mol2frag
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from scipy.spatial.distance import jensenshannon

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
    

class MotifDistMetric(Metric):
    """
    Compute the distribution of molecular motifs (Atom, Ring, Function Group) in a set of molecules.
    Outputs a dictionary with fragment counts.
    """
    name = "MotifDistribution"

    def _getAtomType(self, mol: Chem.Mol):
        return [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    def _getRingType(self, mol: Chem.Mol):
        """
        Statistical analysis of all ring sizes in the molecule
        """
        ring_info = mol.GetRingInfo()
        ring_types = [len(r) for r in ring_info.AtomRings()]
        return ring_types

    def _getFuncGroup(self, mol: Chem.Mol):
        try:
            fgs, _ = mol2frag(mol)
        except:
            return []
        return fgs
    
    def _getAllTypes(self, mol: Chem.Mol):
        atom_types = self._getAtomType(mol)
        ring_types = self._getRingType(mol)
        func_groups = self._getFuncGroup(mol)
        return atom_types, ring_types, func_groups
    
    def _count_and_filter(self, items, min_freq=5):
        count = Counter(items)
        return {k: v for k, v in count.items() if v >= min_freq}
    
    def _evalSubTypeDist(self, ref_dict, gen_dict):
        """
        ref_dict: {motif: count}
        gen_dict: {motif: count}
        """

        total_ref_types = sum(ref_dict.values())
        total_gen_types = sum(gen_dict.values())

        ref_dist = {}
        gen_dist = {}

        for k in ref_dict:
            ref_dist[k] = ref_dict[k] / total_ref_types

            if k in gen_dict:
                gen_dist[k] = gen_dict[k] / total_gen_types
            else:
                gen_dist[k] = 0

        # convert to arrays
        r = np.array(list(ref_dist.values()))
        g = np.array(list(gen_dist.values()))

        r = np.where(r == 0, 1e-10, r)
        g = np.where(g == 0, 1e-10, g)

        jsd = jensenshannon(r, g)

        return jsd

    def compute(
        self, 
        ref_records: List[MoleculeRecord],
        gen_records: List[MoleculeRecord]
    ) -> Dict[str, Dict]:
        """
        Compute motif distributions between reference and generated molecules
        and return JSD + MAE per motif category.

        Args:
            ref_records: list of MoleculeRecord (actives)
            gen_records: list of MoleculeRecord (generated)
        """

        # ---------------- collect motifs ----------------
        ref_atoms, ref_rings, ref_fgs = [], [], []
        gen_atoms, gen_rings, gen_fgs = [], [], []

        for r in ref_records:
            mol = r.rdkit_mol
            if is_valid(mol):
                a, rings, fgs = self._getAllTypes(mol)
                ref_atoms.extend(a)
                ref_rings.extend(rings)
                ref_fgs.extend(fgs)

        for r in gen_records:
            mol = r.rdkit_mol
            if is_valid(mol):
                a, rings, fgs = self._getAllTypes(mol)
                gen_atoms.extend(a)
                gen_rings.extend(rings)
                gen_fgs.extend(fgs)

        # ---------------- count & filter ----------------
        ref_atom_count = self._count_and_filter(ref_atoms)
        gen_atom_count = self._count_and_filter(gen_atoms)

        ref_ring_count = self._count_and_filter(ref_rings)
        gen_ring_count = self._count_and_filter(gen_rings)

        ref_fg_count = self._count_and_filter(ref_fgs)
        gen_fg_count = self._count_and_filter(gen_fgs)

        # ---------------- final metric set ----------------
        result = {
            "Atom Type JS": self._evalSubTypeDist(ref_atom_count, gen_atom_count),
            "Ring Type JS": self._evalSubTypeDist(ref_ring_count, gen_ring_count),
            "Functional Group JS": self._evalSubTypeDist(ref_fg_count, gen_fg_count),
        }

        return result