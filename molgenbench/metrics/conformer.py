import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.metrics.basic import is_valid
from molgenbench.io.types import MoleculeRecord

import prolif as plf
from posebusters import PoseBusters
from posecheck import PoseCheck
from spyrmsd import molecule
from spyrmsd import rmsd as spy_rmsd


class PoseBusterMetric(Metric):
    """
    Metric wrapper for PoseBusters evaluation.
    Runs PoseBusters on predicted ligand poses against the true reference
    and outputs a CSV summary with the geometric validation results.
    """
    name = "PoseBuster"

    def __init__(self, config: str = "dock"):
        self.config = config
        self.buster = PoseBusters(config=config)

    def compute(
        self,
        record: MoleculeRecord,
    ):
        """
        Run PoseBusters on an SDF/PDB input pair and store results to CSV.

        Args:
            records: MoleculeRecord object
            mol_cond: Path to receptor PDB file (used for context)
        """
        
        mol_pred = record.rdkit_mol
        if not is_valid(mol_pred):
            record.metadata[self.name] = None
            return None
        
        mol_cond = record.metadata.get("protein_path", None)
        
        try:
            df = self.buster.bust(
                mol_pred=mol_pred,
                mol_true=None,
                mol_cond=mol_cond,
            ).droplevel("file")
            
            posebuster_results = df.iloc[0].to_dict()
        except:
            posebuster_results = {}
        record.metadata[self.name] = posebuster_results
        return posebuster_results
    

class StrainEnergyMetrics(Metric):
    """
    Compute strain energy metrics for a given molecule using PoseCheck.
    Outputs total strain energy and per-atom strain energy.
    """
    name = "StrainEnergy"

    def compute(self, record: MoleculeRecord):
        """
        Compute strain energy metrics for the molecule.

        Args:
            record: MoleculeRecord object
        """
        strain = None
        mol = record.rdkit_mol
        
        if not is_valid(mol):
            record.metadata[self.name] = strain
            return record.metadata[self.name]
        
        try:
            pc = PoseCheck()
            pc.load_ligands_from_mols([mol])
            strain = pc.calculate_strain_energy()[0]
        except:
            strain = None

        record.metadata[self.name] = strain
        return strain
    

class ClashScoreMetric(Metric):
    """
    Compute clash score metric for a given molecule using PoseCheck.
    Outputs the clash score value.
    """
    name = "ClashScore"
    
    def _filterMol(self, mol: Chem.Mol) -> bool:
        """
        Filter molecule based on conformer.
        """
        if not is_valid(mol):
            return False
        
        if mol.GetNumAtoms() > 0 and not np.isnan(mol.GetConformer().GetPositions()).any():
            return True
        else:
            return False

    def compute(self, record: MoleculeRecord):
        """
        Compute clash score metric for the molecule.

        Args:
            record: MoleculeRecord object
        """
        clash_score = None
        mol = record.rdkit_mol
        
        if not self._filterMol(mol):
            record.metadata[self.name] = clash_score
            return record.metadata[self.name]
        
        mol = Chem.AddHs(mol, addCoords=True)
        protein_path = record.metadata.get("protein_path", None)
        
        try:
            pc = PoseCheck()
            pc.load_ligands_from_mols([mol])
            pc.load_protein_from_pdb(protein_path)
            clash_score = pc.calculate_clashes()[0]
        except:
            clash_score = None

        record.metadata[self.name] = clash_score
        return clash_score


class RMSDMetric(Metric):
    """
    Compute the redocking RMSD between a predicted and a reference molecule.

    This metric first tries to compute the symmetry-corrected RMSD (using spyrmsd),
    and falls back to a simple Cartesian RMSD if symmetry alignment fails.
    The result is stored in record.metadata["RMSD"].
    """

    name = "RMSD"

    def _symmetry_rmsd(self, mol_pred: Chem.Mol, mol_ref: Chem.Mol) -> float:
        """Compute symmetry-corrected RMSD using spyrmsd."""
        try:
            mol_ref = molecule.Molecule.from_rdkit(mol_ref)
            mol_pred = molecule.Molecule.from_rdkit(mol_pred)
            return spy_rmsd.symmrmsd(
                mol_ref.coordinates,
                mol_pred.coordinates,
                mol_ref.atomicnums,
                mol_pred.atomicnums,
                mol_ref.adjacency_matrix,
                mol_pred.adjacency_matrix,
            )
        except Exception as e:
            print(f"[RMSDMetric] Symmetry RMSD failed: {e}")
            return np.nan

    def compute(self, record: MoleculeRecord) -> Dict[str, Any]:
        """
        Compute RMSD between this record's molecule and the reference.

        Args:
            record: MoleculeRecord object with rdkit_mol

        Returns:
            dict: { "RMSD": value }
        """
        mol_ref = record.rdkit_mol
        mol_pred = record.metadata.get("docked_mol", None)
        if mol_pred is None or mol_ref is None:
            record.metadata[self.name] = np.nan
            return np.nan

        # Try symmetry RMSD first, fall back if needed
        rmsd_val = self._symmetry_rmsd(mol_pred, mol_ref)

        record.metadata[self.name] = rmsd_val
        return rmsd_val
    

class InteractionScoreMetric(Metric):
    """
    Compute interaction score metric for a given molecule.
    Outputs the interaction score value.
    """
    name = "InteractionScore"

    
    def _filterMol(self, mol: Chem.Mol) -> bool:
        """
        Filter atoms in a molecule based on their atomic symbol.
        """
        if not is_valid(mol):
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'As':
                return False
        
        if mol.GetNumAtoms() > 0 and not np.isnan(mol.GetConformer().GetPositions()).any():
            return True
        else:
            return False
        
    
    def _nonBondInteractions(self, mol_lig: Chem.Mol, mol_pro: Chem.Mol) -> pd.DataFrame:
        """Compute interaction score using ProLIF."""
        
        all_interactions = plf.Fingerprint.list_available()
        fp = plf.Fingerprint(interactions=all_interactions)
        
        fp.run_from_iterable([mol_lig], mol_pro, progress=False, n_jobs=1)
        df = fp.to_dataframe()
        return df

    def compute(self, record: MoleculeRecord):
        """
        Compute interaction score metric for the molecule.

        Args:
            record: MoleculeRecord object
        """
        interaction_score = None
        mol = record.rdkit_mol
        
        if not self._filterMol(mol):
            record.metadata[self.name] = interaction_score
            return record.metadata[self.name]
        
        protein_path = record.metadata.get("protein_path", None)
        
        mol = Chem.AddHs(mol, addCoords=True)
        mol_pro = Chem.MolFromPDBFile(protein_path, removeHs=False)
        
        mol_lig = plf.Molecule.from_rdkit(mol)
        mol_pro = plf.Molecule(mol_pro)
        
        try:
            results = self._nonBondInteractions(mol_lig, mol_pro)
            interaction_score = results.to_dict()
        except:
            interaction_score = None

        record.metadata[self.name] = interaction_score
        return interaction_score