import numpy as np
from pathlib import Path
from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from molgenbench.metrics.base import MetricBase, Metric
from molgenbench.metrics.basic import is_valid
from molgenbench.io.types import MoleculeRecord

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