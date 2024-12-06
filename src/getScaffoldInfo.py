

import numpy as np
import sys,os
from tqdm import tqdm
from rdkit import Chem, RDLogger
import pandas as pd
RDLogger.DisableLog("rdApp.*")
from spyrmsd import rmsd, molecule
import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

from rdkit.Chem import RemoveHs
def remove_all_hs(mol,santize=None):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    # if santize is not None:
    params.sanitize = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return RemoveHs(mol, params)
def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
        coords1,
        coords2,
        mol.atomicnums,
        mol2_atomicnums,
        mol.adjacency_matrix,
        mol2_adjacency_matrix,
            )
        return RMSD
def find_correct_index_match(
    ref_coord, scaff_coord, index_matches, eps=1e-6
):
    for im in index_matches:
        idx = np.asarray(im, dtype=np.compat.long)
        matched_coord = ref_coord[idx]
        rmsd = np.sqrt(np.sum(np.power(matched_coord - scaff_coord, 2)))
        if rmsd < eps:
            return idx
    return
def get_scaffold_from_index(ligand_mol, indices):
    from rdkit.Geometry.rdGeometry import Point3D

    new_mol = Chem.RWMol(Chem.Mol())
    new_conf = Chem.Conformer(len(indices))
    atom_map = {}
    for idx in indices:
        atom = ligand_mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)
        atom_pos = Point3D(*ligand_mol.GetConformer(0).GetPositions()[idx])
        new_conf.SetAtomPosition(atom_map[idx], atom_pos)

    indices = set(indices)
    for idx in indices:
        a = ligand_mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in indices:
                continue
            bond = ligand_mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():
                # print(a.GetIdx(), b.GetIdx(), bt)
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    scaff_mol = new_mol.GetMol()
    conf = scaff_mol.AddConformer(new_conf, assignId=True)
    return scaff_mol
def get_scaffold_pose(sacffold_smart,ref_mol):
    scaff_ref = Chem.MolFromSmarts(sacffold_smart)

    scaff_idx = ref_mol.GetSubstructMatch(scaff_ref)
    scaff_mol = get_scaffold_from_index(ref_mol, scaff_idx)

    Chem.SanitizeMol(scaff_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    frags = Chem.GetMolFrags(scaff_mol, asMols=True)
    mol = max(frags, key=lambda x: x.GetNumAtoms())
    return mol
    
def get_scaffold_smart(uniprot_id,serise_id,scaffold_info,serise_dir):
    
    serise_id_sdf_docking_pose = os.path.join(serise_dir, f'{uniprot_id}_{serise_id}_with_common_scaffold_docking_pose.sdf')
    selected_mols = Chem.SDMolSupplier(serise_id_sdf_docking_pose)

    # selected the cluster core in docking pose as the reference
    scaff_smart = scaffold_info[scaffold_info['SeriseID']==serise_id].Scaffold.iloc[0]
    scaff_ref = Chem.MolFromSmarts(scaffold_info[scaffold_info['SeriseID']==serise_id].Scaffold.iloc[0])

    scaffold_mols = []
    for ligand_mol in selected_mols:
        scaff_idx = ligand_mol.GetSubstructMatch(scaff_ref)
        scaff_mol = get_scaffold_from_index(ligand_mol, scaff_idx)
        if scaff_mol is None:
            continue
        Chem.SanitizeMol(scaff_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        scaffold_mols.append(scaff_mol)
    # check if the scaffold is the same
    if np.sum(np.array([mol.GetNumAtoms() for mol in scaffold_mols ]) == scaffold_mols[0].GetNumAtoms()) != len(scaffold_mols) or scaffold_mols[0].GetNumAtoms() == 0:
        print('csv Smarts cant match all refernce pose , regenerate smart string.....')
        print(f'{uniprot_id}_{serise_id} scaffold is not the same update it')
        from rdkit.Chem import rdFMCS
        mcs_result = rdFMCS.FindMCS(
            selected_mols,
            threshold=1,  # 要求所有当前分子包含此骨架
            atomCompare=rdFMCS.AtomCompare.CompareElements,  # 原子比较方式
            bondCompare=rdFMCS.BondCompare.CompareOrder,    # 键比较方式
            ringMatchesRingOnly=True,                       # 保证环完整性
            completeRingsOnly=True,                         # 仅匹配完整环
            timeout=30,                                      # 超时时间
        )
        # scaff_ref = Chem.MolFromSmarts(mcs_result.smartsString)
        # print('csv Smarts cant match all refernce pose , regenerate smart string.....')
        scaff_smart = mcs_result.smartsString
        print(f'New Smarts cant match all refernce pose , {scaff_smart}')
    return scaff_smart