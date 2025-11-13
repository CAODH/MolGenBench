

import numpy as np
import sys,os
from tqdm import tqdm
from rdkit import Chem, RDLogger
import pandas as pd
RDLogger.DisableLog("rdApp.*")
import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass
# from rdkit.Geometry.rdGeometry import Point3D
from rdkit import Geometry


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

    new_mol = Chem.RWMol(Chem.Mol())
    new_conf = Chem.Conformer(len(indices))
    atom_map = {}
    for idx in indices:
        atom = ligand_mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)
        atom_pos = Geometry.Point3D(*ligand_mol.GetConformer(0).GetPositions()[idx])
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

def mapSmartMol(scaff_mol):
    new_mol = Chem.RWMol(Chem.Mol())
## map sacff_mol to new_mol
    for atom in scaff_mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_mol.AddAtom(new_atom)
    for bond in scaff_mol.GetBonds():
        new_mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    new_mol = new_mol.GetMol()
    return new_mol
def get_scaffold_pose(sacffold_smart,ref_mol):
    scaff_ref = Chem.MolFromSmarts(sacffold_smart)

    scaff_idx = ref_mol.GetSubstructMatch(scaff_ref)
    scaff_mol = get_scaffold_from_index(ref_mol, scaff_idx)

    Chem.SanitizeMol(scaff_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    frags = Chem.GetMolFrags(scaff_mol, asMols=True)
    mol = max(frags, key=lambda x: x.GetNumAtoms())
    return mol

def get_scaffold_pose_new(sacffold_smart,ref_mol):

    # try:
    scaff_ref = Chem.MolFromSmarts(sacffold_smart)
    # 先match下
    
    # ref_mol.UpdatePropertyCache(strict=False)
    # scaff_ref.UpdatePropertyCache(strict=False)
    matches = ref_mol.GetSubstructMatches(scaff_ref)
    
    if len(matches) < 1:
        raise Exception('Could not find scaffold matches')
    if len(matches) > 1:
        print('Found multiple scaffold matches')
    match = matches[0]
    # update the scaffold with the reference molecule
    for i, atom in enumerate(scaff_ref.GetAtoms()):
        # atom.SetSymbol(ref_mol.GetAtoms()[match[i]].GetSymbol())
        atom.SetFormalCharge(ref_mol.GetAtoms()[match[i]].GetFormalCharge())
        atom.SetNumExplicitHs(ref_mol.GetAtoms()[match[i]].GetNumExplicitHs())
    scaff_ref = mapSmartMol(scaff_ref)
    # raise Exception(Chem.MolToSmiles(scaff_ref))

    # scaff_ref = Chem.MolFromSmiles(Chem.MolToSmiles(scaff_ref))
    scaffold_conformer = transfer_conformers(scaff_ref, ref_mol)
    scaff_ref.AddConformer(scaffold_conformer)
    
    # transform the scaffold to the reference molecule
    Chem.SanitizeMol(scaff_ref, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    frags = Chem.GetMolFrags(scaff_ref, asMols=True)
    mol = max(frags, key=lambda x: x.GetNumAtoms())


    return mol

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)

    if len(matches) < 1:
        raise Exception('Could not find scaffold matches')

    if len(matches) > 1:
        print('Found multiple scaffold matches')
    
    match = matches[0]
    # for match in matches:
    mol_coords = mol.GetConformer().GetPositions()
    scaf_coords = mol_coords[np.array(match)]
    scaf_conformer = create_conformer(scaf_coords)

    return scaf_conformer
    
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