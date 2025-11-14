from pathlib import Path
from typing import Dict, List
from rdkit import Chem
from molgenbench.io.types import MoleculeRecord


def read_sdf_to_records(
    path: Path,
    protein_path: str,
    ref_active_path: str,
) -> List[MoleculeRecord]:
    """
    Read one SDF file and convert all molecules into MoleculeRecord objects.

    Args:
        path (Path): Path to the .sdf file.
        source_name (str): Optional name (e.g., UniProt ID or model name)
                           stored in metadata for provenance.

    Returns:
        List[MoleculeRecord]: List of MoleculeRecord objects.
    """
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    records: List[MoleculeRecord] = []

    for i, mol in enumerate(suppl):
        smiles = Chem.MolToSmiles(mol) if mol is not None else None
        num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) if mol is not None else None
        record = MoleculeRecord(
            id=mol.GetProp("_Name"),
            smiles=smiles,
            rdkit_mol=mol,
            num_rotatable_bonds=num_rotatable_bonds,
            metadata={
                "source_file": str(path),
                "protein_path": protein_path, 
                "ref_active_path": ref_active_path,
            },
        )
        records.append(record)

    return records


def attach_docked_molecules(
    records: List[MoleculeRecord],
    docked_sdf_path,
) -> List[MoleculeRecord]:
    """
    Try to attach docked molecules to existing MoleculeRecords via '_Name' match.
    If docked_sdf_path does not exist, silently skip.

    Args:
        records: List of MoleculeRecord (from read_sdf_file)
        docked_sdf_path: Path to vina-docked SDF file

    Returns:
        Updated list of MoleculeRecords with `metadata["docked_mol"]` if matched.
    """
    docked_sdf_path = Path(docked_sdf_path)
    if not docked_sdf_path.exists():
        return records  # nothing to attach

    docked_suppl = Chem.SDMolSupplier(str(docked_sdf_path))
    docked_dict = {
        mol.GetProp("_Name"): mol
        for mol in docked_suppl
        if mol is not None
    }

    for record in records:
        name = record.id
        if name in docked_dict:
            record.metadata["docked_mol"] = docked_dict[name]

    return records


def read_uniprot_sdf_dir(root_dir: str) -> Dict[str, List[MoleculeRecord]]:
    """
    Traverse a root directory containing multiple UniProt subfolders.
    Each subfolder should contain exactly one .sdf file.
    Example structure:
        TestSamples/
        ├── O14757/
            ├──Round1/
                ├── De_novo_Results/
                │   ├── PocketFlow_generated_molecules/
                │   │   └── O14757_PocketFlow_generated_molecules.sdf
                │   └── Hit_to_Lead_Results/
                        └── Series14139
                            └── DeleteHit2Lead(CrossDock)_Hit_to_Lead
                                └── O14757_Series14139_DeleteHit2Lead(CrossDock)_Hit_to_Lead.sdf
        ├── P12345/
            ├──Round1/
                ├── De_novo_Results/
                    └── ...
                ├── Hit_to_Lead_Results/
                    └── ...

    Args:
        root_dir (str): Path to the root directory.

    Returns:
        Dict[str, List[MoleculeRecord]]: Mapping from UniProt ID to MoleculeRecord list.
    """
    root = Path(root_dir)
    all_records: Dict[str, List[MoleculeRecord]] = {}

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        uniprot_id = subdir.name
        
        sdf_files = list(subdir.glob("*.sdf"))
        if not sdf_files:
            continue

        # take the first SDF file by default
        sdf_file = sdf_files[0]
        records = read_sdf_to_records(sdf_file, source_name=uniprot_id)
        all_records[uniprot_id] = records

    return all_records
