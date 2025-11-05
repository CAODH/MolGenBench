import os
import random
import string
import tempfile
import subprocess
import contextlib
import pandas as pd
import multiprocessing as mp

import AutoDockTools
from joblib import Parallel, delayed
from rdkit import Chem
from vina import Vina
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils

from molgenbench.metrics.basic import is_valid


def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def get_pdbqt_mol(pdbqt_block: str) -> Chem.Mol:
    """Convert pdbqt block to rdkit mol by converting with openbabel"""
    # write pdbqt file
    random_name = get_random_id()
    # random_name = np.random.randint(0, 100000)
    pdbqt_name = f"tmp/test_pdbqt_{random_name}.pdbqt"
    with open(pdbqt_name, "w") as f:
        f.write(pdbqt_block)

    # read pdbqt file from autodock
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    obConversion.ReadFile(mol, pdbqt_name)

    # convert to RDKIT
    mol = Chem.MolFromPDBBlock(obConversion.WriteString(mol))

    # remove tmp file
    os.remove(pdbqt_name)

    return mol

def prepare_and_check(file_path):
    """
    Tries to prepare PQR and PDBQT files. Returns the file path if successful, otherwise None.
    
    :param file_path: Path to the input file
    :return: The input file path if PDBQT preparation is successful, otherwise None
    """
    pqr_file = file_path[:-4] + '.pqr'
    pdbqt_file = file_path[:-4] + '.pdbqt'
    prot = PrepProt(file_path)

    # Generate the PQR file if it does not exist
    
    prot.addH(pqr_file)
    # Generate the PDBQT file if it does not exist
    if not os.path.exists(pdbqt_file):
        prot.get_pdbqt(pdbqt_file)
    # Check if the PDBQT file exists
    if os.path.exists(pdbqt_file):
        return file_path
    return None


def choose_processable_path(protein_fn, pocket_fn):
    """
    Chooses a processable path, prioritizing protein_fn. If not processable, tries pocket_fn.
    
    :param protein_fn: Path to the protein file
    :param pocket_fn: Path to the pocket file
    :return: The path to the processable file
    :raises ValueError: If neither protein_fn nor pocket_fn is processable
    """
    # Try processing the protein file
    protein_result = prepare_and_check(protein_fn)
    if protein_result:
        return protein_result

    # Try processing the pocket file
    pocket_result = prepare_and_check(pocket_fn)
    if pocket_result:
        return pocket_result

    # Raise an error if neither file is processable
    raise ValueError(f"No processable protein or pocket file found: {protein_fn}, {pocket_fn}")



class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string()
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        if not os.path.exists(prot_pqr):
            subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        if not os.path.exists(prot_pdbqt):
            prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
            subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()
    

class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    @property
    def pdbqt_code_map(self):
        return {
            0: '',
            1: 'Ru',
            2: 'B',
            3: 'V',
            4: 'Sc',
            5: 'Mo',
            6: 'Cr',
        }

    def check_pdbqt_str(self, fn):
        with open(fn, 'r') as f:
            lig_str = f.readlines()
        # print(fn)
        for s in lig_str:
            if s.strip().endswith(' Ru'):
                return 1
            if s.strip().endswith(' B'):
                return 2
            if s.strip().endswith(' V'):
                return 3
            if s.strip().endswith(' Sc'):
                return 4
            if s.strip().endswith(' Mo'):
                return 5
            if s.strip().endswith(' Cr'):
                return 6
        return 0


    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, cpu=exhaustiveness, **kwargs)
        
        prot_code = self.check_pdbqt_str(self.prot_pdbqt)
        if prot_code > 0:
            raise ValueError(f'atom type {self.pdbqt_code_map[prot_code]} cannot be parsed')
        else:
            lig_code = self.check_pdbqt_str(self.lig_pdbqt)
            if lig_code > 0:
                raise ValueError(f'atom type {self.pdbqt_code_map[lig_code]} cannot be parsed')
        
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):
    
    @classmethod
    def from_generated_mol(cls, ligand_rdmol, protein_root='Path/to/protein.pdb', **kwargs):

        return cls(protein_root, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0, pos=None):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        # we use the ref ligand to define the center and size of the pocket
        if pos is None:
            # raise ValueError('pos is None')
            pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None


    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'
        ## check if the protein_pdbqt exists or not
        if not os.path.exists(protein_pdbqt):
            prot = PrepProt(self.receptor_path)
            if not os.path.exists(protein_pqr):
                prot.addH(protein_pqr)
            if not os.path.exists(protein_pdbqt):
                prot.get_pdbqt(protein_pdbqt)
        

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        try:
            score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
            return [{'affinity': score, 'pose': pose}]
        except Exception as e:
            return [{'affinity': None, 'pose': None}]


def ligprep_simple(mol: Chem.Mol) -> Chem.Mol:
    if not is_valid(mol):
        return None
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    except Exception as e:
        print(f"Error in ligprep_simple: {e}")
        return mol


def get_vina_results(ligand_mol, protein_path, ref_ligand_mol, docking_mode='vina_dock', exhaustiveness=8):
    try:
        if ligand_mol is None:
            return None, None
        vina_task = VinaDockingTask.from_generated_mol(
            ligand_rdmol=ligand_mol,
            protein_root=protein_path,
            pos=ref_ligand_mol.GetConformer(0).GetPositions(), # !!! we use the ref ligand to define the center and size of the pocket
        )
        
        docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
        affinity = docking_results[0]['affinity']
        pose = docking_results[0]['pose']
        
        return affinity, pose
    
    except Exception as e:
        raise e


def process_ligand(ligand_file, ligand_root, protein_path, ref_ligand_mol, docked_root):
    """
    process a single ligand file, return a dictionary containing the required information.
    """
    ligand_mol = Chem.SDMolSupplier(os.path.join(ligand_root, ligand_file))[0]
    return {
        'file_id': os.path.splitext(ligand_file)[0],  # file_id as the name of the file without extension
        'lig_name': ligand_mol.GetProp('_Name'),
        'protein_path': protein_path,
        'ligand_path': os.path.join(ligand_root, ligand_file),
        'ligand_mol': ligprep_simple(ligand_mol),
        'ref_ligand_mol': ref_ligand_mol,
        'docked_root': docked_root,
        'affinity': None,  # Placeholder for vina_dock results
        'docked_path': None  # Placeholder for vina_dock results
    }

def prepare_output_df(protein_path, ligand_root, ref_ligand_mol, docked_root):
    """
    Prepare a DataFrame to store the docking results.
    """
    ligand_files = os.listdir(ligand_root)
    # parallelize the process
    rows = Parallel(n_jobs=max(1, mp.cpu_count()-10))(
        delayed(process_ligand)(ligand_file, ligand_root, protein_path, ref_ligand_mol, docked_root)
        for ligand_file in ligand_files
    )
    
    output_df = pd.DataFrame(rows)
    return output_df


def process_row(row):
    docked_path = os.path.join(row['docked_root'], f"{row['file_id']}_docked.sdf")
    if os.path.exists(docked_path):
        affinity = Chem.SDMolSupplier(docked_path)[0].GetProp('vina_dock')
        return row.name, affinity, docked_path
    try:
        affinity, docked_pose = get_vina_results(row['ligand_mol'], row['protein_path'], row['ref_ligand_mol'])
        if docked_pose is not None:
            docked_mol = get_pdbqt_mol(docked_pose)
            docked_mol.SetProp('_Name', row['lig_name'])
            docked_mol.SetProp('vina_dock', str(affinity))

            writer = Chem.SDWriter(docked_path)
            writer.write(docked_mol)
            writer.close()
        else:
            affinity = "Failed"
            docked_path = "Failed"
        
        return row.name, affinity, docked_path
    except Exception as e:
        print(f"Error in process_row: {e}")
        return row.name, "Failed", "Failed"