# We analysis bond length angle and torsion angle in this file.
# The docked active molecules of each uniprot are selected as reference molecules

import os
import time
import math
import json
import argparse
import numpy as np
import pandas as pd
from scipy import spatial as sci_spatial

from rdkit import Chem
from src.evaluation import eval_bond_length, eval_bond_angle, eval_torsion_angle


N_BINS = 100
BOND_LENGTH_BINS = np.arange(1.1, 1.70001, 0.005) # 用来算jsd
BOND_ANGLE_BINS = np.arange(100, 140.001, 0.25)
TORSION_ANGLE_BINS = np.arange(-180, 181, 1.5)


def get_allowed_types(profile):
    allowed_types = []
    for k, v in sorted(profile.items(), key=lambda x: len(x[1]), reverse=True):
        freq = len(v)
        if freq < 10:
            continue

        if freq > 200:
            allowed_types.append(k)
            
    return allowed_types


def get_results(generated_file, ref_file):
    ref_bond_lengths = []
    gen_bond_lengths = []
    
    ref_bond_angles = []
    gen_bond_angles = []
    
    ref_torsion_angles = []
    gen_torsion_angles = []
        
    bond_length_jsd_results = {}
    bond_angle_jsd_results = {}
    torsion_angle_jsd_results = {}
    
    
    ref_suppl = Chem.SDMolSupplier(ref_file)
    gen_suppl = Chem.SDMolSupplier(generated_file)
    
    for ref_mol in ref_suppl:
        if ref_mol is None:
            continue
        ref_mol = Chem.RemoveAllHs(ref_mol)
        ref_bond_lengths += eval_bond_length.bond_distance_from_mol(ref_mol) # (bond_type, distance)
        ref_bond_angles += eval_bond_angle.bond_angle_from_mol(ref_mol) # (bond_type, angle)
        ref_torsion_angles += eval_torsion_angle.torsion_angle_from_mol(ref_mol) # (bond_type, angle)
        
    for gen_mol in gen_suppl:
        if gen_mol is None:
            continue
        gen_mol = Chem.RemoveAllHs(gen_mol)
        gen_bond_lengths += eval_bond_length.bond_distance_from_mol(gen_mol) # (bond_type, distance)
        gen_bond_angles += eval_bond_angle.bond_angle_from_mol(gen_mol) # (bond_type, angle)
        gen_torsion_angles += eval_torsion_angle.torsion_angle_from_mol(gen_mol) # (bond_type, angle)
        
    ref_length_profile = eval_bond_length.get_bond_lengths(ref_bond_lengths) # collections.defaultdict(list) : bond_type : [distances]
    gen_length_profile = eval_bond_length.get_bond_lengths(gen_bond_lengths) # collections.defaultdict(list) : bond_type : [distances]
    allowed_bond_type = get_allowed_types(ref_length_profile)
    
    ref_angle_profile = eval_bond_angle.get_bond_angles(ref_bond_angles) # collections.defaultdict(list) : bond_type : [angles]
    gen_angle_profile = eval_bond_angle.get_bond_angles(gen_bond_angles) # collections.defaultdict(list) : bond_type : [angles]
    allowed_angle_type = get_allowed_types(ref_angle_profile)
    
    ref_torsion_angles_profile = eval_torsion_angle.get_torsion_angles(ref_torsion_angles) # collections.defaultdict(list) : bond_type : [angles]
    gen_torsion_angles_profile = eval_torsion_angle.get_torsion_angles(gen_torsion_angles) # collections.defaultdict(list) : bond_type : [angles]
    allowed_torsion_type = get_allowed_types(ref_torsion_angles_profile)
    
    for BOND in allowed_bond_type:
        ref_bond_dist = np.histogram(ref_length_profile[BOND], bins=BOND_LENGTH_BINS, density=True)[0] / 100
        gen_bond_dist = np.histogram(gen_length_profile[BOND], bins=BOND_LENGTH_BINS, density=True)[0] / 100
        bond_length_jsd = sci_spatial.distance.jensenshannon(ref_bond_dist, gen_bond_dist)
        bond_length_jsd_results[eval_bond_length._bond_type_str(BOND)] = bond_length_jsd
        
    for ANGLE in allowed_angle_type:
        ref_angle_dist = np.histogram(ref_angle_profile[ANGLE], bins=BOND_ANGLE_BINS, density=True)[0] / 100
        gen_angle_dist = np.histogram(gen_angle_profile[ANGLE], bins=BOND_ANGLE_BINS, density=True)[0] / 100
        bond_angle_jsd = sci_spatial.distance.jensenshannon(ref_angle_dist, gen_angle_dist)
        bond_angle_jsd_results[eval_bond_angle._angle_type_str(ANGLE)] = bond_angle_jsd
    
    for TORSION in allowed_torsion_type:
        ref_torsion_dist = np.histogram(ref_torsion_angles_profile[TORSION], bins=TORSION_ANGLE_BINS, density=True)[0] / 100
        gen_torsion_dist = np.histogram(gen_torsion_angles_profile[TORSION], bins=TORSION_ANGLE_BINS, density=True)[0] / 100
        torsion_angle_jsd = sci_spatial.distance.jensenshannon(ref_torsion_dist, gen_torsion_dist)
        torsion_angle_jsd_results[eval_torsion_angle._torsion_type_str(TORSION)] = torsion_angle_jsd
    
    bond_length_jsd_results = {k: v for k, v in bond_length_jsd_results.items() if not math.isnan(v)}
    bond_angle_jsd_results = {k: v for k, v in bond_angle_jsd_results.items() if not math.isnan(v)}
    torsion_angle_jsd_results = {k: v for k, v in torsion_angle_jsd_results.items() if not math.isnan(v)}
    
    return bond_length_jsd_results, bond_angle_jsd_results, torsion_angle_jsd_results
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RMSD')
    parser.add_argument('--output_file', type=str,default=None, help='Output file path')
    parser.add_argument('--generated_file', type=str,default = '/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_Pocket2Mol_generated_molecules.sdf', help='File name of the SDF that contains generated 3D molecules')
    parser.add_argument('--ref_file', type=str, default='/home/datahouse1/fanzhehuan/myprojects/SBDDBench/TestSamples/O14757/O14757_all_active_molecules_ligprep_glide_sp_pv.sdf', help='File name of the reference positive 3D molecules')
    
    args = parser.parse_args()
    start =  time.strftime("%Y-%m-%d", time.localtime())

    args.output_file = args.output_file if args.output_file is not None else f'{os.path.dirname(args.generated_file)}/{start}_bond_analysis.csv'

    bond_length_jsd_results, bond_angle_jsd_results, torsion_angle_jsd_results = get_results(args.generated_file, args.ref_file)
    
    jsd_dict = {
        'Bond Length JSD': json.dumps(bond_length_jsd_results),
        'Bond Angle JSD': json.dumps(bond_angle_jsd_results),
        'Torsion Angle JSD': json.dumps(torsion_angle_jsd_results)
    }
    
    df = pd.DataFrame.from_dict(jsd_dict, orient='index', columns=['value']).transpose()

    df.to_csv(args.output_file, index=False)
