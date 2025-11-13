
# MolGenBench

# Environment Setup
```bash
conda create --name MolGenBench python=3.11
mamba install -c conda-forge rdkit numpy pandas seaborn scipy -y
pip install --use-pep517 EFGs
pip install tqdm joblib
pip install pytest
pip install swifter
pip install medchem
mamba install lilly-medchem-rules
mamba install openbabel
pip install posebusters spyrmsd

# for vina docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# for posecheck evaluation
git clone https://github.com/cch1999/posecheck.git
cd posecheck
git checkout 57a1938  # the calculation of strain energy used in our paper
pip install -e .
pip install -r requirements.txt
mamba install -c mx reduce

```
# Test the sample and Environment Setup
```bash

cd ~/MolGenBench
pytest -q molgenbench/pytest/*

# Runing The Evaluation
## Denovo : compute hit rete/fraction/recovery 
preprocess
 python molgenbench/preprocess/reference_process.py --reference_dir /home/data-house-01/cdhofficial/MolGens/TestSamples
 python molgenbench/preprocess/hit_info_preprocess_denovo.py --generated_dir /home/data-house-01/cdhofficial/MolGens/TestSamples

 And then, using notebook to show the final results:
 For Example:
    denovo hit rate
    /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/FigShow/Denovo_hit_recovery/Deonovo_repeats_hit_rate_boxplot.ipynb

    denovo hit fraction
    /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/FigShow/Denovo_hit_recovery/Deonovo_repeats_hit_fraction_boxplot.ipynb
## H2L compute hit rete/fraction/recovery 

    python /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/molgenbench/preprocess/hit_info_preprocess_h2l.py --generated_dir /home/data-house-01/cdhofficial/MolGens/TestSamples 
## interaction score
python /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/molgenbench/preprocess/interaction_score.py --generated_dir /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/TestSamples --root_save_dir /home/data-house-01/cdhofficial/MolGens/MolGenBench-dev/test_interaction
# 
