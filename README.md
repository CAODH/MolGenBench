# 1. 更新了pytest 单元测试要求：每个模块或者预处理模块都要通过测试案例检查
```bash
# cd 到库目录
cd ~/MolGenBench
# 运行所有测试，并通过后提交PR
pytest -q molgenbench/pytest/*

```
# 2. 需要主要添加补充环境安装的命令


# MolGenBench

sbdd Benchmark and evalation metrics

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


######
###
