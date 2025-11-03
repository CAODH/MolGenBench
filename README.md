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
conda install rdkit
mamba install seaborn
pip install --use-pep517 EFGs
pip install tqdm joblib
pip install pytest

```


######
###
