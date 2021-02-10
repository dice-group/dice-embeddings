# DAIKIRI-Embedding
This open-source project contains several Pytorch implementation of knowledge graph embedding approaches

## Installation

First clone the repository:
```
git clone https://github.com/dice-group/DAIKIRI-Embedding.git
```
Then obtain the required libraries:
```
conda env create -f environment.yml
conda activate daikiri_emb
unzip KGs
```

# Usage
In this section, we provide simple examples on using our project. During our examples, we use our newest
scalable neural knowledge graph embedding model (Shallom:([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) ).

1. Train Shallom for 3 epochs by using **32 CPU cores** on YAGO3-10. We expect that
**--path_dataset_folder** leads to a folder containing **train.txt**, **valid.txt** and **test.txt**.
```
python main.py --path_dataset_folder 'KGs/YAGO3-10' --model 'Shallom' --max_epochs 3  --num_workers 32
```
Executing this command results in obtaining 0.981. multi-label test accuracy.

2. Train Shallom for 3 epochs by using **8 GPUs** on YAGO3-10. To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/YAGO3-10' --model 'Shallom' --max_epochs 3
```

