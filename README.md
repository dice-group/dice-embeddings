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

1. Train Shallom for 3 epochs by using 32 CPU cores on YAGO3-10.
```
python main.py --path_train_dataset 'KGs/YAGO3-10/train.txt' --model 'Shallom' --max_epochs 3  --num_workers 32
```
2. Train Shallom for 3 epochs by using 8 GPUs on YAGO3-10. To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8--path_train_dataset 'KGs/YAGO3-10/train.txt' --model 'Shallom' --max_epochs 3 --distributed_backend ddp
```

# Examples

1. Executing ```python main.py --model Shallom --path_dataset_folder 'KGs/UMLS' --max_epochs 20 --num_workers 32``` 
results in obtaining **raw** 0.9726. test accuracy.
   
2. Executing ```python main.py --model Shallom --path_dataset_folder 'KGs/WN18RR' --max_epochs 20 --num_workers 32```
results in obtaining **raw** .952. test accuracy.
   
3. Executing ```python main.py --model Shallom --path_dataset_folder 'KGs/YAGO3-10' --max_epochs 20 --num_workers 32```
results in obtaining **raw** 0.981. test accuracy.