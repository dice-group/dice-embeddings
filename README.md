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

# Examples
In this section, we provide simple examples on using our project. During our examples, we use our newest
scalable neural knowledge graph embedding model (Shallom:([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) ).

Note that we require **--path_dataset_folder** to lead a folder containing **train.txt**, **valid.txt** and **test.txt**.
Soon, we will be able to perform k-fold cross validation on a single dataset.

1. Train Shallom for 1 epochs by using **4 CPU cores** on WN18RR. 
```
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_epochs 1 --check_val_every_n_epoch 5 --num_workers 4
```
Executing this command results in obtaining 0.909. multi-label test accuracy and raw MRR 0.767.

2. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_epochs 5 --check_val_every_n_epoch 5 --num_workers 4
```

# Using [Tensorboard](https://www.tensorflow.org/tensorboard)

If you wish to save log files and analyze them via tensorboard, all you need is run models with different configurations
```
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_epochs 5 --check_val_every_n_epoch 5 --logging True
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_epochs 10 --check_val_every_n_epoch 5 --logging True 
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_epochs 15 --check_val_every_n_epoch 5  --logging True
```
Then, execute ```tensorboard --logdir ./lightng_logs/```

