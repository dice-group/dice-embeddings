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

1. Train Shallom (Shallom:([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) ). for 1 epochs by using **4 CPU cores** on WN18RR. 
```
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 1 --check_val_every_n_epoch 5 --num_workers 4
```

Note that we require **--path_dataset_folder** to lead a folder containing **train.txt**, **valid.txt** and **test.txt**.
If there is no such split found under **--path_dataset_folder**, we apply k-fold cross validation on **train.txt**

3. Train Shallom on Carcinogenesis by using 5-fold cross validation by using  **all available CPU cores** on Carcinogenesis. 
```
python main.py --path_dataset_folder 'KGs/Carcinogenesis' --model 'Shallom' --num_folds_for_cv 5 --max_num_epochs 1
```

4. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 5 --check_val_every_n_epoch 5 --num_workers 4
```

# Using [Tensorboard](https://www.tensorflow.org/tensorboard)

If you wish to save log files and analyze them via tensorboard, all you need is run models with different configurations
```
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 5 --check_val_every_n_epoch 5 --logging True
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 10 --check_val_every_n_epoch 5 --logging True 
python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 15 --check_val_every_n_epoch 5  --logging True
```
Then, execute ```tensorboard --logdir ./lightng_logs/```

