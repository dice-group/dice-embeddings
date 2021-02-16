# DAIKIRI-Embedding
Need a better name. This open-source project contains several Pytorch implementation of knowledge graph embedding approaches

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
# Dataset Format and Usage

1. A dataset must be located in a folder, e.g. 'KGs/Family'.

2. A folder must contain **train.txt**. If the validation and test splits are available, then they must named as **valid.txt** and **test.txt**, respectively.

3. **train.txt**, **valid.txt** and **test.txt** must be in either N-triples format or standard link prediction dataset format.

4. For instance, 'KGs/Family' contains only **train.txt**. To evaluate quality of Shallom ([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) embeddings, we rely on the k-fold cross validation metric.
Concretely, executing ```python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_folds_for_cv 10 --max_num_epochs 1```
results in generating **Mean and standard deviation of raw MRR in 10-fold cross validation => 0.768, 0.023**. Moreover, all necessary information including embeddings are stored in DAIKIRI_Storage folder (if does not exist it will be created).
   
5. Most link prediction benchmark datasets contain the train, validation and test datasets (see 'KGs/FB15K-237', 'KGs/WN18RR' or 'KGs/YAGO3-10').
To evaluate quality of embeddings, we rely on the standard metric, i.e. mean reciprocal rank (MRR). Executing ```python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 1```
results in evaluating quality of SHALLOM embeddings on the test split.
   
   
### More Examples

1. To train Shallom for 1 epochs by using **32 CPU cores** (if available) on UMLS. 
```
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 1
```

2. To train Shallom for 1 epochs on UMLS. All information will be stored in to 'DummyFolder'.
```
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --storage_path DummyFolder --max_num_epochs 1
```

3. To train Shallom on Carcinogenesis by using 5-fold cross validation on Carcinogenesis. 
```
python main.py --path_dataset_folder 'KGs/Carcinogenesis' --model 'Shallom' --num_folds_for_cv 5 --max_num_epochs 1
```

4. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 5 --check_val_every_n_epoch 5 --num_workers 4
```