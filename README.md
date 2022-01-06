# DAIKIRI-Embedding
This open-source project contains scalable implementation of many knowledge graph embedding approaches. Through the command line, models can be trained via using CPUs, GPUs and even TPUs.
Embeddings of knowledge graphs are readily created in csvs after models are trained.

## Installation

First clone the repository:
```
git clone https://github.com/dice-group/DAIKIRI-Embedding.git
```
Then obtain the required libraries:
```
conda env create -f environment.yml
conda activate daikiri
wget https://hobbitdata.informatik.uni-leipzig.de/KG/KGs.zip
unzip KGs.zip
python -m pytest tests
```
# Available Models
1. Our models: [Shallom](https://arxiv.org/pdf/2101.09090.pdf), [ConEx](https://openreview.net/forum?id=6T45-4TFqaX&invitationId=eswc-conferences.org/ESWC/2021/Conference/Research_Track/Paper49/-/Camera_Ready_Revision&referrer=%5BTasks%5D(%2Ftasks)), [QMult](https://proceedings.mlr.press/v157/demir21a.html), [OMult](https://proceedings.mlr.press/v157/demir21a.html), [ConvQ](https://proceedings.mlr.press/v157/demir21a.html), [ConvO](https://proceedings.mlr.press/v157/demir21a.html)

2. [DistMult](https://arxiv.org/pdf/1412.6575.pdf), [ComplEx](https://arxiv.org/pdf/1606.06357.pdf).

# Dataset Format
1. A dataset must be located in a folder, e.g. 'KGs/YAGO3-10'.

2. A folder must contain **train.txt**. If the validation and test splits are available, then they must named as **valid.txt** and **test.txt**, respectively.

3. **train.txt**, **valid.txt** and **test.txt** must be in either [N-triples](https://www.w3.org/2001/sw/RDFCore/ntriples/) format or standard link prediction dataset format (see KGs folder).

# Usage 
1. For instance, 'KGs/Family' contains only **train.txt**. To obtain Shallom embeddings ([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) 
```python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_folds_for_cv 10 --max_num_epochs 1```
This execution results in generating **Mean and standard deviation of raw MRR in 10-fold cross validation => 0.768, 0.023**. Moreover, all necessary information including embeddings are stored in DAIKIRI_Storage folder (if does not exist it will be created).
   
1. Executing  ```python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --max_num_epochs 1 --scoring_technique 'KvsAll'```
   
2. Most link prediction benchmark datasets contain the train, validation and test datasets (see 'KGs/FB15K-237', 'KGs/WN18RR' or 'KGs/YAGO3-10').
To evaluate quality of embeddings, we rely on the standard metric, i.e. mean reciprocal rank (MRR). Executing ```python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 1 --scoring_technique 'KvsAll'```
results in evaluating quality of SHALLOM embeddings on the test split.
   
   
### Examples

1. To train our approaches for 10 epochs by using **32 CPU cores** (if available) on UMLS. 
```
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 10 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --max_num_epochs 10 --scoring_technique 'KvsAll'
```
2. To train our approaches for 10 epochs by using a single GPU.
```
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 10 --scoring_technique 'KvsAll'
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --max_num_epochs 10 --scoring_technique 'KvsAll'
```

3. To train Shallom for 1 epochs on UMLS. All information will be stored in to 'DummyFolder'.
```
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --storage_path DummyFolder --model 'Shallom' --max_num_epochs 10 --scoring_technique 'KvsAll'
```

4. To train Shallom on Carcinogenesis by using 10-fold cross validation on Carcinogenesis.  To check GPU usages, ```watch -n 1 nvidia-smi```
```
python main.py --gpus 1 --path_dataset_folder 'KGs/Carcinogenesis' --model 'Shallom' --num_folds_for_cv 10 --max_num_epochs 10
```
5. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 5
```

6. More examples can be found in run.sh.

## How to cite
If you want to cite the framework, feel free to
```
@inproceedings{demir2021convolutional,
title={Convolutional Complex Knowledge Graph Embeddings},
author={Caglar Demir and Axel-Cyrille Ngonga Ngomo},
booktitle={Eighteenth Extended Semantic Web Conference - Research Track},
year={2021},
url={https://openreview.net/forum?id=6T45-4TFqaX}}
```

For any further questions, please contact:  ```caglar.demir@upb.de```

