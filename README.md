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
1. Our models: [Shallom](https://arxiv.org/pdf/2101.09090.pdf), [ConEx](https://openreview.net/forum?id=6T45-4TFqaX&invitationId=eswc-conferences.org/ESWC/2021/Conference/Research_Track/Paper49/-/Camera_Ready_Revision&referrer=%5BTasks%5D(%2Ftasks)), QMult, OMult, ConvQ, ConvO

2. [DistMult](https://arxiv.org/pdf/1412.6575.pdf), [ComplEx](https://arxiv.org/pdf/1606.06357.pdf).

# Including a new KGE model can not be easier
```python
class DistMult(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'DistMult'
        self.loss = torch.nn.BCELoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward(self, e1_idx, rel_idx):
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_rel_real = self.emb_rel_real(rel_idx)
        score = torch.mm(emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        return torch.sigmoid(score)
```

# Dataset Format
1. A dataset must be located in a folder, e.g. 'KGs/YAGO3-10'.

2. A folder must contain **train.txt**. If the validation and test splits are available, then they must named as **valid.txt** and **test.txt**, respectively.

3. **train.txt**, **valid.txt** and **test.txt** must be in either [N-triples](https://www.w3.org/2001/sw/RDFCore/ntriples/) format or standard link prediction dataset format (see KGs folder).

# Usage 
1. For instance, 'KGs/Family' contains only **train.txt**. To obtain Shallom embeddings ([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) 
```python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_folds_for_cv 10 --max_num_epochs 1```
This execution results in generating **Mean and standard deviation of raw MRR in 10-fold cross validation => 0.768, 0.023**. Moreover, all necessary information including embeddings are stored in DAIKIRI_Storage folder (if does not exist it will be created).
   
1. Executing  ```python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --max_num_epochs 1```
   
2. Most link prediction benchmark datasets contain the train, validation and test datasets (see 'KGs/FB15K-237', 'KGs/WN18RR' or 'KGs/YAGO3-10').
To evaluate quality of embeddings, we rely on the standard metric, i.e. mean reciprocal rank (MRR). Executing ```python main.py --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 1```
results in evaluating quality of SHALLOM embeddings on the test split.
   
   
### More Examples

1. To train our approaches for 10 epochs by using **32 CPU cores** (if available) on UMLS. 
```
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 10
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --max_num_epochs 10
```


2. To train our approaches for 10 epochs by using a single GPU.
```
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 10
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --max_num_epochs 10
```

3. To train Shallom for 1 epochs on UMLS. All information will be stored in to 'DummyFolder'.
```
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --storage_path DummyFolder --model 'Shallom' --max_num_epochs 10
```

4. To train Shallom on Carcinogenesis by using 10-fold cross validation on Carcinogenesis.  To check GPU usages, ```watch -n 1 nvidia-smi```
```
python main.py --gpus 1 --path_dataset_folder 'KGs/Carcinogenesis' --model 'Shallom' --num_folds_for_cv 10 --max_num_epochs 10
```
5. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder 'KGs/WN18RR' --model 'Shallom' --max_num_epochs 5
```

## How to cite
If you want to cite the framework, feel free to
```
@article{demir2021Daikiri,
  title={DAIKIRI Embedding},
  author={Demir, Caglar},
  journal={GitHub. Note: https://github.com/dice-group/DAIKIRI-Embedding},
  volume={1},
  year={2021}
}
```

For any further questions, please contact:  ```caglar.demir@upb.de```

