# Training
## Input KG Format
1. A dataset must be located in a folder, e.g. 'KGs/YAGO3-10'.
2. A folder must contain **train** file. If the validation and test splits are available, then they must named as **valid** and **test**, respectively.
3. **train**, **valid** and **test** must be in either [N-triples](https://www.w3.org/2001/sw/RDFCore/ntriples/) format or standard link prediction dataset format (see KGs folder).
4. **train**, **valid** and **test** contain more than 10^7 triples, you may want to split each file, e.g.,
```
split train.txt -l 100000 train_split
mv train.txt orignal_train.txt
```
This would allow to fully leverage DASK as DASK allow us to read separate files simultaneously

5. Larger **train**, **valid**, and **test** can be stored in any of the following compression techniques [.gz, .bz2, or .zip].
Splitting a large **train.gz** into **train1.gz**, **train2.gz** etc. often decreases the runtimes of reading as in (4)

### Training Features:
1. Combine Kronecker Decomposition with any embedding model to reduce the memory requirements [source](https://arxiv.org/abs/2205.06560).
2. Use noise as a tikhonov regularization (see [source](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)
3. Pseudo-Labelling and Conformal Credal Self-Supervised Learning for KGE ([source](https://arxiv.org/abs/2205.15239)) 
4. If you have something in your mind, please contact us :)

### Available KGE Models
1. Multiplicative based KGE models: [DistMult](https://arxiv.org/pdf/1412.6575.pdf), [ComplEx](https://arxiv.org/pdf/1606.06357.pdf), [QMult](https://proceedings.mlr.press/v157/demir21a.html), and [OMult](https://proceedings.mlr.press/v157/demir21a.html) 
2. Feed Forward Neural Models: [Shallom](https://arxiv.org/pdf/2101.09090.pdf)
3. Convolutional Neural models [ConEx](https://openreview.net/forum?id=6T45-4TFqaX&invitationId=eswc-conferences.org/ESWC/2021/Conference/Research_Track/Paper49/-/Camera_Ready_Revision&referrer=%5BTasks%5D(%2Ftasks)), [ConvQ](https://proceedings.mlr.press/v157/demir21a.html), [ConvO](https://proceedings.mlr.press/v157/demir21a.html)
4. Contact us to add your favorite one :)


### Training
1. To obtain Shallom embeddings ([Research paper](https://arxiv.org/abs/2101.09090) and [conference presentation](https://www.youtube.com/watch?v=LUDpdgdvTQg)) 
```python main.py --path_dataset_folder "KGs/Family" --model "Shallom" --num_folds_for_cv 10 --num_epochs 1```
This execution results in generating **Mean and standard deviation of raw MRR in 10-fold cross validation => 0.768, 0.023**. Moreover, all necessary information including embeddings are stored in DAIKIRI_Storage folder (if does not exist it will be created).
2. Most link prediction benchmark datasets contain the train, validation and test datasets (see 'KGs/FB15K-237', 'KGs/WN18RR' or 'KGs/YAGO3-10').
To evaluate quality of embeddings, we rely on the standard metric, i.e. mean reciprocal rank (MRR). Executing ```python main.py --path_dataset_folder "KGs/WN18RR" --model "QMult" --num_epochs 1 --scoring_technique "KvsAll" ```
results in evaluating quality of SHALLOM embeddings on the test split.

#### Examples
1. To train our approaches for 10 epochs by using all available CPUs on UMLS. 
```
python main.py --path_dataset_folder "KGs/UMLS" --model "ConEx" --num_epochs 10 --scoring_technique "KvsAll"
```
2. To train our approaches for 10 epochs by using a single GPU.
```
python main.py --gpus 1 --path_dataset_folder "KGs/UMLS" --model "DistMult" --num_epochs 10 --scoring_technique "KvsAll"
python main.py --gpus 1 --path_dataset_folder "KGs/UMLS" --model "ComplEx" --num_epochs 10 --scoring_technique "KvsAll"
```

3. To train Shallom for 1 epochs on UMLS. All information will be stored in to 'DummyFolder'.
```
python main.py --gpus 1 --path_dataset_folder 'KGs/UMLS' --storage_path "DummyFolder" --model "ComplEx" --num_epochs 10 --scoring_technique "1vsAll"
```

4. To train Shallom on Carcinogenesis by using 10-fold cross validation on Carcinogenesis.  To check GPU usages, ```watch -n 1 nvidia-smi```
```
python main.py --gpus 1 --path_dataset_folder "KGs/Carcinogenesis" --model "Shallom" --num_folds_for_cv 10 --num_epochs 10
```
5. Train Shallom for 5 epochs by using **8 GPUs** on WN18RR To check GPU usages, ```watch -n 1 nvidia-smi```.
```
python main.py --gpus 8 --distributed_backend ddp --path_dataset_folder "KGs/WN18RR" --model 'Shallom' --num_epochs 5
```
#### Examples for Sharded Training
For Sharded Training, please refer to https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#sharded-training
```
python main.py --path_dataset_folder "dbpedia_only_03_2022.parquet" --embedding_dim 4 --num_epochs 1 --batch_size 64 --scoring_technique 'KvsAll' --plugins ddp_fully_sharded --gpus -1 --precision 16
```