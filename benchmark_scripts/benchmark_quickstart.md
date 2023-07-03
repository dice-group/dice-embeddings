
# Quick Start

## Introduction

Pykeen offers two training loops for graph embedding training: LCWA (using kvsall) and SLCWA (using negative sampling). Once the training is complete, the models trained with these loops are evaluated using the evaluator in the Dice framework. The Dice framework provides three trainers specifically designed for graph embedding training. However, it's important to note that not all models trained with both training loops are compatible with these three trainers. For details, please refer to the paper.  
When working with Pykeen's models, most of the parameters need to be set after `--pykeen_model_kwargs`. Some parameters, such as `embedding_dim`, may need to be set twice for the recording purpose of the Dice framework.

## Trainer

- TorchTrainer: CPU, GPU
- torchDDP: GPU
- PL trainer: GPU

## LCWA training loop (kvsall)

**NOTE:** LCWA training loop only support kvsall scoring technique. By default, the Dice framework uses kvsall, so it is optional to add `--scoring_technique=KvsAll`. `num_core` should be bigger than 0 for multi-processes data loading.

Model training uses GPU:  

- PL trainer:

train and evaluate **Distmult** with kvsall using PL trainer and save embedding representation.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL"  --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --scoring_technique KvsAll
```

- TorchDDP:

train and evaluate **Distmult** with kvsall using TorchDDP trainer and save embedding representation.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None" --scoring_technique KvsAll
```

- TorchTrainer:

train and evaluate **Distmult** with kvsall using TorchTrainer and save embedding representation.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10  --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 1  --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None" --scoring_technique KvsAll
```


train and evaluate **Distmult** with kvsall using TorchTrainer and save embedding representation.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 0 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None" --scoring_technique KvsAll
```

## SLCWA (negative sampling)

All you need to do is to add `--use_SLCWALitModule`. Also, you need to state `--scoring_technique "NegSample"` explicitly. Otherwise, the model will not be evaluated using the correct evaluator.

- PL trainer:

train and evaluate **Distmult** with kvsall using TorchTrainer and save embedding representation.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
```

## Using auto_batch finder

**NOTE:** SLCWA training loop is not supported by auto_batch finder. Do not try to train the large model on the machine without enough memory of GPU, it will crash the machine immediately.

Add `--use_ddp_batch_finder True`.

e.g. train and evaluate **Distmult** of the Dice framework with kvsall using auto-batch finder in torchDDP trainer.

```[bash]
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 50 --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True --scoring_technique KvsAll
```

e.g. train and evaluate **Complex** of Pykeen with LCWA using auto-batch finder in torchDDP trainer.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_ComplEx --num_epochs 50 --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "auto" --optim "Adam" --normalization "None" --use_ddp_batch_finder True --scoring_technique KvsAll
```
