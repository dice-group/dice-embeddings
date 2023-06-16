
# Benchmark Pykeen

A quick start to use Pykeen in the Dice embedding. When using Pykeen, you should also add arguments to the Pykeen models through `--pykeen_model_kwargs`, for example, `--pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss"`.

## LCWA training loop

(Trainning with only 10 epochs)

GPU:

- PL trainer:
  
```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"
```

- TorchDDP:

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
```

- TorchTrainer:

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 1 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None"
```

CPU:

- TorchTrainer

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 0 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None"
```

## SLCWA

Add `--use_SLCWALitModule`. e.g.

```[bash]
python ../main.py --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 10 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
```

## Using auto_batch finder

(Training with 100 epoches, and only with LCWA training loop)

Add `--use_ddp_batch_finder True`. e.g. here we train the DistMult with `TorchDDP` trainer with auto_batch finder.

```[bash]
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
```
