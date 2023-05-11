#!/bin/bash
num_epochs=500
for dataset in "UMLS" "KINSHIP"
do
  for model_name in  "CLf" "QMult" "DistMult" "ComplEx"
  do
    for embedding_dim in 32
    do
      if [ "$model_name" = "CLf" ]
      then
        python -u main.py --model $model_name --p 0 --q 0 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 1 --q 0 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 0 --q 1 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"

        python -u main.py --model $model_name --p 3 --q 0 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 0 --q 3 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 2 --q 1 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 1 --q 2 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"

        python -u main.py --model $model_name --p 0 --q 7 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 7 --q 0 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 6 --q 1 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 1 --q 6 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 2 --q 5 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 5 --q 2 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"

        python -u main.py --model $model_name --p 3 --q 4 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"
        python -u main.py --model $model_name --p 4 --q 3 --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"

     else
        python -u main.py --model $model_name --embedding_dim $embedding_dim  --scoring_technique KvsAll --num_epochs $num_epochs --lr 0.1 --path_dataset_folder "KGs/$dataset" --eval_model "train_val_test"

      fi
    done
  done
done

