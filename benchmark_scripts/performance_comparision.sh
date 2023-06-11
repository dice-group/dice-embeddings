# # DistMult
# i=5

# while ((i>0))
# do
# # GPU
# python ../main.py --storage_path ./distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
# python ../main.py --storage_path ./distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
# python ../main.py --storage_path ./distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# # CPU


# python ../main.py --storage_path ./distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# # complex

# # GPU
# python ../main.py --storage_path ./complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
# python ../main.py --storage_path ./complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
# python ../main.py --storage_path ./complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# # CPU
# python ../main.py --storage_path ./complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll

# ((i--))
# done



# Kinships

# DistMult
i=5
while ((i>0))
do
# GPU
python ../main.py --storage_path ./distmult_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
python ../main.py --storage_path ./distmult_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
python ../main.py --storage_path ./distmult_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# CPU


python ../main.py --storage_path ./distmult_kinships_cpu --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# complex

# GPU
python ../main.py --storage_path ./complex_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
python ../main.py --storage_path ./complex_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" --scoring_technique KvsAll
python ../main.py --storage_path ./complex_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


# CPU
python ../main.py --storage_path ./complex_kinships_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" --scoring_technique KvsAll


((i--))
done




