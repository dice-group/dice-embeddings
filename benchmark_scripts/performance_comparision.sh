# # # DistMult
# i=5

# while ((i>0))
# do
# # GPU
# python ../main.py --storage_path ./distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU


# python ../main.py --storage_path ./distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # complex

# # GPU
# python ../main.py --storage_path ./complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # CPU
# python ../main.py --storage_path ./complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"

# ((i--))
# done



# # Kinships

# # DistMult
# i=5
# while ((i>0))
# do
# # GPU
# python ../main.py --storage_path ./distmult_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./distmult_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./distmult_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU


# python ../main.py --storage_path ./distmult_kinships_cpu --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # complex

# # GPU
# python ../main.py --storage_path ./complex_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./complex_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./complex_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU
# python ../main.py --storage_path ./complex_kinships_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# ((i--))
# done


##################################
# KVSALL
# i=5

# while ((i>0))
# do
# # GPU
# python ../main.py --storage_path ./kvsall_distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU


# python ../main.py --storage_path ./kvsall_distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # complex

# # GPU
# python ../main.py --storage_path ./kvsall_complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./kvsall_complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./kvsall_complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # CPU
# python ../main.py --storage_path ./kvsall_complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"

# ((i--))
# done



# # Kinships

# # DistMult
# i=5
# while ((i>0))
# do
# # GPU
# python ../main.py --storage_path ./kvsall_distmult_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_distmult_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_distmult_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU


# python ../main.py --storage_path ./kvsall_distmult_kinships_cpu --path_dataset_folder ../KGs/KINSHIP --model DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# # complex

# # GPU
# python ../main.py --storage_path ./kvsall_complex_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_complex_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./kvsall_complex_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# # CPU
# python ../main.py --storage_path ./kvsall_complex_kinships_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# ((i--))
# done







# # DistMult
i=5

while ((i>0))
do
# GPU
python ../main.py --storage_path ./distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU


python ../main.py --storage_path ./distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../main.py --storage_path ./complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../main.py --storage_path ./complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../main.py --storage_path ./complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU
python ../main.py --storage_path ./complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"

((i--))
done



# Kinships

# DistMult
i=5
while ((i>0))
do
# GPU
python ../main.py --storage_path ./distmult_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU


python ../main.py --storage_path ./distmult_kinships_cpu --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../main.py --storage_path ./complex_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./complex_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./complex_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU
python ../main.py --storage_path ./complex_kinships_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


((i--))
done



i=5

while ((i>0))
do
# GPU
python ../main.py --storage_path ./distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU


python ../main.py --storage_path ./distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../main.py --storage_path ./complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../main.py --storage_path ./complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../main.py --storage_path ./complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU
python ../main.py --storage_path ./complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"

((i--))
done



# Kinships

# DistMult
i=5
while ((i>0))
do
# GPU
python ../main.py --storage_path ./distmult_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./distmult_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU


python ../main.py --storage_path ./distmult_kinships_cpu --path_dataset_folder ../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../main.py --storage_path ./complex_kinships_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./complex_kinships_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./complex_kinships_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


# CPU
python ../main.py --storage_path ./complex_kinships_cpu --path_dataset_folder ../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 32 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None" 


((i--))
done