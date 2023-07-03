

# i=5

# while ((i>0))
# do

# # # Pykeen_DistMult
# # # GPU
# python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 1  --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # # CPU
# python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # Pykeen_ComplEx
# # # GPU
# python ../main.py --storage_path ./slcwa_pykeen_complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./slcwa_pykeen_complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # CPU
# python ../main.py --storage_path ./slcwa_pykeen_complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

# ((i--))
# done



# i=5

# while ((i>0))
# do

# # Pykeen_DistMult
# # GPU
# python ../main.py --storage_path ./pykeen_distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./pykeen_distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./pykeen_distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" 

# # # CPU
# python ../main.py --storage_path ./pykeen_distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" 

# # Pykeen_ComplEx
# # GPU
# python ../main.py --storage_path ./pykeen_complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./pykeen_complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./pykeen_complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" 

# # CPU
# python ../main.py --storage_path ./pykeen_complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" 

# ((i--))
# done







i=5

while ((i>0))
do

# # # Pykeen_DistMult
# # # GPU
# python ../main.py --storage_path ./slcwa16_pykeen_distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 16  --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./slcwa16_pykeen_distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # # CPU
# python ../main.py --storage_path ./slcwa16_pykeen_distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # Pykeen_ComplEx
# # # GPU
# python ../main.py --storage_path ./slcwa16_pykeen_complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None" 
python ../main.py --storage_path ./slcwa16_pykeen_complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # CPU
# python ../main.py --storage_path ./slcwa16_pykeen_complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 16 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

((i--))
done



# i=5

# while ((i>0))
# do

# # # Pykeen_DistMult
# # # GPU
# python ../main.py --storage_path ./slcwa32_pykeen_distmult_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32  --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --trainer "PL" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_distmult_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./slcwa32_pykeen_distmult_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # # CPU
# python ../main.py --storage_path ./slcwa32_pykeen_distmult_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.01 --embedding_dim 64 --gpus 0 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # Pykeen_ComplEx
# # # GPU
# python ../main.py --storage_path ./slcwa32_pykeen_complex_umls_gpu_1 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --trainer "PL" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
# # python ../main.py --storage_path ./slcwa_pykeen_complex_umls_gpu_3 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None" 
# python ../main.py --storage_path ./slcwa32_pykeen_complex_umls_gpu_2 --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 1 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# # # CPU
# python ../main.py --storage_path ./slcwa32_pykeen_complex_umls_cpu --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.1 --embedding_dim 32 --gpus 0 --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "cpu" --optim "Adam" --normalization "None" --use_SLCWALitModule

# ((i--))
# done