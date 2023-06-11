

i=5

while ((i>0))
do

# Pykeen_DistMult
# GPU
# python ../main.py --storage_path ./pykeen_distmult_kinship_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
python ../main.py --storage_path ./pykeen_distmult_kinship_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./pykeen_distmult_kinship_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 1 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None" --use_SLCWALitModule
# CPU
# python ../main.py --storage_path ./pykeen_distmult_kinship_cpu --path_dataset_folder ../KGs/KINSHIP --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 128 --gpus 0 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --optim "Adam" --normalization "None" --use_SLCWALitModule



# Pykeen_ComplEx
# GPU
# python ../main.py --storage_path ./pykeen_complex_kinship_gpu_1 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --trainer "PL" --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule
python ../main.py --storage_path ./pykeen_complex_kinship_gpu_3 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "auto" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./pykeen_complex_kinship_gpu_2 --path_dataset_folder ../KGs/KINSHIP --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 1 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule

# CPU
# python ../main.py --storage_path ./pykeen_complex_kinship_cpu --path_dataset_folder ../KGs/KINSHIP --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 0 --neg_ratio 16 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" --use_SLCWALitModule


((i--))
done