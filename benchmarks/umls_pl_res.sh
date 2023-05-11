





# Pykeen_DistMult
# GPU
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --trainer "PL" --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" 

python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 1 --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None" 

# # CPU
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 0 --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"

# Pykeen_ComplEx
# GPU
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --trainer "PL" --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"

python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --gpus 1 --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"

# CPU
python ../main.py --path_dataset_folder ../KGs/UMLS --model Pykeen_ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --gpus 0 --neg_ratio 32 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"


