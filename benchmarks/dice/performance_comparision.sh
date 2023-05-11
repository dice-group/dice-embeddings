# DistMult

# GPU
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU


python ../../main.py --path_dataset_folder ../../KGs/UMLS --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.1 --embedding_dim 64 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"




# Kinships

# DistMult

# GPU
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.01 --embedding_dim 128 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.01 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.01 --embedding_dim 128 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU


python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.01 --embedding_dim 128 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# complex

# GPU
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.1 --embedding_dim 128 --trainer "PL" --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.1 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --accelerator "gpu" --optim "Adam" --normalization "None"
python ../../main.py --path_dataset_folder ../../KGs/KINSHIP --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.1 --embedding_dim 128 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"


# CPU
python ../../main.py --path_dataset_folder ../../KGs/UMLS --model ComplEx --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.1 --embedding_dim 128 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --optim "Adam" --normalization "None"