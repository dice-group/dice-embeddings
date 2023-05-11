
# FB15K-237
# DistMult

# GPU
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --neg_ratio 1 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 0 --gpus 1 --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"



# CPU


python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 32 --num_epochs 100 --scoring_technique "NegSample" --batch_size 128 --lr 0.01 --embedding_dim 64 --num_core 0 --gpus 0 --save_embeddings_as_csv --eval_model "train_val_test" --optim "Adam" --normalization "None"


# complex

# GPU
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "PL" --neg_ratio 32 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --optim "Adam" --normalization "None"
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "torchDDP" --neg_ratio 32 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 1 --accelerator "auto" --optim "Adam" --normalization "None"
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --neg_ratio 32 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --gpus 1 --optim "Adam" --normalization "None"



# CPU
python ../main.py --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --neg_ratio 32 --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --gpus 0 --optim "Adam" --normalization "None"