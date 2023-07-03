
# fb15k

#lcwa
# distmult
# python ../main.py --storage_path ./pykeen_distmult_lcwa_distmult_fb15k_autobtach --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True

# # complex
# python ../main.py --storage_path ./pykeen_complex_lcwa_fb15k_autobtach --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True


# # kvsall
# # distmult

# python ../main.py --storage_path ./distmult_kvsall_fb15k_autobtach --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None" --use_ddp_batch_finder True

# # complex
# python ../main.py --storage_path ./complex_kvsall_fb15k_autobtach --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None" --use_ddp_batch_finder True



# yago

#lcwa

# complex
python ../main.py --storage_path ./pykeen_complex_lcwa_autobtach --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True


# # kvsall

# # distmult
python ../main.py --storage_path ./distmult_kvsall_yago_autobtach --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None" --use_ddp_batch_finder True

# # complex
python ../main.py --storage_path ./complex_kvsall_yago_autobtach --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None" --use_ddp_batch_finder True




# test
# python ../main.py --storage_path ./demo --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
# python ../main.py --storage_path ./demo --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True