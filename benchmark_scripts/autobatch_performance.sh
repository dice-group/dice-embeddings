
# python ../main.py --storage_path ./distmult_fb15k_autobatch --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
# python ../main.py --storage_path ./complex_fb15k_autobatch --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --neg_ratio 1 --num_epochs 100 --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True

# python ../main.py --storage_path ./distmult_yago_autobatch --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
# python ../main.py --storage_path ./complex_yago_autobatch --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True


# python ../main.py --storage_path ./distmult_fb15k_autobatch_kvsall --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique KvsAll --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
# python ../main.py --storage_path ./complex_fb15k_autobatch_ksvall --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique KvsAll --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True



# python ../main.py --storage_path ./distmult_yago_autobatch_kvsall --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique KvsAll --batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True
python ../main.py --storage_path ./complex_yago_autobatch_kvsall --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique KvsAll --batch_size 8192 --lr 0.0016718252573185953  --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --use_ddp_batch_finder True


# Pykeen

# FB15K
# python ../main.py --storage_path ./pykeen_distmult_fb15k_autobatch --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True
# python ../main.py --storage_path ./pykeen_complex_fb15k_autobatch --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" --batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True

# YAGO
# change emb_dim, otherwise its too slow
python ../main.py --storage_path ./pykeen_distmult_yago_autobatch --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 4096 --lr 0.00113355532419969 --embedding_dim 128 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" --normalization "None" --use_ddp_batch_finder True

# python ../main.py --storage_path ./pykeen_complex_yago_autobatch --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 8192 --lr 0.0016718252573185953  --embedding_dim 256 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True



