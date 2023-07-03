

# # FB15K-237
# # DistMult

# # GPU
# python ../main.py --storage_path ./distmult_fb15k_gpu_1 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model train --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model train --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"




# # complex

# # GPU
# python ../main.py --storage_path ./complex_fb15k_gpu_1  --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"


# # kvsall
# # GPU

# # Distmult

# python ../main.py --storage_path ./distmult_kvsall_fb15k_gpu_1 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_kvsall_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_kvsall_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"




# # complex

# # GPU
# python ../main.py --storage_path ./complex_kvsall_fb15k_gpu_1 --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_kvsall_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_kvsall_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"





# PYkeen, LCWA

# Distmut
# python ../main.py --storage_path ./pykeen_distmult_lcwa_fb15k_gpu_1 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_distmult_lcwa_distmult_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_distmult_lcwa_distmult_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.001849662035249092 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train_test" --num_core 1 --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss"


# # # Complex
# python ../main.py --storage_path ./pykeen_complex_lcwa_fb15k_gpu_1 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_ComplEx  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train_test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_complex_lcwa_fb15k_gpu_2 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train_test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_complex_lcwa_fb15k_gpu_3 --path_dataset_folder ../KGs/FB15k-237 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.007525067744232913 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train_test" --num_core 1 --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"











# YAGO
# DistMult

# GPU
# python ../main.py --storage_path ./distmult_yago_gpu_1 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"




# # complex

# # GPU
# python ../main.py --storage_path ./complex_yago_gpu_1  --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001723135381847608 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001723135381847608 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --neg_ratio 1 --num_epochs 100 --scoring_technique "NegSample" --batch_size 512 --lr 0.001723135381847608 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"


# # kvsall
# # GPU

# # Distmult

# python ../main.py --storage_path ./distmult_kvsall_yago_gpu_1 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_kvsall_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./distmult_kvsall_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model DistMult --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"




# # complex
# # can only use small batch_size 
# # GPU
# python ../main.py --storage_path ./complex_kvsall_yago_gpu_1 --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "train" --num_core 0 --accelerator "gpu" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_kvsall_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "train" --optim "Adam" --normalization "None"
# python ../main.py --storage_path ./complex_kvsall_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model ComplEx --num_epochs 100 --scoring_technique KvsAll --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "train" --num_core 0 --optim "Adam" --normalization "None"



# # PYkeen, LCWA

# # Distmut
# python ../main.py --storage_path ./pykeen_distmult_lcwa_yago_gpu_1 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --trainer "PL" --save_embeddings_as_csv --eval_model "test" --accelerator "gpu" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --num_core 1
# python ../main.py --storage_path ./pykeen_distmult_lcwa_distmult_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" --use_ddp_batch_finder True
# python ../main.py --storage_path ./pykeen_distmult_lcwa_distmult_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_DistMult --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.00113355532419969 --embedding_dim 64 --gpus 1 --save_embeddings_as_csv --eval_model "test" --num_core 1 --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss"


# # # Complex
# python ../main.py --storage_path ./pykeen_complex_lcwa_yago_gpu_1 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_ComplEx  --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --trainer "PL" --save_embeddings_as_csv --eval_model "test" --num_core 1 --accelerator "gpu" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_complex_lcwa_yago_gpu_2 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --num_core 1 --trainer "torchDDP" --save_embeddings_as_csv --eval_model "test" --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"
# python ../main.py --storage_path ./pykeen_complex_lcwa_yago_gpu_3 --path_dataset_folder ../KGs/YAGO3-10 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "KvsAll" --batch_size 512 --lr 0.0016718252573185953 --embedding_dim 32 --gpus 1 --save_embeddings_as_csv --eval_model "test" --num_core 1 --optim "Adam" --normalization "None" --pykeen_model_kwargs embedding_dim=32 loss="BCEWithLogitsLoss"














