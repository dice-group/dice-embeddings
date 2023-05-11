@REM KINSHIP/Pykeen_DistMult
python ../main.py --path_dataset_folder "../KGs/KINSHIP" --model "Pykeen_DistMult" --num_epochs 20 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.01 --embedding_dim 128 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
@REM KINSHIP/Pykeen_ComplEx
python ../main.py --path_dataset_folder "../KGs/KINSHIP" --model "Pykeen_ComplEx" --num_epochs 20 --scoring_technique "NegSample" ^
--batch_size 128 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=128 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"

@REM KINSHIP/DistMult
python ../main.py --path_dataset_folder "../KGs/KINSHIP" --model "DistMult" --num_epochs 20 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.01 --embedding_dim 128 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
@REM KINSHIP/ComplEx
python ../main.py --path_dataset_folder "../KGs/KINSHIP" --model "ComplEx" --num_epochs 20 --scoring_technique "NegSample" ^
--batch_size 128 --lr 0.1 --embedding_dim 128 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
