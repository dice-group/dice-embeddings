
@REM CALL :StartPykeen1 ../KGs/FB15k-237 , Pykeen_DistMult
@REM CALL :StartPykeen2 ../KGs/FB15k-237 , Pykeen_ComplEx
@REM CALL :StartDice1 ../KGs/FB15k-237 , DistMult
@REM CALL :StartDice2 ../KGs/FB15k-237 , ComplEx


@REM the hyperparameters are choosen according to https://github.com/pykeen/benchmarking


SET DATA="../KGs/FB15k-237"

:StartPykeen1
python ../main.py --path_dataset_folder %DATA% --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=64 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen2
python ../main.py --path_dataset_folder %DATA%1 --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


:StartDice1
python ../main.py --path_dataset_folder %DATA% --model DistMult --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.001849662035249092 --embedding_dim 64 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartDice2
python ../main.py --path_dataset_folder %DATA% --model ComplEx --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 256 --lr 0.007525067744232913 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


