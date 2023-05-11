
@REM CALL :StartPykeen1 ../KGs/YAGO3-10 , Pykeen_DistMult @REM lcwa
@REM CALL :StartPykeen2 ../KGs/YAGO3-10 , Pykeen_ComplEx @REM lcwa
@REM CALL :StartPykeen3 ../KGs/YAGO3-10 , Pykeen_DistMult @REM slcwa
@REM CALL :StartPykeen4 ../KGs/YAGO3-10 , Pykeen_ComplEx @REM slcwa
@REM CALL :StartDice1 ../KGs/YAGO3-10 , DistMult
@REM CALL :StartDice2 ../KGs/YAGO3-10 , ComplEx


@REM the hyperparameters are choosen according to https://github.com/pykeen/benchmarking

SET DATA="../KGs/YAGO3-10"


:StartPykeen1
python ../main.py --path_dataset_folder %DATA% --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen2
python ../main.py --path_dataset_folder %DATA% --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen3
python ../main.py --path_dataset_folder %DATA% --model Pykeen_DistMult --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartPykeen4
python ../main.py --path_dataset_folder %DATA% --model Pykeen_ComplEx --num_epochs 100 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 --pykeen_model_kwargs embedding_dim=256 loss="BCEWithLogitsLoss" ^
--use_SLCWALitModule --save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"

:StartDice1
python ../main.py --path_dataset_folder %DATA% --model DistMult --num_epochs 200 --scoring_technique "NegSample" ^
--batch_size 4096 --lr 0.00113355532419969 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"
:StartDice2
python ../main.py --path_dataset_folder %DATA% --model ComplEx --num_epochs 200 --scoring_technique "NegSample" ^
--batch_size 8192 --lr 0.001723135381847608 --embedding_dim 256 --trainer "PL" --neg_ratio 1 ^
--save_embeddings_as_csv --eval_model "train_val_test" --num_core 0 --accelerator "auto" --devices 1 --optim "Adam"


