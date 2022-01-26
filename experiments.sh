echo 'How good are learned embeddings under number of epochs and number of parameter constraints?'
num_epochs=100
echo "Number of epochs:$num_epochs"
echo "Number of real values to represent a single possible embedding vector 8 for OMult"

python main.py --path_dataset_folder 'KGs/Countries-S1' --model 'OMult' --embedding_dim 1 --num_epochs $num_epochs > OMult.log
python main.py --path_dataset_folder 'KGs/Countries-S1' --model 'QMult' --embedding_dim 2 --num_epochs $num_epochs > QMult.log
python main.py --path_dataset_folder 'KGs/Countries-S1' --model 'ComplEx' --embedding_dim 4 --num_epochs $num_epochs > ComplEx.log
python main.py --path_dataset_folder 'KGs/Countries-S1' --model 'DistMult' --embedding_dim 8 --num_epochs $num_epochs > DistMult.log
python main.py --path_dataset_folder 'KGs/Countries-S1' --model 'KronE' --entity_embedding_dim 6 --rel_embedding_dim 2 --num_epochs $num_epochs > KronE.log
python collect_results_from_logs.py --logs OMult.log QMult.log DistMult.log ComplEx.log KronE.log
echo 'Done!'