echo 'How good are learned embeddings under number of epochs and number of parameter constraints?'

num_epochs=100
dataset_path="KGs/Countries-S1"
storage_path='Exp_Results_Countries-S1'
mkdir $storage_path
echo "Number of epochs:$num_epochs"
echo "Number of real values to represent a single possible embedding vector 8 for OMult"

python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'OMult' --embedding_dim 1 --num_epochs $num_epochs > "$storage_path/OMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'QMult' --embedding_dim 2 --num_epochs $num_epochs > "$storage_path/QMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'ComplEx' --embedding_dim 4 --num_epochs $num_epochs > "$storage_path/ComplEx.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'DistMult' --embedding_dim 8 --num_epochs $num_epochs > "$storage_path/DistMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KronE' --entity_embedding_dim 6 --rel_embedding_dim 2 --num_epochs $num_epochs > "$storage_path/KronE.log"
python core/collect_results_from_logs.py --logs "$storage_path/OMult.log" "$storage_path/QMult.log" "$storage_path/ComplEx.log" "$storage_path/DistMult.log" "$storage_path/KronE.log"
echo 'Done!'