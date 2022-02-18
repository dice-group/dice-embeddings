dataset_path="KGs/$1"
embedding_dim=$2
storage_path="Exp_Results_$1_$2"
lr=.01
num_epochs=1000
storage_path="Exp_Results_$1_$2"
mkdir "$storage_path"
echo "Number of epochs:$num_epochs"
echo "Learning rate:$lr"
echo "embedding_dim :$embedding_dim"

python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'DistMult' --lr $lr --embedding_dim "$embedding_dim"  --num_epochs $num_epochs> "$storage_path/DistMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KPDistMult' --lr $lr --embedding_dim "$embedding_dim"  --num_epochs $num_epochs > "$storage_path/KPDistMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KronE' --lr $lr --embedding_dim "$embedding_dim"  --num_epochs $num_epochs > "$storage_path/KronE.log"
python core/collect_results_from_logs.py --logs  "$storage_path/DistMult.log" "$storage_path/KPDistMult.log" "$storage_path/KronE.log"
echo 'Done!'