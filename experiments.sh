
dataset_path="KGs/$1"
embedding_dim=$2
storage_path="Exp_Results_$1_$2"

lr=.01
num_epochs=1000
input_DR=0.0
hidden_DR=0.0

storage_path="Exp_Results_$1_$2"
# Number of epochs


mkdir "$storage_path"
echo "Number of epochs:$num_epochs"
echo "Learning rate:$lr"
echo "embedding_dim :$embedding_dim"

#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'OMult' --lr $lr --embedding_dim "$((embedding_dim/8))" --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs --num_folds_for_cv "$kfold_cv"> "$storage_path/OMult.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'QMult' --lr $lr --embedding_dim "$((embedding_dim/4))" --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs --num_folds_for_cv "$kfold_cv"> "$storage_path/QMult.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'ComplEx' --lr $lr --embedding_dim "$((embedding_dim/2))" --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs --num_folds_for_cv "$kfold_cv"> "$storage_path/ComplEx.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'DistMult' --lr $lr --embedding_dim "$embedding_dim" --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs> "$storage_path/DistMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KPDistMult' --lr $lr --embedding_dim "$embedding_dim" --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/KPDistMult.log"

python core/collect_results_from_logs.py --logs  "$storage_path/DistMult.log" "$storage_path/KPDistMult.log"

echo 'Done!'