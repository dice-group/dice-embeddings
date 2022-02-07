
dataset_path="KGs/Countries-S1"
storage_path='Exp_Results_X'
# Number of epochs
num_epochs=1
input_DR=0.0
hidden_DR=0.0

embedding_dim=3
entity_embedding_dim=3
rel_embedding_dim=3
lr=.01
rm -rf $storage_path
mkdir $storage_path
echo "Number of epochs:$num_epochs"
echo "Learning rate:$lr"

#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'OMult' --lr $lr --embedding_dim $embedding_dim --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/OMult.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'QMult' --lr $lr --embedding_dim $((2*embedding_dim)) --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/QMult.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'ComplEx' --lr $lr --embedding_dim $((4*embedding_dim)) --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/ComplEx.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'DistMult' --lr $lr --embedding_dim $((8*embedding_dim)) --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/DistMult.log"
#python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KronE' --lr $lr --entity_embedding_dim $entity_embedding_dim --rel_embedding_dim $rel_embedding_dim --input_dropout_rate $input_DR  --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/KronE.log"
#python core/collect_results_from_logs.py --logs "$storage_path/OMult.log" "$storage_path/QMult.log" "$storage_path/ComplEx.log" "$storage_path/DistMult.log" "$storage_path/KronE.log"

python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'DistMult' --lr $lr --embedding_dim $embedding_dim --input_dropout_rate $input_DR --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/DistMult.log"
python main.py --storage_path "$storage_path" --path_dataset_folder "$dataset_path" --model 'KronE' --lr $lr --entity_embedding_dim $entity_embedding_dim --rel_embedding_dim $rel_embedding_dim --input_dropout_rate $input_DR  --hidden_dropout_rate $hidden_DR --num_epochs $num_epochs > "$storage_path/KronE.log"
python core/collect_results_from_logs.py --logs "$storage_path/DistMult.log" "$storage_path/KronE.log"

echo 'Done!'