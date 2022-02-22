#!/bin/sh

# shellcheck disable=SC2164
dir_parent="$(cd "$PWD"; cd ..; pwd)"

path_script="$dir_parent/main.py"

models_name="Shallom"

path_dataset_folder="$dir_parent/KGs/EN_FR_15K_V1"

storage_path=$models_name
# shellcheck disable=SC2039
storage_path+="_EN_FR_15K_V1"
mkdir $storage_path
python "$path_script" --path_dataset_folder "$path_dataset_folder" --storage_path "$storage_path" --model "$models_name" --embedding_dim 300 --num_epochs 300 > "$storage_path/$models_name.log"


path_dataset_folder="$dir_parent/KGs/EN_FR_15K_V2"
storage_path=$models_name
# shellcheck disable=SC2039
storage_path+="_EN_FR_15K_V2"
mkdir $storage_path
python "$path_script" --path_dataset_folder "$path_dataset_folder" --storage_path "$storage_path" --model "$models_name" --embedding_dim 300 --num_epochs 300 > "$storage_path/$models_name.log"



path_dataset_folder="$dir_parent/KGs/EN_FR_100K_V1"
models_name="Shallom"
storage_path=$models_name
# shellcheck disable=SC2039
storage_path+="_EN_FR_100K_V1"
mkdir $storage_path
python "$path_script" --path_dataset_folder "$path_dataset_folder" --storage_path "$storage_path" --model "$models_name" --embedding_dim 300 --num_epochs 300 > "$storage_path/$models_name.log"




path_dataset_folder="$dir_parent/KGs/EN_FR_100K_V2"
storage_path=$models_name
# shellcheck disable=SC2039
storage_path+="_EN_FR_100K_V2"
mkdir $storage_path
python "$path_script" --path_dataset_folder "$path_dataset_folder" --storage_path "$storage_path" --model "$models_name" --embedding_dim 300 --num_epochs 300 > "$storage_path/$models_name.log"
