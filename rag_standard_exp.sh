#!/bin/bash

# Default values
embedding_dim=32
num_epochs=256
batch_size=1024

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_dir)
      dataset="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check if dataset is provided
if [[ -z "$dataset" ]]; then
  echo "Error: --dataset_dir is required"
  exit 1
fi

# Loop through models
for model in "DistMult" "ComplEx" "QMult" "Keci" "DeCaL" "Pykeen_MuRE" "Pykeen_QuatE"
do
  echo "Running model $model on dataset $dataset"
  dicee --model "$model" \
        --dataset_dir "$dataset" \
        --scoring_technique "KvsAll" \
        --embedding_dim "$embedding_dim" \
        --num_epochs "$num_epochs" \
        --batch_size "$batch_size" \
        --input_dropout_rate 0.3 \
        --optim "Adam"
done
