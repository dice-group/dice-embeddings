#!/bin/sh
# (1) main working directory
# shellcheck disable=SC2164
main_wd="$(cd "$PWD"; cd ..; pwd)"
# (2) Script in (1)
python_script_path="$main_wd/main.py"

# shellcheck disable=SC2043
for kgname in "Family"
do
  kg_path="$main_wd/KGs/$kgname"
  for model in "DistMult"
  do
    for epoch in 20
    do
      for dim in 256
      do
        for callback in "Polyak"
          do
          # shellcheck disable=SC2154
          config_name="$kgname-$model-$epoch-$dim-$callback"
          echo "Running $config_name.log"
          python -u "$python_script_path" --num_folds_for_cv 10 --callback "$callback" --path_dataset_folder "$kg_path" --model "$model" --num_epochs "$epoch" --embedding_dim "$dim" > "$config_name.log"
          echo "Done!"
          done
      done
    done
  done
done
