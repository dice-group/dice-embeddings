#!/bin/sh
# (1) main working directory
# shellcheck disable=SC2164
main_wd="$(cd "$PWD"; cd ..; pwd)"
# (2) Script in (1)
python_script_path="$main_wd/main.py"

# shellcheck disable=SC2043
for kgname in "UMLS" "KINSHIP"
do
  kg_path="$main_wd/KGs/$kgname"
  for model in "QMult" "OMult"
  do
    for epoch in 1
    do
      for dim in 25 50
      do
          # shellcheck disable=SC2154
          log_name="$kg_path-$model-$epoch-$dim"
          echo "Running $log_name configuration"
          /bin/bash "$PWD/config_runner.sh" "$python_script_path" "$kg_path" "$model" "$epoch" "$dim" > "$log_name.log"
          echo "Done!"
      done
    done
  done
done
