#!/bin/sh
# (1) main working directory
# shellcheck disable=SC2164
main_wd="$(cd "$PWD"; cd ..; pwd)"
# (2) Script in (1)
python_script_path="$main_wd/main.py"

# shellcheck disable=SC2043
for kgname in "OpenEA_V1.1_EN_FR_15K_V1" "OpenEA_V1.1_EN_FR_15K_V2" "OpenEA_V1.1_EN_FR_100K_V1" "OpenEA_V1.1_EN_FR_100K_V2"
do
  kg_path="$main_wd/KGs/$kgname"
  for model in "Shallom"
  do
    for epoch in 1
    do
      for dim in 25
      do
          # shellcheck disable=SC2154
          config_name="$kgname-$model-$epoch-$dim"
          echo "Running $config_name.log"
          /bin/bash "$PWD/config_runner.sh" "$python_script_path" "$kg_path" "$model" "$epoch" "$dim" > "$config_name.log"
          echo "Done!"
      done
    done
  done
done
