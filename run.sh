#!/bin/sh
python --version
python -u -c 'import torch; print(torch.__version__)'
echo "Start Training......"

# Checked
python main.py --profile True --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3


# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'QMult' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'OMult' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3


# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConvQ' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3


# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ComplEx' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'DistMult' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

echo "Ends......"