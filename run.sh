#!/bin/sh
python --version
python -u -c 'import torch; print(torch.__version__)'
echo "Start Training......"

# To deserialize parsed KG.
#python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConEx' --deserialize_flag '/home/demir/Desktop/work/DAIKIRI_Emb/DAIKIRI_Storage/2021-05-12 12:11:12.095319'
# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConEx'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConEx' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'Shallom' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'Shallom' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'QMult' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'QMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'OMult' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'OMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3


# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConvQ' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvQ' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3


# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ConvO' --max_num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'ComplEx' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'ComplEx' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

# Checked
python main.py --path_dataset_folder 'KGs/UMLS' --model 'DistMult' --num_epochs 3 --scoring_technique 'KvsAll'
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --num_epochs 3 --scoring_technique 'KvsAll' --num_folds_for_cv 3
python main.py --path_dataset_folder 'KGs/Family' --model 'DistMult' --num_epochs 3 --scoring_technique 'NegSample' --negative_sample_ratio 3 --num_folds_for_cv 3

echo "Ends......"