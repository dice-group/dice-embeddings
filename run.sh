#!/bin/sh
python --version
python -u -c 'import torch; print(torch.__version__)'
echo "Start Training......"
python main.py --path_dataset_folder 'KGs/Animals' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Biopax' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Forte' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Carcinogenesis' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Hepatitis' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Lymphography' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Mammographic' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Nctrer' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25
python main.py --path_dataset_folder 'KGs/Mutagenesis' --model 'ConEx' --max_num_epochs 100 --embedding_dim 25

python main.py --path_dataset_folder 'KGs/Animals' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Biopax' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Carcinogenesis' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Forte' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Hepatitis' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Lymphography' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Mammographic' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Mutagenesis' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
python main.py --path_dataset_folder 'KGs/Nctrer' --model 'Shallom' --max_num_epochs 100 --embedding_dim 50
