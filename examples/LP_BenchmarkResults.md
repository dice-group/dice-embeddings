# Link Prediction on Benchmark Datasets

Here, we show that generalization performance of knowledge graph embedding models do not differ much if they are trained well.

# Hyperparameter Setting

Hyperparameters play an important role in the successful applications of knowledge graph embedding models.
In our experiments, we selected such hyperparameter configuration so that experiments can be done less than a minute on UMLS dataset

# Link Prediction Performance Analysis on UMLS

1. Multiplicative models fit the training dataset split of UMLS better than convolutional neural network based models.
2. Replacing the multiplicative connections of conv(h,r) with additive connections leads convolutional neural network based models to fit better the training data.
3. Additive connections decrease the runtimes of the neural network based models.


|                |   MRR | Hits@1 |   Hits@3 |  Hits@10 | Runtime |
|----------------|------:|-------:|---------:|---------:|--------:|
| DistMult-train | 0.997 |  0.995 |    1.000 |    1.000 |      35 |
| DistMult-val   | 0.739 |  0.623 |    0.821 |    0.955 |         |
| DistMult-test  | 0.739 |  0.620 |    0.824 |    0.948 |         |
| ComplEx-train  | 0.998 |  0.996 |    1.000 |    1.000 |      37 |
| ComplEx-val    | 0.725 |  0.604 |    0.810 |    0.950 |         |
| ComplEx-test   | 0.749 |  0.634 |    0.831 |    0.958 |         |
| QMult-train    | 0.997 |  0.995 |    1.000 |    1.000 |      49 |
| QMult-val      | 0.733 |  0.612 |    0.826 |    0.959 |         |
| QMult-test     | 0.733 |  0.611 |    0.827 |    0.961 |         |
| OMult-train    | 0.986 |  0.973 |    0.999 |    1.000 |      40 |
| OMult-val      | 0.754 |  0.630 |    0.856 |    0.959 |         |
| OMult-test     | 0.762 |  0.637 |    0.857 |    0.968 |         |
| ConEx-train    | 0.964 |  0.933 |    0.995 |    0.999 |      50 |
| ConEx-val      | 0.805 |  0.705 |    0.883 |    0.958 |         |
| ConEx-test     | 0.810 |  0.701 |    0.906 |    0.969 |         |
| ConvQ-train    | 0.905 |  0.832 |    0.976 |    0.993 |      37 |
| ConvQ-val      | 0.834 |  0.731 |    0.927 |    0.976 |         |
| ConvQ-test     | 0.827 |  0.711 | 0.939486 |    0.977 |         |
| ConvO-train    | 0.912 |  0.837 |    0.985 |    0.997 |      43 |
| ConvO-val      | 0.832 |  0.722 |    0.934 |    0.978 |         |
| ConvO-test     | 0.830 |  0.715 | 0.933434 | 0.984871 |         |
| AConEx-train   | 0.995 |  0.991 |    1.000 |    1.000 |      38 |
| AConEx-val     | 0.720 |  0.586 | 0.824387 | 0.943252 |         |
| AConEx-test    | 0.734 |  0.610 | 0.824508 |  0.95764 |         |
| AConvQ-train   | 0.995 |  0.991 |    0.976 |    0.993 |      37 |
| AConvQ-val     | 0.710 |  0.568 | 0.819018 | 0.956288 |         |
| AConvQ-test    | 0.737 |  0.607 | 0.850983 | 0.953858 |         |
| AConvO-train   | 0.987 |  0.974 |    0.999 |    1.000 |      43 |
| AConvO-val     | 0.698 |  0.548 | 0.817485 | 0.946319 |         |
| AConvO-test    | 0.732 |  0.594 |  0.83888 | 0.961422 |         |



## ConEx (Convolutional Complex Knowledge Graph Embeddings)
```bash
python main.py --path_dataset_folder KGs/UMLS --model "ConEx" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 38.546 seconds
Evaluate ConEx on Train set: Evaluate ConEx on Train set
{'H@1': 0.932707055214724, 'H@3': 0.995398773006135, 'H@10': 0.9990414110429447, 'MRR': 0.9639930469284392}
Evaluate ConEx on Validation set: Evaluate ConEx on Validation set
{'H@1': 0.7055214723926381, 'H@3': 0.8834355828220859, 'H@10': 0.9578220858895705, 'MRR': 0.8053400891316801}
Evaluate ConEx on Test set: Evaluate ConEx on Test set
{'H@1': 0.7012102874432677, 'H@3': 0.9062027231467473, 'H@10': 0.9689863842662633, 'MRR': 0.8101772584084878}
```

## AConEx (ConEx with Additive Connections)
```bash
python main.py --path_dataset_folder KGs/UMLS --model "AConEx" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 38.555 seconds
Evaluate AConEx on Train set: Evaluate AConEx on Train set
{'H@1': 0.991180981595092, 'H@3': 0.9996165644171779, 'H@10': 1.0, 'MRR': 0.9953668200408997}
Evaluate AConEx on Validation set: Evaluate AConEx on Validation set
{'H@1': 0.5858895705521472, 'H@3': 0.8243865030674846, 'H@10': 0.9432515337423313, 'MRR': 0.7197978156406648}
Evaluate AConEx on Test set: Evaluate AConEx on Test set
{'H@1': 0.6104387291981845, 'H@3': 0.8245083207261724, 'H@10': 0.9576399394856279, 'MRR': 0.734348890030545}
```

## ConvQ
```bash
python main.py --path_dataset_folder KGs/UMLS --model "ConvQ" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 41.013 seconds
Evaluate ConvQ on Train set: Evaluate ConvQ on Train set
{'H@1': 0.8319593558282209, 'H@3': 0.9757476993865031, 'H@10': 0.993194018404908, 'MRR': 0.9047693553188044}
Evaluate ConvQ on Validation set: Evaluate ConvQ on Validation set
{'H@1': 0.7308282208588958, 'H@3': 0.9271472392638037, 'H@10': 0.9762269938650306, 'MRR': 0.834325039087529}
Evaluate ConvQ on Test set: Evaluate ConvQ on Test set
{'H@1': 0.7110438729198184, 'H@3': 0.9394856278366112, 'H@10': 0.9773071104387292, 'MRR': 0.8270827146655849}
```

## AConvQ (ConvQ with with Additive Connections)
```bash
python main.py --path_dataset_folder KGs/UMLS --model "AConvQ" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 41.238 seconds
Evaluate AConvQ on Train set: Evaluate AConvQ on Train set
{'H@1': 0.9907016871165644, 'H@3': 0.999808282208589, 'H@10': 1.0, 'MRR': 0.9951623210633946}
Evaluate AConvQ on Validation set: Evaluate AConvQ on Validation set
{'H@1': 0.5682515337423313, 'H@3': 0.8190184049079755, 'H@10': 0.9562883435582822, 'MRR': 0.7098740236669477}
Evaluate AConvQ on Test set: Evaluate AConvQ on Test set
{'H@1': 0.6074130105900152, 'H@3': 0.850983358547655, 'H@10': 0.953857791225416, 'MRR': 0.737209298931209}
```

## ConvO
```bash
python main.py --path_dataset_folder KGs/UMLS --model "ConvO" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 54.756 seconds
Evaluate ConvO on Train set: Evaluate ConvO on Train set
{'H@1': 0.8376150306748467, 'H@3': 0.9849501533742331, 'H@10': 0.9973159509202454, 'MRR': 0.9120930363683276}
Evaluate ConvO on Validation set: Evaluate ConvO on Validation set
{'H@1': 0.7216257668711656, 'H@3': 0.9340490797546013, 'H@10': 0.977760736196319, 'MRR': 0.8322465353766179}
Evaluate ConvO on Test set: Evaluate ConvO on Test set
{'H@1': 0.7148260211800302, 'H@3': 0.9334341906202723, 'H@10': 0.9848714069591528, 'MRR': 0.8301952668172322}

```

## AConvO
```bash
python main.py --path_dataset_folder KGs/UMLS --model "AConvO" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --kernel_size 3 --num_of_output_channels 3 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 42.910 seconds
Evaluate AConvO on Train set: Evaluate AConvO on Train set
{'H@1': 0.9742139570552147, 'H@3': 0.9990414110429447, 'H@10': 1.0, 'MRR': 0.986693187627812}
Evaluate AConvO on Validation set: Evaluate AConvO on Validation set
{'H@1': 0.5483128834355828, 'H@3': 0.8174846625766872, 'H@10': 0.946319018404908, 'MRR': 0.6984911756060336}
Evaluate AConvO on Test set: Evaluate AConvO on Test set
{'H@1': 0.594553706505295, 'H@3': 0.8388804841149773, 'H@10': 0.9614220877458396, 'MRR': 0.7325348232341585}
```

## DistMult
```bash
python main.py --path_dataset_folder KGs/UMLS --model "DistMult" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0  --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 43.789 seconds
Evaluate DistMult on Train set: Evaluate DistMult on Train set
{'H@1': 0.9949194785276073, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9973958333333333}
Evaluate DistMult on Validation set: Evaluate DistMult on Validation set
{'H@1': 0.6226993865030674, 'H@3': 0.821319018404908, 'H@10': 0.9547546012269938, 'MRR': 0.7389999406726034}
Evaluate DistMult on Test set: Evaluate DistMult on Test set
{'H@1': 0.6202723146747352, 'H@3': 0.8245083207261724, 'H@10': 0.9478063540090772, 'MRR': 0.7389508344813291}
```

## ComplEx
```bash
python main.py --path_dataset_folder KGs/UMLS --model "ComplEx" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0 --feature_map_dropout_rate 0.0 --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 37.448 seconds
Evaluate ComplEx on Train set: Evaluate ComplEx on Train set
{'H@1': 0.9963573619631901, 'H@3': 0.9999041411042945, 'H@10': 1.0, 'MRR': 0.9981067868098159}
Evaluate ComplEx on Validation set: Evaluate ComplEx on Validation set
{'H@1': 0.6042944785276073, 'H@3': 0.8098159509202454, 'H@10': 0.9501533742331288, 'MRR': 0.725286550123257}
Evaluate ComplEx on Test set: Evaluate ComplEx on Test set
{'H@1': 0.6338880484114977, 'H@3': 0.8305597579425114, 'H@10': 0.9583963691376702, 'MRR': 0.7487605914404327}

```
## QMult
```bash
python main.py --path_dataset_folder KGs/UMLS --model "QMult" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0  --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 35.082 seconds
Evaluate QMult on Train set: Evaluate QMult on Train set
{'H@1': 0.9948236196319018, 'H@3': 0.9997124233128835, 'H@10': 1.0, 'MRR': 0.9971481978527608}
Evaluate QMult on Validation set: Evaluate QMult on Validation set
{'H@1': 0.6119631901840491, 'H@3': 0.825920245398773, 'H@10': 0.9585889570552147, 'MRR': 0.7334272790319509}
Evaluate QMult on Test set: Evaluate QMult on Test set
{'H@1': 0.6111951588502269, 'H@3': 0.8267776096822995, 'H@10': 0.9606656580937972, 'MRR': 0.7328362412430357}

```
## OMult
```bash
python main.py --path_dataset_folder KGs/UMLS --model "OMult" --optim Adam --embedding_dim 32 --num_epochs 256 --batch_size 1024 --lr 0.1 --backend 'pandas' --trainer 'PL' --scoring_technique 'KvsAll' --weight_decay 0.0 --input_dropout_rate 0.0 --hidden_dropout_rate 0.0  --normalization LayerNorm --init_param xavier_normal --label_smoothing_rate 0.0 --seed_for_computation 0 --num_core 4
```
```bash
Total computation time: 52.970 seconds
Evaluate OMult on Train set: Evaluate OMult on Train set
{'H@1': 0.9727760736196319, 'H@3': 0.9992331288343558, 'H@10': 1.0, 'MRR': 0.985969452965235}
Evaluate OMult on Validation set: Evaluate OMult on Validation set
{'H@1': 0.629601226993865, 'H@3': 0.8558282208588958, 'H@10': 0.9593558282208589, 'MRR': 0.754292154154622}
Evaluate OMult on Test set: Evaluate OMult on Test set
{'H@1': 0.6369137670196672, 'H@3': 0.857034795763994, 'H@10': 0.9682299546142209, 'MRR': 0.7625450316560369}
```


