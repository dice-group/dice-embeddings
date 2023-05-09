# Link Prediction on Benchmark Datasets

Here, we show that generalization performance of knowledge graph embedding models do not differ much if they are trained well.

# Hyperparameter Setting

Hyperparameters play an important role in the successful applications of knowledge graph embedding models.
In our experiments, we selected such hyperparameter configuration so that experiments can be done less than a minute on UMLS dataset

# Link Prediction Performance Analysis on KINSHIP

| model_name   |   train_mrr |   train_h1 |   train_h3 |   train_h10 |   val_mrr |   val_h1 |   val_h3 |   val_h10 |   test_mrr |   test_h1 |   test_h3 |   test_h10 |   runtime |
|:-------------|------------:|-----------:|-----------:|------------:|----------:|---------:|---------:|----------:|-----------:|----------:|----------:|-----------:|----------:|
| DistMult     |    0.781619 |   0.681706 |   0.853406 |    0.967228 |  0.656955 | 0.516386 | 0.744382 |  0.941948 |   0.640159 |  0.496741 |  0.724395 |   0.938547 |   43.4192 |
| ComplEx      |    0.883142 |    0.81291 |   0.947507 |     0.99011 |  0.757905 | 0.636236 | 0.860019 |   0.96161 |   0.762234 |  0.646182 |  0.851955 |   0.972533 |   45.6701 |
| QMult        |    0.864761 |   0.789033 |   0.930185 |    0.985194 |  0.742049 |  0.61985 |  0.83661 |  0.955993 |    0.75211 |  0.636406 |  0.841713 |   0.956238 |   51.2721 |
| OMult        |     0.85052 |   0.771126 |   0.917427 |    0.981449 |  0.744363 | 0.628277 | 0.834738 |   0.94897 |   0.745078 |  0.625233 |  0.838454 |   0.963687 |   47.2538 |
| ConEx        |    0.726664 |   0.603406 |   0.815485 |    0.952891 |   0.68413 | 0.549157 | 0.779494 |  0.941479 |   0.676157 |  0.544693 |  0.768156 |   0.934358 |   52.2601 |
| ConvQ        |    0.715911 |   0.586142 |   0.807584 |    0.957104 |  0.669007 | 0.525749 | 0.768258 |  0.948502 |   0.667853 |  0.524209 |  0.770019 |   0.935754 |   60.0243 |
| ConvO        |    0.589155 |   0.433462 |   0.677434 |    0.918305 |  0.558573 | 0.401685 | 0.639513 |  0.902154 |    0.55491 |   0.39851 |  0.634078 |   0.902235 |   51.1115 |
| AConvO       |    0.855045 |   0.770365 |   0.931355 |    0.986189 |   0.73474 | 0.609551 | 0.831461 |  0.950843 |   0.741717 |   0.61825 |  0.836127 |   0.962291 |   50.5469 |
| AConEx       |    0.888418 |     0.8204 |    0.95055 |    0.989993 |   0.75704 | 0.637172 | 0.858614 |  0.955993 |   0.753787 |  0.634078 |  0.847765 |   0.967877 |   48.5488 |
| AConvQ       |    0.888232 |   0.822156 |   0.947507 |    0.989876 |   0.74992 | 0.625468 | 0.852528 |  0.961142 |   0.752967 |  0.625698 |  0.856145 |   0.969274 |   46.3772 |


# Link Prediction Performance Analysis on UMLS

1. Multiplicative models fit the training dataset split of UMLS better than convolutional neural network based models.
2. Replacing the multiplicative connections of conv(h,r) with additive connections leads convolutional neural network based models to fit better the training data.
3. Additive connections decrease the runtimes of the neural network based models.

| model_name   |   train_mrr |   train_h1 |   train_h3 |   train_h10 |   val_mrr |   val_h1 |   val_h3 |   val_h10 |   test_mrr |   test_h1 |   test_h3 |   test_h10 |   runtime |
|:-------------|------------:|-----------:|-----------:|------------:|----------:|---------:|---------:|----------:|-----------:|----------:|----------:|-----------:|----------:|
| DistMult     |    0.997396 |   0.994919 |          1 |           1 |     0.739 | 0.622699 | 0.821319 |  0.954755 |   0.738951 |  0.620272 |  0.824508 |   0.947806 |   32.1166 |
| ComplEx      |    0.998107 |   0.996357 |   0.999904 |           1 |  0.725287 | 0.604294 | 0.809816 |  0.950153 |   0.748761 |  0.633888 |   0.83056 |   0.958396 |   39.2883 |
| QMult        |    0.997148 |   0.994824 |   0.999712 |           1 |  0.733427 | 0.611963 |  0.82592 |  0.958589 |   0.732836 |  0.611195 |  0.826778 |   0.960666 |   37.2082 |
| OMult        |    0.985969 |   0.972776 |   0.999233 |           1 |  0.754292 | 0.629601 | 0.855828 |  0.959356 |   0.762545 |  0.636914 |  0.857035 |    0.96823 |   38.2151 |
| ConEx        |    0.963993 |   0.932707 |   0.995399 |    0.999041 |   0.80534 | 0.705521 | 0.883436 |  0.957822 |   0.810177 |   0.70121 |  0.906203 |   0.968986 |   46.6404 |
| ConvO        |    0.912093 |   0.837615 |    0.98495 |    0.997316 |  0.832247 | 0.721626 | 0.934049 |  0.977761 |   0.830195 |  0.714826 |  0.933434 |   0.984871 |   58.9782 |
| ConvQ        |    0.904769 |   0.831959 |   0.975748 |    0.993194 |  0.834325 | 0.730828 | 0.927147 |  0.976227 |   0.827083 |  0.711044 |  0.939486 |   0.977307 |   42.0177 |
| AConEx       |    0.995367 |   0.991181 |   0.999617 |           1 |  0.719798 |  0.58589 | 0.824387 |  0.943252 |   0.734349 |  0.610439 |  0.824508 |    0.95764 |   40.6111 |
| AConvQ       |    0.995162 |   0.990702 |   0.999808 |           1 |  0.709874 | 0.568252 | 0.819018 |  0.956288 |   0.737209 |  0.607413 |  0.850983 |   0.953858 |   42.8576 |
| AConvO       |    0.986693 |   0.974214 |   0.999041 |           1 |  0.698491 | 0.548313 | 0.817485 |  0.946319 |   0.732535 |  0.594554 |   0.83888 |   0.961422 |   44.2366 |



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

