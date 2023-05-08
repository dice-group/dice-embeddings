# Link Prediction on UMLS
1. Embedding dim 32
2. Number of epochs 256
3. Batch size 1024
4. Optimizer Adam 
5. Learning rate 0.1
6. No regularization


# Training Performance Analysis

Additive connections lead models to fit better the training data.

# Testing Performance Analysis
.

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

