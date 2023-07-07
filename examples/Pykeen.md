# Using Pykeen

# Training a Pykeen Model with Pytorch-lightning
```
python main.py --model Pykeen_MuRE --num_epochs 10 --batch_size 256 --lr 0.1 --trainer "PL" --num_core 4 --scoring_technique KvsAll --pykeen_model_kwargs embedding_dim=64
python main.py --model Pykeen_HolE --num_epochs 10 --batch_size 256 --lr 0.1 --trainer "PL" --num_core 4 --scoring_technique KvsAll --pykeen_model_kwargs embedding_dim=64
python main.py --model Pykeen_DistMult --num_epochs 10 --batch_size 256 --lr 0.1 --trainer "PL" --num_core 4 --scoring_technique KvsAll --pykeen_model_kwargs embedding_dim=64
python main.py --model Pykeen_ComplEx --num_epochs 10 --batch_size 256 --lr 0.1 --trainer "PL" --num_core 4 --scoring_technique KvsAll --pykeen_model_kwargs embedding_dim=32
python main.py --model Pykeen_QuatE --num_epochs 10 --batch_size 256 --lr 0.1 --trainer "PL" --num_core 4 --scoring_technique KvsAll --pykeen_model_kwargs embedding_dim=16
python analyse_experiments.py
| model_name      |   train_mrr |   train_h1 |   train_h3 |   train_h10 |   val_mrr |   val_h1 |   val_h3 |   val_h10 |   test_mrr |   test_h1 |   test_h3 |   test_h10 |   runtime |   params |
|:----------------|------------:|-----------:|-----------:|------------:|----------:|---------:|---------:|----------:|-----------:|----------:|----------:|-----------:|----------:|---------:|
| Pykeen_DistMult |    0.827423 |   0.707918 |   0.938267 |    0.992331 |  0.794849 | 0.661043 | 0.918712 |  0.980061 |   0.776859 |  0.639183 |  0.901664 |   0.982602 |   3.9479  |    14528 |
| Pykeen_MuRE     |    0.96874  |   0.94306  |   0.995878 |    1        |  0.79331  | 0.685583 | 0.882669 |  0.969325 |   0.804933 |  0.705749 |  0.887292 |   0.967474 |   4.89179 |    20686 |
| Pykeen_ComplEx  |    0.771764 |   0.673792 |   0.846817 |    0.925422 |  0.709708 | 0.598926 | 0.784509 |  0.895706 |   0.701002 |  0.58472  |  0.7882   |   0.8941   |   3.13179 |    14528 |
| Pykeen_HolE     |    0.945462 |   0.901169 |   0.990798 |    0.999425 |  0.778686 | 0.667945 | 0.868098 |  0.960123 |   0.760102 |  0.641452 |  0.854766 |   0.962935 |   5.46916 |    14528 |
| Pykeen_QuatE    |    0.92043  |   0.870207 |   0.96434  |    0.988976 |  0.817052 | 0.72546  | 0.898006 |  0.95092  |   0.819355 |  0.726929 |  0.894856 |   0.955371 |   4.24787 |    14528 |
```