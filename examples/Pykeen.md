# Using Pykeen

# Training Few Pykeen Models with KvsAll

```
dicee --dataset_dir KGs/UMLS --model Pykeen_MuRE --num_epochs 100 --batch_size 256 --lr 0.1 --scoring_technique KvsAll
dicee --dataset_dir KGs/UMLS --model Pykeen_HolE --num_epochs 100 --batch_size 256 --lr 0.1 --scoring_technique KvsAll
dicee --dataset_dir KGs/UMLS --model Pykeen_DistMult --num_epochs 100 --batch_size 256 --lr 0.1 --scoring_technique KvsAll
dicee --dataset_dir KGs/UMLS --model Pykeen_ComplEx --num_epochs 100 --batch_size 256 --lr 0.1 --scoring_technique KvsAll
dicee --dataset_dir KGs/UMLS --model Pykeen_QuatE --num_epochs 100 --batch_size 256 --lr 0.1 --scoring_technique KvsAll
```

```
python dicee/analyse_experiments.py --dir Experiments --features "model" "trainMRR" "testMRR"

\begin{tabular}{lrrr}
\toprule
model & trainMRR & testMRR & NumParam \\
\midrule
Pykeen_MuRE & 0.879 & 0.836 & 10478 \\
Pykeen_HolE & 0.830 & 0.689 & 7264 \\
Pykeen_QuatE & 1.000 & 0.683 & 29056 \\
Pykeen_ComplEx & 1.000 & 0.648 & 14528 \\
Pykeen_DistMult & 0.818 & 0.588 & 7264 \\
\bottomrule
\end{tabular}


```