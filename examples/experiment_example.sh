d=128
num_epochs=300

for dataset in "UMLS" "KINSHIP" "NELL-995-h100" "NELL-995-h75" "NELL-995-h25" "WN18RR" "FB15k-237" "YAGO3-10"
do
echo $dataset
for model in "DistMult" "ComplEx" "QMult" "Keci"
do
echo $dataset
# train a KG model on a single node/server having two GPUs in the DDP setting
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --trainer torchDDP --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll &
# train a KG model with SWA on a single node/server having two GPUs in the DDP setting
python -m main --trainer PL --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-SWA"  --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --lr 0.1 --optim Adam --scoring_technique KvsAll --stochastic_weight_avg True --accelerator gpu --devices 2 &
# train a KG model with ASWA on a single node/server having two GPUs in the DDP setting
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --trainer torchDDP --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-ASWA" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll --adaptive_swa
done
done
