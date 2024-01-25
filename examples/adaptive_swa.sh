d=128
num_epochs=300

for dataset in "NELL-995-h50" "NELL-995-h75"
do

for model in "DistMult" "ComplEx" "QMult" "Keci"
do

echo $dataset-$model

# Torch DDP
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 dicee/scripts/run.py --trainer torchDDP --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll > "$dataset-$model.txt" 
# PL
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python dicee/scripts/run.py --trainer PL --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-SWA"  --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024  --lr 0.1 --optim Adam --scoring_technique KvsAll --adaptive_swa > "$dataset-$model-SWA.txt" 

# Torch DDP
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 dicee/scripts/run.py --trainer torchDDP --p 0 --q 1 --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-ASWA" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll --adaptive_swa > "$dataset-$model-ASWA.txt" 
done
done
