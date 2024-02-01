d=128
num_epochs=300

for dataset in "Countries-S1" "Countries-S2" "Countries-S3 "
do

for model in "DistMult" "ComplEx" "QMult" "Keci"
do

echo $dataset-$model


# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1
dicee --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll > "$dataset-$model.txt"
# PL
dicee --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-SWA"  --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024  --lr 0.1 --optim Adam --scoring_technique KvsAll --adaptive_swa > "$dataset-$model-SWA.txt"

# Torch DDP
dicee --path_to_store_single_run "$dataset-$model-E$num_epochs-D$d-ASWA" --dataset_dir "KGs/$dataset" --model $model --embedding_dim $d --num_epochs $num_epochs --batch_size 1024 --lr 0.1 --optim Adam --scoring_technique KvsAll --adaptive_swa > "$dataset-$model-ASWA.txt"
done
done
