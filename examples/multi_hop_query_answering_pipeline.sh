#!/bin/bash
# To run this script, from the parent directory of examples,
# cp examples/multi_hop_query_answering_pipeline.sh run.sh && sh run.sh && rm MultiHopQuery && rm run.sh

# This script create multi-hop queries
# (1) Generate queries
# (2) Train a KGE model
# (3) Evaluates (2) on multi-hop queries
# To use this script, move it to the parent directory of examples directory
dir_exp="MultiHopQuery"

# (1) Generate Queries:
query_path="$dir_exp"
cp -r KGs/UMLS "$query_path"
python -m dicee.mappings --datapath "$query_path" --map_to_ids --indexify_files
python -m dicee.create_queries --dataset "$query_path" --gen_test_num 100 --gen_test --save_name --gen_all
python -m dicee.mappings --datapath "$query_path" --unmap_to_text --join_queries --file_type unmapped


# (2) Train Models:
for model in "DistMult" "ComplEx" "QMult"
do
  python -m dicee.run --model "$model" --trainer PL --scoring_technique "KvsAll" --embedding_dim 256 --num_epochs 256 --batch_size 1024 --path_dataset_folder "KGs/UMLS" --path_to_store_single_run "$dir_exp/UMLS_PPE_$model" --callbacks  "{\"PPE\":{\"last_percent_to_consider\":20}}"
  python -m dicee.run --model "$model" --trainer PL --scoring_technique "KvsAll" --embedding_dim 256 --num_epochs 256 --batch_size 1024 --path_dataset_folder "KGs/UMLS" --path_to_store_single_run "$dir_exp/UMLS_$model"
done

# (3) Report the LP results
for model in "DistMult" "ComplEx" "QMult"
do
  echo  "$model LP results"
  cat "$dir_exp/UMLS_$model/eval_report.json"
  cat "$dir_exp/UMLS_PPE_$model/eval_report.json"
done
# (4) Eval complex query answering
for model in "DistMult" "ComplEx" "QMult"
do
  echo  "$model Multi-hop Results"
  python -m dicee.complex_query_answering --datapath "$query_path" --experiment "$dir_exp/UMLS_$model" --tnorm 'prod' --neg_norm 'yager' --k_ 10 --lambda_ 0.4
  python -m dicee.complex_query_answering --datapath "$query_path" --experiment "$dir_exp/UMLS_PPE_$model" --tnorm 'prod' --neg_norm 'yager' --k_ 10 --lambda_ 0.4
done

# (3)
