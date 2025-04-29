# Retrieval Augmented Link Prediction

# Link Prediction Results via Static RALP
´´´
python -m retrieval_aug_predictors.models.static_ralp --enrich_train --dataset_dir KGs/Countries-S3 --out "countries_s3_results.json" && cat countries_s3_results.json
´´´

´´´
cp -r KGs/Countries-S3 KGs/Enriched_Countries-S3
cat KGs/Enriched_Countries-S3/missing_triples.txt >> KGs/Enriched_Countries-S3/train.txt
bash rag_standard_exp.sh --dataset_dir KGs/Enriched_Countries-S3
´´´
#### Latex Table 
´´´
python dicee/analyse_experiments.py --dir Experiments --features "model" "testMRR" "testH@1" "testH@3" "testH@10"
´´´
do not forget to delete.
´´´
rm -rf Experiments
´´´


bash rag_standard_exp.sh --dataset_dir KGs/Countries-S3