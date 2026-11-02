# Evaluating multi-hop query answering

`benchmarks/multihop_query_eval.py` evaluates a trained model on multi-hop queries. It reuses a model you
already trained for link prediction, so nothing needs retraining.

## Run

```bash
python benchmarks/multihop_query_eval.py \
    --path_to_model Experiments/<run> \
    --dataset_dir KGs/FB15k-237 \
    --query_types 1p 2p 3p 2i 3i ip pi 2u up \
    --gen_num 1000 --tnorm min --out results.json
```

`--path_to_model` is the directory of a trained dicee run. `--dataset_dir` holds `train.txt`, `valid.txt`
and `test.txt`. Other options: `--k` (intermediate entities kept for multi-hop projections), `--seed`,
`--probabilities`. Run `--help` for the full list.

## What it does

1. `dicee.QueryGenerator` samples queries from the dataset, for example `(e, (r1, r2))` for `2p`. Each
   query has *easy* answers (reachable in train + valid) and *hard* answers (reachable only with test
   triples).
2. `KGE.answer_multi_hop_query(query_type=..., query=..., only_scores=True)` scores every entity.
3. Each hard answer is ranked among all entities that are not answers. Other easy and hard answers are
   filtered out.
4. MRR and Hits@1/3/10 are averaged over hard answers, then over queries. The result is one row per query
   type, printed and optionally written to `--out`.

How `QueryGenerator` samples queries is described in
[docs/guides/multi_hop_queries.md](../docs/guides/multi_hop_queries.md#generating-queries-and-evaluating).

## Things to watch

- **Use `--tnorm min`** (the script default). The library default, `tnorm="prod"` on raw logits, multiplies
  scores that can be negative and misranks answers. On UMLS the `2i` MRR was 0.009 with `prod` and about
  0.63 with `min`. Combining sigmoid probabilities (`--probabilities`) also avoids the problem.
- **1p corresponds to tail prediction only.** The link-prediction MRR dicee reports averages tail
  `(h, r, ?)` and head `(?, r, t)` prediction, so it is not directly comparable. Compare 1p with tail-only
  MRR. On FB15k-237 with DistMult the tail MRR is 0.318, the head MRR 0.136 and the reported MRR 0.227
  (see [#479](https://github.com/dice-group/dice-embeddings/issues/479)).
- **Filtering changes the numbers.** The same model scored 0.214 unfiltered and 0.318 filtered against all
  splits. Use the same protocol when comparing.
- **Results come from a sample of queries** (`--gen_num`). Small samples varied between runs, so use many
  queries for stable numbers.
- **Existing query files.** To evaluate on a benchmark's own query and answer files instead of generated
  ones, the ranking step is the same, but the script would need an adapter to load that format.
- The script has only been tested on UMLS so far.
