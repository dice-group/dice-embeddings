#!/usr/bin/env bash
# WN18RR sweep: type_relation {schema, none} x cardinality_cutoff {100, 0}
#               x use_hop_distance_tokens {yes, no}. 50 epochs each, eval after.
set -uo pipefail

cd /home/lukef/Documents/GitHub/dice-embeddings
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/lukef/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate dicee

KG=KGs/WN18RR
FMT=head_relation_tail
EPOCHS=50
LOGDIR=ilp/runs/wn18rr_schema_cutoff_hop_sweep
mkdir -p "$LOGDIR"

# Schema = WordNet taxonomy-bearing relations (same set used in wn18rr_v1.yaml).
SCHEMA_RELS='[_hypernym, _instance_hypernym, _synset_domain_topic_of]'

run_one () {
    local tag="$1"; shift
    local save="$LOGDIR/${tag}.pt"
    local log="$LOGDIR/${tag}.log"
    local json="$LOGDIR/${tag}_results.json"
    echo "=== [$tag] train ===" | tee "$log"
    python -m ilp train --kg-dir "$KG" --triple-format "$FMT" \
        --epochs "$EPOCHS" --save "$save" \
        "$@" 2>&1 | tee -a "$log"
    echo "=== [$tag] eval ===" | tee -a "$log"
    python -m ilp eval --model "$save" --kg-dir "$KG" \
        --json-out "$json" 2>&1 | tee -a "$log"
}

for schema in yes no; do
  if [ "$schema" = "yes" ]; then
    schema_set=(--set "type_relation=${SCHEMA_RELS}")
  else
    schema_set=(--set 'type_relation=""')
  fi
  for cut in 100 0; do
    for hop in yes no; do
      if [ "$hop" = "yes" ]; then hop_val=true; else hop_val=false; fi
      tag="schema-${schema}_cut${cut}_hop-${hop}"
      run_one "$tag" \
        "${schema_set[@]}" \
        --set cardinality_cutoff=${cut} \
        --set use_hop_distance_tokens=${hop_val}
    done
  done
done

echo "=== ALL DONE ==="
