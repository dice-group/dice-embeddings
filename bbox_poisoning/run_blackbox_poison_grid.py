# run_blackbox_poison_grid.py
from pathlib import Path
import json

from blackbox_poison_pipeline import RunnerConfig, run_blackbox_poison

# -----------------------
# Grid configuration
# -----------------------

DBS    = ["KINSHIP", "UMLS"]  # , "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = ["DistMult", "ComplEx", "Pykeen_TransE", "Pykeen_TransH",
          "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"]

# Oracles: ORACLE_ROOT/DB/MODEL/SEED/
ORACLE_ROOT  = Path("./saved_models/without_recipriocal/")
ORACLE_SEEDS = [694443915, 831769172, 2430986565]

# Poisoning ratios used for both ADD and DEL
PERCENTAGES = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


# Victim training hyperparameters
NUM_EPOCHS = "100"
BATCH_SIZE = "256"
LR         = "0.01"
EMB_DIM    = "32"
LOSS_FN    = "BCELoss"
SCORING    = "KvsAll"
OPTIM      = "Adam"

# Output roots (datasets will be saved by the pipeline under ./saved_datasets)
RESULTS_ROOT = Path("./final_results/without_recipriocal/blackbox")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helpers
# -----------------------

def resolve_mu_csvs(oracle_path: Path, model_name: str) -> tuple[str, str] | None:
    """
    Try common filename patterns inside the oracle folder; if not present, try a generic glob.
    Returns (entity_csv, relation_csv) or None if missing.
    """
    # 1) Exact model prefix (e.g., "DistMult_entity_embeddings.csv")
    ent1 = oracle_path / f"{model_name}_entity_embeddings.csv"
    rel1 = oracle_path / f"{model_name}_relation_embeddings.csv"
    if ent1.exists() and rel1.exists():
        return str(ent1), str(rel1)

    # 2) Generic names
    ent2 = oracle_path / "entity_embeddings.csv"
    rel2 = oracle_path / "relation_embeddings.csv"
    if ent2.exists() and rel2.exists():
        return str(ent2), str(rel2)

    # 3) Globs inside the folder
    ents = list(oracle_path.glob("*entity*embeddings*.csv"))
    rels = list(oracle_path.glob("*relation*embeddings*.csv"))
    if len(ents) == 1 and len(rels) == 1:
        return str(ents[0]), str(rels[0])

    # Not found
    return None

# -----------------------
# Main loop
# -----------------------

def main():
    summary = []
    for DB in DBS:
        for MODEL in MODELS:
            for SEED in ORACLE_SEEDS:
                oracle_path = ORACLE_ROOT / DB / MODEL / str(SEED)
                if not oracle_path.exists():
                    print(f"[SKIP] Oracle not found: {oracle_path}")
                    summary.append({"DB": DB, "MODEL": MODEL, "SEED": SEED, "status": "missing_oracle"})
                    continue

                mu_files = resolve_mu_csvs(oracle_path, MODEL)
                if mu_files is None:
                    print(f"[SKIP] µ-embeddings CSVs not found under: {oracle_path}")
                    summary.append({"DB": DB, "MODEL": MODEL, "SEED": SEED, "status": "missing_mu_csvs"})
                    continue
                ent_csv, rel_csv = mu_files

                # Build config and run
                cfg = RunnerConfig(
                    DB=DB,
                    MODEL=MODEL,
                    ORACLE_PATH=str(oracle_path),
                    ENTITY_CSV=ent_csv,
                    RELATION_CSV=rel_csv,
                    ADD_PCTS=PERCENTAGES,
                    DEL_PCTS=PERCENTAGES,
                    NUM_EPOCHS=NUM_EPOCHS,
                    BATCH_SIZE=BATCH_SIZE,
                    LR=LR,
                    EMB_DIM=EMB_DIM,
                    LOSS_FN=LOSS_FN,
                    SCORING_TECH=SCORING,
                    OPTIM=OPTIM,
                    SAVED_DATASETS_ROOT="./saved_datasets/without_recipriocal/",
                    RUNS_ROOT="./running_experiments",
                    RESULTS_ROOT=str(RESULTS_ROOT),
                )

                print(f"\n=== BLACKBOX | DB={DB} | MODEL={MODEL} | SEED={SEED} ===")
                try:
                    # Use a deterministic per-run master seed if desired:
                    # master_seed = (hash(f"{DB}::{MODEL}::{SEED}") & 0xFFFFFFFF)
                    master_seed = 12345
                    run_blackbox_poison(cfg, master_seed=master_seed)
                    status = "ok"
                except Exception as e:
                    print(f"[ERROR] {DB}/{MODEL}/seed={SEED}: {e}")
                    status = f"error: {e}"

                summary.append({"DB": DB, "MODEL": MODEL, "SEED": SEED, "status": status})

    # Write a small JSON summary
    out_json = RESULTS_ROOT / "summary-blackbox-grid.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] Summary written to: {out_json}")

if __name__ == "__main__":
    main()
