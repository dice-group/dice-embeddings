from pathlib import Path
import sys
import ast
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dicee.executer import run_dicee_eval

robust_kge_dir = Path(__file__).resolve().parent
project_root = robust_kge_dir.parent.resolve()
sys.path.insert(0, str(robust_kge_dir))
sys.path.insert(0, str(project_root))


NUM_EPOCHS = 100
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "test"


def abs_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def normalize_dataset_key(dataset_str: str) -> str:
    parts = [p for p in dataset_str.strip().replace("\\", "/").split("/") if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-2]}/{parts[-1]}"


def parse_report_file(report_path: Path):
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    rows = []
    with open(report_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "Params:" not in line:
                continue

            params_idx = line.find("Params:")
            dataset_idx = line.find("Dataset:")
            model_idx = line.find("Model:")
            loss_idx = line.find("Loss:")
            if min(params_idx, dataset_idx, model_idx, loss_idx) == -1:
                continue

            value = None
            if line.startswith("Value:"):
                value_str = line[len("Value:"):params_idx].strip().rstrip(",")
                try:
                    value = float(value_str)
                except ValueError:
                    value = None

            params_str = line[params_idx + len("Params:"):dataset_idx].strip().rstrip(",")
            dataset_str = line[dataset_idx + len("Dataset:"):model_idx].strip().rstrip(",")
            model_str = line[model_idx + len("Model:"):loss_idx].strip().rstrip(",")
            loss_str = line[loss_idx + len("Loss:"):].strip().rstrip(",")

            params = ast.literal_eval(params_str)
            if not isinstance(params, dict):
                continue

            rows.append(
                {
                    "Dataset": normalize_dataset_key(dataset_str),
                    "Model": model_str.strip(),
                    "Loss": loss_str.strip(),
                    "Value": value,
                    "Params": params,
                }
            )

    best_by_key = {}
    for row in rows:
        key = (row["Dataset"], row["Model"], row["Loss"])
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = row
            continue

        prev_value = prev["Value"]
        curr_value = row["Value"]
        if prev_value is None and curr_value is not None:
            best_by_key[key] = row
        elif curr_value is not None and prev_value is not None and curr_value > prev_value:
            best_by_key[key] = row

    return list(best_by_key.values())


def build_zero_point_plan(entries):
    grouped = {}
    for entry in entries:
        dataset_key = entry["Dataset"]
        db = dataset_key.split("/", 1)[0]
        key = (db, entry["Model"], entry["Loss"])
        grouped.setdefault(key, []).append(entry)

    plan = []
    for (db, model, loss_fn), group_entries in grouped.items():
        source = None
        for entry in group_entries:
            if entry["Dataset"] == f"{db}/0.0":
                source = entry
                break
        if source is None:
            for entry in group_entries:
                if entry["Dataset"] == db:
                    source = entry
                    break

        if source is None:
            continue

        plan.append(
            {
                "DB": db,
                "Model": model,
                "Loss": loss_fn,
                "Params": source["Params"],
                "SourceDataset": source["Dataset"],
            }
        )

    return plan


def resolve_dataset_targets(datasets_root: Path, db: str):
    db_path = (datasets_root / db).resolve()
    if not db_path.exists():
        return []
    candidates = []
    for d in db_path.iterdir():
        if not d.is_dir():
            continue
        try:
            val = float(d.name)
        except ValueError:
            continue
        if val <= 0.08:
            candidates.append((val, d.name))
    subdirs = [name for _, name in sorted(candidates, key=lambda x: x[0])]
    return [f"{db}/{subdir}" for subdir in subdirs]


def create_results_table(records):
    df = pd.DataFrame(records)
    if df.empty:
        return df, pd.DataFrame()

    pivot_df = df.pivot_table(
        index=["Model", "Loss"],
        columns="Dataset",
        values="Test_MRR",
        aggfunc="first",
    )
    return df, pivot_df


def create_visualization(pivot_df, output_dir: Path):
    if pivot_df.empty:
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot_df.columns) * 1.8), max(10, len(pivot_df.index) * 0.5))
    )

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        cbar_kws={"label": "Test MRR"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=1.0,
    )

    ax.set_title("MRR Comparison: (Model, Loss) vs Datasets (Test Set)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model / Loss", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"mrr_comparison_heatmap_from_report_{timestamp}.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap visualization saved to: {image_path}")


def save_pretty_table(pivot_df: pd.DataFrame, out_path: Path, title: str):
    line = "=" * 80
    with open(out_path, "w") as f:
        f.write(f"{line}\n")
        f.write(f"{title}\n")
        f.write(f"{line}\n")
        f.write(pivot_df.to_string())
        f.write(f"\n{line}\n")
    print(f"Formatted table saved to: {out_path}")


def save_results(df, pivot_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_path = output_dir / f"detailed_results_from_report_{timestamp}.csv"
    df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")

    pivot_path = output_dir / f"comparison_table_from_report_{timestamp}.csv"
    pivot_df.to_csv(pivot_path)
    print(f"Comparison table saved to: {pivot_path}")

    table_txt_path = output_dir / f"comparison_table_from_report_{timestamp}.txt"
    save_pretty_table(pivot_df, table_txt_path, "MRR Comparison Table (Test Set)")

    create_visualization(pivot_df, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run test experiments from BO report params.")
    parser.add_argument("--report_file", type=str, default="bayesian_optimization_report_100_trials.txt")
    parser.add_argument("--datasets_root", type=str, default=str(project_root / "Datasets_Perturbed"))
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--scoring_technique", type=str, default=SCORING_TECH)
    parser.add_argument("--optim", type=str, default=OPTIM)
    parser.add_argument("--eval_model", type=str, default=EVAL_MODEL)
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=str(project_root / "robust-kge" / "results"))
    parser.add_argument("--saved_models_dir", type=str, default=str(robust_kge_dir / "saved_models_from_report"))
    args = parser.parse_args()

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    report_file = abs_path(args.report_file, project_root)
    datasets_root = abs_path(args.datasets_root, project_root)
    results_dir = abs_path(args.results_dir, project_root)
    saved_models_dir = abs_path(args.saved_models_dir, project_root)

    entries = parse_report_file(report_file)
    plan = build_zero_point_plan(entries)
    if not plan:
        raise ValueError("No valid (DB, model, loss) plan could be built from report file.")

    records = []
    for entry in plan:
        db = entry["DB"]
        model = entry["Model"]
        loss_fn = entry["Loss"]
        params = entry["Params"]
        source_dataset = entry["SourceDataset"]

        targets = resolve_dataset_targets(datasets_root, db)
        for dataset_name in targets:
            dataset_folder = (datasets_root / dataset_name).resolve()
            store_path = (saved_models_dir / dataset_name / model / loss_fn).resolve()

            print(
                f"Running: dataset={dataset_name}, model={model}, loss={loss_fn}, "
                f"source_params={source_dataset}, params={params}"
            )
            result = run_dicee_eval(
                dataset_folder=str(dataset_folder),
                model=model,
                num_epochs=args.num_epochs,
                loss_function=loss_fn,
                path_to_store_single_run=str(store_path),
                scoring_technique=args.scoring_technique,
                optim=args.optim,
                eval_model=args.eval_model,
                trainer=args.trainer,
                accelerator=args.accelerator,
                devices=devices,
                precision=args.precision,
                **params,
            )
            test_mrr = result.get("Test", {}).get("MRR", None)

            records.append(
                {
                    "Dataset": dataset_name,
                    "Model": model,
                    "Loss": loss_fn,
                    "Test_MRR": test_mrr,
                }
            )

    df, pivot_df = create_results_table(records)
    save_results(df, pivot_df, results_dir)
