"""Publish completed KGFM runs to JSON records and the separate README table."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("MRR", "H@1", "H@3", "H@10")


def bold_best_metrics(section):
    """Highlight each dataset's column maxima, including displayed ties."""
    lines = section.splitlines()
    rows = []
    best = {}
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 7 or cells[1] not in ("ULTRA-3g", "TRIX", "Flock"):
            continue
        cells[3:] = [cell.strip("*") for cell in cells[3:]]
        values = [None if cell == "—" else float(cell) for cell in cells[3:]]
        maxima = best.setdefault(cells[0], [float("-inf")] * len(METRICS))
        for column, value in enumerate(values):
            if value is not None:
                maxima[column] = max(maxima[column], value)
        rows.append((index, cells, values))
    for index, cells, values in rows:
        for column, value in enumerate(values):
            if value is not None and value == best[cells[0]][column]:
                cells[column + 3] = f"**{cells[column + 3]}**"
        lines[index] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()
    before, remainder = readme.split("## KGFM Link Prediction with Frozen Checkpoints\n", 1)
    section, after = remainder.split("## Link Prediction Benchmarks\n", 1)
    records = ROOT / "benchmarks/results/kgfm-zero-shot"
    records.mkdir(parents=True, exist_ok=True)
    updated = 0
    for path in sorted(args.runs_dir.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            continue
        if result["configuration"].get("tie_policy") not in ("sort", "DICE sort position"):
            raise ValueError("This README table uses sort ties; publish other policies separately")
        metrics = result["metrics"]
        assert all(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1 for key in METRICS)
        assert metrics["H@1"] <= metrics["H@3"] <= metrics["H@10"]
        assert metrics["H@1"] <= metrics["MRR"]
        assert result["ranked_queries"] == 2 * result["test_triples"]
        assert result["weights_unchanged"] and result["configuration"]["optimizer_updates"] == 0
        label = "ULTRA-3g" if result["model"] == "ULTRA" else result["model"]
        prefix = f'| {result["dataset"]} | {label} |'
        lines = section.splitlines()
        matching = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one README row: {prefix}")
        seen = "Yes" if result["target_graph_in_pretraining"] else "No"
        lines[matching[0]] = f'{prefix} {seen} | ' + " | ".join(f"{metrics[key]:.4f}" for key in METRICS) + " |"
        section = "\n".join(lines) + "\n"
        (records / f'{result["dataset"]}-{result["model"]}.json').write_text(json.dumps(result, indent=2) + "\n")
        updated += 1
    section = bold_best_metrics(section)
    readme_path.write_text(before + "## KGFM Link Prediction with Frozen Checkpoints\n" + section
                           + "## Link Prediction Benchmarks\n" + after)
    print(f"Published {updated} completed KGFM results; original benchmark section unchanged.")


if __name__ == "__main__":
    main()
