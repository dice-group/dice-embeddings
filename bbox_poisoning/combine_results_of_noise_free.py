import os
import json
import numpy as np
import re

# ======= CONFIG =======
ROOT_DIR = "/home/adel/Desktop/final_results_www26/enexa1/saved_models/WN18RR/"
RUN_DIRS = ["831769172", "2430986565"]
REPORT_NAME = "eval_report.json"
OUT_TXT = "mrr_summary.txt"
# ======================

def drop_leading_zero(s: str) -> str:
    """Remove '0.' at the very start of a string (so 0.52 -> .52)."""
    return re.sub(r'^0\.', '.', s)

def extract_test_mrr(data):
    if isinstance(data, dict) and "Test" in data and isinstance(data["Test"], dict):
        if "MRR" in data["Test"]:
            return float(data["Test"]["MRR"])
    if "Test.MRR" in data:
        return float(data["Test.MRR"])
    # case-insensitive fallback
    lower_map = {str(k).lower(): k for k in data.keys()}
    if "test" in lower_map:
        test_obj = data[lower_map["test"]]
        if isinstance(test_obj, dict):
            lm = {str(k).lower(): k for k in test_obj.keys()}
            if "mrr" in lm:
                return float(test_obj[lm["mrr"]])
    if "test.mrr" in lower_map:
        return float(data[lower_map["test.mrr"]])
    raise KeyError("Could not find Test.MRR in report JSON")

def main():
    model_dirs = [d for d in os.listdir(ROOT_DIR)
                  if os.path.isdir(os.path.join(ROOT_DIR, d))]

    rows = []
    for model in sorted(model_dirs):
        values = []
        for run in RUN_DIRS:
            report_path = os.path.join(ROOT_DIR, model, run, REPORT_NAME)
            if not os.path.isfile(report_path):
                print(f"[WARN] Missing: {report_path}")
                continue
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mrr = extract_test_mrr(data)
                values.append(float(mrr))
            except Exception as e:
                print(f"[WARN] Failed to read MRR from {report_path}: {e}")

        if not values:
            print(f"[SKIP] No MRR values for model {model}")
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0

        # format mean and std
        mean_str = drop_leading_zero(f"{mean:.2f}")
        std_str = f"{std:.1f}"
        rows.append((model, f"{mean_str}±{std_str}"))

    # write output
    out_path = os.path.join(ROOT_DIR, OUT_TXT)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Model\tMRR\n")
        for model, val in rows:
            f.write(f"{model}\t{val}\n")

    print(f"[OK] Wrote {out_path}")

if __name__ == "__main__":
    main()
