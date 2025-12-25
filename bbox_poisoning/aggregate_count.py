#!/usr/bin/env python3
"""
Aggregate 4 summary CSV/TSV files where each cell is 'X/Y (Z%)' by summing counts:
e.g., 7/12 + 10/12 -> 17/24, then recompute the percentage.

- Expected columns:  method, 0.02, 0.04, 0.08  (comma or tab separated is fine)
- Output: writes aggregated CSV to AGG_OUT

Edit the FILES list below to your 4 file paths.
"""

import re
from typing import Dict, Tuple, List
import pandas as pd

# ---------------------- EDIT THESE PATHS ---------------------- #
FILES = [
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/without_recipriocal/UMLS/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/without_recipriocal/KINSHIP/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/NELL-995-h100/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/FB15k-237/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/WN18RR/summary_add.csv",
]
case = "add-without_recipriocal"

"""
FILES = [
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/without_recipriocal/UMLS/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/without_recipriocal/KINSHIP/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/NELL-995-h100/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/FB15k-237/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/without_recipriocal/WN18RR/summary_delete.csv",
]
case = "delete-without_recipriocal"
"""

"""
FILES = [
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/with_recipriocal/UMLS/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/with_recipriocal/KINSHIP/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/NELL-995-h100/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/FB15k-237/summary_add.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/WN18RR/summary_add.csv",
]
case = "add-with_recipriocal"
"""

"""
FILES = [
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/with_recipriocal/UMLS/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/256_01/with_recipriocal/KINSHIP/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/NELL-995-h100/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/FB15k-237/summary_delete.csv",
    r"/home/adel/Desktop/final_results_www26/FINAL/512_02/with_recipriocal/WN18RR/summary_delete.csv",
]
case = "delete-with_recipriocal"
"""


AGG_OUT = f"aggregated_summary_{case}.csv"
# -------------------------------------------------------------- #

RATIOS = ["0.02", "0.04", "0.08"] 


def read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV with auto-delimiter detection. Ensure first col is 'method' and strip."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    if first.lower() != "method":
        df = df.rename(columns={first: "method"})
    df["method"] = df["method"].astype(str).str.strip()
    return df


def parse_cell(cell: str) -> Tuple[int, int]:
    """Parse 'X/Y (Z%)' or 'X/Y (Z\\%)' or 'X/Y' -> (X, Y). Returns (0,0) if missing/unparsable."""
    if cell is None:
        return (0, 0)
    s = str(cell).strip()
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 0)


def aggregate_files(paths: List[str]) -> pd.DataFrame:
    """Sum numerators/denominators per method×ratio across files; recompute %."""
    counts: Dict[Tuple[str, str], List[int]] = {}
    methods = set()

    for p in paths:
        df = read_table(p)
        keep = ["method"] + [c for c in RATIOS if c in df.columns]
        df = df[keep]
        print(df)
        print("-------------", p)

        for _, row in df.iterrows():
            m = str(row["method"]).strip()
            methods.add(m)
            for r in RATIOS:
                if r not in row or pd.isna(row[r]):
                    continue
                x, y = parse_cell(row[r])
                key = (m, r)
                if key not in counts:
                    counts[key] = [0, 0]
                counts[key][0] += x
                counts[key][1] += y

    print(counts)

    out_rows = []
    for m in sorted(methods):
        row = {"method": m}
        sx = 0
        sy = 0
        for r in RATIOS:
            x, y = counts.get((m, r), (0, 0))
            sx += x
            sy += y
            row[r] = f"{x}/{y} ({(100.0*x/y):.1f}%)" if y > 0 else "0/0 (NA)"
        row["AVG"] = f"{sx}/{sy} ({(100.0*sx/sy):.1f}%)" if sy > 0 else "0/0 (NA)"
        out_rows.append(row)

    cols = ["method"] + RATIOS + ["AVG"]
    return pd.DataFrame(out_rows)[cols]


def main():
    if len(FILES) != 5:
        raise SystemExit("Please set FILES to exactly 5 paths.")
    agg = aggregate_files(FILES)
    agg.to_csv(AGG_OUT, index=False)
    print(f"Wrote: {AGG_OUT}")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(agg)


if __name__ == "__main__":
    main()
