import os
import glob
import numpy as np
import pandas as pd
import re


ROOT_DIR = "/home/adel/Desktop/final_results_www26/FINAL/256_01/without_recipriocal/UMLS/"

MODELS = ["DistMult", "ComplEx", "Keci", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"]
CELL_FORMAT = "{mean:.2f}±{std:.2f}"

EMPTY_CELL = ""       

ALLOWED_RATIOS = ["0.02", "0.04", "0.08"]  
DESIRED_ORDER  = ["0.02", "0.04", "0.08"]    


def drop_leading_zero(s: str) -> str:
    if s is None:
        return s
    return re.sub(r'(?<!\d)0\.', '.', str(s))

def read_numeric_df(path):
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def mean_std_two(df1, df2):
    a, b = df1.align(df2, join="outer", axis=None)
    mean = (a + b) / 2.0
    diff_sq_sum = (a - mean) ** 2 + (b - mean) ** 2
    std = np.sqrt(diff_sq_sum)
    return mean, std

def format_result(mean_df, std_df, cell_format=CELL_FORMAT, empty_cell=EMPTY_CELL):
    m, s = mean_df.align(std_df, join="outer", axis=None)
    out = m.copy().astype(object)
    for r in m.index:
        for c in m.columns:
            mv = m.at[r, c]
            sv = s.at[r, c]
            if pd.isna(mv) or pd.isna(sv):
                out.at[r, c] = empty_cell
            else:
                formatted = cell_format.format(mean=mv, std=sv)
                out.at[r, c] = drop_leading_zero(formatted)
    return out

def try_numeric_sort_labels(labels):
    try:
        pairs = []
        for lbl in labels:
            val = float(str(lbl).strip().rstrip('%'))
            pairs.append((val, lbl))
        pairs.sort(key=lambda x: x[0])
        return [lbl for _, lbl in pairs], True
    except Exception:
        return list(labels), False

def write_ratio_lines_txt(result_df, out_txt_path):
    methods = list(result_df.index)
    ratio_cols = list(result_df.columns)
    ratio_cols, _ = try_numeric_sort_labels(ratio_cols)

    lines = []

    header = "ratio & " + " & ".join(methods)
    lines.append(header)

    for ratio in ratio_cols:
        vals = []
        for m in methods:
            if ratio in result_df.columns:
                v = str(result_df.at[m, ratio])
                vals.append(drop_leading_zero(v))   
            else:
                vals.append(EMPTY_CELL)
        line = f"{drop_leading_zero(str(ratio))} & " + " & ".join(vals)
        lines.append(line)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")  

    flat_values = []
    for ratio in ratio_cols:
        for m in methods:
            if ratio in result_df.columns:
                v = str(result_df.at[m, ratio])
                flat_values.append(drop_leading_zero(v))
            else:
                flat_values.append(EMPTY_CELL)

    with open(out_txt_path, "a", encoding="utf-8") as f:
        f.write(" & ".join(flat_values) + "\n")


def bold_min_per_ratio(formatted_df: pd.DataFrame, mean_df: pd.DataFrame) -> pd.DataFrame:
    out = formatted_df.copy()
    for col in mean_df.columns:
        col_vals = mean_df[col]
        if col_vals.notna().any():
            m = col_vals.min(skipna=True)
            mask = (col_vals == m) & col_vals.notna()
            for r in col_vals.index[mask]:
                val = str(out.at[r, col])
                if not (val.startswith("\\textbf{") and val.endswith("}")):
                    out.at[r, col] = "\\textbf{" + val + "}"
    return out


def process_leaf_dir(leaf_dir):
    csv1 = glob.glob(os.path.join(leaf_dir, "*831769172*.csv"))
    csv2 = glob.glob(os.path.join(leaf_dir, "*2430986565*.csv"))

    if len(csv1) != 1 or len(csv2) != 1:
        print(f"[SKIP] Expected exactly one match for each suffix in: {leaf_dir}")
        print(f"       Found for 831769172: {csv1}")
        print(f"       Found for 2430986565: {csv2}")
        return

    df_a = read_numeric_df(csv1[0])
    df_b = read_numeric_df(csv2[0])

    mean_df, std_df = mean_std_two(df_a, df_b)

    mean_df = mean_df.loc[:, mean_df.columns.astype(str).isin(ALLOWED_RATIOS)]
    std_df  = std_df.loc[:,  std_df.columns.astype(str).isin(ALLOWED_RATIOS)]

    keep_cols = [c for c in DESIRED_ORDER if c in mean_df.columns.astype(str).tolist()]
    mean_df = mean_df.loc[:, [c for c in mean_df.columns if str(c) in keep_cols]]
    std_df  = std_df.loc[:,  [c for c in std_df.columns  if str(c) in keep_cols]]



    result_df = format_result(mean_df, std_df, CELL_FORMAT, EMPTY_CELL)
    result_df = bold_min_per_ratio(result_df, mean_df)  


    out_csv = os.path.join(leaf_dir, "result.csv")
    result_df.to_csv(out_csv)

    out_txt = os.path.join(leaf_dir, "result.txt")
    write_ratio_lines_txt(result_df, out_txt)

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {out_txt}")

def main():
    if MODELS is None:
        model_dirs = [d for d in os.listdir(ROOT_DIR)
                      if os.path.isdir(os.path.join(ROOT_DIR, d))]
    else:
        model_dirs = MODELS

    for model in model_dirs:
        for sub in ("add", "delete"):
            leaf = os.path.join(ROOT_DIR, model, sub)
            if os.path.isdir(leaf):
                process_leaf_dir(leaf)
            else:
                print(f"[WARN] Missing directory: {leaf}")

if __name__ == "__main__":
    main()
