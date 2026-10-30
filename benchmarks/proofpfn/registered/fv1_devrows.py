"""Development rows for `fv1_analysis.py precision` (docs/synthetic_kg_fresh_v1_registration.md, section 13). Runs on kiowl.

Development data only: the 91 main graphs of pcu-eval-20261029/plan.json (the graphs and test facts of the S rows),
no fresh-v1 file. Per graph, system and model: the raw MRR (per-seed raw ranks, no self-loop rule, no GT2), the metric
of the registration. Source cluster: the 'source' field of graph-audit-v1/graphs.json (the graph key if it has none),
as step5/s5_power.py. Every row file of a graph must rank the same queries as the S rows; a graph without a system's
file is left out for that system (reported).
Output: CSV in the format of fv1_analysis.py --mrr (block empty): graph, block, source, system, seed, mrr, and a JSON
report with the files and their SHA-256. Usage (from ~/kgfm-synth): fv1_devrows.py --out DEV.csv
"""
import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = HERE.parent
ROAD = K / 'icml-roadmap-20261025'
OURS = ('42', '43', '44')
MERIT = ('1024', '2025', '2026', '4000', '8000')
MERIT_TAG = 'merit1024-merit2025-merit2026-merit4000-merit8000'


def sources(g: dict) -> list:
    """(system key, seeds, row-key pattern, list of candidate files) for one graph."""
    c, n = g['collection'], g['name'] + '.json'
    return [
        ('rec10axlepcunoa', OURS, 'rec10axlepcunoa.seed{s}:raw', [g['s_rows']]),
        ('rec10axlecunoa', OURS, 'rec10axlecunoa.seed{s}:raw', [g['s_rows']]),
        ('rec10axleulnoa', OURS, 'rec10axleulnoa.seed{s}:raw', [g['s_rows']]),
        ('pcur', OURS, 'pcur.seed{s}:raw', [ROAD / 'pcu-eval-20261029/rows-ag' / c / 'pcur-pcum' / n]),
        ('pcum', OURS, 'pcum.seed{s}:raw', [ROAD / 'pcu-eval-20261029/rows-ag' / c / 'pcur-pcum' / n]),
        ('pcun', OURS, 'pcun.seed{s}:raw', [ROAD / 'prior-transfer-20261028/rows' / c / 'pcun' / n]),
        ('pcuht', OURS, 'pcuht.seed{s}:raw', sorted(glob.glob(str(ROAD / 'pcu-ht-20261029/rows-ag' / c / 'pcuht' / n)))),
        ('rec10axlenoa', OURS, 'rec10axlenoa.seed{s}:raw', [K / 'allgraph-v1/rows-seedraw' / c / 'rec10axlenoa-m2rec10axlenoa-ro2noa' / n]),
        ('htrec10axlenoa', OURS, 'htrec10axlenoa.seed{s}:raw', sorted(glob.glob(str(ROAD / 'overnight-20261026/results/ht/*' / c / 'htrec10axlenoa' / n)))),
        ('trix', ('0',), 'trix:raw', [K / 'allgraph-v1/rows' / c / 'trix-ultra3g' / n,
                                       ROAD / 'recovery-20261027/results/ag' / c / 'trix-ultra3g' / n,
                                       K / 'prior-fix-v1/v7-read/rows' / c / 'trix-ultra3g' / n]),
        ('ultra3g', ('0',), 'ultra3g:raw', [K / 'allgraph-v1/rows' / c / 'trix-ultra3g' / n,
                                             ROAD / 'recovery-20261027/results/ag' / c / 'trix-ultra3g' / n,
                                             K / 'prior-fix-v1/v7-read/rows' / c / 'trix-ultra3g' / n]),
        ('flockoff', ('0',), 'flockoff:raw', [K / 'allgraph-v1/rows' / c / 'flockoff' / n]),
        ('merit', MERIT, 'merit{s}:raw', sorted(glob.glob(str(ROAD / 'overnight-20261026/results/merit/gpu*' / c / MERIT_TAG / n)))),
        ('kgpfn', ('0',), 'kgpfn:raw', [K / 'allgraph-v1/rows-kgpfn' / c / 'kgpfn' / n]),
        ('anyburl', ('0',), 'anyburl:raw', [K / 'allgraph-v1/rows-anyburl' / c / 'anyburl' / n]),
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queries(path) -> dict:
    d = json.loads(Path(path).read_text())
    out = {(r['index'], bool(r['head'])): r for r in d['rows']}
    assert len(out) == len(d['rows']), f'{path}: duplicate queries'
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((ROAD / 'pcu-eval-20261029/plan.json').read_text())
    audit = {g['key']: g for g in json.loads((K / 'graph-audit-v1/graphs.json').read_text())['graphs']}
    graphs = plan['ag_graphs']
    assert len(graphs) == 91, len(graphs)
    rows, files, missing = [], {}, {}
    for g in graphs:
        src = audit[g['key']].get('source') or g['key']
        base = queries(g['s_rows'])
        for system, seeds, pattern, cands in sources(g):
            found = [Path(p) for p in cands if Path(p).is_file()]
            if not found:
                missing.setdefault(system, []).append(g['key'])
                continue
            assert len(found) == 1, (system, g['key'], found)
            q = queries(found[0])
            assert set(q) == set(base), f'{system} {g["key"]}: other queries than the S rows'
            files[str(found[0])] = sha256(found[0])
            for s in seeds:
                key = pattern.format(s=s)
                rr = [1.0 / q[x][key] for x in sorted(base)]
                rows.append((g['key'], '', src, system, s, repr(sum(rr) / len(rr))))
    rows.sort()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['graph', 'block', 'source', 'system', 'seed', 'mrr'])
        w.writerows(rows)
    report = dict(graphs=len(graphs), sources=len({r[2] for r in rows}), rows=len(rows), missing={k: len(v) for k, v in missing.items()},
                  missing_graphs=missing, files=files, out_sha256=sha256(a.out), script_sha256=sha256(Path(__file__)))
    a.out.with_name(a.out.name + '.report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(f'{len(graphs)} graphs, {report["sources"]} sources, {len(rows)} rows; graphs without rows: {report["missing"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
