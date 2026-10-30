"""dropped.json of the fresh-v1 run from the dispatcher's exit records (docs/synthetic_kg_fresh_v1_registration.md,
section 14, option B). It reads RUN/plan.json and RUN/state.json only, never a row file or a score.

A (graph, system) enters if its task is 'failed' and its attempts cover every GPU of the plan (a CPU task: two
attempts). It refuses a run with a task that is pending, running, waiting for a retry or stopped by a setup error, and
a failed task whose attempts do not cover every GPU. Output: {graph: {system label: reason}}, the format of
fv1_analysis.py --dropped. The reason lists every attempt (GPU, caps, exit code).
Usage: fv1_drops.py RUN --out dropped.json
"""
import argparse
import json
import sys
from pathlib import Path


def drops(plan: dict, state: dict) -> dict:
    open_tasks = sorted(k for k, s in state.items() if s['status'] in ('pending', 'running', 'retry', 'setup_error'))
    if open_tasks:
        raise SystemExit(f'{len(open_tasks)} tasks are not finished (for example {open_tasks[:3]}): no dropped.json yet')
    out = {}
    gpus = set(plan['gpus'])
    for t in plan['tasks']:
        s = state[t['id']]
        if s['status'] == 'done':
            continue
        assert s['status'] == 'failed', (t['id'], s['status'])
        tried = s['attempts']
        covered = {a['gpu'] for a in tried}
        if t['kind'] == 'gpu' and not gpus <= covered:
            raise SystemExit(f'{t["id"]}: failed, but not on every GPU of the plan ({sorted(covered)} of {sorted(gpus)})')
        if t['kind'] == 'cpu' and len(tried) < 2:
            raise SystemExit(f'{t["id"]}: failed after one attempt only')
        reason = 'failed on every attempt: ' + ', '.join(
            f'{"CPU" if a["gpu"] < 0 else "GPU " + str(a["gpu"])} at {a["caps"]["gpu_gib"]} GiB exit {a["exit"]}' for a in tried)
        out.setdefault(t['graph'], {})[t['system']] = reason
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads((a.run / 'plan.json').read_text())
    state = json.loads((a.run / 'state.json').read_text())
    d = drops(plan, state)
    a.out.write_text(json.dumps(d, indent=1, sort_keys=True) + '\n')
    print(f'{sum(len(v) for v in d.values())} dropped (graph, system) pairs on {len(d)} graphs -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
