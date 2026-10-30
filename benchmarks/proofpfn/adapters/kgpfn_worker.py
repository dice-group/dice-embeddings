"""KGPFN scoring worker: runs in KGPFN's own Python environment, started by ``kgpfn.py`` (never imported by the benchmark).

Started as ``<KGPFN python> -B -P <this file> --meta META ...`` with ``PYTHONPATH`` set to the KGPFN code folder (a clone
of github.com/HKUST-KnowComp/KGPFN), so KGPFN's packages ``pfn``, ``model`` and ``script`` import as in its own scripts;
``-P`` keeps this file's folder off the module path. It imports only the standard library, numpy, torch, yaml,
torch_geometric and KGPFN's modules.

Port of the research runner ``kgpfn-ref-v1-runtime/kgpfn_eval.py`` in its all-graph mode (``--meta``) with the
answer-blind context and context seed 0:

- Model: ``kgpfn_icl_3g.pth`` (frozen ULTRA-3g structure encoder, TabICL feature transformer), built by KGPFN's
  ``create_model`` from ``config/script/test.yaml`` plus ``config/kgpfn/kgpfn_icl.yaml`` (context graph 3, context tail),
  with ``fast_test`` off, ``num_epoch`` 0 and ``init_model`` off. The TabICL yaml is written from the TabICL
  checkpoint's ``config`` entry as ``script/download.py`` does, loaded with ``weights_only=True``. KGPFN loads its
  checkpoint with ``strict=False``; every tensor of the model must then equal the checkpoint's, with none left over.
- Graph: KGPFN's ``TransductiveDataset`` layout: entities and relations numbered by first appearance in train, valid and
  test (a line splits at tabs if it has one, else at whitespace), train edges plus inverse edges (relation ``r + R``),
  the relation graph of ``tasks.build_relation_graph``. The parse must hash as the host's (``parse_sha256`` of the meta).
- Queries: the ranked facts of the meta's ``positions`` (all rows for ``null``), in the order of KGPFN's test loader
  (``DistributedSampler``, shuffle, seed 0) restricted to them, in batches; each fact is asked as the tail query
  ``(h, r, ?)`` and the head query as KGPFN's inverse query ``(t, r^-1, ?)``. random, numpy and torch are seeded with 0
  right before the loop.
- Context (answer-blind): ``tasks.build_context_relation_aware`` gets the query with its answer replaced by ``-1`` (20
  positives and the meta's negatives of the query relation from the visible graph); a context that cannot be filled
  stops the run.
- Scores: ``script/test_kgpfn.py::_score_all_candidate_tails``. Ranks (``our_ranks``): pessimistic over the allowed
  candidates (every known answer of train, valid and test except the target is removed), a row with a non-finite score
  gets the worst rank; the rule view sets the anchor to ``-inf`` unless the base relation is one of the meta's
  ``reflexive``. KGPFN's native ranks (``tasks.compute_ranking`` on ``tasks.strict_negative_mask``) are compared.

Output: one JSON with the per-position ranks (``tail_rank``, ``head_rank``, ``tail_rank_rule``, ``head_rank_rule``, the
optimistic ranks as a tie diagnostic), counts, settings, the strict-load check, times, GPU memory and provenance. It
holds no metric, and the log prints none. ``--dump-scores`` also writes the score rows (small graphs, for the host's
check). Exit code 3: CUDA out of memory (an error JSON is written next to the output).
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import types
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
from torch.utils.data import DistributedSampler

logger = logging.getLogger('kgpfn_worker')

CHECKPOINT, TABICL = 'kgpfn_icl_3g.pth', 'tabicl-regressor-v2-20260212.ckpt'
SEED = 0
EXIT_OOM = 3
Split = list[tuple[str, str, str]]


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def code_commit(code: Path) -> str:
    """The commit checked out in the KGPFN folder, read from ``.git``."""
    git = code / '.git'
    head = (git / 'HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return head
    ref = head[5:]
    if (git / ref).exists():
        return (git / ref).read_text().strip()
    return next(line.split()[0] for line in (git / 'packed-refs').read_text().splitlines() if line.endswith(' ' + ref))


def read_split(path: Path) -> Split:
    """The non-empty lines of a split file as name triples (tab split if a line has a tab, else whitespace split)."""
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            fields = [x.strip() for x in (line.split('\t') if '\t' in line else line.split())]
            if len(fields) != 3:
                raise ValueError(f'{path}: not a triple: {line!r}')
            rows.append((fields[0], fields[1], fields[2]))
    return rows


def parse_sha(rows: Split) -> str:
    """SHA-256 of the parsed triples in file order (the host hashes its parse the same way)."""
    return hashlib.sha256('\n'.join('\t'.join(t) for t in rows).encode('utf-8')).hexdigest()


def build_graphs(splits: dict[str, Split], name: str, tasks: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor], int, int, dict[str, int]]:
    """Train, valid and test ``Data`` objects as KGPFN's ``TransductiveDataset.process`` builds them (vocabulary in file order).

    Returns:
        ``(graphs, ids, entities, relations, relation ids)``; ``ids`` holds ``(h, t, r)`` rows per split.
    """
    from torch_geometric.data import Data  # type: ignore[import-not-found]
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}
    ids = {}
    for s in ('train', 'valid', 'test'):
        out = []
        for h, r, t in splits[s]:
            entities.setdefault(h, len(entities))
            entities.setdefault(t, len(entities))
            relations.setdefault(r, len(relations))
            out.append((entities[h], entities[t], relations[r]))
        ids[s] = torch.tensor(out, dtype=torch.long).view(-1, 3)
    n, num_relations, train = len(entities), len(relations), ids['train']
    edge_index = torch.cat([train[:, :2].t(), train[:, :2].t().flip(0)], dim=1)
    edge_type = torch.cat([train[:, 2], train[:, 2] + num_relations])
    graphs = {}
    for s in ('train', 'valid', 'test'):
        graphs[s] = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n, num_relations=2 * num_relations,
                         target_edge_index=ids[s][:, :2].t().contiguous(), target_edge_type=ids[s][:, 2].contiguous())
        graphs[s].dataset = name
    graphs['test'] = tasks.build_relation_graph(graphs['test'])
    return graphs, ids, n, num_relations, relations


def our_ranks(pred: torch.Tensor, batch: torch.Tensor, target: torch.Tensor, known_t: dict, known_h: dict, gate: torch.Tensor,
              anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The ranks of a batch: rows ``0..b-1`` are the tail queries ``(h, r, ?)``, rows ``b..2b-1`` the head queries.

    Args:
        pred: ``[2b, |V|]`` scores.
        batch: ``[b, 3]`` facts as ``(h, t, r)``.
        target: ``[2b]`` targets.
        known_t: ``{(h, r): tails}`` of all splits.
        known_h: ``{(t, r): heads}`` of all splits.
        gate: ``[2b]`` rows whose anchor gets ``-inf`` in the rule view.
        anchor: ``[2b]`` anchors.

    Returns:
        ``(pessimistic ranks, optimistic ranks, rule-view ranks, rows with a non-finite score)``; a row with a non-finite
        score gets the number of allowed candidates in every view.
    """
    b, s = len(batch), pred.double()
    allowed = torch.ones_like(s, dtype=torch.bool)
    rows, cols = [], []
    for j, (h, t, r) in enumerate(batch.tolist()):
        for row, answers, answer in ((j, known_t[h, r], t), (b + j, known_h[t, r], h)):
            other = [e for e in answers if e != answer]
            rows += [row] * len(other)
            cols += other
    if rows:
        allowed[torch.tensor(rows, device=s.device), torch.tensor(cols, device=s.device)] = False
    score = s.gather(1, target[:, None])
    rank, optimistic = ((s >= score) & allowed).sum(1), 1 + ((s > score) & allowed).sum(1)
    gated, rows_t = s.clone(), gate.nonzero().view(-1)
    gated[rows_t, anchor[rows_t]] = -float('inf')
    rule = ((gated >= gated.gather(1, target[:, None])) & allowed).sum(1)
    bad = ~torch.isfinite(s).all(1)
    rank[bad] = optimistic[bad] = rule[bad] = allowed[bad].sum(1)
    return rank, optimistic, rule, int(bad.sum())


def write(path: Path, value: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, default=str))
    temporary.replace(path)


def package_version(name: str) -> Optional[str]:
    """The installed version of a distribution, or ``None``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """The worker's options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--graph-dir', type=Path, required=True, help='the staged train.txt, valid.txt and test.txt')
    p.add_argument('--name', required=True, help='the graph name (KGPFN dataset name)')
    p.add_argument('--meta', type=Path, required=True, help="the host's meta.json (positions, reflexive, parse and file pins)")
    p.add_argument('--code', type=Path, required=True, help='the KGPFN code folder')
    p.add_argument('--checkpoints', type=Path, required=True, help='the folder of kgpfn_icl_3g.pth and the TabICL checkpoint')
    p.add_argument('--work', type=Path, required=True, help='a writable folder (the TabICL yaml)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, required=True, help='test facts per batch')
    p.add_argument('--entity-chunk', type=int, required=True, help='NBFNet heads per chunk (memory only)')
    p.add_argument('--eval-chunk', type=int, required=True, help='candidates per scoring chunk (memory only)')
    p.add_argument('--num-neg', type=int, required=True, help='context negatives')
    p.add_argument('--gpu-gb', type=float, default=0.0, help='cap of the CUDA allocator in GB (0: none)')
    p.add_argument('--dump-scores', type=Path, default=None, help='write the score rows (small graphs)')
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Score one graph; returns the exit code."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s kgpfn_worker: %(message)s')
    a = parse_args(argv)
    began = time.time()
    import yaml  # type: ignore[import-untyped]
    from pfn import tasks, util  # type: ignore[import-not-found]
    from script import test_kgpfn  # type: ignore[import-not-found]
    code = a.code.resolve()
    origins = [Path(m.__file__).resolve() for m in (test_kgpfn, tasks, util)]
    if not all(o.is_relative_to(code) for o in origins):
        raise SystemExit(f'KGPFN imported from outside {code}: {origins}')
    meta = json.loads(a.meta.read_text())
    checkpoint, tabicl = a.checkpoints / CHECKPOINT, a.checkpoints / TABICL
    commit = code_commit(code)
    if commit != meta['pins']['code_commit']:
        raise SystemExit(f'KGPFN code at {commit}, the pin {meta["pins"]["code_commit"]}')
    sums = {f.name: sha256(f) for f in (checkpoint, tabicl)}
    if sums != {CHECKPOINT: meta['pins'][CHECKPOINT], TABICL: meta['pins']['tabicl']}:
        raise SystemExit(f'checkpoint SHA-256 differ from the pins: {sums}')

    a.work.mkdir(parents=True, exist_ok=True)
    tab_cfg = torch.load(tabicl, map_location='cpu', weights_only=True)['config']
    if not isinstance(tab_cfg, dict):
        raise SystemExit(f'the TabICL checkpoint config is a {type(tab_cfg).__name__}')
    tab_yaml = a.work / 'tabicl.yaml'
    write_yaml = tab_yaml.with_suffix(f'.{os.getpid()}.tmp')
    write_yaml.write_text(yaml.dump(tab_cfg, default_flow_style=False))
    write_yaml.replace(tab_yaml)
    cfg = util.load_config(str(code / 'config/script/test.yaml'), context={'gpus': [0]})
    cfg.model_config_path = str(code / 'config/kgpfn/kgpfn_icl.yaml')
    util.apply_model_config(cfg)
    if 'fast_test' in cfg.train:
        cfg.train.pop('fast_test')
    cfg.train.update(num_epoch=0, init_model=False, tabicl_config_path=str(tab_yaml), kgpfn_checkpoint=str(checkpoint))
    if (cfg.model.context_graph, cfg.model.context_tail, cfg.model.feature_transformer) != (3, True, 'tabicl'):
        raise SystemExit('the KGPFN preset is not context graph 3 with context tail and TabICL')
    cfg.model.entity_model.entity_chunk_size = a.entity_chunk

    t0 = time.time()
    model = test_kgpfn.create_model(cfg, init=False, ckpt_path=str(checkpoint), map_location='cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    msd = model.state_dict()
    check: dict[str, Any] = {'model_tensors': len(msd), 'checkpoint_tensors': len(state), 'missing': [k for k in msd if k not in state],
                             'shape_mismatch': [k for k in msd if k in state and msd[k].shape != state[k].shape],
                             'unexpected': [k for k in state if k not in msd]}
    check['value_mismatch'] = [k for k in msd if k in state and k not in check['shape_mismatch']
                               and not (msd[k].dtype == state[k].dtype and torch.equal(msd[k], state[k]))]
    if any(check[k] for k in ('missing', 'shape_mismatch', 'value_mismatch', 'unexpected')):
        raise SystemExit(f'strict load check failed: {json.dumps(check)[:3000]}')
    logger.info('strict load check passed: all %d model tensors equal the checkpoint', len(msd))
    del state, msd
    device = torch.device('cuda:0')
    if a.gpu_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory), 0)
    model = model.to(device).eval()
    t1 = time.time()
    t_model = t1 - t0
    import model.ultra.rspmm as rspmm  # type: ignore[import-not-found]  # the JIT build of the rspmm kernel, outside the timed loop
    t_rspmm = time.time() - t1
    if not Path(rspmm.__file__).resolve().is_relative_to(code):
        raise SystemExit(f'the rspmm kernel imported from outside {code}: {rspmm.__file__}')

    t1 = time.time()

    splits = {s: read_split(a.graph_dir / f'{s}.txt') for s in ('train', 'valid', 'test')}
    bad_parse = [s for s in splits if parse_sha(splits[s]) != meta['parse_sha256'][s]]
    if bad_parse:
        raise SystemExit(f"the parse of {bad_parse} differs from the host's")
    graphs, ids, n, num_relations, relation_ids = build_graphs(splits, a.name.replace(':', '-'), tasks)
    filters = tasks.make_filtered_ranking_graph(graphs['test'], graphs['train'], graphs['test'], graphs['valid'])
    known_t: dict[tuple[int, int], set[int]] = defaultdict(set)
    known_h: dict[tuple[int, int], set[int]] = defaultdict(set)
    for x in ids.values():
        for h, t, r in x.tolist():
            known_t[h, r].add(t)
            known_h[t, r].add(h)
    graph, filters = graphs['test'].to(device), filters.to(device)
    t_data = time.time() - t1

    num_pos = int(cfg.task.num_pos)
    both = types.SimpleNamespace(dataset='', num_relations=graph.num_relations)   # head queries on every graph
    test = ids['test']
    nt = len(test)
    order: list[int] = list(DistributedSampler(cast(Any, test), num_replicas=1, rank=0))   # KGPFN's test loader (shuffle, seed 0)
    positions = meta['positions']
    if positions is not None:
        keep = set(positions)
        order = [i for i in order if i in keep]
        if len(order) != len(keep) or max(keep) >= nt:
            raise SystemExit('the positions of the meta are not rows of test.txt')
    missing = [r for r in meta['reflexive'] if r not in relation_ids]
    if missing:
        raise SystemExit(f'reflexive relations not in the graph: {missing}')
    reflexive = torch.tensor(sorted(relation_ids[r] for r in meta['reflexive']), dtype=torch.long, device=device)
    keys = ('tail', 'head', 'tail_opt', 'head_opt', 'tail_rule', 'head_rule', 'native_tail', 'native_head')
    out: dict[str, list[Optional[int]]] = {k: [None] * nt for k in keys}
    dump: Optional[dict[str, list]] = {'index': [], 'head': [], 'scores': []} if a.dump_scores else None
    secs: dict[str, float] = defaultdict(float)
    done = nonfinite = mismatch = 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    t_loop = last = time.time()
    try:
        with torch.no_grad():
            for start in range(0, len(order), a.batch_size):
                idx = order[start:start + a.batch_size]
                b = len(idx)
                batch = test[idx].to(device)
                eval_batch = torch.cat([batch, tasks.inverse_relation_queries(both, batch)])
                query = eval_batch.clone()
                query[:, 1] = -1   # the answer-blind context
                c0 = time.time()
                ctx_x, ctx_y = tasks.build_context_relation_aware(graph, query.unsqueeze(1), num_pos=num_pos, num_neg=a.num_neg)
                ctx_x, ctx_y = [c.to(device) for c in ctx_x], [y.to(device) for y in ctx_y]
                c1 = time.time()
                pred = test_kgpfn._score_all_candidate_tails(model, graph, eval_batch, ctx_x, ctx_y, a.eval_chunk)
                torch.cuda.synchronize(device)
                c2 = time.time()
                target = eval_batch[:, 1]
                t_mask, _ = tasks.strict_negative_mask(filters, eval_batch)
                native = tasks.compute_ranking(pred, target, t_mask)
                gate = ~torch.isin(batch[:, 2], reflexive).repeat(2)
                ours, optimistic, rule, bad = our_ranks(pred, batch, target, known_t, known_h, gate, eval_batch[:, 0])
                nonfinite += bad
                mismatch += int((ours != native).sum())
                ours_l, opt_l, rule_l, native_l = ours.tolist(), optimistic.tolist(), rule.tolist(), native.tolist()
                if dump is not None:
                    dump['index'] += idx * 2
                    dump['head'] += [False] * b + [True] * b
                    dump['scores'].append(pred.float().cpu().numpy())
                for j, i in enumerate(idx):
                    out['tail'][i], out['head'][i] = ours_l[j], ours_l[b + j]
                    out['tail_opt'][i], out['head_opt'][i] = opt_l[j], opt_l[b + j]
                    out['tail_rule'][i], out['head_rule'][i] = rule_l[j], rule_l[b + j]
                    out['native_tail'][i], out['native_head'][i] = native_l[j], native_l[b + j]
                done += b
                secs['context'] += c1 - c0
                secs['score'] += c2 - c1
                secs['rank'] += time.time() - c2
                if time.time() - last > 60 or done == len(order):
                    last = time.time()
                    logger.info('%s: %d of %d facts, %.0f s', a.name, done, len(order), last - t_loop)
    except torch.cuda.OutOfMemoryError as error:
        write(a.out.with_name(a.out.stem + '.error.json'), dict(graph=a.name, status='oom', batch_size=a.batch_size, done=done,
                                                                error=str(error).splitlines()[0][:400],
                                                                peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30))
        logger.error('%s: CUDA out of memory at batch size %d', a.name, a.batch_size)
        return EXIT_OOM
    t_loop = time.time() - t_loop
    scored = [i for i in range(nt) if out['tail'][i] is not None]
    ties = sum(out[f'{side}'][i] != out[f'{side}_opt'][i] for i in scored for side in ('tail', 'head'))
    result = dict(
        graph=a.name, status='complete', context_mode='blind', seed=SEED,
        counts=dict(entities=n, relations=num_relations, train=len(splits['train']), valid=len(splits['valid']), test=nt,
                    test_scored=len(scored), queries_scored=2 * len(scored), nonfinite_rows=nonfinite),
        rank_mismatches_vs_native=mismatch, queries_with_target_tie=ties,
        settings=dict(batch_size=a.batch_size, eval_chunk_size=a.eval_chunk, entity_chunk_size=a.entity_chunk, num_pos=num_pos,
                      num_neg=a.num_neg, init_model=False, num_epoch=0, gpu_gb=a.gpu_gb, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      order='torch DistributedSampler(num_replicas=1, rank=0, shuffle=True, seed=0), as their test loader',
                      seeding='random, numpy, torch (CPU+CUDA) seeded right before the scoring loop',
                      ranked_positions=None if positions is None else len(positions), reflexive=sorted(meta['reflexive']),
                      meta_sha256=sha256(a.meta), model=cfg.model, task=cfg.task),
        strict_load=check,
        times=dict(total_s=time.time() - began, model_load_s=t_model, rspmm_load_s=t_rspmm, data_s=t_data, loop_s=t_loop,
                   context_s=secs['context'], score_s=secs['score'], rank_s=secs['rank']),
        gpu=dict(name=torch.cuda.get_device_name(device), peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                 peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30),
        provenance=dict(code_commit=commit, checkpoint_sha256=sums, worker_sha256=sha256(Path(__file__)), python=sys.version.split()[0],
                        torch=torch.__version__, cuda=torch.version.cuda, tabicl=package_version('tabicl'),
                        torch_geometric=package_version('torch_geometric'), image=os.environ.get('KGPFN_IMAGE'),
                        image_id=os.environ.get('KGPFN_IMAGE_ID'),
                        split_sha256={s: sha256(a.graph_dir / f'{s}.txt') for s in ('train', 'valid', 'test')}),
        tail_rank=out['tail'], head_rank=out['head'], tail_rank_opt=out['tail_opt'], head_rank_opt=out['head_opt'],
        tail_rank_rule=out['tail_rule'], head_rank_rule=out['head_rule'])
    if mismatch:
        result.update(native_tail_rank=out['native_tail'], native_head_rank=out['native_head'])
    if dump is not None and a.dump_scores is not None:
        np.savez_compressed(a.dump_scores, index=np.asarray(dump['index']), head=np.asarray(dump['head']), scores=np.concatenate(dump['scores']))
    write(a.out, result)
    logger.info('%s: %d facts scored in %.0f s -> %s', a.name, len(scored), t_loop, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
