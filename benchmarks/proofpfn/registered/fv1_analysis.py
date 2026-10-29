"""fresh-v1: the registered analysis of the new-graph test (ICML roadmap step 7). Frozen with the registration.

Registration: docs/synthetic_kg_fresh_v1_registration.md (sections 6 to 13 define what this file computes).
Validation on simulated data: test_fv1_analysis.py. Python 3.12 with numpy only. CPU only. The script reads no graph
file, loads no model and is deterministic (SEED; every population draws its own stream from SEED and its name).

Commands
  analyse    the registered tests, the sensitivity analyses and the descriptive tables -> one JSON file
  precision  the precision estimate from development rows for a planned design (registration, section 13)

Input of `analyse` (CSV files with a header row)
  --mrr      graph, block, source, system, seed, mrr
             One row per graph, system and seed (one model). mrr is the raw MRR of that model on the graph's test
             facts in both directions, filtered with train, valid and test, ties pessimistic, without the self-loop
             rule and without GT2. block is 1a, 1b, 2 or 3. source is the source cluster (one cluster per source,
             also across blocks). system is a key of SYSTEMS. Seeds: 42, 43, 44 for our families, the five release
             seeds for MERIT, 0 for a single checkpoint. A system outside SYSTEMS (an ensemble, the rule view, the
             double null) enters only the descriptive levels.
  --queries  graph, query, r1_at, rank_peer, r2_at
             The xrel predictors of every query of blocks 1a, 1b and 2 (the cells of section 8), as floats: r1_at and
             r2_at are sums of rule confidences, rank_peer is the filtered rank with ties at their mean.
  --ranks    graph, query, system, seed, rank
             The per-query raw ranks of the systems of CELL_SYSTEMS. They must reproduce the --mrr values.
  --gates    JSON {"kgpfn_parity": "pass" | "fail", "n_validity": "pass" | "fail"}, both fixed before the fresh read.
  --dropped  JSON {graph: {system label: reason}}: the failures (option B, section 14). A failed system's graph leaves
             every hypothesis, contrast and table that uses that system, and nothing else.
  --manifest KGs/fresh-v1/MANIFEST.json (required). Cross-check of the graphs, blocks and sources of blocks 1a, 1b, 2.
  --cost     graph, system, seed, seconds. Optional. The measured inference cost beside H4b (descriptive).

Input of `precision`
  --dev      development rows in the format of --mrr (block may be empty), with the development source clusters
  --manifest or --design (CSV: source, graphs, block) for the planned fresh design

Output of `analyse` (one JSON file, keys sorted)
  families.<family>.hypotheses.<H>   estimate, se, ci95, ci90 (unadjusted, the union of the two tests' intervals),
                                     lower (one-sided bounds at the family level and at the Bonferroni level),
                                     p = max(p_bootstrap, p_t), p_studentized (diagnostic), mc_se, p_adjusted,
                                     rejected, rejection_order. The intersection-union hypothesis H4a holds its
                                     components.
  families.primary.claims.complementarity    H2a and H2b both rejected (section 6), with the larger adjusted p-value
  families.mechanism.hypotheses.N1.wording   the wording flags of amendment A1
  sensitivity     margin_0_01, seed_inclusive, source_equal, blocks (with H1_published_block), leave_one_source_out
  descriptive     levels, contrasts_primary, per_source, per_graph, seed_effects, cells, block3, cost
  also            populations, gates, dropped, inputs (with SHA-256), script_sha256, draft, seed, bootstrap_replicates,
                  environment (Python and numpy versions against the registered ones)
"""
import argparse
import csv
import functools
import hashlib
import json
import math
import platform
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.dont_write_bytecode = True

VERSION = 'fv1-analysis draft 4 (29 October 2026, failure rule option B, H2 as the fixed sequence H2a -> H2b)'
DRAFT = False                 # frozen with docs/synthetic_kg_fresh_v1_registration.md (29 October 2026)
SEED = 20261029
B = 50_000
ALPHA = dict(primary=0.025, h4=0.025, mechanism=0.05, component=0.05)   # one-sided, family-wise
MARGIN = 0.02                 # H1, H3, H4b noninferiority margin
MARGIN_SENSITIVITY = 0.01     # the fixed sensitivity margin
NO_MATERIAL = 0.005           # amendment A1: "no material transfer" needs the 90% interval of S - N inside this
SMALL_K = 10                  # an analysis with fewer sources is flagged as small (registration, section 11.6)
ENUMERATE_K = 16              # the sign-flip test enumerates all sign vectors up to this many sources
CONFIRMATORY = ('1a', '1b', '2')
BLOCKS = ('1a', '1b', '2', '3')
SUBPOPULATIONS = {'published': ('1a', '1b'), 'converted': ('2',), 'block1a': ('1a',), 'block1b': ('1b',)}
QUERIES_PER_GRAPH = 2000      # 1,000 test facts, both directions (blocks 1a, 1b, 2)
SEED_PAIR_DIVISOR = 3         # A1: a registered statistic averages 3 seed pairs, so its seed variance is s^2 / 3
REGISTERED_ENV = dict(python='3.12.2', numpy='1.26.4')   # freeze condition (registration, section 16)

OUR_SEEDS = (42, 43, 44)
MERIT_SEEDS = (1024, 2025, 2026, 4000, 8000)
SYSTEMS = {  # label: (key in the table, seeds)
    'S': ('rec10axlepcunoa', OUR_SEEDS),   # PCUltra, synthetic only: the primary model
    'R': ('pcur', OUR_SEEDS),              # PCUltra, real only (matched control)
    'M': ('pcum', OUR_SEEDS),              # PCUltra, 50% real / 50% synthetic
    'N': ('pcun', OUR_SEEDS),              # PCUltra, structural null of four families (amendment A1)
    'HT': ('pcuht', OUR_SEEDS),            # PCUltra, hard targets (one sampled answer per query)
    'CU': ('rec10axlecunoa', OUR_SEEDS),   # CountULTRA, S's recipe
    'UL': ('rec10axleulnoa', OUR_SEEDS),   # ULTRA backbone, S's recipe
    'S0': ('rec10axlenoa', OUR_SEEDS),     # the old S with the context adapter: the control of DT and PV
    'DT': ('rec10axledtnoa', OUR_SEEDS),   # old architecture, prior without twin relations
    'PV': ('rec10axlepvnoa', OUR_SEEDS),   # old architecture, 0.10 of plain worlds replaced by peer worlds
    'TRIX': ('trix', (0,)),
    'ULTRA': ('ultra3g', (0,)),
    'FLOCK': ('flockoff', (0,)),           # official walk count and passes
    'MERIT': ('merit', MERIT_SEEDS),       # the five release checkpoints; the value is the mean of their MRRs
    'KGPFN': ('kgpfn', (0,)),              # kgpfn_icl_3g, answer-blind context, context seed 0
    'ANYBURL': ('anyburl', (0,)),          # KR 2026 setting, 30 s per graph on CPU, trained per graph
}
BLOCK3_SYSTEMS = ('S', 'R', 'M', 'TRIX', 'ULTRA', 'MERIT', 'KGPFN', 'FLOCK', 'ANYBURL')
CELL_SYSTEMS = ('S', 'N', 'S0', 'PV', 'DT')
CELLS = ('one_step', 'peer', 'neither', 'union', 'two_hop', 'freq_only')

CONTRASTS = {  # name: (minuend, subtrahend); the value per graph is the difference of the family values
    'S-TRIX': ('S', 'TRIX'), 'M-S': ('M', 'S'), 'M-R': ('M', 'R'), 'S-R': ('S', 'R'),
    'M-ULTRA': ('M', 'ULTRA'), 'M-TRIX': ('M', 'TRIX'), 'M-MERIT': ('M', 'MERIT'), 'M-KGPFN': ('M', 'KGPFN'),
    'M-FLOCK': ('M', 'FLOCK'), 'S-N': ('S', 'N'), 'S0-DT': ('S0', 'DT'), 'S-HT': ('S', 'HT'), 'S-CU': ('S', 'CU'),
    'CU-UL': ('CU', 'UL'),
    # descriptive only
    'S-ULTRA': ('S', 'ULTRA'), 'S-MERIT': ('S', 'MERIT'), 'S-KGPFN': ('S', 'KGPFN'), 'S-FLOCK': ('S', 'FLOCK'),
    'R-TRIX': ('R', 'TRIX'), 'S-ANYBURL': ('S', 'ANYBURL'), 'M-ANYBURL': ('M', 'ANYBURL'),
    'PV-S0': ('PV', 'S0'), 'S-S0': ('S', 'S0'),
}
PRIMARY_CONTRASTS = ('S-TRIX', 'M-S', 'M-R', 'S-R')
NI_CONTRASTS = ('S-TRIX', 'S-R', 'M-FLOCK')   # the statistics of H1, H3 and H4b.NI (null boundary -margin)
CELL_STATS = {'S-N': ('S', 'N'), 'PV-S0': ('PV', 'S0'), 'S0-DT': ('S0', 'DT')}

# Registered families (registration, sections 6 to 9). kind: 'sup' (difference > 0), 'ni' (difference > -margin),
# 'iu' (every component > 0; p = the largest component p-value; H4a only).
# slots: the divisor of the Bonferroni bounds when it differs from the number of hypotheses. claims: a claim that
# needs all the listed hypotheses rejected.
FAMILIES = {
    # Holm over H1, H2 and H3 with H2 as the fixed sequence H2a -> H2b inside its slot (section 6, user, 29 October)
    'primary': dict(alpha=ALPHA['primary'], procedure='graph', hypotheses={
        'H1': dict(kind='ni', stat='S-TRIX'),
        'H2a': dict(kind='sup', stat='M-R'),
        'H2b': dict(kind='sup', stat='M-S'),
        'H3': dict(kind='ni', stat='S-R')},
        weights={'H1': 1 / 3, 'H2a': 1 / 3, 'H3': 1 / 3, 'H2b': 0.0},
        transitions={('H1', 'H2a'): 0.5, ('H1', 'H3'): 0.5, ('H3', 'H1'): 0.5, ('H3', 'H2a'): 0.5,
                     ('H2a', 'H2b'): 1.0, ('H2b', 'H1'): 0.5, ('H2b', 'H3'): 0.5},
        # no 'slots': the Bonferroni bounds are at 0.025 / 4, simultaneous for H1, H2a, H2b and H3 (section 6, lead)
        claims={'complementarity': ('H2a', 'H2b')}),
    'h4': dict(alpha=ALPHA['h4'], procedure='graph', hypotheses={
        'H4a': dict(kind='iu', components=('M-ULTRA', 'M-TRIX', 'M-MERIT', 'M-KGPFN')),   # KGPFN only if its gate passes
        'H4b.NI': dict(kind='ni', stat='M-FLOCK'),
        'H4b.S': dict(kind='sup', stat='M-FLOCK')},
        weights={'H4a': 0.5, 'H4b.NI': 0.5, 'H4b.S': 0.0},
        transitions={('H4a', 'H4b.NI'): 1.0, ('H4b.NI', 'H4b.S'): 1.0, ('H4b.S', 'H4a'): 1.0}),
    'mechanism': dict(alpha=ALPHA['mechanism'], procedure='graph', hypotheses={
        'N1': dict(kind='sup', stat='S-N'),
        'N2': dict(kind='sup', stat='S-N@union-neither'),
        'DT': dict(kind='sup', stat='S0-DT'),
        'PV': dict(kind='sup', stat='PV-S0@peer')},
        weights={'N1': 0.5, 'N2': 0.0, 'DT': 0.25, 'PV': 0.25},
        transitions={('N1', 'N2'): 1.0, ('N2', 'DT'): 0.5, ('N2', 'PV'): 0.5, ('DT', 'N1'): 0.5, ('DT', 'PV'): 0.5,
                     ('PV', 'N1'): 0.5, ('PV', 'DT'): 0.5}),
    'component': dict(alpha=ALPHA['component'], procedure='holm', hypotheses={
        'RBW': dict(kind='sup', stat='S-HT'),
        'PAIR': dict(kind='sup', stat='S-CU'),
        'COUNT': dict(kind='sup', stat='CU-UL')}),
}
# If the validity gate of N fails (amendment A1), N1 and N2 leave the family before the read. DT and PV form a Holm pair.
MECHANISM_WITHOUT_N = dict(weights={'DT': 0.5, 'PV': 0.5}, transitions={('DT', 'PV'): 1.0, ('PV', 'DT'): 1.0})

# Training-seed SDs s of the tested statistics for the seed-inclusive sensitivity (section 11.8). Scale (A1): s is the SD
# of one seed-pair contrast (one model of A minus one model of B), so the registered statistic, a mean over 3 seed
# pairs, has the seed variance s^2 / SEED_PAIR_DIVISOR. The three S - N values are fixed by amendment A1
# (step5/expert-discussion/A-scratch/a_round2_z.py). The other three are the development values of 29 October 2026
# (`precision` on the PCUltra rows of the 91 development graphs, 89 for TRIX, field dev_seed_sd; registration,
# section 13.3). A statistic without a registered value uses the fresh-data estimate on the same scale, and the output
# labels it so.
SEED_SD = {'S-N': 0.0016, 'S-N@union': 0.0035, 'S-N@union-neither': 0.0042,
           'S-TRIX': 0.0014, 'S-CU': 0.0035, 'CU-UL': 0.0011}


# --- small helpers ------------------------------------------------------------------------------------------------

def derive_seed(*parts: str) -> np.random.SeedSequence:
    """A fixed random stream per name (independent of Python's hash randomization)."""
    return np.random.SeedSequence([SEED] + [zlib.crc32(p.encode()) for p in parts])


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path) -> list:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean(value):
    """JSON-safe values: numpy scalars become Python numbers, non-finite floats become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def environment() -> dict:
    """The Python and numpy versions of this run and whether they equal the registered ones (section 16)."""
    env = dict(python=platform.python_version(), numpy=np.__version__)
    return dict(env, registered=REGISTERED_ENV, matches_registration=env == REGISTERED_ENV)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for a Student t variable with df degrees of freedom."""
    if math.isnan(t):
        return float('nan')
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


_ERFC = np.frompyfunc(math.erfc, 1, 1)


def nct_sf(c: float, df: float, delta: float, n: int = 8001) -> float:
    """P(T' > c) for the noncentral t with df degrees of freedom and noncentrality delta (the power of a t test).

    T' = (Z + delta) / U with U = sqrt(V / df), V ~ chi^2(df), so P(T' > c) = E[P(Z > c U - delta)]. The expectation is
    integrated over the density of U, 2 (df/2)^(df/2) u^(df-1) exp(-df u^2 / 2) / Gamma(df/2), by Simpson's rule on
    [0, 1 + 40 / sqrt(2 df)]. At delta = 0 it equals t_sf to about 1e-10 (test_fv1_analysis.py).
    """
    if df <= 0:
        return float('nan')
    half = df / 2.0
    u = np.linspace(0.0, 1.0 + 40.0 / math.sqrt(2.0 * df), n)
    with np.errstate(divide='ignore', invalid='ignore'):   # u = 0: f[0] is set below
        logf = math.log(2.0) + half * math.log(half) - math.lgamma(half) + (df - 1.0) * np.log(u) - half * u * u
    f = np.exp(logf)
    f[0] = math.exp(math.log(2.0) + half * math.log(half) - math.lgamma(half)) if df == 1 else 0.0
    g = 0.5 * _ERFC((c * u - delta) / math.sqrt(2.0)).astype(np.float64)
    y = f * g
    h = u[1] - u[0]
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


@functools.lru_cache(maxsize=256)
def nct_noncentrality(c: float, df: float, power: float) -> float:
    """The noncentrality delta at which a one-sided t test with critical value c reaches the given power (bisection)."""
    lo, hi = -10.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if nct_sf(c, df, mid) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@functools.lru_cache(maxsize=4096)
def t_quantile(q: float, df: float) -> float:
    """Student t quantile by bisection on t_sf (accurate to 1e-12)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - t_sf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


# --- input --------------------------------------------------------------------------------------------------------

def parse_dropped(dropped) -> tuple:
    """dropped.json under option B (registration, section 14): {graph: {system label of SYSTEMS: reason}}.

    Returns ({graph: {label: reason}}, problems). The old global format {graph: reason} is refused with its own message.
    """
    problems, out = [], {}
    if dropped is None:
        return out, problems
    if not isinstance(dropped, dict):
        return out, ['dropped.json must be a JSON object {graph: {system label: reason}}']
    key_to_label = {key: label for label, (key, _) in SYSTEMS.items()}
    for g, entry in dropped.items():
        if not isinstance(entry, dict):
            problems.append(f'{g}: dropped.json uses the old global format {{graph: reason}}. The registered failure rule '
                            f'(option B, registration, section 14) drops a graph only from the analyses of the failed '
                            f'system and needs {{graph: {{system label: reason}}}}, for example '
                            f'{{"{g}": {{"FLOCK": "out of memory on GPUs 0 and 1"}}}}')
            continue
        if not entry:
            problems.append(f'{g}: an empty drop entry')
        for label, reason in entry.items():
            if label not in SYSTEMS:
                hint = f' (the key {label} has the label {key_to_label[label]})' if label in key_to_label else ''
                problems.append(f'{g}: unknown system label {label!r}{hint}. Labels: {", ".join(SYSTEMS)}')
                continue
            if not isinstance(reason, str) or not reason.strip():
                problems.append(f'{g} {label}: a drop needs a reason')
                continue
            out.setdefault(str(g), {})[label] = reason.strip()
    return {g: dict(sorted(v.items())) for g, v in sorted(out.items())}, problems


class Table:
    """Per-graph, per-seed MRR of every system, and each graph's block and source (registration, section 5).

    values[label][seed][graph] holds the registered systems (labels of SYSTEMS), extra[system][seed][graph] the others.
    family(label) is the per-graph family value: the mean of the single-model MRRs, never a score ensemble.
    dropped is {graph: {system label: reason}} (option B, registration, section 14). A drop removes that system's
    results on that graph: its rows there are ignored (and counted), and no analysis of that system uses the graph. A
    graph stays in the table. A drop next to complete results, a missing result without a drop, an unknown graph or
    label and the old format {graph: reason} are refused.
    """

    def __init__(self, rows: list, dropped: Optional[dict] = None, registered: bool = True, require_block: bool = True):
        self.dropped, problems = parse_dropped(dropped)
        self.block, self.source = {}, {}
        cells = defaultdict(dict)
        key_to_label = {key: label for label, (key, _) in SYSTEMS.items()}
        for i, r in enumerate(rows, start=2):
            try:
                g, src, system = r['graph'].strip(), r['source'].strip(), r['system'].strip()
                blk = (r.get('block') or '').strip() or ('' if require_block else 'dev')
                seed, mrr = int(r['seed']), float(r['mrr'])
            except (KeyError, ValueError, AttributeError) as e:
                problems.append(f'line {i}: unreadable row ({e})')
                continue
            if not g or not src or not system:
                problems.append(f'line {i}: empty graph, source or system')
                continue
            if require_block and blk not in BLOCKS:
                problems.append(f'line {i}: block {blk!r} is not one of {BLOCKS}')
            if not math.isfinite(mrr) or not 0.0 < mrr <= 1.0:
                problems.append(f'line {i}: mrr {mrr} outside (0, 1]')
            for field, store, value in (('block', self.block, blk), ('source', self.source, src)):
                if store.setdefault(g, value) != value:
                    problems.append(f'{g}: two values of {field} ({store[g]!r}, {value!r})')
            if seed in cells[(g, system)]:
                problems.append(f'{g} {system} seed {seed}: duplicate row')
            cells[(g, system)][seed] = mrr
        self.graphs = sorted(self.block)
        for g, entry in self.dropped.items():
            if g in self.block:
                for label in entry:
                    key, seeds = SYSTEMS[label]
                    if set(cells.get((g, key), {})) == set(seeds):
                        problems.append(f'{g} {label}: dropped, but the table holds its complete results (a drop needs a '
                                        f'failure on every GPU, section 14)')
            elif not (set(entry) >= set(SYSTEMS) or set(entry) >= set(BLOCK3_SYSTEMS)):
                problems.append(f'{g}: dropped, but the table has no row of this graph (a graph without any result '
                                f'needs every system of its block in dropped.json)')
        self.values = {label: defaultdict(dict) for label in SYSTEMS}
        self.extra = defaultdict(lambda: defaultdict(dict))
        self.ignored = defaultdict(int)   # rows of a dropped (graph, system), by "graph label"
        for (g, system), seeds in cells.items():
            label = key_to_label.get(system)
            if label is not None and label in self.dropped.get(g, {}):
                self.ignored[f'{g} {label}'] += len(seeds)
                continue
            for seed, mrr in seeds.items():
                if label is None:
                    self.extra[system][seed][g] = mrr
                else:
                    self.values[label][seed][g] = mrr
        self.ignored = dict(sorted(self.ignored.items()))
        if registered:
            for g in self.graphs:
                required = BLOCK3_SYSTEMS if self.block[g] == '3' else tuple(SYSTEMS)
                for label in required:
                    if label in self.dropped.get(g, {}):
                        continue
                    key, seeds = SYSTEMS[label]
                    have = set(cells.get((g, key), {}))
                    if have != set(seeds):
                        problems.append(f'{g} {key}: seeds {sorted(have)}, the plan needs {list(seeds)}, and '
                                        f'dropped.json has no entry {{"{g}": {{"{label}": reason}}}}')
        for system, by_seed in self.extra.items():
            covered = [set(v) for v in by_seed.values()]
            if any(x != covered[0] for x in covered):
                problems.append(f'unregistered system {system}: its seeds cover different graphs')
        if problems:
            raise ValueError('input table refused:\n  ' + '\n  '.join(problems[:40]) +
                             (f'\n  ... {len(problems) - 40} more' if len(problems) > 40 else ''))

    def family(self, label: str, graphs: list) -> np.ndarray:
        seeds = SYSTEMS[label][1]
        return np.array([np.mean([self.values[label][s][g] for s in seeds]) for g in graphs])

    def single(self, label: str, seed: int, graphs: list) -> np.ndarray:
        return np.array([self.values[label][seed][g] for g in graphs])

    def has(self, label: str, graphs: list) -> bool:
        return bool(graphs) and all(g in self.values[label][s] for s in SYSTEMS[label][1] for g in graphs)

    def missing_reason(self, label: str, g: str) -> str:
        """Why a system has no results on a graph: its drop reason, or 'not in the plan on this block'."""
        return self.dropped.get(g, {}).get(label, 'not in the plan on this block')

    def drops_of(self, graphs: list, labels) -> dict:
        """{graph: {label: reason}}: the drops of these systems on these graphs (section 14)."""
        out = {}
        for g in graphs:
            hit = {lab: self.dropped[g][lab] for lab in sorted(labels) if lab in self.dropped.get(g, {})}
            if hit:
                out[g] = hit
        return out

    def check_manifest(self, manifest: dict) -> None:
        problems = []
        listed = manifest['graphs']
        mine = {g for g in self.block if self.block[g] in CONFIRMATORY}
        for g, info in listed.items():
            if g not in mine:
                if set(self.dropped.get(g, {})) >= set(SYSTEMS):
                    continue   # every system failed on this graph: no analysis uses it
                problems.append(f'{g}: in the manifest, not in the table (a graph without any result needs every '
                                f'system in dropped.json)')
            elif str(info['block']) != self.block[g] or info['source'] != self.source[g]:
                problems.append(f'{g}: block or source differs from the manifest')
        for g in sorted(mine - set(listed)):
            problems.append(f'{g}: in the table, not in the manifest')
        for g in self.dropped:
            if g not in listed and g not in self.block:
                problems.append(f'{g}: dropped, but neither in the manifest nor in the table')
        if problems:
            raise ValueError('manifest check failed:\n  ' + '\n  '.join(problems[:40]))


class Cells:
    """Per-query reciprocal ranks of CELL_SYSTEMS and the query cells (registration, section 8.3).

    one_step: r1_at > 0. peer: r1_at = 0 and rank_peer <= 10. neither: r1_at = 0 and rank_peer > 10.
    union: one_step or peer. two_hop: r1_at = 0 and r2_at > 0. freq_only: neither and r2_at = 0.
    The predictors are floats: r1_at and r2_at are sums of rule confidences (at least 0), rank_peer is the filtered rank
    with ties at their mean (xr_diag.rank_of: 1 + #greater + 0.5 (#equal - 1), at least 1). A non-finite or
    out-of-range value and a duplicate (graph, query) row are refused, as is a duplicate (graph, query, system, seed)
    rank row. The queries must cover every graph of `graphs`. Ranks are required where the table has the system's
    results. Ranks of a dropped (graph, system) are ignored (section 14).
    """

    def __init__(self, queries: list, ranks: list, table: Table, graphs: list):
        problems = []
        q_of = defaultdict(dict)
        for i, r in enumerate(queries, start=2):
            try:
                g, q = r['graph'].strip(), r['query'].strip()
                vals = (float(r['r1_at']), float(r['rank_peer']), float(r['r2_at']))
            except (KeyError, ValueError, AttributeError) as e:
                problems.append(f'queries line {i}: unreadable ({e})')
                continue
            if q in q_of[g]:
                problems.append(f'queries line {i}: duplicate query {g} {q}')
                continue
            r1, rp, r2 = vals
            if not all(math.isfinite(v) for v in vals):
                problems.append(f'queries line {i}: a non-finite predictor ({r1}, {rp}, {r2})')
                continue
            if r1 < 0 or r2 < 0:
                problems.append(f'queries line {i}: r1_at {r1} or r2_at {r2} below 0 (sums of rule confidences)')
                continue
            if rp < 1:
                problems.append(f'queries line {i}: rank_peer {rp} below 1')
                continue
            q_of[g][q] = vals
        rr = defaultdict(dict)
        key_to_label = {SYSTEMS[label][0]: label for label in CELL_SYSTEMS}
        wanted = set(graphs)
        for i, r in enumerate(ranks, start=2):
            try:
                g, q, label = r['graph'].strip(), r['query'].strip(), key_to_label.get(r['system'].strip())
                seed, rank = int(r['seed']), int(r['rank'])
            except (KeyError, ValueError, AttributeError) as e:
                problems.append(f'ranks line {i}: unreadable ({e})')
                continue
            if label is None or g not in wanted or label in table.dropped.get(g, {}):
                continue
            if rank < 1:
                problems.append(f'ranks line {i}: rank {rank} below 1')
                continue
            if q in rr[(g, label, seed)]:
                problems.append(f'ranks line {i}: duplicate rank row {g} {q} {SYSTEMS[label][0]} seed {seed}')
                continue
            rr[(g, label, seed)][q] = 1.0 / rank
        self.cell, self.rr = {}, {}
        for g in graphs:
            qs = sorted(q_of.get(g, {}))
            if len(qs) != QUERIES_PER_GRAPH:
                problems.append(f'{g}: {len(qs)} queries with cells, the protocol has {QUERIES_PER_GRAPH}')
            pred = np.array([q_of[g][q] for q in qs], dtype=np.float64).reshape(-1, 3)
            r1, peer, r2 = pred[:, 0], pred[:, 1], pred[:, 2]
            one_step = r1 > 0
            peer_c = (r1 == 0) & (peer <= 10)
            neither = (r1 == 0) & (peer > 10)
            self.cell[g] = dict(one_step=one_step, peer=peer_c, neither=neither, union=one_step | peer_c,
                                two_hop=(r1 == 0) & (r2 > 0), freq_only=neither & (r2 == 0))
            for label in CELL_SYSTEMS:
                if not table.has(label, [g]):
                    continue   # a dropped system on this graph: no cell statistic of it uses the graph
                for seed in SYSTEMS[label][1]:
                    got = rr.get((g, label, seed), {})
                    if set(got) != set(qs):
                        problems.append(f'{g} {SYSTEMS[label][0]} seed {seed}: ranks do not cover the queries')
                        continue
                    vec = np.array([got[q] for q in qs])
                    self.rr[(g, label, seed)] = vec
                    if abs(vec.mean() - table.values[label][seed][g]) > 1e-9:
                        problems.append(f'{g} {SYSTEMS[label][0]} seed {seed}: ranks give MRR {vec.mean():.12f}, '
                                        f'the table {table.values[label][seed][g]:.12f}')
        if problems:
            raise ValueError('cell input refused:\n  ' + '\n  '.join(problems[:40]))

    def rr_of(self, label: str, g: str, seed: Optional[int]) -> np.ndarray:
        if seed is not None:
            return self.rr[(g, label, seed)]
        return np.mean([self.rr[(g, label, s)] for s in SYSTEMS[label][1]], axis=0)

    def sums(self, a: str, b: str, cell: str, graphs: list, seed_a=None, seed_b=None) -> tuple:
        """Per graph: the sum over the cell's queries of RR_a - RR_b divided by the graph's queries, and the share."""
        s, w = [], []
        for g in graphs:
            d = self.rr_of(a, g, seed_a) - self.rr_of(b, g, seed_b)
            m = self.cell[g][cell]
            s.append(float(d[m].sum()) / len(m))
            w.append(float(m.sum()) / len(m))
        return np.array(s), np.array(w)


# --- populations and the cluster bootstrap --------------------------------------------------------------------------

class Population:
    """Graphs with their source clusters. Statistics are graph-macro (each graph weight 1) unless stated."""

    def __init__(self, name: str, graphs: list, source: dict):
        self.name = name
        self.graphs = sorted(graphs)
        self.sources = sorted({source[g] for g in self.graphs})
        index = {s: k for k, s in enumerate(self.sources)}
        self.cluster = np.array([index[source[g]] for g in self.graphs], dtype=np.int64)
        self.K, self.G = len(self.sources), len(self.graphs)
        self.sizes = np.bincount(self.cluster, minlength=self.K)
        mmax = int(self.sizes.max()) if self.K else 1
        self.padded = np.zeros((self.K, mmax), dtype=np.int64)
        for k in range(self.K):
            members = np.flatnonzero(self.cluster == k)
            self.padded[k, :len(members)] = members
        self.m_of_graph = self.sizes[self.cluster] if self.K else np.zeros(0, dtype=np.int64)

    def summary(self, block: dict) -> dict:
        blocks = defaultdict(int)
        for g in self.graphs:
            blocks[block.get(g, '?')] += 1
        return dict(graphs=self.G, sources=self.K, graphs_by_block=dict(sorted(blocks.items())),
                    sources_by_size={str(m): int((self.sizes == m).sum()) for m in sorted(set(self.sizes.tolist()))},
                    largest_source_share=float(self.sizes.max() / self.G) if self.G else None,
                    small=self.K < SMALL_K)


class Bootstrap:
    """The registered two-stage cluster bootstrap (registration, section 11.2).

    Stage 1 draws K sources with replacement. Stage 2 draws, inside every drawn source, as many graphs as it has, with
    replacement. Test facts stay fixed: every system is scored on the same facts, in both directions of every fact.
    within=False gives the one-stage variant (sources only). The registered analysis never uses it. It serves only
    comparisons (the drafter's exploratory run and the statistical review, stats-review/simlib.py).
    """

    def __init__(self, pop: Population, b: int, seed, within: bool = True):
        rng = np.random.default_rng(seed)
        self.pop, self.b = pop, b
        k, mmax = pop.K, pop.padded.shape[1]
        src = rng.integers(0, k, size=(b, k))
        u = rng.random((b, k, mmax))
        m = pop.sizes[src]
        if within:
            pos = np.minimum((u * m[..., None]).astype(np.int64), m[..., None] - 1)
        else:
            pos = np.broadcast_to(np.arange(mmax), (b, k, mmax))
        del u
        # compact storage (one bootstrap per population, section 14): the same values as int64 indices and a float mask
        self.src = src.astype(np.int32)
        self.gidx = pop.padded[src[..., None], pos].astype(np.int32)
        self.mask = np.arange(mmax)[None, None, :] < m[..., None]
        self.z = rng.standard_normal(b)   # the seed-inclusive sensitivity adds seed_sd * z to every replicate


class Estimate:
    """A statistic sum_r lam_r * (sum_g a_gr / sum_g b_gr) on a population, observed and over the bootstrap.

    The graph-macro mean is one term with a = the per-graph differences and b = 1. A cell statistic has a = the cell
    sums and b = the cell shares. The standard error is the cluster-robust (CR1) linearization over sources.
    """

    def __init__(self, terms: list, pop: Population, boot: Optional[Bootstrap]):
        self.terms = terms
        self.k = k = pop.K
        theta, u = 0.0, np.zeros(k)
        for lam, a, b in terms:
            ak = np.bincount(pop.cluster, weights=a, minlength=k)
            bk = np.bincount(pop.cluster, weights=b, minlength=k)
            with np.errstate(invalid='ignore', divide='ignore'):   # an empty cell: pivot_test reports the cause
                ratio = ak.sum() / bk.sum()
                theta += lam * ratio
                u += lam * (ak - ratio * bk) / bk.sum()
        self.theta = float(theta)
        self.se = math.sqrt(k / (k - 1) * float((u ** 2).sum())) if k > 1 else float('nan')
        self.theta_b = self.se_b = self.z = None
        if boot is not None and k > 1:
            tb, ub = 0.0, 0.0
            for lam, a, b in terms:
                aj = (a[boot.gidx] * boot.mask).sum(-1)
                bj = (b[boot.gidx] * boot.mask).sum(-1)
                at, bt = aj.sum(1), bj.sum(1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    ratio = at / bt
                    tb = tb + lam * ratio
                    ub = ub + lam * (aj - ratio[:, None] * bj) / bt[:, None]
            self.theta_b = np.asarray(tb, dtype=np.float64)
            self.se_b = np.sqrt(k / (k - 1) * (ub ** 2).sum(1))
            self.z = boot.z

    def leave_one_out(self, pop: Population) -> list:
        """The estimate with each source removed in turn."""
        out = []
        for k in range(pop.K):
            keep = pop.cluster != k
            out.append(float(sum(lam * a[keep].sum() / b[keep].sum() for lam, a, b in self.terms)))
        return out


def order_stat(sorted_t: np.ndarray, k: int) -> float:
    """The k-th smallest value (1-based), clipped to the sample."""
    return float(sorted_t[min(max(k, 1), len(sorted_t)) - 1])


def pivot_test(est: Estimate, theta0: float, seed_sd: float = 0.0, levels=(0.025,)) -> dict:
    """The registered one-sided test of H0: theta <= theta0 against theta > theta0 (registration, section 11.3).

    Two tests must both reject. Their p-values:
      p_B  the cluster bootstrap test: p_B = (1 + #{theta*_b <= theta0}) / (B + 1). This inverts the percentile
           interval of the two-stage cluster bootstrap. Its lower bound at level a is the order statistic
           theta*_(floor(a (B + 1))), so p_B <= a holds exactly when that bound exceeds theta0 (without ties).
      p_T  the cluster-robust t test: T = (theta_hat - theta0) / SE (CR1 over sources), referred to Student t with
           K - 1 degrees of freedom. Its lower bound is theta_hat - t_(K-1, 1-a) SE.
    p = max(p_B, p_T). The bound of the registered test is the smaller of the two bounds, the interval the union.
    Reason (section 11.3, simulations in test_fv1_analysis.py and the statistical review): for an upper one-sided test
    both parts are liberal under left skew and conservative under right skew, the percentile part with the smaller
    skewness error. Under symmetric heavy tails in designs with mostly single-graph sources, the percentile part is
    liberal and the t part is not. The maximum keeps the smaller error. p_studentized is a diagnostic only.
    seed_sd = s > 0 adds A1's training-seed term on A1's scale (s: the SD of one seed-pair contrast): SE^2 + s^2 / 3
    in T, and (s / sqrt(3)) z_b in every replicate (SEED_PAIR_DIVISOR = 3).
    A replicate whose statistic is undefined (an empty cell in every drawn graph) counts as non-rejecting: it enters
    p_B as a replicate at or below theta0 and the bounds as -inf (lower) or +inf (upper). Their number is reported.
    Without a test, p is None and 'note' gives the cause.
    """
    theta, se = est.theta, est.se
    out = dict(estimate=theta, se=se, theta0=theta0, seed_sd=seed_sd)
    cause = None
    if est.k < 2:
        cause = f'fewer than two sources (K = {est.k})'
    elif not math.isfinite(theta):
        cause = 'the statistic is undefined on the observed data (for a cell statistic: the cell is empty in every graph)'
    elif not math.isfinite(se):
        cause = 'the standard error is undefined'
    elif est.theta_b is None:
        cause = 'no bootstrap replicates (B = 0)'
    if cause:
        out.update(p=None, note=f'no test: {cause}')
        return out
    b = len(est.theta_b)
    k = est.k
    seed_var = seed_sd ** 2 / SEED_PAIR_DIVISOR
    se_t = math.sqrt(se ** 2 + seed_var)
    theta_b = est.theta_b + math.sqrt(seed_var) * est.z if seed_sd > 0 else est.theta_b
    undefined = ~np.isfinite(theta_b)
    n_undefined = int(undefined.sum())
    if se_t > 0:
        tobs = (theta - theta0) / se_t
    else:
        tobs = math.inf if theta > theta0 else (-math.inf if theta < theta0 else 0.0)
    p_b = (1 + int((theta_b[~undefined] <= theta0).sum()) + n_undefined) / (b + 1)
    p_t = t_sf(tobs, k - 1)
    sorted_low = np.sort(np.where(undefined, -np.inf, theta_b))
    sorted_high = np.sort(np.where(undefined, np.inf, theta_b))
    # the studentized bootstrap (diagnostic only); undefined replicates count as non-rejecting here too
    se_b = np.sqrt(est.se_b ** 2 + seed_var)
    dev = theta_b - theta
    with np.errstate(divide='ignore', invalid='ignore'):
        tstar = np.where(se_b > 0, dev / se_b, np.sign(dev) * np.inf)
    tstar = np.where(np.isnan(tstar) & ~undefined, 0.0, tstar)
    p_s = (1 + int((tstar[~undefined] >= tobs).sum()) + n_undefined) / (b + 1)

    def lower(a: float) -> float:
        lb = order_stat(sorted_low, int(math.floor(a * (b + 1))))
        return min(lb, theta - t_quantile(1 - a, k - 1) * se_t)

    def upper(a: float) -> float:
        ub = order_stat(sorted_high, b + 1 - int(math.floor(a * (b + 1))))
        return max(ub, theta + t_quantile(1 - a, k - 1) * se_t)

    p = max(p_b, p_t)
    out.update(t=tobs, df=k - 1, p=p, p_bootstrap=p_b, p_t=p_t, p_studentized=p_s, undefined_replicates=n_undefined,
               mc_se=math.sqrt(p_b * (1 - p_b) / b), ci95=[lower(0.025), upper(0.025)], ci90=[lower(0.05), upper(0.05)],
               lower={f'{a:.6g}': lower(a) for a in levels})
    return out


def signflip_p(d: np.ndarray, pop: Population, theta0: float, draws: int, seed) -> dict:
    """The wild cluster restricted bootstrap-t with Rademacher weights (Cameron, Gelbach and Miller 2008) for the
    graph-macro mean: the small-cluster check of section 11.6, run at the hypothesis's own theta0.

    The null is imposed: the source deviations T_k - m_k theta0 get random signs. The statistic is the CR1 t statistic.
    Up to ENUMERATE_K sources every sign vector is enumerated. The test is then exact if the deviations are symmetric
    and independent. It is a diagnostic: no decision uses it.
    """
    k, g = pop.K, pop.G
    if k < 2:
        return dict(p=None, note='fewer than two sources')
    tk = np.bincount(pop.cluster, weights=d, minlength=k)
    mk = pop.sizes.astype(np.float64)
    dk = tk - mk * theta0
    if k <= ENUMERATE_K:
        eps = 1.0 - 2.0 * ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        eps = np.vstack([np.ones(k), rng.choice([-1.0, 1.0], size=(draws, k))])
        exact = False
    tk_star = mk * theta0 + eps * dk
    th = tk_star.sum(1) / g
    resid = tk_star - mk * th[:, None]
    se = np.sqrt(k / (k - 1) * (resid ** 2).sum(1)) / g
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, (th - theta0) / se, 0.0)
    p = float((t >= t[0] - 1e-12).mean())   # row 0 is the identity sign vector in both branches
    return dict(p=p, exact=exact, sign_vectors=int(len(eps)))


# --- multiplicity ---------------------------------------------------------------------------------------------------

def graph_procedure(p: dict, weights: dict, transitions: dict, alpha: float) -> dict:
    """Sequentially rejective weighted Bonferroni graph (Bretz et al. 2009), with their adjusted p-values (Algorithm 2).

    p: {hypothesis: p-value or None}. A hypothesis with None is never rejected and passes no weight.
    Returns {hypothesis: dict(p, p_adjusted, rejected, order)}. Ties are broken by the name, so the result is fixed.
    """
    hyps = list(weights)
    w = {h: float(weights[h]) for h in hyps}
    gm = {(i, j): float(transitions.get((i, j), 0.0)) for i in hyps for j in hyps if i != j}
    assert sum(w.values()) <= 1 + 1e-12, 'initial weights exceed 1'
    for i in hyps:
        assert sum(gm[(i, j)] for j in hyps if j != i) <= 1 + 1e-12, f'outgoing weights of {i} exceed 1'
    active = set(hyps)
    result = {h: dict(p=p.get(h), p_adjusted=1.0, rejected=False, order=None) for h in hyps}

    def ratio(h: str) -> float:
        ph = p.get(h)
        return math.inf if ph is None or w[h] <= 0 else ph / w[h]

    running, step = 0.0, 0
    while active:
        j = min(sorted(active), key=ratio)
        r = ratio(j)
        if not math.isfinite(r):
            break
        running = max(running, min(1.0, r))
        step += 1
        result[j].update(p_adjusted=running, order=step)
        active.remove(j)
        for l in active:
            w[l] += w[j] * gm[(j, l)]
        new = {}
        for l in active:
            for k in active:
                if l != k:
                    den = 1.0 - gm[(l, j)] * gm[(j, l)]
                    new[(l, k)] = (gm[(l, k)] + gm[(l, j)] * gm[(j, k)]) / den if den > 1e-15 else 0.0
        gm.update(new)
        w[j] = 0.0
    for h in hyps:
        result[h]['rejected'] = bool(result[h]['order'] is not None and result[h]['p_adjusted'] <= alpha)
    return result


def holm_graph(hyps: list) -> tuple:
    m = len(hyps)
    weights = {h: 1.0 / m for h in hyps}
    transitions = {(i, j): 1.0 / (m - 1) for i in hyps for j in hyps if i != j} if m > 1 else {}
    return weights, transitions


# --- the statistics ----------------------------------------------------------------------------------------------

class Bank:
    """Builds and caches the estimates of one population (registration, sections 6 to 10)."""

    def __init__(self, table: Table, pop: Population, b: int, cells: Optional[Cells] = None, within: bool = True):
        self.table, self.pop, self.cells = table, pop, cells
        self.boot = Bootstrap(pop, b, derive_seed('population', pop.name), within) if pop.K > 1 and b > 0 else None
        self.cache = {}
        self.sf_cache = {}

    def available(self, stat: str) -> bool:
        name, _, cell = stat.partition('@')
        if cell and self.cells is None:
            return False
        a, b = (CELL_STATS if cell else CONTRASTS)[name]
        return self.table.has(a, self.pop.graphs) and self.table.has(b, self.pop.graphs)

    def _values(self, label: str, seed: Optional[int]) -> np.ndarray:
        g = self.pop.graphs
        return self.table.family(label, g) if seed is None else self.table.single(label, seed, g)

    def terms(self, stat: str, weighting: str = 'graph', seeds: tuple = (None, None)) -> list:
        """seeds=(seed_a, seed_b) uses single models (seed effects). weighting 'source' gives every source weight 1."""
        name, _, cell = stat.partition('@')
        n = self.pop.G
        if not cell:
            a, b = CONTRASTS[name]
            d = self._values(a, seeds[0]) - self._values(b, seeds[1])
            if weighting == 'source':
                m = self.pop.m_of_graph.astype(np.float64)
                return [(1.0, d / m, 1.0 / m)]
            return [(1.0, d, np.ones(n))]
        assert weighting == 'graph', 'cell statistics use graph weights'
        a, b = CELL_STATS[name]
        parts = [(1.0, cell)] if '-' not in cell else [(1.0, cell.split('-')[0]), (-1.0, cell.split('-')[1])]
        out = []
        for lam, c in parts:
            s, w = self.cells.sums(a, b, c, self.pop.graphs, seeds[0], seeds[1])
            out.append((lam, s, w))
        return out

    def estimate(self, stat: str, weighting: str = 'graph') -> Estimate:
        key = (stat, weighting)
        if key not in self.cache:
            self.cache[key] = Estimate(self.terms(stat, weighting), self.pop, self.boot)
        return self.cache[key]

    def point(self, stat: str, seeds: tuple) -> float:
        return Estimate(self.terms(stat, seeds=seeds), self.pop, None).theta

    def contributions(self, stat: str, seeds: tuple) -> np.ndarray:
        """Per-graph linearized contributions x_g with mean theta: x_g = theta + G sum_r lam_r (a_gr - R_r b_gr) / B_r.

        For a graph-macro contrast x_g is the per-graph difference itself. This is A1's linearization of a cell
        statistic (step5/expert-discussion/A-scratch/a_round2_z.py, function linear).
        """
        g = self.pop.G
        theta, x = 0.0, np.zeros(g)
        for lam, a, b in self.terms(stat, seeds=seeds):
            ratio = a.sum() / b.sum()
            theta += lam * ratio
            x += lam * (a - ratio * b) / b.sum()
        return theta + g * x

    @staticmethod
    def sigma_u2(x: np.ndarray) -> float:
        """A1's global seed component: x_gs = tau_g + u_s + v_gs over graphs g and seed pairs s.

        sigma_u^2 = max(0, Var_s(T_s) - sigma_v^2 / G), T_s = the mean of column s, sigma_v^2 from the two-way residuals.
        """
        g, s = x.shape
        if s < 2:
            return 0.0
        col = x.mean(0)
        resid = x - x.mean(1, keepdims=True) - col[None, :] + x.mean()
        sv2 = float((resid ** 2).sum()) / ((g - 1) * (s - 1)) if g > 1 else 0.0
        return max(float(np.var(col, ddof=1)) - sv2 / g, 0.0)

    def seed_sd(self, stat: str) -> dict:
        """The training-seed SD s of a statistic on A1's scale (registration, section 11.8).

        s is the SD of the global seed component of one seed-pair contrast, so the statistic (a mean over its seed
        pairs) has the seed variance s^2 / SEED_PAIR_DIVISOR. Estimator (A1's): the two-way decomposition sigma_u2 of
        the per-graph contributions of each seed pair.
          - Both systems with the same seeds: the pairs (A_s, B_s).
          - One system a single checkpoint: the pairs (A_s, B) or (A, B_s).
          - Different seed sets (M against MERIT's five checkpoints): each system's own decomposition against the other's
            mean, combined to the seed variance v = sigma_A^2 / n_A + sigma_B^2 / n_B of the statistic.
        In every case s = sqrt(SEED_PAIR_DIVISOR * v), so that s^2 / 3 is the seed variance of the statistic. A single
        checkpoint on both sides gives s = 0: no training variation is observable.
        """
        name, _, cell = stat.partition('@')
        a, b = (CELL_STATS if cell else CONTRASTS)[name]
        sa, sb = SYSTEMS[a][1], SYSTEMS[b][1]
        if sa == sb or len(sa) == 1 or len(sb) == 1:
            if sa == sb:
                pairs = [(s, s) for s in sa]
            elif len(sb) == 1:
                pairs = [(s, sb[0]) for s in sa]
            else:
                pairs = [(sa[0], s) for s in sb]
            if len(pairs) < 2:
                return dict(s=0.0, seed_variance=0.0, method='single checkpoints: no training variation observable')
            x = np.column_stack([self.contributions(stat, p) for p in pairs])
            v = self.sigma_u2(x) / len(pairs)
            method = f'A1 two-way decomposition over {len(pairs)} seed pairs'
        else:
            xa = np.column_stack([self.contributions(stat, (s, None)) for s in sa])
            xb = np.column_stack([self.contributions(stat, (None, s)) for s in sb])
            v = self.sigma_u2(xa) / len(sa) + self.sigma_u2(xb) / len(sb)
            method = f'A1 decomposition per system ({len(sa)} and {len(sb)} models), combined'
        return dict(s=math.sqrt(SEED_PAIR_DIVISOR * v), seed_variance=v, method=method)

    def registered_or_fresh_sd(self, stat: str) -> dict:
        if stat in SEED_SD:
            return dict(sd=SEED_SD[stat], source='registered (A1 scale)')
        return dict(sd=self.seed_sd(stat)['s'], source='fresh-data estimate on the A1 scale (no registered value)')

    def signflip(self, stat: str, theta0: float) -> Optional[float]:
        """The sign-flip check (section 11.6) of a graph-macro statistic at its own theta0, cached."""
        if '@' in stat or self.pop.K < 2:
            return None
        key = (stat, round(theta0, 12))
        if key not in self.sf_cache:
            d = self.terms(stat)[0][1]
            self.sf_cache[key] = signflip_p(d, self.pop, theta0, B, derive_seed('signflip', self.pop.name, stat,
                                                                                f'{theta0:.6g}'))['p']
        return self.sf_cache[key]

    def test(self, stat: str, kind: str, margin: float = MARGIN, weighting: str = 'graph', seed_sd: float = 0.0,
             levels=(0.025,)) -> dict:
        theta0 = 0.0 if kind == 'sup' else -margin
        out = pivot_test(self.estimate(stat, weighting), theta0, seed_sd, levels)
        out.update(stat=stat, kind=kind, weighting=weighting, population=self.pop.name, sources=self.pop.K,
                   graphs=self.pop.G)
        if weighting == 'graph' and out.get('p') is not None:
            out['signflip_p'] = self.signflip(stat, theta0)
        return out


def family_spec(name: str, gates: dict) -> tuple:
    """The hypotheses, weights and transitions of a family after the gates (both fixed before the read)."""
    family = FAMILIES[name]
    hyps = dict(family['hypotheses'])
    weights = dict(family.get('weights', {}))
    transitions = dict(family.get('transitions', {}))
    notes = []
    if name == 'mechanism' and gates.get('n_validity') != 'pass':
        hyps.pop('N1')
        hyps.pop('N2')
        weights, transitions = dict(MECHANISM_WITHOUT_N['weights']), dict(MECHANISM_WITHOUT_N['transitions'])
        notes.append('validity gate of N not passed: N1 and N2 are descriptive only (amendment A1)')
    if name == 'h4' and gates.get('kgpfn_parity') != 'pass':
        hyps['H4a'] = dict(hyps['H4a'], components=tuple(c for c in hyps['H4a']['components'] if c != 'M-KGPFN'))
        notes.append('KGPFN parity gate not passed: KGPFN is descriptive only')
    if family['procedure'] == 'holm':
        weights, transitions = holm_graph(list(hyps))
    return hyps, weights, transitions, notes


def stat_systems(stat: str) -> tuple:
    """The two systems that a statistic compares (a graph-macro contrast or a cell statistic)."""
    name, _, cell = stat.partition('@')
    return (CELL_STATS if cell else CONTRASTS)[name]


class Context:
    """The populations of one analysis under the failure rule option B (registration, section 14).

    A hypothesis or a contrast uses the graphs of a base set on which all the systems it compares have results. Every
    distinct graph set is one population with its own bank. Its name is the base name when no graph is missing, and
    '<base>|without:<the missing graphs>' otherwise. Its bootstrap stream derives from SEED and that name, so a drop
    changes only the results whose systems failed.
    """

    def __init__(self, table: Table, cells: Optional[Cells], b: int):
        self.table, self.cells, self.b = table, cells, b
        self.conf = [g for g in table.graphs if table.block[g] in CONFIRMATORY]
        self.banks = {}

    def bank(self, base: str, graphs: list, labels, source: Optional[dict] = None) -> Bank:
        avail = [g for g in graphs if all(self.table.has(lab, [g]) for lab in labels)]
        missing = sorted(set(graphs) - set(avail))
        name = base if not missing else f"{base}|without:{','.join(missing)}"
        if name not in self.banks:
            self.banks[name] = Bank(self.table, Population(name, avail, source or self.table.source), self.b, self.cells)
        return self.banks[name]

    def info(self, bank: Bank, graphs: list, labels) -> dict:
        """K, G, the systems and the drops (with their reasons) of a hypothesis's or contrast's population."""
        return dict(population=bank.pop.name, sources=bank.pop.K, graphs=bank.pop.G, systems=sorted(labels),
                    dropped=self.table.drops_of(graphs, labels))


def family_tests(ctx: Context, name: str, gates: dict, margin: float = MARGIN, weighting: str = 'graph',
                 seed_inclusive: bool = False) -> dict:
    """The tests of one registered family, then its multiplicity procedure.

    Every hypothesis uses its own population (section 14): the confirmatory graphs on which all the systems it compares
    have results, for an intersection-union hypothesis all the systems of its components. A hypothesis whose population
    has fewer than SMALL_K sources is not tested (section 11.6). It is reported with its K, G and drops, takes no
    decision and passes no weight in the procedure. A tested hypothesis without a p-value stops the analysis with the
    cause. A claim of the family (the complementarity claim of section 6) holds when all its hypotheses are rejected.
    """
    alpha = FAMILIES[name]['alpha']
    hyps, weights, transitions, notes = family_spec(name, gates)
    m = FAMILIES[name].get('slots', len(hyps))
    levels = (alpha, alpha / m)
    tests, pvals = {}, {}
    for h, spec in hyps.items():
        stats = spec['components'] if spec['kind'] == 'iu' else (spec['stat'],)
        kind = 'sup' if spec['kind'] == 'iu' else spec['kind']
        labels = sorted({lab for s in stats for lab in stat_systems(s)})
        bank = ctx.bank('primary', ctx.conf, labels)
        info = ctx.info(bank, ctx.conf, labels)
        if bank.pop.K < SMALL_K:
            pvals[h] = None
            tests[h] = dict(kind=spec['kind'], p=None, not_tested=f'{bank.pop.K} sources after the drops. A registered '
                            f'hypothesis needs at least {SMALL_K} (registration, sections 11.6 and 14)', **info)
            continue
        comps = {}
        for s in stats:
            if not bank.available(s):
                raise ValueError(f'family {name}, hypothesis {h}: the statistic {s} lacks input rows (or the cell input)')
            sd = bank.registered_or_fresh_sd(s) if seed_inclusive else dict(sd=0.0, source='none')
            r = bank.test(s, kind, margin, weighting, sd['sd'], levels)
            if r.get('p') is None:
                raise ValueError(f"family {name}, hypothesis {h}, statistic {s}: {r.get('note', 'no p-value')}")
            r['seed_sd_source'] = sd['source']
            r['margin'] = margin if kind == 'ni' else 0.0
            comps[s] = r
        if spec['kind'] == 'iu':
            pvals[h] = max(c['p'] for c in comps.values())
            tests[h] = dict(kind='iu', components=comps, p=pvals[h], rule='p = the largest component p-value', **info)
        else:
            pvals[h] = comps[stats[0]]['p']
            tests[h] = dict(comps[stats[0]], systems=info['systems'], dropped=info['dropped'])
    decided = graph_procedure(pvals, weights, transitions, alpha)
    for h in hyps:
        tests[h].update(p_adjusted=decided[h]['p_adjusted'], rejected=decided[h]['rejected'],
                        rejection_order=decided[h]['order'])
    out = dict(alpha=alpha, one_sided=True, procedure=FAMILIES[name]['procedure'], weights=weights,
               transitions={f'{i}->{j}': v for (i, j), v in sorted(transitions.items())}, margin=margin,
               weighting=weighting, seed_inclusive=seed_inclusive, notes=notes, hypotheses=tests,
               bonferroni_level=alpha / m)
    if FAMILIES[name].get('claims'):
        # the smallest level at which all parts are rejected: the largest adjusted p-value of the parts
        out['claims'] = {c: dict(requires=list(parts), rejected=all(tests[h]['rejected'] for h in parts),
                                 p_adjusted=max(tests[h]['p_adjusted'] for h in parts),
                                 rule=f"{' and '.join(parts)} both rejected (registration, section 6)")
                         for c, parts in FAMILIES[name]['claims'].items()}
    return out


# --- the analysis ----------------------------------------------------------------------------------------------

def levels_table(table: Table, graphs: list) -> dict:
    """Each system's level on the graphs of `graphs` where it has results (section 14). 'missing' lists the others."""
    out = {}
    if not graphs:
        return out
    for label, (key, seeds) in SYSTEMS.items():
        avail = [g for g in graphs if table.has(label, [g])]
        if not avail:
            continue
        per_seed = {str(s): float(table.single(label, s, avail).mean()) for s in seeds}
        entry = dict(key=key, family=float(table.family(label, avail).mean()), per_seed=per_seed,
                     sd_over_models=float(np.std(list(per_seed.values()), ddof=1)) if len(seeds) > 1 else None,
                     graphs=len(avail))
        have = set(avail)
        missing = {g: table.missing_reason(label, g) for g in graphs if g not in have}
        if missing:
            entry['missing'] = missing
        out[label] = entry
    for system, by_seed in sorted(table.extra.items()):
        if all(g in v for v in by_seed.values() for g in graphs):
            per_seed = {str(s): float(np.mean([v[g] for g in graphs])) for s, v in sorted(by_seed.items())}
            out[f'unregistered:{system}'] = dict(key=system, family=float(np.mean(list(per_seed.values()))),
                                                 per_seed=per_seed, note='descriptive only')
    return out


def contrast_rows(ctx: Context, base: str, graphs: list, stats: list, sign_flip: bool = True, weighting: str = 'graph',
                  source: Optional[dict] = None) -> dict:
    """Estimate, SE and 95% interval of each statistic (unadjusted: no claim), each on its own population (the graphs
    of `graphs` where its two systems have results), with the sign-flip check at 0 and, for the statistics of H1, H3
    and H4b.NI, at -margin."""
    out = {}
    for s in stats:
        labels = stat_systems(s)
        bank = ctx.bank(base, graphs, labels, source)
        if not bank.available(s):
            continue
        r = bank.test(s, 'sup', weighting=weighting)
        row = dict(estimate=r['estimate'], se=r.get('se'), ci95=r.get('ci95'), p_sup=r.get('p'),
                   undefined_replicates=r.get('undefined_replicates'), note=r.get('note'), sources=bank.pop.K,
                   graphs=bank.pop.G, dropped=ctx.table.drops_of(graphs, labels))
        if sign_flip and weighting == 'graph':
            row['signflip_p_sup'] = bank.signflip(s, 0.0)
            if s in NI_CONTRASTS:
                row['signflip_p_ni'] = bank.signflip(s, -MARGIN)
        out[s] = row
    return out


def per_source(table: Table, pop: Population) -> list:
    """The four primary contrasts per source, each on the source's graphs where its systems have results. A contrast
    without such a graph is None. 'missing' lists the drops of the contrasts' systems on the source's graphs."""
    labels = sorted({lab for s in PRIMARY_CONTRASTS for lab in CONTRASTS[s]})
    rows = []
    for k, src in enumerate(pop.sources):
        graphs = [pop.graphs[i] for i in np.flatnonzero(pop.cluster == k)]
        row = dict(source=src, graphs=graphs, blocks=sorted({table.block[g] for g in graphs}))
        for s in PRIMARY_CONTRASTS:
            a, b = CONTRASTS[s]
            avail = [g for g in graphs if table.has(a, [g]) and table.has(b, [g])]
            row[s] = float((table.family(a, avail) - table.family(b, avail)).mean()) if avail else None
        missing = table.drops_of(graphs, labels)
        if missing:
            row['missing'] = missing
        rows.append(row)
    return rows


def seed_effects(ctx: Context) -> dict:
    """Every pairing of single models (cross), the same-seed pairs, and the fresh-data seed SD of the contrast on A1's
    scale (s, the SD of one seed-pair contrast) with the seed variance s^2 / 3 of the statistic. Each contrast uses its
    own population."""
    out = {}
    for s in PRIMARY_CONTRASTS + ('S-N', 'S-HT', 'S-CU', 'CU-UL', 'S0-DT', 'M-FLOCK', 'M-ULTRA', 'M-MERIT'):
        a, b = CONTRASTS[s]
        bank = ctx.bank('primary', ctx.conf, (a, b))
        if not bank.available(s):
            continue
        sa, sb = SYSTEMS[a][1], SYSTEMS[b][1]
        cross = {f'{x}|{y}': bank.point(s, (x, y)) for x in sa for y in sb}
        same = {str(x): cross[f'{x}|{x}'] for x in sa} if sa == sb else {}
        sd = bank.seed_sd(s)
        out[s] = dict(cross=cross, same_seed=same, min=min(cross.values()), max=max(cross.values()),
                      fresh_seed_sd_a1_scale=sd['s'], fresh_seed_variance_of_statistic=sd['seed_variance'],
                      seed_sd_method=sd['method'], sources=bank.pop.K, graphs=bank.pop.G)
    return out


def cell_tables(ctx: Context) -> dict:
    """The cell effects of the cell statistics, each on the graphs where its two systems have rows."""
    out = {}
    if ctx.cells is None:
        return out
    for name, (a, b) in CELL_STATS.items():
        bank = ctx.bank('primary', ctx.conf, (a, b))
        if not bank.available(f'{name}@union'):
            continue
        g = bank.pop.graphs
        rows = dict(population=ctx.info(bank, ctx.conf, (a, b)))
        same_seeds = SYSTEMS[a][1] if SYSTEMS[a][1] == SYSTEMS[b][1] else ()
        for c in CELLS + ('union-neither',):
            stat = f'{name}@{c}'
            est = bank.estimate(stat)
            row = dict(delta=est.theta, ci95=pivot_test(est, 0.0).get('ci95'),
                       same_seed={str(s): bank.point(stat, (s, s)) for s in same_seeds})
            if c in CELLS:
                share = np.array([ctx.cells.cell[x][c].mean() for x in g])
                row.update(queries=int(sum(int(ctx.cells.cell[x][c].sum()) for x in g)), mean_share=float(share.mean()),
                           graphs_nonempty=int((share > 0).sum()))
            rows[c] = row
        out[name] = rows
    return out


def cost_table(rows: list, graphs: list) -> dict:
    per = defaultdict(lambda: defaultdict(list))
    keep = set(graphs)
    for r in rows:
        if r['graph'].strip() in keep:
            per[r['system'].strip()][r['graph'].strip()].append(float(r['seconds']))
    single = {s: {g: float(np.mean(v)) for g, v in gs.items()} for s, gs in per.items()}
    out = dict(per_system={s: dict(total_seconds=float(sum(v.values())), median_seconds=float(np.median(list(v.values()))),
                                   graphs=len(v)) for s, v in sorted(single.items())})
    f, m = SYSTEMS['FLOCK'][0], SYSTEMS['M'][0]
    if f in single and m in single:
        common = sorted(set(single[f]) & set(single[m]))
        if common:
            out['flock_over_m'] = dict(total=sum(single[f][g] for g in common) / sum(single[m][g] for g in common),
                                       median_per_graph=float(np.median([single[f][g] / single[m][g] for g in common])),
                                       graphs=len(common), note='one M model (the mean over its seeds) against one Flock run')
    return out


def analyse(table: Table, cells: Optional[Cells], gates: dict, b: int = B, cost_rows: Optional[list] = None) -> dict:
    for key in ('kgpfn_parity', 'n_validity'):
        if gates.get(key) not in ('pass', 'fail'):
            raise ValueError(f'gate {key} must be "pass" or "fail" (fixed before the read)')
    ctx = Context(table, cells, b)
    conf = ctx.conf
    primary = ctx.bank('primary', conf, ()).pop
    env = environment()
    if not DRAFT and not env['matches_registration']:
        raise ValueError(f'environment {env} differs from the registered one (registration, section 16)')
    res = dict(version=VERSION, draft=DRAFT, seed=SEED, bootstrap_replicates=b, gates=gates, dropped=table.dropped,
               ignored_rows_of_dropped_systems=table.ignored, failure_rule='option B (registration, section 14)',
               registered_seed_sd=SEED_SD, seed_sd_scale='A1: the SD of one seed-pair contrast, seed variance s^2 / 3',
               environment=env, populations={'primary': primary.summary(table.block)})

    # the registered families (sections 6 to 9), every hypothesis on its own population (section 14)
    res['families'] = {name: family_tests(ctx, name, gates) for name in FAMILIES}

    # sensitivity analyses (section 11.7 to 11.10), on the same per-hypothesis populations: no claim rests on them
    sens = dict(margin_0_01={n: family_tests(ctx, n, gates, margin=MARGIN_SENSITIVITY) for n in ('primary', 'h4')},
                seed_inclusive={n: family_tests(ctx, n, gates, seed_inclusive=True) for n in FAMILIES},
                source_equal=dict(primary=family_tests(ctx, 'primary', gates, weighting='source'),
                                  contrasts=contrast_rows(ctx, 'primary', conf, list(PRIMARY_CONTRASTS), False, 'source')))
    blocks = {}
    for name, blks in SUBPOPULATIONS.items():
        graphs = [g for g in conf if table.block[g] in blks]
        if not graphs:
            continue
        pop = ctx.bank(name, graphs, ()).pop
        entry = dict(population=pop.summary(table.block), contrasts=contrast_rows(ctx, name, graphs, list(CONTRASTS)))
        if name == 'published':
            sub = ctx.bank(name, graphs, ('S', 'TRIX'))
            if sub.pop.K > 1:
                h1 = sub.test('S-TRIX', 'ni', levels=(ALPHA['primary'],))
                h1.update(note='pre-specified sensitivity of H1 (decision C2). No claim rests on it',
                          small=sub.pop.K < SMALL_K, systems=['S', 'TRIX'], dropped=table.drops_of(graphs, ('S', 'TRIX')))
                entry['H1_published_block'] = h1
        blocks[name] = entry
    sens['blocks'] = blocks
    loo = {}
    for s in PRIMARY_CONTRASTS + ('M-FLOCK', 'S-N', 'S-HT', 'S-CU', 'CU-UL'):
        bank = ctx.bank('primary', conf, CONTRASTS[s])
        if bank.available(s) and bank.pop.K > 1:
            vals = bank.estimate(s).leave_one_out(bank.pop)
            i, j = int(np.argmin(vals)), int(np.argmax(vals))
            loo[s] = dict(min=vals[i], min_without=bank.pop.sources[i], max=vals[j], max_without=bank.pop.sources[j],
                          sources=bank.pop.K, graphs=bank.pop.G)
    sens['leave_one_source_out'] = loo
    res['sensitivity'] = sens

    # the wording rules of amendment A1 (section 8.5)
    mech = res['families']['mechanism']['hypotheses']
    if 'N1' in mech and mech['N1'].get('p') is not None:
        s_n = ctx.bank('primary', conf, ('S', 'N')).test('S-N', 'sup')
        seed_n1 = sens['seed_inclusive']['mechanism']['hypotheses']['N1']
        mech['N1']['wording'] = dict(
            causal_wording_allowed=bool(mech['N1']['rejected'] and seed_n1['rejected'] and gates['n_validity'] == 'pass'),
            seed_inclusive_rejected=bool(seed_n1['rejected']),
            ci90=s_n['ci90'],
            no_material_transfer=bool(gates['n_validity'] == 'pass' and s_n['ci90'][0] > -NO_MATERIAL
                                      and s_n['ci90'][1] < NO_MATERIAL))

    # descriptive analyses (section 10): every system and contrast on the graphs where its systems have results
    desc = dict(levels={'primary': levels_table(table, conf)})
    for name, blks in SUBPOPULATIONS.items():
        desc['levels'][name] = levels_table(table, [g for g in conf if table.block[g] in blks])
    desc['contrasts_primary'] = contrast_rows(ctx, 'primary', conf, list(CONTRASTS))
    desc['per_source'] = per_source(table, primary)
    desc['per_graph'] = [dict(graph=g, block=table.block[g], source=table.source[g],
                              values={lab: float(table.family(lab, [g])[0]) for lab in SYSTEMS if table.has(lab, [g])},
                              **({'missing': dict(table.dropped[g])} if g in table.dropped else {}))
                         for g in table.graphs]
    desc['seed_effects'] = seed_effects(ctx)
    desc['cells'] = cell_tables(ctx)
    b3 = [g for g in table.graphs if table.block[g] == '3']
    if b3:
        own = {g: g for g in b3}
        pop3 = ctx.bank('block3', b3, (), own).pop
        desc['block3'] = dict(population=pop3.summary(table.block), levels=levels_table(table, b3),
                              contrasts=contrast_rows(ctx, 'block3', b3, list(PRIMARY_CONTRASTS) +
                                                      ['M-FLOCK', 'M-TRIX', 'M-ULTRA', 'M-MERIT', 'S-ANYBURL'], source=own),
                              note='unseen graphs from development sources (decision C1): graph-level clusters, no test')
    if cost_rows:
        desc['cost'] = cost_table(cost_rows, conf)
    res['descriptive'] = desc
    res['method'] = dict(bootstrap='two-stage cluster bootstrap: sources with replacement, then graphs within each drawn '
                                   'source with replacement, test facts fixed. One population and stream per set of '
                                   'graphs (failure rule option B)',
                         test='p = max(p_B, p_T). p_B = (1 + #{theta* <= theta0} + #undefined) / (B + 1). '
                              'p_T = P(t(K-1) >= T), T = (theta_hat - theta0) / SE, SE = CR1 over sources. The '
                              'studentized bootstrap p and the sign-flip p (wild cluster restricted bootstrap-t, '
                              'Rademacher) are diagnostics only',
                         seed_inclusive='SE^2 + s^2 / 3, s on the A1 scale', small_cluster_threshold=SMALL_K)
    return clean(res)


# --- precision (section 13) ------------------------------------------------------------------------------------------

def variance_components(d: np.ndarray, cluster: np.ndarray) -> dict:
    """One-way random-effects ANOVA estimates (unbalanced): between-source tau^2 and within-source sigma_w^2."""
    k = int(cluster.max()) + 1
    n = len(d)
    nk = np.bincount(cluster, minlength=k).astype(np.float64)
    mk = np.bincount(cluster, weights=d, minlength=k) / nk
    ssw = float(((d - mk[cluster]) ** 2).sum())
    ssb = float((nk * (mk - d.mean()) ** 2).sum())
    msw = ssw / (n - k) if n > k else float('nan')
    msb = ssb / (k - 1) if k > 1 else float('nan')
    n0 = (n - (nk ** 2).sum() / n) / (k - 1) if k > 1 else float('nan')
    if not math.isfinite(msw):
        return dict(tau2=max(0.0, msb), sigma_w2=0.0, icc=1.0, sources=k, graphs=n, note='one graph per source')
    tau2 = max(0.0, (msb - msw) / n0)
    return dict(tau2=tau2, sigma_w2=msw, icc=tau2 / (tau2 + msw) if tau2 + msw > 0 else 0.0, sources=k, graphs=n)


def skewness(x: np.ndarray) -> Optional[float]:
    """Sample skewness m3 / m2^1.5 (None for fewer than 3 values or no spread)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return None
    c = x - x.mean()
    m2 = float((c ** 2).mean())
    return float((c ** 3).mean() / m2 ** 1.5) if m2 > 0 else None


def predicted_se(tau2: float, sw2: float, sizes: list) -> float:
    """SE of a graph-macro mean for sources of the given graph counts: sqrt(tau2 sum m_k^2 / G^2 + sigma_w2 / G)."""
    g = float(sum(sizes))
    return math.sqrt(tau2 * sum(x * x for x in sizes) / g ** 2 + sw2 / g)


def precision(dev: Table, designs: dict) -> dict:
    """Expected precision of every registered graph-macro contrast on the planned fresh design (section 13).

    It also gives the skewness of the development differences, per graph and per source. Strong left skew (below -1)
    makes the registered test liberal (registration, section 18.3): left_skew_warning marks it before the freeze.
    Power and the smallest detectable effect use the noncentral t with K - 1 degrees of freedom (K = the sources of the
    design): the power of the t part of the registered test, an upper guide for the registered test itself.
    dev_seed_sd is on A1's scale (the SD s of one seed-pair contrast, seed variance s^2 / 3): the value for SEED_SD.
    """
    pop = Population('development', dev.graphs, dev.source)
    bank = Bank(dev, pop, 0)
    out = dict(development=dict(graphs=pop.G, sources=pop.K), environment=environment(),
               designs={k: dict(sources=len(v), graphs=int(sum(v)), sizes=v) for k, v in designs.items()}, contrasts={},
               method='power = P(t\'(K-1, delta) > t(K-1, 1-a)) with delta = margin / SE (noncentral t), MDE80 = delta80 SE; '
                      'seed-inclusive SE = sqrt(SE^2 + s^2 / 3)')
    for s in CONTRASTS:
        if not bank.available(s):
            continue
        d = bank.terms(s)[0][1]
        est = bank.estimate(s)
        vc = variance_components(d, pop.cluster)
        seed = bank.seed_sd(s)
        sd_seed = seed['s']
        tk = np.bincount(pop.cluster, weights=d, minlength=pop.K)
        src_skew = skewness(tk - pop.sizes * d.mean())
        row = dict(dev_estimate=est.theta, dev_se=est.se, per_graph_sd=float(np.std(d, ddof=1)), dev_seed_sd=sd_seed,
                   dev_seed_sd_method=seed['method'], per_graph_skewness=skewness(d), source_skewness=src_skew,
                   left_skew_warning=bool((src_skew or 0.0) < -1.0), designs={}, **vc)
        for name, sizes in designs.items():
            k = len(sizes)
            se = predicted_se(vc['tau2'], vc['sigma_w2'], sizes)
            se_seed = math.sqrt(se ** 2 + sd_seed ** 2 / SEED_PAIR_DIVISOR)
            entry = dict(se=se, se_scaled_by_sources=est.se * math.sqrt(pop.K / k), se_seed_inclusive=se_seed,
                         df=k - 1, half_width_95=t_quantile(0.975, k - 1) * se if k > 1 else None)
            for a in (ALPHA['primary'] / 3, ALPHA['primary'] / 2, ALPHA['primary'], 0.05):
                if k < 2:
                    continue
                c = t_quantile(1 - a, k - 1)
                d80 = nct_noncentrality(c, k - 1, 0.8)
                entry[f'alpha={a:.5f}'] = dict(power_ni_margin_0_02_true_0=nct_sf(c, k - 1, MARGIN / se),
                                               power_ni_margin_0_01_true_0=nct_sf(c, k - 1, MARGIN_SENSITIVITY / se),
                                               mde80_superiority=d80 * se,
                                               mde80_superiority_seed_inclusive=d80 * se_seed)
            row['designs'][name] = entry
        out['contrasts'][s] = row
    return clean(out)


def designs_from(manifest: Optional[dict], design_rows: Optional[list]) -> dict:
    """Graphs per source of the confirmatory population (blocks 1a, 1b, 2) and of the published block (1a, 1b)."""
    src = defaultdict(lambda: defaultdict(int))
    if manifest is not None:
        for info in manifest['graphs'].values():
            src[str(info['block'])][info['source']] += 1
    for r in design_rows or []:
        src[(r.get('block') or '2').strip()][r['source'].strip()] += int(r['graphs'])

    def sizes(blocks: tuple) -> list:
        per = defaultdict(int)
        for blk in blocks:
            for s, n in src.get(blk, {}).items():
                per[s] += n
        return sorted(per.values(), reverse=True)

    out = {'primary': sizes(CONFIRMATORY), 'published': sizes(('1a', '1b'))}
    return {k: v for k, v in out.items() if v}


# --- command line ---------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyse')
    for opt in ('--mrr', '--queries', '--ranks', '--gates', '--manifest', '--out'):
        a.add_argument(opt, required=True)
    for opt in ('--dropped', '--cost'):
        a.add_argument(opt)
    a.add_argument('--b', type=int, default=B, help='bootstrap replicates (the registered value is the default)')
    p = sub.add_parser('precision')
    p.add_argument('--dev', required=True)
    p.add_argument('--manifest')
    p.add_argument('--design')
    p.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.command == 'analyse':
        dropped = json.loads(Path(args.dropped).read_text()) if args.dropped else {}
        table = Table(read_csv(args.mrr), dropped)
        table.check_manifest(json.loads(Path(args.manifest).read_text()))
        conf = [g for g in table.graphs if table.block[g] in CONFIRMATORY]
        cells = Cells(read_csv(args.queries), read_csv(args.ranks), table, conf)
        gates = json.loads(Path(args.gates).read_text())
        res = analyse(table, cells, gates, args.b, read_csv(args.cost) if args.cost else None)
        names = ('mrr', 'queries', 'ranks', 'gates', 'dropped', 'manifest', 'cost')
        res['inputs'] = {k: dict(path=getattr(args, k), sha256=sha256_file(getattr(args, k))) for k in names
                         if getattr(args, k)}
        res['script_sha256'] = sha256_file(Path(__file__).resolve())
        Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
        for name, fam in res['families'].items():
            for h, t in fam['hypotheses'].items():
                est = '' if t.get('estimate') is None else f"estimate {t['estimate']:+.4f} "
                print(f"{name:9s} {h:7s} {est}p {t['p']} adjusted {t['p_adjusted']:.5f} rejected {t['rejected']}")
            for c, cl in fam.get('claims', {}).items():
                print(f"{name:9s} claim {c} ({' and '.join(cl['requires'])}) adjusted {cl['p_adjusted']:.5f} "
                      f"rejected {cl['rejected']}")
    else:
        dev = Table(read_csv(args.dev), registered=False, require_block=False)
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
        designs = designs_from(manifest, read_csv(args.design) if args.design else None)
        if not designs:
            raise SystemExit('name a design (--manifest or --design)')
        res = precision(dev, designs)
        res['inputs'] = dict(dev=dict(path=args.dev, sha256=sha256_file(args.dev)))
        res['script_sha256'] = sha256_file(Path(__file__).resolve())
        Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
        for s, row in res['contrasts'].items():
            d = row['designs'].get('primary') or next(iter(row['designs'].values()))
            print(f"{s:10s} dev {row['dev_estimate']:+.4f} (SE {row['dev_se']:.4f}); fresh SE {d['se']:.4f}; "
                  f"seed SD {row['dev_seed_sd']:.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
