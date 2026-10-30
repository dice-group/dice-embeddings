"""Command line of the ProofPFN benchmark: ``python -m benchmarks.proofpfn <command>``.

Commands:
    list        the graphs of a manifest view
    check       CPU only: the graphs' hash checks and counts, and the systems' files and pins; no score
    run         rank the graphs with the systems and write one row file per graph (``evaluate.evaluate_graph``)
    cells       the query cells of the graphs (``queries.csv``, ``cells.write_queries``)
    gate        the N validity gate of amendment A1: S and N on held-out worlds of the prior (``gate.gate``)
    fv1-models  the ``--models`` file of the registered ``fv1_tables.py`` from ``models.json``
    adapter     an external baseline (``kgpfn``, ``merit``, ``anyburl``) on the graphs: one row file per graph (``adapters``)

A family is ``--family [KEY=]ARM:TEMPLATE``: ARM is S, R, M, N, HT, CU or UL; TEMPLATE is the checkpoint or research
export path with ``{seed}`` (seeds 42, 43, 44); KEY defaults to the arm's registered key, whose pins every file must
match. References: ``--references trix,ultra3g,flockoff`` (checkpoints below ``--checkpoint-root``).

An adapter takes its external code from ``--code`` or its environment variable (``$KGPFN_CODE``, ``$MERIT_HOME``,
``$ANYBURL_HOME``); ``--check`` checks the setup, the pins and the graphs without a run.

Exit codes: 0 done; 2 wrong input (graph, system, hash, pins, adapter setup); 3 a graph ran out of CUDA memory, or an
adapter failed on a graph and went on with the next one (the graph's error record is in ``.errors/`` of the row folder);
76 stopped at ``--max-minutes`` (the part file is saved); 143 stopped by SIGTERM.
"""
import argparse
import json
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

from .manifests import REPO, BlindSetError, InputError, read_manifest, read_models

logger = logging.getLogger(__name__)
EXIT_INPUT, EXIT_OOM, EXIT_PARTIAL, EXIT_TERM = 2, 3, 76, 143
EXIT_FAILED = EXIT_OOM   # a graph failed in an adapter's external step
ADAPTERS = ('kgpfn', 'merit', 'anyburl')


class UsageError(InputError):
    """A command line option that is malformed or that does not fit the command."""


def parse_family(spec: str, models: Any) -> tuple[str, str, str]:
    """``(key, arm, template)`` of a ``[KEY=]ARM:TEMPLATE`` option; the key defaults to the arm's registered key.

    Raises:
        UsageError: For a malformed option or an arm without a registered key.
    """
    key, rest = spec.split('=', 1) if '=' in spec.split(':', 1)[0] else ('', spec)
    if ':' not in rest:
        raise UsageError(f'--family {spec!r}: expected [KEY=]ARM:TEMPLATE')
    arm, template = rest.split(':', 1)
    if not key:
        if arm not in models.families:
            raise UsageError(f'--family {spec!r}: arm {arm!r} has no registered key')
        key = models.families[arm].key
    return key, arm, template


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the commands."""
    parser = argparse.ArgumentParser(prog='python -m benchmarks.proofpfn', description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'check', 'run', 'cells'):
        sub = commands.add_parser(name)
        sub.add_argument('--set', required=True, help='a manifest name (dev121, t1, gate) or a manifest file')
        sub.add_argument('--view', default=None, help='a view of the manifest (dev121: main, contamination, apart, all)')
        sub.add_argument('--graphs', default=None, help='comma list of graph names or keys')
        sub.add_argument('--codes', default=None, help='comma list of audit group codes (E, L, B, S, O, R, C, T)')
        sub.add_argument('--release', default=None, help='the name of the blind test set that this run may open')
        sub.add_argument('--data-root', type=Path, default=REPO, help='the folder that the manifest paths are relative to')
        if name in ('check', 'run'):
            sub.add_argument('--family', action='append', default=[], help='[KEY=]ARM:TEMPLATE with {seed}; repeat for more families')
            sub.add_argument('--references', default='', help='comma list of trix, ultra3g, flockoff')
            sub.add_argument('--checkpoint-root', type=Path, default=REPO / 'checkpoints')
            sub.add_argument('--models', type=Path, default=None, help='the model pins (default: manifests/models.json)')
            sub.add_argument('--flock-passes-in-sequence', action='store_true',
                             help="run official Flock's passes of a query one after another, not as one microbatch")
        if name == 'run':
            sub.add_argument('--out', type=Path, required=True)
            sub.add_argument('--device', default='cuda')
            sub.add_argument('--shard', default='1/1', help='i/k: block i of k contiguous blocks of the query groups')
            sub.add_argument('--parts', action='store_true', help='keep a part file (and resume from it) also without shards')
            sub.add_argument('--save-minutes', type=float, default=10.0)
            sub.add_argument('--max-minutes', type=float, default=0.0, help='save and stop after this time (0: no limit)')
        if name == 'cells':
            sub.add_argument('--out', type=Path, required=True, help='the queries.csv file')
    gate = commands.add_parser('gate')
    gate.add_argument('--family', action='append', default=[], help='[KEY=]S:TEMPLATE and [KEY=]N:TEMPLATE, each with {seed}')
    gate.add_argument('--families', default=None, help='comma list of held-out families (default: rec1, attr2, list1, categorical2, plain)')
    gate.add_argument('--per-family', type=int, default=100, help='held-out worlds per family (at most 100)')
    gate.add_argument('--device', default='cuda')
    gate.add_argument('--models', type=Path, default=None, help='the model pins (default: manifests/models.json)')
    gate.add_argument('--out', type=Path, required=True, help='the result file (JSON)')
    fv1 = commands.add_parser('fv1-models')
    fv1.add_argument('--models', type=Path, default=None)
    fv1.add_argument('--out', type=Path, required=True)
    adapter = commands.add_parser('adapter', help='run an external baseline (README, section External baselines)')
    adapter.add_argument('name', choices=ADAPTERS)
    adapter.add_argument('--set', required=True, help='a manifest name (dev121, t1, gate) or a manifest file')
    adapter.add_argument('--view', default=None)
    adapter.add_argument('--graphs', default=None, help='comma list of graph names or keys')
    adapter.add_argument('--codes', default=None, help='comma list of audit group codes')
    adapter.add_argument('--release', default=None, help='the name of the blind test set that this run may open')
    adapter.add_argument('--data-root', type=Path, default=REPO, help='the folder that the manifest paths are relative to')
    adapter.add_argument('--out', type=Path, required=True, help='the output root of the row files')
    adapter.add_argument('--code', type=Path, default=None, help='the external code folder (default: $KGPFN_CODE, $MERIT_HOME, $ANYBURL_HOME)')
    adapter.add_argument('--work', type=Path, default=None, help='the work root (default: <out>/.work)')
    adapter.add_argument('--models', type=Path, default=None, help='the pins (default: manifests/models.json)')
    adapter.add_argument('--check', action='store_true', help='check the setup, the pins and the graphs; run nothing')
    adapter.add_argument('--python', default=None, help="kgpfn: the command that starts KGPFN's Python ($KGPFN_PYTHON); "
                                                        "merit: the Python of MERIT's environment ($MERIT_PYTHON, default: this one)")
    adapter.add_argument('--checkpoints', type=Path, default=None, help="kgpfn: the folder of KGPFN's two checkpoints ($KGPFN_CHECKPOINT_DIR)")
    adapter.add_argument('--gpu-gb', type=float, default=0.0, help='kgpfn: cap of the CUDA allocator in GB (0: none)')
    adapter.add_argument('--checkpoint-root', type=Path, default=REPO / 'checkpoints', help='merit: the root of merit/ckpts/merit/')
    adapter.add_argument('--device', default='cuda', help='merit: the scoring device')
    adapter.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)   # merit: this process runs in MERIT's environment
    adapter.add_argument('--java', default='java', help='anyburl: the Java 11 executable')
    adapter.add_argument('--no-scope', action='store_true', help='anyburl: run Java without the systemd-run memory scope')
    return parser


def _entries(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    manifest = read_manifest(args.set, release=args.release)
    graphs = [g for g in args.graphs.split(',') if g] if args.graphs else None
    codes = args.codes.split(',') if args.codes else None
    try:
        return manifest, manifest.select(view=args.view or (None if graphs else 'all'), graphs=graphs, codes=codes)
    except KeyError as error:   # a graph or view that the manifest does not hold
        raise UsageError(error.args[0]) from None


def _systems(args: argparse.Namespace) -> list[Any]:
    from .systems import family, reference
    models = read_models(args.models)
    systems: list[Any] = [family(*parse_family(spec, models), models) for spec in args.family]
    systems += [reference(key, models, args.checkpoint_root, one_microbatch=not args.flock_passes_in_sequence)
                for key in args.references.split(',') if key]
    if not systems:
        raise UsageError('give at least one --family or --references')
    return systems


def _check_graphs(manifest: Any, entries: Sequence[Any], root: Path, what: str, flock: Sequence[str] = ()) -> int:
    """Hash-check, load and count every graph, with official Flock's setting for the keys ``flock``; print one line per
    graph and return the exit code (``EXIT_INPUT`` when a graph fails)."""
    from .data import load_graph, prepare_queries
    from .systems import official_setting
    failed = []
    for entry in entries:
        try:
            graph = load_graph(manifest, entry, root)
            queries = prepare_queries(graph, manifest.sample_rule)
            setting = official_setting(entry, graph, flock) if flock else None
        except (InputError, BlindSetError) as error:
            failed.append(entry.key)
            print(f'{entry.key:45s} FAILED: {error}')
            continue
        walks = f" flock walks {setting['walks']} passes {setting['passes']}" if setting else ''
        print(f'{entry.key:45s} entities {graph.num_entities:7d} relations {graph.num_relations:5d} train {len(graph.train):8d} '
              f'test rows {queries.test_rows:6d} ranked {len(queries.facts):5d} query groups {len(queries.groups):6d}{walks} ok')
    if failed:
        print(f'check FAILED: {len(failed)} of {len(entries)} graphs ({", ".join(failed)})')
        return EXIT_INPUT
    print(f'check passed ({len(entries)} graphs, {what})')
    return 0


def _check(args: argparse.Namespace) -> int:
    from .systems import OfficialFlock
    manifest, entries = _entries(args)
    systems = _systems(args)
    for system in systems:
        system.check()
        print(f'{system.key}: {json.dumps(system.model_sha256())}')
    flock = [s.key for s in systems if isinstance(s, OfficialFlock)]
    return _check_graphs(manifest, entries, args.data_root, f'systems {",".join(s.key for s in systems)}', flock)


def _run(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import evaluate_graph, row_file_path, write_error
    manifest, entries = _entries(args)
    systems = _systems(args)
    shard = tuple(int(x) for x in args.shard.split('/') if x.isdigit())
    if len(shard) != 2 or args.shard.count('/') != 1:
        raise UsageError(f'--shard {args.shard!r}: expected i/k')
    statuses = {}
    for entry in entries:
        try:
            statuses[entry.key] = evaluate_graph(manifest, entry, systems, args.out, root=args.data_root, device=args.device,
                                                 shard=(shard[0], shard[1]), parts=args.parts, save_minutes=args.save_minutes,
                                                 max_minutes=args.max_minutes)
        except torch.cuda.OutOfMemoryError as error:
            keys = [s.key for s in systems]
            write_error(row_file_path(args.out, entry, keys), entry, systems=keys, stage='score', error=str(error).splitlines()[0][:400])
            torch.cuda.empty_cache()
            statuses[entry.key] = 'oom'
        print(entry.key, statuses[entry.key], flush=True)
    if 'oom' in statuses.values():
        return EXIT_OOM
    return EXIT_PARTIAL if 'stopped' in statuses.values() else 0


def _cells(args: argparse.Namespace) -> int:
    from .cells import write_queries
    manifest, entries = _entries(args)
    meta = write_queries(manifest, entries, args.out, args.data_root)
    print(f"{len(meta['graphs'])} graphs, {sum(meta['graphs'].values())} queries -> {args.out}")
    return 0


def _adapter(args: argparse.Namespace, argv: Sequence[str]) -> int:
    manifest, entries = _entries(args)
    if args.name == 'merit' and not args.worker:   # the scoring runs in a child process with MERIT's folder on its path
        from .adapters import merit
        return merit.run_in_child(argv, code=args.code, models=args.models, checkpoint_root=args.checkpoint_root, python=args.python)
    if args.name == 'anyburl':
        from .adapters import anyburl
        system: Any = anyburl.setup(args.code, args.models, java=args.java, scope=not args.no_scope)
    elif args.name == 'kgpfn':
        from .adapters import kgpfn
        system = kgpfn.setup(args.code, args.checkpoints, args.python, args.models, gpu_gb=args.gpu_gb)
    else:   # merit, in MERIT's environment (the child process of run_in_child)
        from .adapters import merit
        system = merit.setup(args.code, args.models, args.checkpoint_root)
    print(f'{args.name}: {json.dumps(system.record())}')
    if args.check:
        return _check_graphs(manifest, entries, args.data_root, f'adapter {args.name}')
    from .adapters import ExternalStepError
    from .evaluate import error_path, row_file_path, write_error
    keys = list(merit.KEYS) if args.name == 'merit' else [args.name]
    statuses = {}
    for entry in entries:
        final = row_file_path(args.out, entry, keys)
        try:
            if args.name == 'anyburl':
                statuses[entry.key] = anyburl.evaluate_graph(manifest, entry, args.out, system, root=args.data_root, work=args.work)
            elif args.name == 'kgpfn':
                statuses[entry.key] = kgpfn.evaluate_graph(manifest, entry, args.out, system, root=args.data_root, work=args.work)
            else:
                statuses[entry.key] = merit.evaluate_graph(manifest, entry, args.out, system, root=args.data_root, device=args.device)
        except (InputError, BlindSetError):
            raise
        except Exception as error:   # the graph gets an error record, and the run goes on with the next graph
            record = error.record if isinstance(error, ExternalStepError) else dict(
                stage='adapter', reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc().splitlines()[-30:])
            logger.error('%s: %s failed (%s) -> %s', entry.key, args.name, record['reason'], write_error(final, entry, adapter=args.name, **record))
            statuses[entry.key] = 'failed'
        if statuses[entry.key] == 'done':
            error_path(final).unlink(missing_ok=True)   # the record of an earlier failure
        print(entry.key, statuses[entry.key], flush=True)
    return EXIT_FAILED if 'failed' in statuses.values() else 0


def _gate(args: argparse.Namespace) -> int:
    from dicee.synthetic_kg.checkpoint import atomic_write_bytes

    from .gate import FAMILIES, PER_FAMILY, gate
    from .systems import family
    models = read_models(args.models)
    systems = {arm: family(key, arm, template, models) for key, arm, template in (parse_family(spec, models) for spec in args.family)}
    if len(args.family) != 2 or set(systems) != {'S', 'N'}:
        raise UsageError('the gate needs one --family of arm S and one of arm N')
    families = args.families.split(',') if args.families else FAMILIES
    if not set(families) <= set(FAMILIES) or not 1 <= args.per_family <= PER_FAMILY:
        raise UsageError(f'the gate has 1 to {PER_FAMILY} held-out worlds (--per-family) of each of the families {FAMILIES}')
    record = gate(systems['S'], systems['N'], families, args.per_family, args.device)
    atomic_write_bytes(args.out, (json.dumps(record, indent=1) + '\n').encode())
    print('| family | worlds | MRR S | MRR N | D = S - N | SE | reading |')
    for name, value in record['results'].items():
        print(f"| {name} | {value['worlds']} | {value['mrr_a']:.4f} | {value['mrr_b']:.4f} | {value['D']:+.4f} | {value['se']:.4f} | {value['reading']} |")
    if 'reading' in record:
        print(f"n_validity: {record['reading']['n_validity']}")
    print(f'-> {args.out}')
    return 0


def _fv1_models(args: argparse.Namespace) -> int:
    from dicee.synthetic_kg.checkpoint import atomic_write_bytes
    atomic_write_bytes(args.out, (json.dumps(read_models(args.models).fv1_models(), indent=1, sort_keys=True) + '\n').encode())
    print(f'-> {args.out}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a command; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_TERM))
    try:
        if args.command == 'list':
            for entry in _entries(args)[1]:
                print(entry.key)
            return 0
        if args.command in ('run', 'gate', 'adapter'):   # the commands that can start CUDA, which needs the setting first
            from dicee.synthetic_kg.determinism import cublas_workspace
            with cublas_workspace():
                return _adapter(args, argv) if args.command == 'adapter' else (_run if args.command == 'run' else _gate)(args)
        commands = dict(check=_check, cells=_cells)
        commands['fv1-models'] = _fv1_models
        return commands[args.command](args)
    except (InputError, BlindSetError) as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    finally:
        signal.signal(signal.SIGTERM, previous)
