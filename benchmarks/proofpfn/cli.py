"""Command line of the ProofPFN benchmark: ``python -m benchmarks.proofpfn <command>`` (the commands: ``COMMANDS``, ``--help``).

A model is ``--model NAME=TEMPLATE``: TEMPLATE is the path of a checkpoint (``proofpfn-checkpoint-v1``). With the field
``{seed}`` it names one checkpoint per seed of ``--seeds`` (default 42, 43, 44), labelled ``NAME.seed<S>``. References:
``--references trix,ultra3g,flock`` (the released checkpoints below ``--checkpoint-root``).

Exit codes: 0 done; 2 wrong input (graph selection, device, system, hash); 3 a system ran out of CUDA memory on a graph
(the other graphs and systems are done); 143 stopped by SIGTERM. An internal error ends with a traceback and exit code 1.
"""
import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .manifests import REPO, InputError, read_manifest

logger = logging.getLogger(__name__)
EXIT_INPUT, EXIT_OOM, EXIT_TERM = 2, 3, 143
COMMANDS = {
    'list': 'the graphs of a manifest view',
    'check': "CPU only: the graphs' hash checks and counts, and the systems' files; no score",
    'run': 'rank the test facts of the graphs with the systems: one result file per graph and system (evaluate.evaluate_graph)',
    'table': 'the mean MRR and Hits@10 of the result files of a graph set, per label and for the seeds of a name (evaluate.summarize)',
    'prior': 'score checkpoints on held-out worlds of the synthetic prior, per world family (prior.evaluate)',
}


class UsageError(InputError):
    """A command line option that is malformed or that does not fit the command."""


def _graph_options(sub: argparse.ArgumentParser) -> None:
    """The options that select the graphs of a manifest."""
    sub.add_argument('--set', required=True, help='a manifest name (ultra54, heldout44) or a manifest file')
    sub.add_argument('--view', default=None, help='a view of the manifest (ultra54: trans, ind_e, ind_er; heldout44: published, converted; default: all)')
    sub.add_argument('--graphs', default=None, help='comma list of graph names')
    sub.add_argument('--data-root', type=Path, default=REPO, help='the folder that the manifest paths are relative to (default: the repository)')


def _system_options(sub: argparse.ArgumentParser) -> None:
    """The options that name the systems."""
    sub.add_argument('--model', action='append', default=[], help='NAME=TEMPLATE, a checkpoint path, with {seed} for one per seed; repeat for more')
    sub.add_argument('--seeds', default='42,43,44', help='comma list of the seeds of a template with {seed} (default: 42,43,44)')


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the commands.

    Returns:
        The parser, with one subcommand per entry of ``COMMANDS``.
    """
    parser = argparse.ArgumentParser(prog='python -m benchmarks.proofpfn', description='Zero-shot link prediction on frozen graph sets (README.md).',
                                     epilog='exit codes: 0 done, 1 internal error, 2 wrong input (one error line), 3 a system ran out of CUDA '
                                            'memory on a graph, 143 stopped by SIGTERM')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = {name: commands.add_parser(name, help=text, description=text) for name, text in COMMANDS.items()}
    for name in ('list', 'check', 'run', 'table'):
        _graph_options(sub[name])
    for name in ('check', 'run'):
        _system_options(sub[name])
        sub[name].add_argument('--references', default='', help='comma list of trix, ultra3g, flock')
        sub[name].add_argument('--checkpoint-root', type=Path, default=REPO / 'checkpoints', help='the folder of the reference checkpoints (default: checkpoints/)')
        sub[name].add_argument('--flock-passes-in-sequence', action='store_true',
                               help="run official Flock's passes of a query one after another, not as one microbatch")
    sub['run'].add_argument('--out', type=Path, required=True, help='the output root of the result files')
    sub['run'].add_argument('--device', default='cuda', help='the scoring device (default: cuda)')
    sub['table'].add_argument('--rows', type=Path, required=True, help='the output root of a run (its --out)')
    sub['table'].add_argument('--direction', default='both', choices=('both', 'tail', 'head'), help='the queries of the metrics (default: both)')
    sub['table'].add_argument('--json', type=Path, default=None, help='also write the summary to this file')
    prior = sub['prior']
    _system_options(prior)
    prior.add_argument('--families', default=None, help='comma list of world families (default: rec1, attr2, list1, categorical2, plain)')
    prior.add_argument('--per-family', type=int, default=100, help='held-out worlds per family (default and maximum: 100)')
    prior.add_argument('--device', default='cuda', help='the scoring device (default: cuda)')
    prior.add_argument('--out', type=Path, required=True, help='the result file (JSON)')
    return parser


def _entries(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    manifest = read_manifest(args.set)
    graphs = [g for g in args.graphs.split(',') if g] if args.graphs else None
    try:
        entries = manifest.select(view=args.view or (None if graphs else 'all'), graphs=graphs)
    except KeyError as error:   # a graph or view that the manifest does not hold
        raise UsageError(error.args[0]) from None
    if not entries:
        raise UsageError(f'no graph of {manifest.name} matches the selection')
    return manifest, entries


def _device(name: str) -> str:
    """The device option, refused when this machine lacks it or when CUDA runs without deterministic cuBLAS."""
    from dicee.synthetic_kg.determinism import device_problem
    problem = device_problem(name)
    if problem:
        raise UsageError(problem)
    return name


def _seeds(args: argparse.Namespace) -> list[int]:
    try:
        seeds = [int(s) for s in args.seeds.split(',') if s]
    except ValueError:
        raise UsageError(f'--seeds {args.seeds!r}: expected a comma list of integers') from None
    if not seeds or len(set(seeds)) != len(seeds):
        raise UsageError(f'--seeds {args.seeds!r}: expected distinct integers')
    return seeds


def _checkpoint_groups(args: argparse.Namespace) -> dict[str, list[Any]]:
    """Name to the checkpoint systems of each ``--model`` option, in option order."""
    from .systems import checkpoints
    groups: dict[str, list[Any]] = {}
    for spec in args.model:
        members = checkpoints(spec, _seeds(args))
        name = spec.partition('=')[0]
        if name in groups:
            raise UsageError(f'model name {name!r} is given twice')
        groups[name] = members
    return groups


def _systems(args: argparse.Namespace) -> list[Any]:
    from .systems import reference
    systems: list[Any] = [system for members in _checkpoint_groups(args).values() for system in members]
    systems += [reference(label, args.checkpoint_root, one_microbatch=not args.flock_passes_in_sequence) for label in args.references.split(',') if label]
    if not systems:
        raise UsageError('give at least one --model or --references')
    labels = [s.label for s in systems]
    if len(set(labels)) != len(labels):
        raise UsageError(f'system labels must be unique: {labels}')
    return systems


def _check(args: argparse.Namespace) -> int:
    from .data import load_graph, prepare_queries
    from .systems import OfficialFlock, official_setting
    manifest, entries = _entries(args)
    systems = _systems(args)
    for system in systems:
        system.check()
        print(f'{system.label}: {json.dumps(system.spec())} sha256 {system.sha256()}')
    flock = any(isinstance(s, OfficialFlock) for s in systems)
    failed = []
    for entry in entries:
        try:
            graph = load_graph(entry, args.data_root)
            queries = prepare_queries(graph, manifest.sample_rule)
            setting = official_setting(entry, graph) if flock else None
        except InputError as error:
            failed.append(entry.name)
            print(f'{entry.name:32s} FAILED: {error}')
            continue
        walks = f" flock walks {setting['walks']} passes {setting['passes']}" if setting else ''
        print(f'{entry.name:32s} entities {graph.num_entities:7d} relations {graph.num_relations:5d} train {len(graph.train):8d} '
              f'test rows {queries.test_rows:6d} ranked {len(queries.facts):5d} query groups {len(queries.groups):6d}{walks} ok')
    if failed:
        print(f'check FAILED: {len(failed)} of {len(entries)} graphs ({", ".join(failed)})')
        return EXIT_INPUT
    print(f'check passed ({len(entries)} graphs, systems {",".join(s.label for s in systems)})')
    return 0


def _run(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_graph
    manifest, entries = _entries(args)
    device = _device(args.device)
    systems = _systems(args)
    oom = False
    for entry in entries:
        statuses = evaluate_graph(manifest, entry, systems, args.out, root=args.data_root, device=device)
        oom = oom or 'oom' in statuses.values()
        print(entry.name, ' '.join(f'{label}:{status}' for label, status in statuses.items()), flush=True)
    return EXIT_OOM if oom else 0


def _table(args: argparse.Namespace) -> int:
    from dicee.synthetic_kg.checkpoint import atomic_write_bytes

    from .evaluate import summarize
    manifest, entries = _entries(args)
    summary = summarize(args.rows, manifest, entries, args.direction)
    if not summary['labels']:
        raise UsageError(f'no result file of {manifest.name} below {args.rows}')
    views = [*summary['views'], 'all']

    def cell(value: Optional[dict[str, Any]], key: str) -> str:
        if value is None:
            return '-'
        v = value[key]
        return f"{100 * v['mean']:.1f} ± {100 * v['sd']:.1f}" if isinstance(v, dict) else f'{100 * v:.1f}'

    print(f"{manifest.name}, {len(entries)} graphs, {args.direction} directions: MRR / Hits@10 in points")
    print('| system | graphs | ' + ' | '.join(views) + ' |')
    print('|---|---:|' + '---:|' * len(views))
    for name, group in summary['groups'].items():
        print(f"| {name} (seeds {', '.join(map(str, group['seeds']))}) | {len(entries)} | "
              + ' | '.join(f"{cell(group.get(v), 'MRR')} / {cell(group.get(v), 'H@10')}" for v in views) + ' |')
    for label, row in summary['labels'].items():
        print(f"| {label} | {row['found']} | " + ' | '.join(f"{cell(row.get(v), 'MRR')} / {cell(row.get(v), 'H@10')}" for v in views) + ' |')
    if args.json is not None:
        atomic_write_bytes(args.json, (json.dumps(summary, indent=1) + '\n').encode())
        print(f'-> {args.json}')
    return 0


def _prior(args: argparse.Namespace) -> int:
    from dicee.synthetic_kg.checkpoint import atomic_write_bytes

    from .prior import FAMILIES, PER_FAMILY, evaluate
    groups = _checkpoint_groups(args)
    if not groups:
        raise UsageError('give at least one --model')
    families = args.families.split(',') if args.families else list(FAMILIES)
    if not set(families) <= set(FAMILIES) or not 1 <= args.per_family <= PER_FAMILY:
        raise UsageError(f'there are 1 to {PER_FAMILY} held-out worlds (--per-family) of each of the families {FAMILIES}')
    device = _device(args.device)
    for members in groups.values():
        for system in members:
            system.check()
    record = evaluate(groups, families, args.per_family, device)
    atomic_write_bytes(args.out, (json.dumps(record, indent=1) + '\n').encode())
    print('| family | worlds | ' + ' | '.join(f'MRR {name}' for name in groups) + ' |')
    for family in families:
        print(f"| {family} | {args.per_family} | " + ' | '.join(f"{record['results'][name][family]['MRR']:.4f}" for name in groups) + ' |')
    if 'difference' in record:
        d = record['difference']
        print(f"| family | D = {d['a']} - {d['b']} | SE | reading |")
        for family, value in d['families'].items():
            print(f"| {family} | {value['D']:+.4f} | {value['se']:.4f} | {value['reading']} |")
    print(f'-> {args.out}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a command.

    While the command runs, SIGTERM exits with ``EXIT_TERM``. The commands ``run`` and ``prior`` run inside
    ``determinism.cublas_workspace``.

    Args:
        argv: The arguments after the program name, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit code (see the module docstring). An ``InputError`` prints one ``error:`` line on stderr and gives
        ``EXIT_INPUT``.

    Raises:
        SystemExit: From ``argparse`` for ``--help`` (code 0) or a malformed command line (code 2), and at SIGTERM
            (code ``EXIT_TERM``).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_TERM))
    try:
        if args.command == 'list':
            for entry in _entries(args)[1]:
                print(entry.name)
            return 0
        commands: dict[str, Callable[[argparse.Namespace], int]] = {'check': _check, 'run': _run, 'table': _table, 'prior': _prior}
        if args.command in ('run', 'prior'):
            from dicee.synthetic_kg.determinism import cublas_workspace
            with cublas_workspace():   # dicee's models score on CUDA, which needs the setting first
                return commands[args.command](args)
        return commands[args.command](args)
    except InputError as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    finally:
        signal.signal(signal.SIGTERM, previous)
