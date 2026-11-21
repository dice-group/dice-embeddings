"""Command line of the ProofPFN benchmark: ``python -m benchmarks.proofpfn <command>`` (the commands: ``COMMANDS``, ``--help``).

A family is ``--family [KEY=]ARM:TEMPLATE``: ARM is S, R, M, N, HT, CU or UL; TEMPLATE is the checkpoint or research
export path with ``{seed}`` (seeds 42, 43, 44); KEY defaults to the arm's registered key, whose pins every file must
match. References: ``--references trix,ultra3g,flockoff`` (checkpoints below ``--checkpoint-root``).

Exit codes: 0 done; 2 wrong input (graph selection, device, system, hash, pins); 3 a graph ran out of CUDA memory (the
graph's error record is in ``.errors/`` of the row folder); 76 stopped at ``--max-minutes`` (the part file is saved);
143 stopped by SIGTERM (with ``--parts`` or a shard, the part file is saved). An internal error ends with a traceback and
exit code 1.
"""
import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .manifests import REPO, BlindSetError, InputError, read_manifest, read_models

logger = logging.getLogger(__name__)
EXIT_INPUT, EXIT_OOM, EXIT_PARTIAL, EXIT_TERM = 2, 3, 76, 143
COMMANDS = {
    'list': 'the graphs of a manifest view',
    'check': "CPU only: the graphs' hash checks, counts and official Flock's setting, and the systems' files and pins; no score",
    'run': 'rank the graphs with the systems and write one row file per graph (evaluate.evaluate_graph)',
    'gate': 'the N validity gate of amendment A1: S and N on held-out worlds of the prior (gate.gate)',
}


class UsageError(InputError):
    """A command line option that is malformed or that does not fit the command."""


def parse_family(spec: str, models: Any) -> tuple[str, str, str]:
    """``(key, arm, template)`` of a ``[KEY=]ARM:TEMPLATE`` option; the key defaults to the arm's registered key.

    The key is the text before the first ``=`` when that ``=`` comes before the first ``:``. The arm ends at the first
    ``:``, and the template is the rest.

    Args:
        spec: The value of a ``--family`` option.
        models: The model pins (``manifests.ModelPins``), which hold the registered keys.

    Returns:
        The key, the arm and the template.

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


def _graph_options(sub: argparse.ArgumentParser) -> None:
    """The options that select the graphs of a manifest."""
    sub.add_argument('--set', required=True, help='a manifest name (t1, gate) or a manifest file')
    sub.add_argument('--view', default=None, help='a view of the manifest (t1: trans, ind_e, ind_er; default: all)')
    sub.add_argument('--graphs', default=None, help='comma list of graph names or keys')
    sub.add_argument('--release', default=None, help='the name of the blind test set that this run opens')
    sub.add_argument('--data-root', type=Path, default=REPO, help='the folder that the manifest paths are relative to (default: the repository)')


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the commands.

    Returns:
        The parser, with one subcommand per entry of ``COMMANDS``.
    """
    parser = argparse.ArgumentParser(prog='python -m benchmarks.proofpfn', description='Zero-shot link prediction on frozen graph sets (README.md).',
                                     epilog='exit codes: 0 done, 1 internal error, 2 wrong input (one error line), 3 a graph failed (its record is '
                                            'in .errors/), 76 stopped at --max-minutes, 143 stopped by SIGTERM (README.md, Quick start)')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = {name: commands.add_parser(name, help=text, description=text) for name, text in COMMANDS.items()}
    for name in ('list', 'check', 'run'):
        _graph_options(sub[name])
    models = 'the model pins (default: manifests/models.json)'
    for name in ('check', 'run'):
        sub[name].add_argument('--family', action='append', default=[], help='[KEY=]ARM:TEMPLATE with {seed}; repeat for more families')
        sub[name].add_argument('--references', default='', help='comma list of trix, ultra3g, flockoff')
        sub[name].add_argument('--checkpoint-root', type=Path, default=REPO / 'checkpoints', help='the folder of the reference checkpoints (default: checkpoints/)')
        sub[name].add_argument('--models', type=Path, default=None, help=models)
        sub[name].add_argument('--flock-passes-in-sequence', action='store_true',
                               help="run official Flock's passes of a query one after another, not as one microbatch")
    run = sub['run']
    run.add_argument('--out', type=Path, required=True, help='the output root of the row files')
    run.add_argument('--device', default='cuda', help='the scoring device (default: cuda)')
    run.add_argument('--shard', default='1/1', help='i/k: rank block i of k contiguous blocks of the query groups (default: 1/1)')
    run.add_argument('--parts', action='store_true', help='keep a part file (and resume from it) also without shards')
    run.add_argument('--save-minutes', type=float, default=10.0, help='minutes between two saves of a part file (default: 10)')
    run.add_argument('--max-minutes', type=float, default=0.0, help='save and stop after this time (default: 0, no limit)')
    gate = sub['gate']
    gate.add_argument('--family', action='append', default=[], help='[KEY=]S:TEMPLATE and [KEY=]N:TEMPLATE, each with {seed}')
    gate.add_argument('--families', default=None, help='comma list of held-out families (default: rec1, attr2, list1, categorical2, plain)')
    gate.add_argument('--per-family', type=int, default=100, help='held-out worlds per family (default and maximum: 100)')
    gate.add_argument('--device', default='cuda', help='the scoring device (default: cuda)')
    gate.add_argument('--models', type=Path, default=None, help=models)
    gate.add_argument('--out', type=Path, required=True, help='the result file (JSON)')
    return parser


def _entries(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    manifest = read_manifest(args.set, release=args.release)
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

    from .evaluate import error_path, evaluate_graph, row_file_path, write_error
    manifest, entries = _entries(args)
    device = _device(args.device)
    systems = _systems(args)
    shard = tuple(int(x) for x in args.shard.split('/') if x.isdigit())
    if len(shard) != 2 or args.shard.count('/') != 1:
        raise UsageError(f'--shard {args.shard!r}: expected i/k')
    keys = [s.key for s in systems]
    statuses = {}
    for entry in entries:
        final = row_file_path(args.out, entry, keys)
        try:
            statuses[entry.key] = evaluate_graph(manifest, entry, systems, args.out, root=args.data_root, device=device,
                                                 shard=(shard[0], shard[1]), parts=args.parts, save_minutes=args.save_minutes,
                                                 max_minutes=args.max_minutes)
        except torch.cuda.OutOfMemoryError as error:
            write_error(final, entry, systems=keys, stage='score', error=str(error).splitlines()[0][:400])
            torch.cuda.empty_cache()
            statuses[entry.key] = 'oom'
        if statuses[entry.key] == 'done':
            error_path(final).unlink(missing_ok=True)   # the record of an earlier failure
        print(entry.key, statuses[entry.key], flush=True)
    if 'oom' in statuses.values():
        return EXIT_OOM
    return EXIT_PARTIAL if 'stopped' in statuses.values() else 0


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
    record = gate(systems['S'], systems['N'], families, args.per_family, _device(args.device))
    atomic_write_bytes(args.out, (json.dumps(record, indent=1) + '\n').encode())
    print('| family | worlds | MRR S | MRR N | D = S - N | SE | reading |')
    for name, value in record['results'].items():
        print(f"| {name} | {value['worlds']} | {value['mrr_a']:.4f} | {value['mrr_b']:.4f} | {value['D']:+.4f} | {value['se']:.4f} | {value['reading']} |")
    if 'reading' in record:
        print(f"n_validity: {record['reading']['n_validity']}")
    print(f'-> {args.out}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a command.

    While the command runs, SIGTERM exits with ``EXIT_TERM``. The commands ``run`` and ``gate`` run inside
    ``determinism.cublas_workspace``.

    Args:
        argv: The arguments after the program name, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit code (see the module docstring). An ``InputError`` or a ``BlindSetError`` prints one ``error:`` line
        on stderr and gives ``EXIT_INPUT``.

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
                print(entry.key)
            return 0
        commands: dict[str, Callable[[argparse.Namespace], int]] = {'check': _check, 'run': _run, 'gate': _gate}
        if args.command in ('run', 'gate'):
            from dicee.synthetic_kg.determinism import cublas_workspace
            with cublas_workspace():   # dicee's models score on CUDA, which needs the setting first
                return commands[args.command](args)
        return commands[args.command](args)
    except (InputError, BlindSetError) as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    finally:
        signal.signal(signal.SIGTERM, previous)
