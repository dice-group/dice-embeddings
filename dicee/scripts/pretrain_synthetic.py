"""ProofPFN pretraining on synthetic knowledge graphs: ``dicee_pretrain_synthetic <command>``.

Commands:
    train    pretrain one variant and seed into a run folder (``dicee.synthetic_kg.pretraining.pretrain``); a folder with
             a ``last.pt`` resumes its run
    sources  build the real source graphs of the real and the mixed prior from their split files (``priors.real.build_source``)

Examples::

    dicee_pretrain_synthetic train --variant PCUltra-S --seed 42 --out runs/PCUltra-S/42
    dicee_pretrain_synthetic sources --graphs KGs --out data/real-sources
    dicee_pretrain_synthetic train --variant PCUltra-R --seed 42 --out runs/PCUltra-R/42 --real_sources data/real-sources

The command sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (deterministic cuBLAS) unless it is set. SIGINT and SIGTERM stop a
run after the current update with ``last.pt``. Exit codes: 0 done, 1 error, 2 wrong input (one ``error:`` line: options,
a device that this machine lacks, a folder of another run or with changed sources, real source graphs that are missing
or hold other content than the pinned files, split files that miss or hold other triples than the pinned files),
3 stopped by a signal (rerun the same command to resume).
"""
import argparse
import dataclasses
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from dicee.synthetic_kg.determinism import cublas_workspace
from dicee.synthetic_kg.pretraining import EXPORTS, PretrainConfig, RunInputError, pretrain
from dicee.synthetic_kg.priors.real import NAMES, build_source
from dicee.synthetic_kg.recipe import TOTAL_UPDATES
from dicee.synthetic_kg.variants import VARIANTS

logger = logging.getLogger(__name__)
EXIT_INPUT, EXIT_STOPPED = 2, 3
DEFAULTS = {field.name: field.default for field in dataclasses.fields(PretrainConfig)}
"""The defaults of the run options (``PretrainConfig``)."""


def updates(text: str) -> tuple[int, ...]:
    """The update numbers of a comma list.

    Args:
        text: For example ``'1,2,4000'``.

    Returns:
        The numbers in list order.
    """
    return tuple(int(part) for part in text.split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the commands.

    Returns:
        The parser; the defaults of ``train`` are those of ``PretrainConfig``.
    """
    parser = argparse.ArgumentParser(prog='dicee_pretrain_synthetic', description='ProofPFN pretraining on synthetic knowledge graphs.',
                                     epilog='exit codes: 0 done, 1 error, 2 wrong input (one error line), 3 stopped by a signal (rerun to resume)')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='pretrain one variant and seed; a folder with a last.pt resumes its run')
    train.add_argument('--variant', required=True, choices=tuple(VARIANTS), help='the model variant')
    train.add_argument('--seed', type=int, required=True, help='the run seed (the thesis runs: 42, 43, 44)')
    train.add_argument('--out', required=True, help='the run folder')
    train.add_argument('--workers', type=int, default=DEFAULTS['workers'],
                       help='worker processes that prepare the worlds ahead; 0 prepares them in this process (default: %(default)s)')
    train.add_argument('--device', default=DEFAULTS['device'], help='the training device: cpu, cuda or cuda:<index> (default: %(default)s)')
    train.add_argument('--real_sources', default=None,
                       help='the folder of the real source graphs of the real and the mixed prior (<name>/data.pt; default: $PROOFPFN_REAL_SOURCES)')
    train.add_argument('--threads', type=int, default=DEFAULTS['threads'],
                       help='torch threads of the training process; CPU results depend on them (default: %(default)s)')
    train.add_argument('--save_every', type=int, default=DEFAULTS['save_every'], help='updates between two writes of last.pt (default: %(default)s)')
    train.add_argument('--allow_code_change', action='store_true',
                       help='resume although the sources that the run binds, the torch version or the GPU changed since it started')
    tests = train.add_argument_group('test options')
    tests.add_argument('--stop_after', type=int, default=None, help='stop after this update with last.pt; rerun to resume')
    tests.add_argument('--exports', type=updates, default=None,
                       help=f"comma list of export updates (default: {','.join(map(str, EXPORTS))}); update {TOTAL_UPDATES} is always exported")
    sources = commands.add_parser('sources', help='build the real source graphs of the real and the mixed prior from their split files')
    sources.add_argument('--graphs', default='KGs', help='the folder with train.txt, valid.txt and test.txt of each graph in <name>/ (default: %(default)s)')
    sources.add_argument('--out', required=True, help='the real-source folder; each graph goes to <name>/data.pt')
    sources.add_argument('--names', default=','.join(NAMES), help='comma list of the graphs to build (default: %(default)s)')
    return parser


def _train(args: argparse.Namespace) -> int:
    requested: list[int] = []

    def request_stop(number: int, _frame: Any) -> None:
        requested.append(number)
        logger.info('signal %d: stopping after the current update', number)

    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        config = PretrainConfig(args.variant, args.seed, stop_after=args.stop_after, exports=args.exports,
                                save_every=args.save_every, workers=args.workers, device=args.device, threads=args.threads,
                                real_sources=args.real_sources, allow_code_change=args.allow_code_change)
        result = pretrain(config, args.out, stop=lambda: bool(requested))
    except RunInputError as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    logger.info('%s after update %d: %s', result.status, result.done, result.folder)
    return EXIT_STOPPED if result.status == 'stopped' else 0


def _sources(args: argparse.Namespace) -> int:
    for name in (part for part in args.names.split(',') if part):
        try:
            path = build_source(name, Path(args.graphs) / name, args.out)
        except ValueError as error:
            print(f'error: {error}', file=sys.stderr)
            return EXIT_INPUT
        logger.info('%s -> %s', name, path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a command.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.

    Returns:
        The exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    with cublas_workspace():
        return dict(train=_train, sources=_sources)[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
