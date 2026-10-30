"""ProofPFN pretraining on synthetic knowledge graphs: ``dicee_pretrain_synthetic <command>``.

Commands:
    train    pretrain one arm and seed into a run folder (``dicee.synthetic_kg.pretraining.pretrain``); a folder with a
             ``last.pt`` resumes its run
    convert  convert a research ``export-<N>.pt`` into a ``proofpfn-checkpoint-v1`` file after its inertness checks
             (``dicee.synthetic_kg.checkpoint.from_research_export``)

Examples::

    dicee_pretrain_synthetic train --arm S --seed 42 --out runs/S/42
    dicee_pretrain_synthetic train --arm R --seed 42 --out runs/R/42 --real-sources data/real-sources
    dicee_pretrain_synthetic convert --export export-24000.pt --arm S --out S-42.pt

The command sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (deterministic cuBLAS) unless it is set. SIGINT and SIGTERM stop a
run after the current update with ``last.pt``. Exit codes: 0 done, 1 error, 2 wrong input (options, or a research export
that the converter refuses), 3 stopped by a signal (rerun the same command to resume).
"""
import argparse
import logging
import os
import signal
import sys
from typing import Any, Optional, Sequence

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.checkpoint import CheckpointError, from_research_export, save_checkpoint
from dicee.synthetic_kg.pretraining import PretrainConfig, pretrain

logger = logging.getLogger(__name__)
EXIT_INPUT, EXIT_STOPPED = 2, 3


def updates(text: str) -> tuple[int, ...]:
    """A comma list of update numbers."""
    return tuple(int(part) for part in text.split(',') if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the commands."""
    parser = argparse.ArgumentParser(prog='dicee_pretrain_synthetic', description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='pretrain one arm and seed')
    train.add_argument('--arm', required=True, choices=tuple(ARMS))
    train.add_argument('--seed', type=int, required=True)
    train.add_argument('--out', required=True, help='the run folder')
    train.add_argument('--workers', type=int, default=4, help='worker processes that prepare the worlds ahead (0: in this process)')
    train.add_argument('--device', default='cuda')
    train.add_argument('--real-sources', default=None, help='the folder of the real source graphs (<name>/data.pt; arms R and M)')
    train.add_argument('--threads', type=int, default=2, help='torch threads of the training process (CPU results depend on it)')
    train.add_argument('--save-every', type=int, default=250, help='last.pt every this many updates')
    train.add_argument('--allow-code-change', action='store_true', help='resume although the bound sources changed')
    tests = train.add_argument_group('test options')
    tests.add_argument('--stop-after', type=int, default=None, help='stop after this update with last.pt (rerun to resume)')
    tests.add_argument('--exports', type=updates, default=None, help='comma list of export updates (default: 4000,8000,16000,20000,24000)')
    convert = commands.add_parser('convert', help='convert a research export into a checkpoint')
    convert.add_argument('--export', required=True, help='the research export-<N>.pt')
    convert.add_argument('--arm', required=True, choices=tuple(ARMS))
    convert.add_argument('--out', required=True, help='the checkpoint file')
    convert.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    return parser


def _train(args: argparse.Namespace) -> int:
    try:
        config = PretrainConfig(args.arm, args.seed, stop_after=args.stop_after, exports=args.exports,
                                save_every=args.save_every, workers=args.workers, device=args.device, threads=args.threads,
                                real_sources=args.real_sources, allow_code_change=args.allow_code_change)
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    requested: list[int] = []

    def request_stop(number: int, _frame: Any) -> None:
        requested.append(number)
        logger.info('signal %d: stopping after the current update', number)

    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = pretrain(config, args.out, stop=lambda: bool(requested))
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    logger.info('%s after update %d: %s', result.status, result.done, result.folder)
    return EXIT_STOPPED if result.status == 'stopped' else 0


def _convert(args: argparse.Namespace) -> int:
    try:
        checkpoint = from_research_export(args.export, args.arm, args.weights)
    except CheckpointError as error:
        print(f'error: {error}', file=sys.stderr)
        return EXIT_INPUT
    path = save_checkpoint(checkpoint, args.out)
    logger.info('arm %s, seed %d, update %d, %s weights -> %s', checkpoint['arm'], checkpoint['seed'], checkpoint['updates'],
                checkpoint['weights'], path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a command; returns the exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace is None:   # deterministic cuBLAS needs it before CUDA starts
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        return _train(args) if args.command == 'train' else _convert(args)
    finally:
        if workspace is None:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']


if __name__ == '__main__':
    sys.exit(main())
