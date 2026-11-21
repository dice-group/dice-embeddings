"""Set up and evaluate the complex-query benchmarks.

    python -m benchmarks.cqa {plus_h,ultraquery} COMMAND [options]

Commands run in this process, which needs dicee and PyTorch, or with ``--image``
inside the pinned Docker runtime; the host then needs only Python 3 and Docker.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import docker, inputs
from .manifests import REPO, SUITES, default_recipes, prepare_manifest

# Longest expected runtimes start first, so short entries fill the remaining slots.
RUNTIME_ORDER = ('trix-adapter', 'ultra-adapter', 'gnnqe', 'cqd-hybrid', 'cqd', 'qto')
# Container GPU access per command; other commands always run on the host.
CONTAINER_GPU = {'evaluate': 'runtime', 'train': 'none'}
HOST_OPTIONS = {'image', 'gpu', 'dry_run', 'models_archive', 'input_root', 'output', 'no_setup'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m benchmarks.cqa', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('suite', choices=SUITES)
    commands = parser.add_subparsers(dest='command', required=True)
    add = {}

    def command(name, help):
        add[name] = commands.add_parser(name, help=help, description=help)
        add[name].add_argument('--input-root', type=Path, default=REPO, help='Root of datasets and checkpoints (default: repository)')
        return add[name]

    def recipes(p):
        p.add_argument('--manifests', type=Path, nargs='+', action='extend', help='Recipe manifests (default: public recipes)')
        p.add_argument('--profile', choices=('bounded', 'reference'), default='bounded',
                       help='CQD batching: bounded (8 GB GPUs) or dense reference batches')
        p.add_argument('--hardware-profile', choices=('default', 'h100'), default='default',
                       help='KGFM batch and cache settings: recipe defaults, or those measured on H100 GPUs')
        p.add_argument('--kgfm-batch-size', type=int, help='Atomic row and backbone batch of KGFM adapters')
        p.add_argument('--answer-filter', choices=('released', 'corrected'),
                       help='+H default: corrected with released controls; UltraQuery: released only')
        p.add_argument('--observed-facts', nargs='+', action='extend', choices=('none', 'atomic', 'both', 'all'),
                       help='KGFM observed-fact ablations with unchanged adapter weights')
        p.add_argument('--inference-graphs', nargs='+', action='extend', choices=('train', 'train+valid'),
                       help='+H only: test inference graphs (default: train+valid)')
        p.add_argument('--entries', nargs='+', action='extend', help='Entry IDs')
        p.add_argument('--methods', nargs='+', action='extend', help='Methods; omit or use all for every recipe')
        p.add_argument('--datasets', nargs='+', action='extend', help='Datasets; omit or use all for every recipe')
        p.add_argument('--query-types', nargs='+', action='extend', help='Query types or groups: all, epfo, negation')
        p.add_argument('--atomic-negation', action='store_true', help='CQD/CQD-Hybrid signed-atom negation for custom recipes')
        p.add_argument('--models-archive', type=Path, help="Official +H weights ZIP if the authors' host is unavailable")

    def container(p):
        p.add_argument('--image', help='Run inside this pinned Docker image (tag or image ID)')
        p.add_argument('--gpu', choices=('runtime', 'cdi', 'none'), help='Docker GPU access')
        p.add_argument('--dry-run', action='store_true', help='Print the command without running it')

    p = command('setup', 'Download missing datasets and checkpoints and install shipped adapters; no inference')
    recipes(p)
    p.add_argument('--check', action='store_true', help='Only report missing inputs and checksum mismatches')
    p.add_argument('--dry-run', action='store_true', help='Print sources without downloading')

    p = command('evaluate', 'Evaluate recipes, or any subset of them')
    recipes(p)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', choices=('valid', 'test'), required=True,
                   help='Query split; required, so the test split is only ever evaluated on request')
    p.add_argument('--max-queries-per-shape', type=int, help='Uniform sample of each query type')
    p.add_argument('--seed', type=int)
    p.add_argument('--threads', type=int)
    p.add_argument('--no-setup', action='store_true', help='Require existing inputs instead of downloading them')
    p.add_argument('--gpus', nargs='+', help='Run entries concurrently, one listed GPU (index or UUID) per worker')
    p.add_argument('--workers-per-gpu', type=int, default=1, help='Concurrent entries per GPU (default: 1)')
    container(p)

    p = command('train', 'Train UltraQuery baselines in the author runtime (UltraQuery suite)')
    from .train import add_arguments
    add_arguments(p)
    container(p)

    p = command('build', 'Build the pinned Docker runtime, or the CPU author-training runtime with --training')
    p.add_argument('--image', required=True, help='Tag for the new image')
    p.add_argument('--training', metavar='BASE_IMAGE', help='Build the training runtime on top of this benchmark image')
    p.add_argument('--cache', type=Path, help='Wheel cache (default: ~/.cache/dicee/cqa)')
    p.add_argument('--network', default='default', help='Use host if Docker bridge DNS is unavailable')
    return parser, add


def arguments(parser, args, *, translate=None, skip=()):
    """Command-line options that reproduce the non-default values of ``args``."""
    def text(value):
        return translate(value) if translate is not None and isinstance(value, Path) else str(value)

    argv = []
    for action in parser._actions:
        if not action.option_strings or action.dest in skip:
            continue
        value = getattr(args, action.dest, action.default)
        if value is None or value == action.default:
            continue
        flag = action.option_strings[-1]
        if action.nargs == 0:
            argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, *map(text, value)]
        else:
            argv += [flag, text(value)]
    return argv


def resolve(args):
    """Recipes selected by ``args``, with the suite's default manifests and answer filter."""
    return prepare_manifest(args.manifests or default_recipes(args.suite), profile=args.profile,
                            entries=args.entries, methods=args.methods, datasets=args.datasets,
                            query_types=args.query_types, atomic_negation=args.atomic_negation,
                            observed_facts=args.observed_facts, inference_graphs=args.inference_graphs,
                            answer_filter=args.answer_filter or SUITES[args.suite]['answer_filter'], suite=args.suite,
                            hardware_profile=args.hardware_profile, kgfm_batch_size=args.kgfm_batch_size)


def in_container(parser, args):
    """Run the same command in Docker: inputs read-only at /inputs, OUTPUT writable at /results."""
    root, output = args.input_root.resolve(), args.output.resolve()
    if not root.is_dir() or output == root or not output.is_relative_to(root):
        raise ValueError('--output must be a directory strictly below an existing --input-root')

    def mapped(value):
        path = Path(value).resolve()
        if path.is_relative_to(output):
            return Path('/results', path.relative_to(output)).as_posix()
        if path.is_relative_to(root):
            return Path('/inputs', path.relative_to(root)).as_posix()
        raise ValueError(f'{value} must be below --input-root')

    argv = [args.suite, args.command, '--input-root', '/inputs', '--output', '/results',
            *arguments(parser, args, translate=mapped, skip=HOST_OPTIONS | {'manifest_output_root'})]
    if args.command == 'train':
        # Generated manifests refer to checkpoints by their path below the input root.
        argv += ['--manifest-output-root', Path('/inputs', output.relative_to(root)).as_posix()]
    if args.command == 'evaluate':
        # Containers have no network: the host fetches missing inputs first.
        argv.append('--no-setup')
        if not args.dry_run and not args.no_setup:
            inputs.ensure_inputs(resolve(args), root, suite=args.suite, models_archive=args.models_archive)
    gpu = args.gpu or CONTAINER_GPU[args.command]
    if gpu == 'none' and getattr(args, 'device', 'cpu') != 'cpu':
        raise ValueError('Without container GPU access (--gpu none), use --device cpu')
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    docker.run(args.image, argv, inputs=root, results=output, gpu=gpu, dry_run=args.dry_run)


def run_workers(jobs, *, status, gpus=None, workers_per_gpu=1):
    """Run (entry, method, command, log) jobs in fresh processes, optionally one GPU per worker.

    Results do not depend on scheduling: each worker has its own checkpoint and
    output. After a failure no new entry starts; running workers finish.
    Returns the failed (entry, exit code) pairs.
    """
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)] if gpus else [None]
    pending = sorted(jobs, key=lambda job: RUNTIME_ORDER.index(job[1]) if job[1] in RUNTIME_ORDER else len(RUNTIME_ORDER))
    running, failed = {}, []

    def report():
        write_json(status, dict(state='running', running={entry: dict(pid=process.pid, gpu=gpu)
                                                          for process, (entry, gpu, _) in running.items()},
                                pending=[job[0] for job in pending], failed=[entry for entry, _ in failed]))

    while running or (pending and not failed):
        while pending and slots and not failed:
            entry, _, command, log = pending.pop(0)
            gpu = slots.pop(0)
            print(json.dumps(dict(entry=entry, state='starting', gpu=gpu)), flush=True)
            stream = log.open('ab', buffering=0)
            environment = os.environ if gpu is None else dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            running[process] = entry, gpu, stream
            report()
        finished = [process for process in running if process.poll() is not None]
        if not finished:
            time.sleep(0.5)
            continue
        for process in finished:
            entry, gpu, stream = running.pop(process)
            stream.close()
            slots.append(gpu)
            if process.returncode:
                failed.append((entry, process.returncode))
            print(json.dumps(dict(entry=entry, state='failed' if process.returncode else 'complete',
                                  exit_code=process.returncode)), flush=True)
        report()
    return failed


def entry_jobs(parser, args, entries, output):
    """One fresh worker process per entry, repeating this command for that entry alone."""
    common = arguments(parser, args, skip={'entries', 'methods', 'datasets', 'gpus', 'workers_per_gpu', 'no_setup', 'dry_run'})
    jobs = []
    for entry in entries:
        (output / entry['id']).mkdir(parents=True, exist_ok=True)
        jobs.append((entry['id'], entry['method'],
                     [sys.executable, '-u', '-m', 'benchmarks.cqa', args.suite, args.command, *common,
                      '--entries', entry['id'], '--no-setup'],
                     output / entry['id'] / 'worker.log'))
    return jobs


def setup(parser, args):
    manifest = resolve(args)
    if args.dry_run:
        inputs.describe(manifest, args.input_root.resolve(), suite=args.suite)
    elif args.check:
        if inputs.check(manifest, args.input_root.resolve(), suite=args.suite):
            raise SystemExit(1)
    else:
        inputs.ensure_inputs(manifest, args.input_root, suite=args.suite, models_archive=args.models_archive)
        print('Inputs ready. No model was loaded and no benchmark was started.')


def evaluate(parser, args):
    manifest = resolve(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    # Validation-only evaluation never reads the +H test reduction labels.
    inputs.ensure_inputs(manifest, args.input_root, suite=args.suite, models_archive=args.models_archive,
                         download_missing=not args.no_setup, test_labels=args.split != 'valid')
    if args.gpus:
        failed = run_workers(entry_jobs(parser, args, manifest['entries'], args.output), status=args.output / 'status.json',
                             gpus=args.gpus, workers_per_gpu=args.workers_per_gpu)
        if failed:
            write_json(args.output / 'status.json', dict(state='failed', exit_codes=dict(failed)))
            raise SystemExit('Failed entries (see worker.log): ' + ', '.join(entry for entry, _ in failed))
        write_json(args.output / 'status.json', dict(state='complete', entries=[entry['id'] for entry in manifest['entries']]))
        return
    from .evaluate import evaluate_manifest
    evaluate_manifest(manifest, input_root=args.input_root, output=args.output, device=args.device, split=args.split,
                      limit=args.max_queries_per_shape, seed=args.seed, threads=args.threads)


def train(parser, args):
    if args.suite != 'ultraquery':
        raise ValueError('Baseline training belongs to the UltraQuery suite')
    from .train import run as run_training
    run_training(args)


def build(parser, args):
    docker.build(args.image, cache=args.cache, network=args.network, training_base=args.training)


HANDLERS = {'setup': setup, 'evaluate': evaluate, 'train': train, 'build': build}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser, commands = build_parser()
    args = parser.parse_args(argv)
    command = commands[args.command]
    if getattr(args, 'gpus', None) is not None or getattr(args, 'workers_per_gpu', 1) != 1:
        if args.workers_per_gpu < 1 or not args.gpus or len(set(args.gpus)) != len(args.gpus):
            command.error('--gpus needs distinct devices and --workers-per-gpu a positive count')
        if args.device not in ('cuda', 'cuda:0'):
            command.error('With --gpus, use --device cuda; each worker sees only its assigned GPU')
    for name in ('kgfm_batch_size', 'max_queries_per_shape'):
        if getattr(args, name, None) is not None and getattr(args, name) < 1:
            command.error(f'--{name.replace("_", "-")} must be positive')
    try:
        if getattr(args, 'image', None) and args.command in CONTAINER_GPU:
            return in_container(command, args)
        return HANDLERS[args.command](command, args)
    except ValueError as error:
        command.error(str(error))
