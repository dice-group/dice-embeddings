"""Prepare, verify, run and report the reproducible complex-query benchmarks.

    python -m benchmarks.cqa {plus_h,ultraquery} COMMAND [options]
    python -m benchmarks.cqa {reproduce,fit,analysis,render} [options]

Commands run in this process, which needs dicee and PyTorch, or with ``--image``
inside the pinned Docker runtime; the host then needs only Python 3 and Docker.
``reproduce`` drives these commands for the final studies of the paper and thesis
(benchmarks/cqa/REPRODUCE.md); ``fit``, ``analysis`` and ``render`` regenerate
their adapters, analyses and tables.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import docker, inputs
from .manifests import REPO, SUITES, default_recipes, prepare_manifest, read_manifest, select_entries, suite_directory

# Longest expected runtimes start first, so short entries fill the remaining slots.
RUNTIME_ORDER = ('trix-adapter', 'ultra-adapter', 'gnnqe', 'cqd-hybrid', 'cqd', 'qto')
# Container GPU access per command; other commands always run on the host.
CONTAINER_GPU = {'evaluate': 'runtime', 'prepare': 'none', 'verify': 'runtime', 'run': 'runtime',
                 'report': 'none', 'difficulty-report': 'none', 'train': 'none'}
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

    def command(name, help, *, hidden=False):
        add[name] = commands.add_parser(name, help=argparse.SUPPRESS if hidden else help, description=help)
        add[name].add_argument('--input-root', type=Path, default=REPO, help='Root of datasets and checkpoints (default: repository)')
        return add[name]

    def selection(p, *, query_types=True):
        p.add_argument('--entries', nargs='+', action='extend', help='Entry IDs')
        p.add_argument('--methods', nargs='+', action='extend', help='Methods; omit or use all for every recipe')
        p.add_argument('--datasets', nargs='+', action='extend', help='Datasets; omit or use all for every recipe')
        if query_types:
            p.add_argument('--query-types', nargs='+', action='extend', help='Query types or groups: all, epfo, negation')
            p.add_argument('--atomic-negation', action='store_true', help='CQD/CQD-Hybrid signed-atom negation for custom recipes')

    def recipes(p, *, ablations=False):
        p.add_argument('--manifests', type=Path, nargs='+', action='extend', help='Recipe manifests (default: public recipes)')
        p.add_argument('--profile', choices=('bounded', 'reference'), default='bounded',
                       help='CQD batching: bounded (8 GB GPUs) or dense reference batches')
        p.add_argument('--hardware-profile', choices=('default', 'h100'), default='default',
                       help='KGFM batch/cache starting point; requires fresh verification')
        p.add_argument('--kgfm-batch-size', type=int, help='Atomic row and backbone batch of KGFM adapters')
        p.add_argument('--answer-filter', choices=('released', 'corrected'),
                       help='+H default: corrected with released controls; UltraQuery: released only')
        if ablations:
            p.add_argument('--observed-facts', nargs='+', action='extend', choices=('none', 'atomic', 'both', 'all'),
                           help='KGFM observed-fact ablations with unchanged adapter weights')
            p.add_argument('--inference-graphs', nargs='+', action='extend', choices=('train', 'train+valid'),
                           help='+H only: test inference graphs (default: train+valid)')

    def inputs_options(p):
        p.add_argument('--no-setup', action='store_true', help='Require existing inputs instead of downloading them')
        p.add_argument('--models-archive', type=Path, help="Official +H weights ZIP if the authors' host is unavailable")

    def container(p):
        p.add_argument('--image', help='Run inside this pinned Docker image (tag or image ID)')
        p.add_argument('--gpu', choices=('runtime', 'cdi', 'none'), help='Docker GPU access')
        p.add_argument('--dry-run', action='store_true', help='Print the command without running it')

    def parallel(p):
        p.add_argument('--gpus', nargs='+', help='Run entries concurrently, one listed GPU (index or UUID) per worker')
        p.add_argument('--workers-per-gpu', type=int, default=1, help='Concurrent entries per GPU (default: 1)')

    p = command('setup', 'Download missing datasets and checkpoints and install shipped adapters; no inference')
    recipes(p, ablations=True)
    selection(p)
    p.add_argument('--models-archive', type=Path, help="Official +H weights ZIP if the authors' host is unavailable")
    p.add_argument('--check', action='store_true', help='Only report missing inputs and checksum mismatches')
    p.add_argument('--dry-run', action='store_true', help='Print sources without downloading')

    p = command('evaluate', 'Evaluate recipe subsets directly, without frozen-study verification')
    recipes(p)
    selection(p)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', choices=('valid', 'test'), required=True,
                   help='Query split; required, so the test split is only ever evaluated on request')
    p.add_argument('--max-queries-per-shape', type=int, help='Uniform sample of each query type')
    p.add_argument('--seed', type=int)
    p.add_argument('--threads', type=int)
    inputs_options(p)
    parallel(p)
    container(p)

    p = command('prepare', 'Resolve recipes and freeze inputs, query batches and source in OUTPUT/bundle')
    recipes(p, ablations=True)
    selection(p)
    p.add_argument('--output', type=Path, required=True, help='Study directory, below the input root for Docker')
    inputs_options(p)
    container(p)

    p = command('verify', 'Check validation parity and pin the evidence in OUTPUT/verified-bundle')
    p.add_argument('--output', type=Path, required=True, help='Study directory containing bundle/')
    p.add_argument('--references', type=Path, help='Directory of exported baseline oracles, REFERENCES/ENTRY.pt')
    p.add_argument('--comparison-references', type=Path, help='Optional dense CQD oracles for comparison reporting')
    p.add_argument('--device', default='cuda')
    p.add_argument('--study-path', help=argparse.SUPPRESS)
    parallel(p)
    container(p)

    p = command('run', 'Run frozen entries: a validation pilot, or the test after verification')
    p.add_argument('--output', type=Path, required=True, help='Study directory; results go to OUTPUT/PHASE')
    selection(p, query_types=False)
    p.add_argument('--phase', choices=('pilot', 'test'), default='test', help='pilot uses bundle/, test verified-bundle/')
    p.add_argument('--device', default='cuda')
    p.add_argument('--pilot-queries', type=int, default=2, help='Minimum queries per type, rounded up to whole reference batches')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    inputs_options(p)
    parallel(p)
    container(p)

    p = command('report', 'Recompute tables and paired deltas from completed results; no inference')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--published', type=Path, help="Published scores (default: the suite's published_results.json)")
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=1, help='Processes for traces and effects; results do not depend on it')
    container(p)

    p = command('difficulty-report', 'Answer-level difficulty from completed frozen rank traces; CPU only')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory outside frozen results and bundles')
    p.add_argument('--labels-root', type=Path, help='Separate extraction root for released reduction labels')
    p.add_argument('--entries', nargs='+', action='extend', help='Completed result IDs, including control suffixes')
    p.add_argument('--query-types', nargs='+', action='extend', help='Report-only query-type subset')
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

    p = command('parity', 'Verify one baseline against its exported oracle', hidden=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--entry', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--comparison-reference', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')

    p = command('integration', 'Verify one KGFM adapter pipeline against the independent reference', hidden=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--entry', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--probes', type=int, default=2, help='Reference-checked queries per type')
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
                            query_types=getattr(args, 'query_types', None), atomic_negation=getattr(args, 'atomic_negation', False),
                            observed_facts=getattr(args, 'observed_facts', None), inference_graphs=getattr(args, 'inference_graphs', None),
                            answer_filter=args.answer_filter or SUITES[args.suite]['answer_filter'], suite=args.suite,
                            hardware_profile=args.hardware_profile, kgfm_batch_size=args.kgfm_batch_size)


def ensure_inputs(args, manifest):
    # Validation-only evaluation never reads the +H test reduction labels.
    test_labels = getattr(args, 'split', 'test') != 'valid'
    if not args.no_setup:
        inputs.ensure_inputs(manifest, args.input_root, suite=args.suite, models_archive=args.models_archive,
                             test_labels=test_labels)
    else:
        inputs.ensure_inputs(manifest, args.input_root, suite=args.suite, download_missing=False, test_labels=test_labels)


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
            *arguments(parser, args, translate=mapped, skip=HOST_OPTIONS | {'manifest_output_root', 'study_path'})]
    if args.command == 'train':
        # Generated manifests refer to checkpoints by their path below the input root.
        argv += ['--manifest-output-root', Path('/inputs', output.relative_to(root)).as_posix()]
    if args.command == 'verify':
        argv += ['--study-path', output.relative_to(root).as_posix()]
    if any(action.dest == 'no_setup' for action in parser._actions):
        argv.append('--no-setup')
        if not args.dry_run and not args.no_setup:
            manifest = (read_manifest(output / ('verified-bundle' if args.phase == 'test' else 'bundle') / 'bundle.json')
                        if args.command == 'run' else resolve(args))
            inputs.ensure_inputs(manifest, root, suite=args.suite, models_archive=args.models_archive)
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


def entry_jobs(parser, args, entries, output, *flags):
    """One fresh worker process per entry, repeating this command for that entry alone."""
    common = arguments(parser, args, skip={'entries', 'methods', 'datasets', 'gpus', 'workers_per_gpu', 'no_setup', 'dry_run'})
    jobs = []
    for entry in entries:
        (output / entry['id']).mkdir(parents=True, exist_ok=True)
        jobs.append((entry['id'], entry['method'],
                     [sys.executable, '-u', '-m', 'benchmarks.cqa', args.suite, args.command, *common,
                      '--entries', entry['id'], '--no-setup', *flags],
                     output / entry['id'] / 'worker.log'))
    return jobs


def finish(jobs, args, output, ids):
    failed = run_workers(jobs, status=output / 'status.json', gpus=args.gpus, workers_per_gpu=args.workers_per_gpu)
    if failed:
        write_json(output / 'status.json', dict(state='failed', exit_codes=dict(failed)))
        raise SystemExit('Failed entries (see worker.log): ' + ', '.join(entry for entry, _ in failed))
    write_json(output / 'status.json', dict(state='complete', entries=ids))


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
    ensure_inputs(args, manifest)
    if args.gpus:
        finish(entry_jobs(parser, args, manifest['entries'], args.output), args, args.output,
               [entry['id'] for entry in manifest['entries']])
        return
    from .evaluate import evaluate_manifest
    evaluate_manifest(manifest, input_root=args.input_root, output=args.output, device=args.device, split=args.split,
                      limit=args.max_queries_per_shape, seed=args.seed, threads=args.threads)


def prepare(parser, args):
    if (args.output / 'bundle' / 'bundle.json').exists():
        raise ValueError('Use a new output directory to change a frozen profile')
    manifest = resolve(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    ensure_inputs(args, manifest)
    from .study import freeze
    bundle = freeze(manifest, args.output / 'bundle', args.input_root)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(profile=args.profile, entries=len(manifest['entries']), source=bundle['source_sha256'])))


def verify(parser, args):
    """Run every entry's parity check in a fresh process, then freeze the passing evidence.

    Passing evidence of an unchanged entry (a KGFM integration check, or a
    parity check against the same oracles) is reused, so an interrupted
    verification resumes and checks can run as separate jobs. With ``--gpus``
    the checks run concurrently, one GPU per worker, logged to
    ``verification/ENTRY.log``; the evidence does not depend on the order.
    """
    from .study import freeze, integration_passed, parity_passed, verify_bundle
    study, root = args.output.resolve(), args.input_root.resolve()
    if (study / 'verified-bundle' / 'bundle.json').exists():
        raise ValueError('A verified bundle already exists; use a new output directory for new evidence')
    # Evidence is pinned by its path below the input root, valid on the host and in containers.
    if args.study_path is None and not study.is_relative_to(root):
        raise ValueError('The study directory must be below --input-root, where evidence paths are recorded')
    relative = Path(args.study_path) if args.study_path is not None else study.relative_to(root)
    bundle = verify_bundle(study / 'bundle', root)
    manifest = bundle['manifest']
    jobs = []
    for entry in manifest['entries']:
        common = [sys.executable, '-u', '-m', 'benchmarks.cqa', args.suite]
        options = ['--bundle', str(study / 'bundle'), '--input-root', str(root), '--entry', entry['id'], '--device', args.device]
        if entry['method'].endswith('-adapter'):
            target = study / 'verification' / entry['id']
            if integration_passed(target / 'integration.json', bundle, entry):
                continue
            if (target / 'pilot').exists():
                raise ValueError(f'KGFM verification requires a fresh directory: {target}')
            jobs.append((entry['id'], entry['method'], [*common, 'integration', *options, '--output', str(target)]))
            continue
        reference = args.references / f'{entry["id"]}.pt' if args.references else None
        if reference is None or not reference.is_file():
            raise ValueError(f'Missing baseline oracle {entry["id"]}.pt; pass --references')
        evidence = study / 'evidence' / f'{entry["id"]}.json'
        command = [*common, 'parity', *options, '--reference', str(reference), '--output', str(evidence)]
        comparison = None
        if args.comparison_references and entry['method'] in ('cqd', 'cqd-hybrid'):
            comparison = args.comparison_references / f'{entry["id"]}.pt'
            if not comparison.is_file():
                raise ValueError(f'Missing comparison oracle: {comparison}')
            command += ['--comparison-reference', str(comparison)]
        if not parity_passed(evidence, bundle, entry, reference, comparison):
            jobs.append((entry['id'], entry['method'], command))
    if args.gpus and jobs:
        logs = study / 'verification'
        logs.mkdir(parents=True, exist_ok=True)
        failed = run_workers([(entry, method, command, logs / f'{entry}.log') for entry, method, command in jobs],
                             status=logs / 'status.json', gpus=args.gpus, workers_per_gpu=args.workers_per_gpu)
        if failed:
            raise SystemExit('Verification failed (see verification/ENTRY.log): ' + ', '.join(entry for entry, _ in failed))
    else:
        for entry, _, command in jobs:
            print(json.dumps(dict(entry=entry, state='verifying')), flush=True)
            if subprocess.run(command).returncode:
                raise SystemExit(f'Verification of {entry} failed; see its output above')
    for entry in manifest['entries']:
        evidence = study / 'evidence' / f'{entry["id"]}.json'
        if entry['method'].endswith('-adapter'):
            evidence.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(study / 'verification' / entry['id'] / 'integration.json', evidence)
        if not json.loads(evidence.read_text()).get('passed'):
            raise ValueError(f'Verification failed: {entry["id"]}')
        entry['verification'] = (relative / 'evidence' / evidence.name).as_posix()
    write_json(study / 'verified-manifest.json', manifest)
    freeze(manifest, study / 'verified-bundle', root)
    print(json.dumps(dict(verified=len(manifest['entries']), bundle=str(study / 'verified-bundle'))))


def run(parser, args):
    from dicee.query_answering._checkpoint import BenchmarkCheckpoint, checksum

    from .study import run_job, verify_bundle
    bundle_dir = args.output / ('verified-bundle' if args.phase == 'test' else 'bundle')
    results = args.output / args.phase
    if not args.worker:
        ensure_inputs(args, read_manifest(bundle_dir / 'bundle.json'))
    bundle = verify_bundle(bundle_dir, args.input_root)
    entries = select_entries(bundle['manifest']['entries'], entry_ids=args.entries, methods=args.methods, datasets=args.datasets)
    if args.worker:
        if len(entries) != 1:
            raise ValueError('A worker takes one entry')
        result = run_job(bundle_dir, entries[0]['id'], args.input_root, results / entries[0]['id'],
                         device=args.device, phase=args.phase, pilot_queries=args.pilot_queries)
        print(json.dumps(dict(entry=entries[0]['id'], queries=result['queries'], averages=result['averages'])), flush=True)
        return
    # Parallelism is excluded from the suite identity: it cannot change results.
    identity = dict(bundle=checksum(bundle_dir / 'bundle.json'), phase=args.phase, device=args.device, pilot_queries=args.pilot_queries)
    with BenchmarkCheckpoint(results / 'suite', identity):
        finish(entry_jobs(parser, args, entries, results, '--worker'), args, results, [entry['id'] for entry in entries])


def report(parser, args):
    from .reports import export_reports
    published = args.published or suite_directory(args.suite) / 'published_results.json'
    rows = export_reports(args.results, published if published.is_file() else None, args.output,
                          bootstrap_samples=args.bootstrap_samples, seed=args.seed, title=SUITES[args.suite]['title'],
                          workers=args.workers)
    print(json.dumps(dict(results=len(rows), output=str(args.output))))


def difficulty_report(parser, args):
    from .difficulty import export_difficulty_reports
    result = export_difficulty_reports(args.results, args.bundle, args.input_root, args.output,
                                       labels_root=args.labels_root, entries=args.entries, query_types=args.query_types)
    print(json.dumps(dict(rows=len(result['rows']), pending_main_entries=len(result['pending_main_entries']), output=str(args.output))))


def train(parser, args):
    if args.suite != 'ultraquery':
        raise ValueError('Baseline training belongs to the UltraQuery suite')
    from .train import run as run_training
    run_training(args)


def build(parser, args):
    docker.build(args.image, cache=args.cache, network=args.network, training_base=args.training)


def parity(parser, args):
    from .oracles import verify_predictions
    result = verify_predictions(args.bundle, args.entry, args.input_root, args.reference, args.output,
                                device=args.device, comparison_reference=args.comparison_reference)
    print(json.dumps(dict(passed=result['passed'], queries=len(result['queries']))))
    if not result['passed']:
        raise SystemExit(1)


def integration(parser, args):
    from .verification.verify_kgfm import verify as verify_kgfm
    verify_kgfm(args.bundle, args.entry, args.input_root, args.output, device=args.device, probes=args.probes)


HANDLERS = {'setup': setup, 'evaluate': evaluate, 'prepare': prepare, 'verify': verify, 'run': run, 'report': report,
            'difficulty-report': difficulty_report, 'train': train, 'build': build, 'parity': parity,
            'integration': integration}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ('reproduce', 'fit', 'analysis', 'render'):
        from .reproduction.cli import main as reproduction
        return reproduction(argv)
    parser, commands = build_parser()
    args = parser.parse_args(argv)
    command = commands[args.command]
    if getattr(args, 'gpus', None) is not None or getattr(args, 'workers_per_gpu', 1) != 1:
        if args.workers_per_gpu < 1 or not args.gpus or len(set(args.gpus)) != len(args.gpus):
            command.error('--gpus needs distinct devices and --workers-per-gpu a positive count')
        if args.device not in ('cuda', 'cuda:0'):
            command.error('With --gpus, use --device cuda; each worker sees only its assigned GPU')
    for name in ('kgfm_batch_size', 'max_queries_per_shape', 'pilot_queries', 'bootstrap_samples', 'probes', 'workers'):
        if getattr(args, name, None) is not None and getattr(args, name) < (0 if name == 'bootstrap_samples' else 1):
            command.error(f'--{name.replace("_", "-")} must be positive')
    try:
        if getattr(args, 'image', None) and args.command in CONTAINER_GPU:
            return in_container(command, args)
        return HANDLERS[args.command](command, args)
    except ValueError as error:
        command.error(str(error))
