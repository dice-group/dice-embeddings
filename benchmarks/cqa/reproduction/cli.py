"""Reproduce the final CQA studies, their reports, the thesis analyses and the paper tables from tracked files.

    python -m benchmarks.cqa reproduce STUDY... | --all | --list   [--output ROOT] [--stages STAGE...] [--dry-run]
    python -m benchmarks.cqa fit ADAPTER... | --all | --list       --output DIR [--gpus GPU...]
    python -m benchmarks.cqa analysis NAME [options]               --output FILE.json
    python -m benchmarks.cqa render [--reports ROOT]               [-o tables.tex] [--figures DIR] [--thesis DIR]

A study runs fit (with --fit), train, manifest, prepare, oracles, verify, run, report and difficulty; --all
also combines each suite's results into one report, runs the analyses and renders the tables and figures.
Every stage writes below ROOT and is skipped once its output exists, so repeating a command resumes it;
--dry-run prints the commands and file actions that would run, assuming every earlier step succeeds.
With --image the benchmark commands run in the pinned Docker runtime; fits, oracle exports, analyses and
rendering run on the host. See benchmarks/cqa/REPRODUCE.md.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .. import cli as benchmark
from ..manifests import REPO, catalog
from . import registry
from .registry import ADAPTER_ROOT, ALL_FITS, FITS, SOURCE_URLS, SOURCES, STUDIES, Study

STUDY_STAGES = ('train', 'manifest', 'prepare', 'oracles', 'verify', 'run', 'report', 'difficulty')
STAGES = ('fit', *STUDY_STAGES, 'combine', 'analyses', 'render')
REPORTS = ('comparison', 'adapter-effects', 'filter-effects', 'graph-effects')
# The +H answer-difficulty reports the paper and thesis render, in the order they were rendered.
DIFFICULTY = ('plus_h-baselines', 'plus_h-qto', 'plus_h-kgfm-b64-ablations', 'plus_h-kgfm-b64', 'plus_h-kgfm-b64-kgicl')
# Where the final studies read the bracket and KG-ICL adapters on the GPU host; Experiments/final-adapters/adapters is a copy.
SCREEN_ADAPTERS = 'Experiments/screens/adapters/'
MODULE = [sys.executable, '-m', 'benchmarks.cqa']


@dataclass
class Context:
    """Settings shared by every stage of one reproduction, and the outputs a dry run assumes."""

    input_root: Path
    root: Path
    adapter_root: str = ADAPTER_ROOT
    device: str = 'cuda'
    image: str | None = None
    training_image: str | None = None
    gpus: list[str] | None = None
    workers_per_gpu: int = 1
    report_workers: int = 8
    upstream: Path | None = None
    kgicl_datasets: Path | None = None
    thesis: Path | None = None
    dry_run: bool = False
    produced: set = field(default_factory=set)

    def say(self, text: str) -> None:
        print(f'# {text}', flush=True)

    def run(self, command: list, *, produces=(), path: str | None = None) -> None:
        """Print a command and, unless this is a dry run, run it; a dry run assumes its outputs ``produces`` exist."""
        shown = shlex.join(map(str, command))
        print(f'PATH={shlex.quote(path)}:"$PATH" {shown}' if path else shown, flush=True)
        if self.dry_run:
            self.produced.update(produces)
            return
        environment = dict(os.environ, PATH=f'{path}{os.pathsep}{os.environ.get("PATH", "")}') if path else None
        if subprocess.run(list(map(str, command)), env=environment).returncode:
            raise SystemExit(f'Failed: {shown}\nRepeat the reproduction to resume once the cause is fixed.')

    def remove(self, path: Path) -> None:
        """Remove a derived directory or file."""
        print(shlex.join(['rm', '-rf', str(path)]), flush=True)
        if not self.dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)

    def exists(self, path: Path) -> bool:
        return path in self.produced or path.exists()

    def complete(self, path: Path) -> bool:
        """Whether ``path`` is (or, in a dry run, will be) a completed result."""
        return path in self.produced or (path.is_file() and bool(json.loads(path.read_text()).get('queries')))

    def gpu_options(self) -> list[str]:
        return ['--gpus', *self.gpus, '--workers-per-gpu', str(self.workers_per_gpu)] if self.gpus else []

    def image_options(self) -> list[str]:
        return ['--image', self.image] if self.image else []

    def study(self, study: Study) -> Path:
        return self.root / study.id

    def entries(self, study: Study) -> list[dict]:
        entries: list[dict] = registry.resolved_manifest(study, self.adapter_root)['entries']
        return entries


def stage_fit(studies: list[Study], context: Context) -> None:
    """Fit every adapter the studies read into ROOT/adapters; ``fit`` runs one worker per group of shared score banks."""
    paths = list(dict.fromkeys(path for study in studies for path in registry.study_adapters(study)))
    missing = [path for path in paths if not context.exists(context.root / 'adapters' / path)]
    if missing:
        context.run([*MODULE, 'fit', *missing, '--input-root', context.input_root, '--output', context.root / 'adapters',
                     '--device', context.device, *context.gpu_options()], produces=[context.root / 'adapters' / path for path in missing])


def stage_train(study: Study, context: Context) -> None:
    """Train missing UltraQuery comparison weights (inductive GNN-QE, QTO on FB15k) with pinned author code."""
    missing = [e for e in context.entries(study) if e.get('training')
               and not all(context.exists(context.input_root / e[key]) for key in ('checkpoint', 'training'))]
    if missing:
        image = ['--image', context.training_image] if context.training_image else []
        context.run([*MODULE, study.suite, 'train', '--methods', *dict.fromkeys(e['method'] for e in missing),
                     '--datasets', *dict.fromkeys(e['dataset'] for e in missing), '--input-root', context.input_root,
                     '--output', context.input_root / Path(missing[0]['checkpoint']).parents[1], '--device', 'cpu' if image else context.device,
                     *image], produces=[context.input_root / e[key] for e in missing for key in ('checkpoint', 'training')])


def stage_manifest(study: Study, context: Context) -> None:
    """Write the recipes ``prepare`` reads; a prepared study must match the registry exactly."""
    directory = context.study(study)
    if (directory / 'bundle' / 'bundle.sha256.json').exists():
        if json.loads((directory / 'manifest.json').read_text()) != registry.resolved_manifest(study, context.adapter_root):
            raise ValueError(f'{directory} was prepared from other recipes or adapters; use a new --output')
        return
    path, text = directory / 'recipes.json', json.dumps(registry.recipe_manifest(study, context.adapter_root), indent=2) + '\n'
    if path.is_file() and path.read_text() == text:
        return
    context.say(f'write {path} ({study.entries} entries)')
    if not context.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def stage_prepare(study: Study, context: Context) -> None:
    directory = context.study(study)
    if not context.exists(directory / 'bundle' / 'bundle.sha256.json'):
        context.run([*MODULE, study.suite, 'prepare', '--input-root', context.input_root, '--output', directory,
                     '--manifests', directory / 'recipes.json', *registry.prepare_options(study), *context.image_options()],
                    produces=[directory / 'bundle' / 'bundle.sha256.json'])


def missing_oracles(study: Study, context: Context) -> list[dict]:
    """Baseline entries of an unverified study without an exported oracle."""
    directory = context.study(study)
    if context.exists(directory / 'verified-bundle' / 'bundle.json'):
        return []
    return [e for e in context.entries(study) if not e['method'].endswith('-adapter') and not context.exists(directory / 'references' / f'{e["id"]}.pt')]


def checkouts_needed(entries: list[dict]) -> str:
    """The commands that clone the pinned upstream checkouts of ``entries`` below UPSTREAM."""
    pins = {(repository.split('/')[1], repository, commit) for method, (repository, commit) in catalog().REFERENCES.items()
            if method in {e['method'] for e in entries}}
    return '\n  '.join(f'git clone https://github.com/{repository} UPSTREAM/{name} && git -C UPSTREAM/{name} checkout {commit}'
                       for name, repository, commit in sorted(pins))


def stage_oracles(study: Study, context: Context) -> None:
    """Export a validation oracle per baseline entry with its pinned upstream code, in this (author-dependency) environment."""
    directory = context.study(study)
    missing = missing_oracles(study, context)
    if not missing:
        return
    upstream = context.upstream
    if upstream is None:
        message = f'{study.id} needs {len(missing)} upstream oracles; pass --upstream UPSTREAM with these checkouts:\n  ' + checkouts_needed(missing)
        if not context.dry_run:
            raise ValueError(message)
        context.say(message.replace('\n', '\n# '))
        upstream = Path('UPSTREAM')
    for entry in missing:
        target = directory / 'references' / f'{entry["id"]}.pt'
        partial = target.with_suffix('.pt.partial')
        if not context.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
        # Upstream code that compiles extensions (ULTRA, TorchDrug) needs ninja from this environment on PATH.
        context.run([sys.executable, REPO / 'benchmarks/cqa/verification/export_reference.py', '--bundle', directory / 'bundle',
                     '--input-root', context.input_root, '--entry', entry['id'],
                     '--upstream', upstream / catalog().REFERENCES[entry['method']][0].split('/')[1],
                     '--device', context.device, '--output', partial], produces=[target], path=str(Path(sys.executable).parent))
        # Written under a temporary name, so an interrupted export never leaves a partial oracle behind.
        print(shlex.join(['mv', str(partial), str(target)]), flush=True)
        if not context.dry_run:
            os.replace(partial, target)


def stage_verify(study: Study, context: Context) -> None:
    directory = context.study(study)
    if context.exists(directory / 'verified-bundle' / 'bundle.json'):
        return
    entries = context.entries(study)
    for entry in entries:
        check = directory / 'verification' / entry['id']
        evidence = check / 'integration.json'
        # An interrupted KGFM check restarts in a fresh directory; verify reuses the passing ones.
        if entry['method'].endswith('-adapter') and check.is_dir() and not (evidence.is_file() and json.loads(evidence.read_text()).get('passed')):
            context.remove(check)
    references = ['--references', directory / 'references'] if any(not e['method'].endswith('-adapter') for e in entries) else []
    context.run([*MODULE, study.suite, 'verify', '--input-root', context.input_root, '--output', directory, '--device', context.device,
                 *references, *context.gpu_options(), *context.image_options()], produces=[directory / 'verified-bundle' / 'bundle.json'])


def stage_run(study: Study, context: Context) -> None:
    directory = context.study(study)
    results = [directory / 'test' / e['id'] / 'result.json' for e in context.entries(study)]
    if not all(map(context.complete, results)):
        context.run([*MODULE, study.suite, 'run', '--input-root', context.input_root, '--output', directory, '--device', context.device,
                     *context.gpu_options(), *context.image_options()], produces=results)


def stage_report(study: Study, context: Context) -> None:
    directory = context.study(study)
    if not context.exists(directory / 'reports' / 'comparison.json'):
        context.run([*MODULE, study.suite, 'report', '--input-root', context.input_root, '--results', directory / 'test',
                     '--output', directory / 'reports', '--workers', str(context.report_workers), *context.image_options()],
                    produces=[directory / 'reports' / 'comparison.json'])


def stage_difficulty(study: Study, context: Context) -> None:
    """Answer-level difficulty of a +H study's rank traces, for the paper's hardness figure and appendix (CPU)."""
    directory = context.study(study)
    if study.difficulty and not context.exists(directory / 'difficulty' / 'inference-difficulty.json'):
        context.run([*MODULE, study.suite, 'difficulty-report', '--input-root', context.input_root, '--results', directory / 'test',
                     '--bundle', directory / 'verified-bundle', '--output', directory / 'difficulty', *context.image_options()],
                    produces=[directory / 'difficulty' / 'inference-difficulty.json'])


STUDY_STAGE_RUNNERS = {'train': stage_train, 'manifest': stage_manifest, 'prepare': stage_prepare, 'oracles': stage_oracles,
                       'verify': stage_verify, 'run': stage_run, 'report': stage_report, 'difficulty': stage_difficulty}


def stage_combine(suites: list[str], context: Context) -> None:
    """One report per suite over every completed study, from hard links of their results (no copies, no inference)."""
    for suite in suites:
        sources, missing = {}, []
        for study in (s for s in STUDIES if s.suite == suite):
            entries = [e['id'] for e in context.entries(study)]
            if all(context.complete(context.study(study) / 'test' / entry / 'result.json') for entry in entries):
                sources[study.id] = entries
            else:
                missing.append(study.id)
        if not sources:
            context.say(f'combine {suite}: no completed study yet')
            continue
        if missing:
            context.say(f'combine {suite}: left out until complete: {", ".join(missing)}')
        combined, reports = context.root / f'{suite}-all', context.root / f'{suite}-all-reports'
        listing = combined / 'sources.json'
        if listing.is_file() and json.loads(listing.read_text()) == sources and (reports / 'comparison.json').is_file():
            continue
        for path in (combined, reports):
            if path.exists():
                context.remove(path)
        context.say(f'link {sum(map(len, sources.values()))} entry directories of {len(sources)} studies into {combined}')
        if not context.dry_run:
            for study_id, entries in sources.items():
                for entry in entries:
                    shutil.copytree(context.root / study_id / 'test' / entry, combined / study_id / entry, copy_function=os.link)
            listing.write_text(json.dumps(sources, indent=2) + '\n')
        context.run([*MODULE, suite, 'report', '--input-root', context.input_root, '--results', combined, '--output', reports,
                     '--workers', str(context.report_workers), *context.image_options()],
                    produces=[reports / f'{name}.json' for name in REPORTS])


def analysis_jobs(context: Context) -> list[tuple[str, list[str]]]:
    """(output file, ``analysis`` arguments) of every thesis analysis."""
    adapters = ['--adapter-root', str(context.input_root / context.adapter_root)] if context.adapter_root != ADAPTER_ROOT else []
    jobs = [(f'calibration-{backbone}.json', ['calibration-profile', '--backbone', backbone, '--device', context.device, *adapters])
            for backbone in registry.BACKBONES]
    jobs += [('observed-links.json', ['observed-links']), ('adapter-weights.json', ['adapter-weights', *adapters]),
             ('negation-probe.json', ['negation-probe', '--device', context.device, *adapters]),
             ('pretraining-overlap-ultra.json', ['pretraining-overlap', '--pretraining', 'ultra']),
             ('answer-classes.json', ['answer-classes', '--results', str(context.root)])]
    if context.kgicl_datasets is not None:
        jobs.append(('pretraining-overlap-kgicl.json', ['pretraining-overlap', '--pretraining', 'kgicl', '--kgicl-datasets', str(context.kgicl_datasets)]))
    return jobs


def stage_analyses(context: Context) -> None:
    for name, arguments in analysis_jobs(context):
        target = context.root / 'analyses' / name
        if not context.exists(target):
            context.run([*MODULE, 'analysis', *arguments, '--input-root', context.input_root, '--output', target], produces=[target])
    if context.kgicl_datasets is None:
        context.say("analyses: KG-ICL's pretraining overlap needs --kgicl-datasets (datasets.zip of the pinned KG-ICL checkout)")


def saved_reports(root: Path, produced=frozenset()) -> list[Path]:
    """The combined suite reports and +H difficulty reports below ``root``, in the layout ``reproduce`` writes."""
    paths = [root / f'{suite}-all-reports' / f'{name}.json' for suite in ('ultraquery', 'plus_h') for name in REPORTS]
    paths += [root / study / 'difficulty' / 'inference-difficulty.json' for study in DIFFICULTY]
    return [path for path in paths if path in produced or path.is_file()]


def stage_render(context: Context) -> None:
    reports = saved_reports(context.root, context.produced)
    if not reports:
        context.say(f'render: no saved reports below {context.root} yet')
        return
    paper = context.root / 'paper'
    context.run([sys.executable, '-m', 'benchmarks.cqa.paper', *reports, '-o', paper / 'tables.tex', '--figures', paper / 'figures',
                 *(['--thesis', context.thesis] if context.thesis else [])])


def listing(studies: list[Study]) -> None:
    """Print each study's entries, prerequisites and measured test cost."""
    print(f'{"study":36} {"entries":>7} {"test h":>7}  prerequisites; contents')
    for suite in dict.fromkeys(s.suite for s in studies):
        chosen = [s for s in studies if s.suite == suite]
        for study in chosen:
            entries = registry.resolved_manifest(study)['entries']
            counts = {'adapter': len(registry.study_adapters(study)), 'oracle': sum(not e['method'].endswith('-adapter') for e in entries),
                      'trained checkpoint': sum(bool(e.get('training')) for e in entries)}
            needs = [f'{count} {name}{"s" if count > 1 else ""}' for name, count in counts.items() if count]
            needs += ['difficulty report'] if study.difficulty else []
            print(f'{study.id:36} {study.entries:>7} {study.hours:>7.1f}  {", ".join(needs)}; {study.summary}')
        print(f'{suite + " total":36} {sum(s.entries for s in chosen):>7} {sum(s.hours for s in chosen):>7.1f}')


def compare(studies: list[Study], frozen: Path, frozen_adapters: Path) -> bool:
    """Print how each regenerated manifest relates to the frozen one; return whether every scientific field agrees."""
    def frozen_sha256(path):
        return registry.sha256(frozen_adapters / path.removeprefix(SCREEN_ADAPTERS) if path.startswith(SCREEN_ADAPTERS) else REPO / path)

    agree = True
    for study in studies:
        path = frozen / study.id / 'manifest.json'
        if not path.is_file():
            print(f'{study.id:36} no frozen manifest at {path}')
            continue
        result = registry.compare_manifest(study, json.loads(path.read_text()), frozen_sha256)
        agree &= result['scientific']
        notes = [f'{len(result[key])} {label}' for key, label in (('adapter_paths', 'adapter paths (same SHA-256)'),
                 ('reference', 'reference notes'), ('superseded', 'superseded entries left out')) if result[key]]
        verdict = ('identical' + (' except ' + ', '.join(notes) if notes else '')) if result['scientific'] else 'DIFFERS'
        print(f'{study.id:36} {result["entries"]:>4} entries: {verdict}')
    return agree


def reproduce(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='python -m benchmarks.cqa reproduce', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('studies', nargs='*', metavar='STUDY', help='Study IDs, names in both suites or patterns, e.g. plus_h-kgfm-b64, qto, "plus_h-*"')
    parser.add_argument('--all', action='store_true', help='Every study, then the combined reports, analyses and tables')
    parser.add_argument('--list', action='store_true', help='List the studies with entries, prerequisites and measured test hours')
    parser.add_argument('--output', type=Path, default=REPO / 'results' / 'final', help='Root of every study directory (default: results/final)')
    parser.add_argument('--input-root', type=Path, default=REPO, help='Root of datasets, checkpoints and adapters (default: repository)')
    parser.add_argument('--stages', nargs='+', choices=STAGES, metavar='STAGE', help=f'Stages to run, in this order: {" ".join(STAGES)} '
                        f'(default: {" ".join(STUDY_STAGES)}; with --all every stage but fit)')
    parser.add_argument('--fit', action='store_true', help='Refit every adapter from scratch into OUTPUT/adapters and use those, not the shipped ones')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpus', nargs='+', help='Parallel workers for fit, verify and run, one listed GPU (index or UUID) per worker')
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--image', help='Run prepare, verify, run, report and difficulty in this pinned Docker image')
    parser.add_argument('--training-image', help='Author-training image for train (python -m benchmarks.cqa ultraquery build --training)')
    parser.add_argument('--upstream', type=Path, help='Pinned upstream checkouts for the baseline oracles, UPSTREAM/REPOSITORY')
    parser.add_argument('--kgicl-datasets', type=Path, help="datasets.zip of the pinned KG-ICL checkout, for KG-ICL's pretraining overlap")
    parser.add_argument('--report-workers', type=int, default=8, help='Processes per report; results do not depend on it')
    parser.add_argument('--thesis', type=Path, help='Also write the thesis tables and figures into this thesis project')
    parser.add_argument('--compare-manifests', type=Path, metavar='FROZEN', help='Only compare the resolved manifests with FROZEN/STUDY/manifest.json')
    parser.add_argument('--frozen-adapters', type=Path, default=REPO / 'Experiments/final-adapters/adapters',
                        help=f'Copy of {SCREEN_ADAPTERS} for --compare-manifests')
    parser.add_argument('--dry-run', action='store_true', help='Print every command and file action without running or writing anything')
    args = parser.parse_args(argv)
    if bool(args.studies) == args.all and not args.list:
        parser.error('Name studies or pass --all')
    if args.workers_per_gpu < 1 or args.report_workers < 1 or (args.gpus and (len(set(args.gpus)) != len(args.gpus) or args.device != 'cuda')):
        parser.error('--gpus needs distinct devices and --device cuda; worker counts must be positive')
    stages = args.stages or [*(['fit'] if args.fit else []), *(STAGES[1:] if args.all else STUDY_STAGES)]
    if 'fit' in stages and not args.fit:
        parser.error('The fit stage refits the adapters of this reproduction; pass --fit')
    try:
        studies = registry.select_studies(args.studies or None)
        if args.list:
            return listing(studies)
        if args.compare_manifests:
            if not compare(studies, args.compare_manifests, args.frozen_adapters):
                raise SystemExit(1)
            return
        context = Context(args.input_root.resolve(), args.output.resolve(), device=args.device, image=args.image,
                          training_image=args.training_image, gpus=args.gpus, workers_per_gpu=args.workers_per_gpu,
                          report_workers=args.report_workers, upstream=args.upstream.resolve() if args.upstream else None,
                          kgicl_datasets=args.kgicl_datasets.resolve() if args.kgicl_datasets else None,
                          thesis=args.thesis.resolve() if args.thesis else None, dry_run=args.dry_run)
        if context.root == context.input_root or not context.root.is_relative_to(context.input_root):
            parser.error('--output must be strictly below --input-root, where verification records its evidence paths')
        if args.fit:
            context.adapter_root = (context.root / 'adapters').relative_to(context.input_root).as_posix()
        if 'oracles' in stages and context.upstream is None and not args.dry_run:
            # Stop before any GPU work rather than after the studies that need no oracles.
            missing = [entry for study in studies for entry in missing_oracles(study, context)]
            if missing:
                parser.error(f'{len(missing)} baseline entries need upstream oracles; pass --upstream UPSTREAM with these checkouts:\n  '
                             + checkouts_needed(missing))
        if 'fit' in stages:
            stage_fit(studies, context)
        for study in studies:
            context.say(f'{study.id}: {study.entries} entries, {study.hours} test hours measured')
            for stage in (s for s in STUDY_STAGES if s in stages):
                STUDY_STAGE_RUNNERS[stage](study, context)
        if 'combine' in stages:
            stage_combine(list(dict.fromkeys(s.suite for s in studies)), context)
        if 'analyses' in stages:
            stage_analyses(context)
        if 'render' in stages:
            stage_render(context)
    except ValueError as error:
        parser.error(str(error))


def fit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='python -m benchmarks.cqa fit', description='Fit or build adapters of the final studies from scratch.')
    parser.add_argument('adapters', nargs='*', metavar='ADAPTER', help='Adapter paths relative to benchmarks/adapters (see --list)')
    parser.add_argument('--all', action='store_true', help='Every adapter the final studies read')
    parser.add_argument('--list', action='store_true', help='List the adapters, how each was fitted and its group')
    parser.add_argument('--output', type=Path, help='Adapter root of the fitted adapters; OUTPUT/cache keeps data, score banks and state')
    parser.add_argument('--input-root', type=Path, default=REPO, help="Root of checkpoints/, KGs/ and the suites' datasets")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpus', nargs='+', help='Run fit groups concurrently, one listed GPU per worker')
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list:
        for path, spec in FITS.items():
            details = {key: value for key, value in vars(spec).items() if key != 'backbone' and value not in ((), None)}
            print(f'{path:62} {spec.backbone:6} group {spec.group:42} {json.dumps(details)}')
        return
    if args.output is None or bool(args.adapters) == args.all:
        parser.error('Name adapters or pass --all, and pass --output')
    paths = list(FITS) if args.all else args.adapters
    unknown = [path for path in paths if path not in ALL_FITS]
    if unknown:
        parser.error(f'No fit recipe for {", ".join(unknown)}; see --list')
    if args.workers_per_gpu < 1 or (args.gpus and (len(set(args.gpus)) != len(args.gpus) or args.device != 'cuda')):
        parser.error('--gpus needs distinct devices and --device cuda; --workers-per-gpu must be positive')
    output, input_root = args.output.resolve(), args.input_root.resolve()
    if args.worker:
        from .fit import fit_all
        return fit_all(paths, input_root, output, device=args.device)
    pending = [path for path in paths if not (output / path).is_file()]
    if not args.dry_run and any(ALL_FITS[path].kind == 'source' for path in pending):
        for name, (relative, digest, _) in SOURCES.items():
            if not (input_root / relative).is_file() or registry.sha256(input_root / relative) != digest:
                parser.error(f'Missing or changed adapter source {input_root / relative} (SHA-256 {digest}); get it from {SOURCE_URLS[name]}')
    groups: dict[str, list[str]] = {}
    for path in pending:
        groups.setdefault(ALL_FITS[path].group, []).append(path)
    jobs = [(group, 'fit', [MODULE[0], '-u', *MODULE[1:], 'fit', *members, '--input-root', str(input_root), '--output', str(output),
                            '--device', args.device, '--worker'], output / 'logs' / f'{group}.log') for group, members in groups.items()]
    for _, _, command, log in jobs:
        print(f'{shlex.join(command)} > {shlex.quote(str(log))} 2>&1', flush=True)
    if args.dry_run or not jobs:
        return
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    failed = benchmark.run_workers(jobs, status=output / 'status.json', gpus=args.gpus, workers_per_gpu=args.workers_per_gpu)
    if failed:
        raise SystemExit(f'Failed fit groups (see {output / "logs"}): ' + ', '.join(group for group, _ in failed))


def analysis(argv: list[str]) -> None:
    from . import analyses
    parser = argparse.ArgumentParser(prog='python -m benchmarks.cqa analysis', description=analyses.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    names = parser.add_subparsers(dest='name', required=True)

    def command(name, help):
        p = names.add_parser(name, help=help, description=help)
        p.add_argument('--input-root', type=Path, default=REPO, help="Root of checkpoints/, KGs/ and the suites' datasets")
        p.add_argument('--output', type=Path, required=True, help='JSON report')
        return p

    p = command('calibration-profile', 'Raw and adapted memberships of validation 1p atoms, per graph')
    p.add_argument('--backbone', choices=registry.BACKBONES, required=True)
    p.add_argument('--datasets', nargs='+', help='Default: all 23 UQ-23 and three +H datasets')
    p.add_argument('--atoms', type=int, default=300)
    p.add_argument('--device', default='cuda')
    p.add_argument('--adapter-root', type=Path, default=REPO / ADAPTER_ROOT)
    p = command('observed-links', 'Share of true answers of 2i, 3i, 2in and 3in test queries with an observed link')
    p.add_argument('--datasets', nargs='+', help='Default: all 26 datasets')
    p = command('adapter-weights', 'Adapter weights over the five training seeds of each backbone')
    p.add_argument('--adapter-root', type=Path, default=REPO / ADAPTER_ROOT)
    p = command('negation-probe', 'Raw rows and 2in queries without beam pruning, with and without the adapter')
    p.add_argument('--backbone', choices=registry.BACKBONES, default='kgicl')
    p.add_argument('--datasets', nargs='+', help='Default: FB15k237LogicalQuery NELL995LogicalQuery WikiTopicsQuery:art')
    p.add_argument('--device', default='cpu')
    p.add_argument('--adapter-root', type=Path, default=REPO / ADAPTER_ROOT)
    p = command('pretraining-overlap', "Target triples in a backbone's pretraining graphs")
    p.add_argument('--pretraining', choices=('ultra', 'kgicl'), default='ultra', help='ULTRA 3g and TRIX, or KG-ICL')
    p.add_argument('--kgicl-datasets', type=Path, help='datasets.zip of the pinned KG-ICL checkout (with --pretraining kgicl)')
    p.add_argument('--datasets', nargs='+', help='Default: all 26 datasets')
    p = command('answer-classes', "FB15k and FB15k237 1p/2p test answers by cheapest grounding, with every method's MRR per class")
    p.add_argument('--results', type=Path, default=REPO / 'results' / 'final', help='Root of the study directories (default: results/final)')
    p.add_argument('--datasets', nargs='+', help='Transductive UQ-23 datasets (default: FB15kLogicalQuery FB15k237LogicalQuery)')
    args = parser.parse_args(argv)
    root = args.input_root.resolve()
    try:
        if args.name == 'calibration-profile':
            report = analyses.calibration_profile(args.backbone, root, names=args.datasets, atoms=args.atoms, device=args.device,
                                                  adapter_root=args.adapter_root)
        elif args.name == 'observed-links':
            report = analyses.observed_links(root, names=args.datasets)
        elif args.name == 'adapter-weights':
            report = analyses.adapter_weights(args.adapter_root)
        elif args.name == 'negation-probe':
            report = analyses.negation_probe(root, backbone=args.backbone, names=args.datasets, device=args.device, adapter_root=args.adapter_root)
        elif args.name == 'pretraining-overlap':
            report = analyses.pretraining_overlap(root, pretraining=args.pretraining, kgicl_datasets=args.kgicl_datasets, names=args.datasets)
        else:
            report = analyses.answer_classes(root, args.results.resolve(), names=args.datasets)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def render(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='python -m benchmarks.cqa render',
                                     description='Render the paper (and thesis) tables and figures from saved reports, on CPU. '
                                                 'Other options (--figures DIR, --thesis DIR, --preview, ...) go to python -m benchmarks.cqa.paper.')
    parser.add_argument('--reports', type=Path, default=REPO / 'results' / 'final',
                        help='Root with {ultraquery,plus_h}-all-reports/ and STUDY/difficulty/ (default: results/final)')
    parser.add_argument('-o', '--output', type=Path, default=REPO / 'results' / 'paper' / 'tables.tex',
                        help='LaTeX file (default: results/paper/tables.tex)')
    parser.add_argument('--dry-run', action='store_true')
    args, options = parser.parse_known_args(argv)
    reports = saved_reports(args.reports.resolve())
    if not reports:
        parser.error(f'No saved reports below {args.reports}; expected {{ultraquery,plus_h}}-all-reports/ and STUDY/difficulty/')
    arguments = [*map(str, reports), '-o', str(args.output), *options]
    print(shlex.join([sys.executable, '-m', 'benchmarks.cqa.paper', *arguments]), flush=True)
    if not args.dry_run:
        from ..paper.tables import main as paper
        paper(arguments)


COMMANDS = {'reproduce': reproduce, 'fit': fit, 'analysis': analysis, 'render': render}


def main(argv: list[str]) -> None:
    """Dispatch ``python -m benchmarks.cqa {reproduce,fit,analysis,render} ...``."""
    COMMANDS[argv[0]](argv[1:])
