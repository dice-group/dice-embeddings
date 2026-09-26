"""Fit query adapters and run benchmarks: python -m dicee.query_answering --help."""

import argparse
import json
from pathlib import Path

from .adapter import QueryScoreAdapter
from .benchmark import add_benchmark_parser, run_benchmark_cli
from .context import QueryContext
from .training import TRAINING_SHAPES, AdapterTrainingData, fit_query_adapter, prepare_adapter_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    add_benchmark_parser(commands)
    prepare = commands.add_parser('prepare', help='Mask source facts and generate disjoint logical queries')
    prepare.add_argument('--source', type=Path, required=True, help='QueryContext JSON: triples, num_entities, num_relations, inverse_relations')
    prepare.add_argument('--name', default='source')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--mask-fraction', type=float, default=.3)
    prepare.add_argument('--train-per-shape', type=int, default=96)
    prepare.add_argument('--validation-per-shape', type=int, default=32)
    prepare.add_argument('--seed', type=int, default=2026090851)
    prepare.add_argument('--shapes', nargs='+', choices=TRAINING_SHAPES, default=['2i', '3i'])
    fit = commands.add_parser('fit', help='Fit adapters for a frozen DICE experiment')
    fit.add_argument('--experiment', required=True)
    fit.add_argument('--data', type=Path, nargs='+', required=True)
    fit.add_argument('--output', type=Path, required=True, help='Output directory for adapter, report, and score banks')
    fit.add_argument('--feature-mode', choices=['global', 'context', 'context_scores_v1'])
    fit.add_argument('--epochs', type=int, default=20)
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--row-batch-size', type=int, default=8)
    fit.add_argument('--learning-rate', type=float, default=.02)
    fit.add_argument('--observed-mix', type=float, default=1.)
    fit.add_argument('--bias-bound', type=float, default=4.)
    fit.add_argument('--scale-bound', type=lambda v: None if v == 'none' else float(v), default=2.)
    fit.add_argument('--beam-size', type=int, default=64)
    fit.add_argument('--tnorm', choices=['prod', 'min'], default='prod')
    fit.add_argument('--normalization', choices=['none', 'standard'], default='none')
    fit.add_argument('--hidden-dim', type=int, default=0)
    fit.add_argument('--validation-every', type=int)
    fit.add_argument('--early-stopping-patience', type=int, help='Epochs without source-validation MRR improvement')
    fit.add_argument('--seed', type=int, default=2026090851)
    fit.add_argument('--samples', type=int)
    fit.add_argument('--device', default='cpu')
    fit.add_argument('--training-device', choices=['cpu', 'cuda'], default='cpu')
    fit.add_argument('--validation-device', choices=['cpu', 'cuda'])
    fit.add_argument('--cpu-threads', type=int)
    fit.add_argument('--training-cache-mb', type=int, default=1536)
    fit.add_argument('--device-cache-mb', type=int, default=512)
    fit.add_argument('--compare-global', action='store_true', help='Also fit a two-parameter baseline, reusing frozen score banks')
    args = parser.parse_args()
    if args.command == 'benchmark':
        run_benchmark_cli(args)
        return
    if args.command == 'prepare':
        data = prepare_adapter_data(QueryContext(**json.loads(args.source.read_text())), name=args.name,
                                    mask_fraction=args.mask_fraction, train_per_shape=args.train_per_shape,
                                    validation_per_shape=args.validation_per_shape, seed=args.seed, shapes=args.shapes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.save(args.output)
        print(f'Saved {len(data.train)} training and {len(data.validation)} validation queries to {args.output}')
        return
    from ..knowledge_graph_embeddings import KGE
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error('--cpu-threads must be positive')
        import torch
        torch.set_num_threads(args.cpu_threads)
    kge = KGE(path=args.experiment)
    kge.to(args.device)
    sources = [AdapterTrainingData.load(path) for path in args.data]
    args.output.mkdir(parents=True, exist_ok=True)
    modes = [args.feature_mode]
    if args.compare_global and args.feature_mode != 'global':
        modes.append('global')
    reports = {}
    for mode in modes:
        result = fit_query_adapter(kge.model, sources, feature_mode=mode, observed_mix=args.observed_mix,
                                   epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                   row_batch_size=args.row_batch_size, seed=args.seed, samples=args.samples,
                                   cache_dir=args.output / 'score-banks', bias_bound=args.bias_bound,
                                   scale_bound=args.scale_bound, beam_size=args.beam_size, tnorm=args.tnorm,
                                   normalization=args.normalization, hidden_dim=args.hidden_dim,
                                   validation_every=args.validation_every,
                                   early_stopping_patience=args.early_stopping_patience,
                                   training_device=args.training_device, training_cache_bytes=args.training_cache_mb * 2**20,
                                   validation_device=args.validation_device,
                                   device_cache_bytes=args.device_cache_mb * 2**20,
                                   checkpoint_path=args.output / f'{mode or "default"}-fit-state.pt')
        name = result.adapter.feature_mode
        path = args.output / f'{name}.json'
        result.adapter.save(path)
        QueryScoreAdapter.load(path, model=kge.model)
        reports[name] = dict(history=result.history, validation=result.validation, metadata=result.adapter.metadata)
        print(f'Saved {name} adapter to {path}')
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
