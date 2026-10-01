"""Evaluate complex-query answering methods from JSON recipes.

    python -m dicee.scripts.evaluate_query_methods --manifest recipe.json --output DIR

A manifest holds one recipe or a list of recipes with unique ``id`` fields;
results go to ``DIR/ID``. The reproducible benchmark suites, with frozen inputs
and verification, run through ``python -m benchmarks.cqa``.
"""

import argparse
import json
from pathlib import Path

from dicee.query_answering.method_evaluation import evaluate_method


def main(argv: list[str] | None = None) -> None:
    """Evaluate each recipe of ``--manifest`` into ``--output/ID``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True, help='JSON recipe or list of recipes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument('--max-queries-per-shape', type=int, help='Uniform sample of each query type')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args(argv)
    recipes = json.loads(args.manifest.read_text())
    recipes = recipes if isinstance(recipes, list) else [recipes]
    names = [recipe.get('id', f'{recipe["method"]}-{recipe["dataset"]}') for recipe in recipes]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ('.', '..') for name in names):
        parser.error('Recipe IDs must be unique, plain directory names')
    for name, recipe in zip(names, recipes):
        recipe = dict(recipe)
        query_order = recipe.pop('query_order', 'relation')
        report = evaluate_method(recipe, output=args.output / name, device=args.device, split=args.split,
                                 limit=args.max_queries_per_shape, seed=args.seed, threads=args.threads, query_order=query_order)
        print(json.dumps(dict(id=name, queries=report['queries'], seconds=report['seconds'], averages=report['averages'])), flush=True)


if __name__ == '__main__':
    main()
