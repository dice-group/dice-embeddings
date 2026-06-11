"""Thin dispatcher: `python -m ilp {train,eval,predict,splits} ...`.

Each subcommand is a CLI-first module that also runs standalone, e.g.
`python -m ilp.train ...` is equivalent to `python -m ilp train ...`.
"""
import sys

_COMMANDS = {
    "train": "ilp.train",
    "eval": "ilp.eval",
    "predict": "ilp.predict",
    "splits": "ilp.make_splits",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        prog = "python -m ilp"
        print(f"usage: {prog} {{{','.join(_COMMANDS)}}} ...\n")
        print("Commands:")
        print("  train    Train a model (CLI flags and/or --config), then auto-eval.")
        print("  eval     Filtered MRR / Hits@K of a model bundle on a held-out split.")
        print("  predict  Top-k inference / single-triple scoring from a bundle.")
        print("  splits   Generate transductive/inductive/semi-inductive splits.")
        print(f"\nEach also runs standalone, e.g. `{prog}.train --help`.")
        raise SystemExit(2 if len(sys.argv) >= 2 else 0)

    import importlib

    cmd = sys.argv.pop(1)
    module = importlib.import_module(_COMMANDS[cmd])
    sys.argv[0] = f"{sys.argv[0]} {cmd}"
    module.main()


if __name__ == "__main__":
    main()
