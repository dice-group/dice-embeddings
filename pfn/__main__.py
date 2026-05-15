"""Unified PFN package CLI.

Usage examples:
    python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 10 --save model.pt
    python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
    python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria
    python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
    python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
"""

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "PFN unified CLI\n\n"
            "Commands:\n"
            "  train            Run training (same args as pfn_train.py)\n"
            "  infer            Run inference (same args as pfn_inference.py infer)\n"
            "  score            Run triple scoring (same args as pfn_inference.py score)\n"
            "  eval rank        Ranking evaluation (MRR/Hits)\n"
            "  eval bce         BCE evaluation\n\n"
            "Examples:\n"
            "  python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 10 --save model.pt\n"
            "  python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5\n"
            "  python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria\n"
            "  python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt\n"
            "  python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt"
        )
        return

    cmd = sys.argv[1]
    if cmd == "train":
        from pfn_train import main as train_main

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
        return

    if cmd in ("infer", "score"):
        from pfn_inference import main as inference_main

        sys.argv = [sys.argv[0], cmd] + sys.argv[2:]
        inference_main()
        return

    if cmd == "eval":
        from pfn_evaluate import main as evaluate_main

        if len(sys.argv) < 3:
            sys.argv = [sys.argv[0], "rank"]
        else:
            sub = sys.argv[2]
            if sub not in ("rank", "bce"):
                raise SystemExit("Unknown eval subcommand. Use: rank or bce")
            sys.argv = [sys.argv[0], sub] + sys.argv[3:]
        evaluate_main()
        return

    raise SystemExit(f"Unknown PFN command: {cmd}")


if __name__ == "__main__":
    main()
