"""Unified PFN package CLI.

Usage examples:
    python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 10 --save model.pt
    python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
    python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria
    python -m pfn visualize --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria
    python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
    python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt
"""

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "PFN unified CLI\n\n"
            "Commands:\n"
            "  train            Run training\n"
            "  infer            Run inference (top-k tail prediction)\n"
            "  score            Run triple scoring\n"
            "  visualize        Visualize triple scoring with context\n"
            "  eval rank        Ranking evaluation (MRR/Hits)\n"
            "  eval bce         BCE evaluation\n\n"
            "Examples:\n"
            "  python -m pfn train --kg-dir KGs/Countries-S1/ --epochs 10 --save model.pt\n"
            "  python -m pfn infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5\n"
            "  python -m pfn score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria\n"
            "  python -m pfn visualize --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria\n"
            "  python -m pfn eval rank --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt\n"
            "  python -m pfn eval bce --model model.pt --train-file KGs/Countries-S1/train.txt --test-file KGs/Countries-S1/test.txt"
        )
        return

    cmd = sys.argv[1]
    if cmd == "train":
        from pfn.train import main as train_main

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
        return

    if cmd in ("infer", "score"):
        from pfn.inference import main as inference_main

        sys.argv = [sys.argv[0], cmd] + sys.argv[2:]
        inference_main()
        return

    if cmd == "visualize":
        import argparse
        import torch
        import random
        from pfn.model import TriplePFN
        from pfn.inference import visualize_triple_scoring

        parser = argparse.ArgumentParser(description="Visualize triple scoring with attention heatmap")
        parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
        parser.add_argument(
            "--triple", type=str, nargs=3, required=True,
            metavar=("HEAD", "RELATION", "TAIL"), help="Triple to visualize"
        )
        parser.add_argument("--data", type=str, required=True, help="Data file for support context")
        parser.add_argument("--context-size", type=int, default=32, help="Number of support triples (default: 32)")
        parser.add_argument("--top-k", type=int, default=15, help="Number of top support triples to show in text (default: 15)")
        parser.add_argument("--save", type=str, default=None, help="Save attention plot to file (e.g., attention.png)")
        parser.add_argument("--no-show", action="store_true", help="Don't display plot interactively (only save)")
        
        args = parser.parse_args(sys.argv[2:])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from {args.model} ...")
        ckpt = torch.load(args.model, map_location=device)
        if isinstance(ckpt, dict) and "hparams" in ckpt:
            model = TriplePFN(**ckpt["hparams"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = TriplePFN()
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        
        # Load support triples
        print(f"Loading support triples from {args.data} ...")
        support = []
        with open(args.data) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 3:
                    support.append((parts[0], parts[1], parts[2]))
        
        print(f"Loaded {len(support):,} triples.")
        
        if len(support) > args.context_size:
            print(f"Sampling {args.context_size} random triples for context ...")
            support = random.sample(support, args.context_size)
        
        h, r, t = args.triple
        visualize_triple_scoring(
            model, h, r, t, support, 
            device=device, 
            top_k_support=args.top_k,
            save_path=args.save,
            show_plot=not args.no_show
        )
        return

    if cmd == "eval":
        from pfn.evaluate import main as evaluate_main

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
