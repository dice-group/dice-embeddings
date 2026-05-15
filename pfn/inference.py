"""PFN inference module wrapper.

Keeps imports stable under the pfn package namespace.
"""

from pfn_inference import evaluate, infer, main, score_triple

__all__ = ["evaluate", "infer", "main", "score_triple"]


if __name__ == "__main__":
    main()
