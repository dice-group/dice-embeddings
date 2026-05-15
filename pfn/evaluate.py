"""PFN evaluation module wrapper.

Keeps imports stable under the pfn package namespace.
"""

from pfn_evaluate import evaluate_bce, main

__all__ = ["evaluate_bce", "main"]


if __name__ == "__main__":
    main()
