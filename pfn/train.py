"""PFN training module wrapper.

Keeps imports stable under the pfn package namespace.
"""

from pfn_train import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
