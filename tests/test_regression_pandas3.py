"""Regression tests for pandas >= 3.0.0 compatibility (issue #374).

pandas 3.0 changed the behaviour of ``pd.read_csv(..., index_col=0, dtype=str)``:
it now applies ``dtype=str`` to *all* columns including the index column,
turning the integer row-index values into strings.  The fix is to pass a
per-column dtype dict so only the data column is cast to str.

These tests guard against that regression without requiring a full training
run.
"""

import os
import tempfile
import pytest
import pandas as pd


class TestPandas3Compat:
    """Ensure entity/relation CSV loading produces integer-keyed dicts.

    The critical invariant is::

        sorted(entity_to_idx.values()) == list(range(len(entity_to_idx)))

    i.e. the values (indices) must be *integers*, not strings.
    """

    def _write_idx_csv(self, path: str, column: str, names: list) -> None:
        """Write a minimal entity_to_idx / relation_to_idx CSV."""
        df = pd.DataFrame({column: names})
        df.to_csv(path, index=True)  # writes integer row-index as first column

    def test_entity_idx_values_are_integers(self):
        """Values of entity_to_idx must be ints, not strings."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "entity_to_idx.csv")
            entities = ["Alice", "Bob", "Charlie"]
            self._write_idx_csv(csv_path, "entity", entities)

            # Reproduce the fixed loading pattern from static_funcs.load_model
            entity_to_idx = {
                v["entity"]: k
                for k, v in pd.read_csv(csv_path, index_col=0, dtype={"entity": str})
                .to_dict(orient="index")
                .items()
            }

            assert len(entity_to_idx) == len(entities)
            # Values must be integers
            assert all(isinstance(v, int) for v in entity_to_idx.values()), (
                f"entity_to_idx values must be int, got: {set(type(v) for v in entity_to_idx.values())}"
            )
            # Values must cover 0..N-1
            assert sorted(entity_to_idx.values()) == list(range(len(entities))), (
                f"entity_to_idx values are not a contiguous range: {sorted(entity_to_idx.values())}"
            )

    def test_relation_idx_values_are_integers(self):
        """Values of relation_to_idx must be ints, not strings."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "relation_to_idx.csv")
            relations = ["hasChild", "worksAt", "livesIn"]
            self._write_idx_csv(csv_path, "relation", relations)

            relation_to_idx = {
                v["relation"]: k
                for k, v in pd.read_csv(csv_path, index_col=0, dtype={"relation": str})
                .to_dict(orient="index")
                .items()
            }

            assert all(isinstance(v, int) for v in relation_to_idx.values()), (
                f"relation_to_idx values must be int, got: {set(type(v) for v in relation_to_idx.values())}"
            )
            assert sorted(relation_to_idx.values()) == list(range(len(relations)))

    def test_dtype_str_with_index_col_breaks_on_pandas3(self):
        """Document the root cause: dtype=str applies to the index in pandas >= 3.

        This test asserts the *old* broken behaviour so we detect if a future
        pandas version changes the semantics again in either direction.
        """
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "entity_to_idx.csv")
            self._write_idx_csv(csv_path, "entity", ["A", "B", "C"])

            raw = pd.read_csv(csv_path, index_col=0, dtype=str)
            idx_dtype = raw.index.dtype

            major = int(pd.__version__.split(".")[0])
            if major >= 3:
                # In pandas >= 3, dtype=str makes the integer index into object/string
                assert idx_dtype == object, (
                    f"Expected object index dtype with dtype=str in pandas {pd.__version__}, "
                    f"got {idx_dtype}"
                )
            else:
                # In pandas < 3, the integer index stays as int64 despite dtype=str
                assert str(idx_dtype).startswith("int"), (
                    f"Expected int index dtype in pandas {pd.__version__}, got {idx_dtype}"
                )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_end_to_end_training_produces_integer_entity_idx(self):
        """Full training run — verifies the KGE class loads correct integer mappings."""
        from dicee.executer import Execute
        from dicee.config import Namespace
        from dicee import KGE

        args = Namespace()
        args.model = "Keci"
        args.scoring_technique = "KvsAll"
        args.p = 0
        args.q = 1
        args.dataset_dir = "KGs/UMLS"
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = "None"

        result = Execute(args).start()
        path = result["path_experiment_folder"]

        pre = KGE(path=path)
        assert all(isinstance(v, int) for v in pre.entity_to_idx.values()), (
            "entity_to_idx values must be integers after loading via KGE"
        )
        assert all(isinstance(v, int) for v in pre.relation_to_idx.values()), (
            "relation_to_idx values must be integers after loading via KGE"
        )
        assert sorted(pre.entity_to_idx.values()) == list(range(pre.num_entities))
        assert sorted(pre.relation_to_idx.values()) == list(range(pre.num_relations))
