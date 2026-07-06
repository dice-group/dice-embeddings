"""Unit tests for Parquet reading in dicee.read_preprocess_save_load_kg.util.

Covers https://github.com/dice-group/dice-embeddings/issues/410: `read_only_few`
must only read approximately that many rows from a Parquet file, not load the
full file and slice afterwards.
"""

import pandas as pd
import pytest

from dicee.read_preprocess_save_load_kg.util import read_with_pandas


@pytest.fixture
def dummy_parquet_path(tmp_path):
    n = 1000
    df = pd.DataFrame({
        "subject": [f"e{i}" for i in range(n)],
        "relation": ["r0"] * n,
        "object": [f"e{(i + 1) % n}" for i in range(n)],
    })
    path = tmp_path / "dummy.parquet"
    df.to_parquet(path)
    return path, df


class TestReadParquetHead:
    def test_read_only_few_returns_exact_row_count(self, dummy_parquet_path):
        path, full_df = dummy_parquet_path
        df = read_with_pandas(str(path), read_only_few=10, separator="\t")
        assert len(df) == 10

    def test_read_only_few_returns_leading_rows_in_order(self, dummy_parquet_path):
        path, full_df = dummy_parquet_path
        df = read_with_pandas(str(path), read_only_few=10, separator="\t")
        expected = full_df.head(10).reset_index(drop=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

    def test_read_only_few_larger_than_file_returns_all_rows(self, dummy_parquet_path):
        path, full_df = dummy_parquet_path
        df = read_with_pandas(str(path), read_only_few=10_000, separator="\t")
        assert len(df) == len(full_df)

    def test_no_read_only_few_returns_full_file(self, dummy_parquet_path):
        path, full_df = dummy_parquet_path
        df = read_with_pandas(str(path), read_only_few=None, separator="\t")
        assert len(df) == len(full_df)

    def test_read_only_few_does_not_load_full_file_into_memory(self, dummy_parquet_path):
        """Regression test for #410: reading a small slice must not require
        peak memory anywhere near the size of a full-file read."""
        import tracemalloc

        path, full_df = dummy_parquet_path

        tracemalloc.start()
        read_with_pandas(str(path), read_only_few=10, separator="\t")
        _, peak_slice = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        read_with_pandas(str(path), read_only_few=None, separator="\t")
        _, peak_full = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak_slice < peak_full / 2
