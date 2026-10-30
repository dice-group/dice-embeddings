"""CSV compatibility and identifier preservation for the Polars reader."""

import polars as pl
import pytest

from dicee.read_preprocess_save_load_kg.util import read_with_polars


@pytest.mark.parametrize('separator', ['\t', ','])
@pytest.mark.parametrize('extra_column', [False, True])
@pytest.mark.parametrize('read_only_few', [None, 2])
def test_read_polars_preserves_string_identifiers(tmp_path, separator, extra_column, read_only_few):
    triples = [('001', '002', '003'), ('004', '005', '006'), ('007', '008', '009')]
    path = tmp_path / 'triples.txt'
    path.write_text(''.join(separator.join(row + (('.',) if extra_column else ())) + '\n' for row in triples))

    result = read_with_polars(str(path), read_only_few=read_only_few, separator=separator)

    assert result.columns == ['subject', 'relation', 'object']
    assert result.dtypes == [pl.String] * 3
    assert result.rows() == triples[:read_only_few]


def test_read_polars_samples_requested_fraction(tmp_path):
    triples = [(f'entity{i}', 'relation', f'entity{i + 1}') for i in range(4)]
    path = tmp_path / 'triples.txt'
    path.write_text(''.join('\t'.join(row) + '\n' for row in triples))

    result = read_with_polars(str(path), sample_triples_ratio=0.5, separator='\t')

    assert len(result) == 2
    assert set(result.rows()).issubset(triples)


def test_read_polars_filters_rdf_literals_with_trailing_column(tmp_path):
    path = tmp_path / 'triples.nt'
    path.write_text('<a>\t<r>\t<b>\t.\n<a>\t<r>\t"literal"\t.\n')

    result = read_with_polars(str(path), separator='\t')

    assert result.rows() == [('<a>', '<r>', '<b>')]
