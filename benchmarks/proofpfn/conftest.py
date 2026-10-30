"""Pytest settings of the ProofPFN benchmark.

The registered analysis files are freeze anchors. Pytest never collects their test scripts
(``registered/test_fv1_analysis.py``, ``registered/test_fv1_tables.py``): importing them would run their module-level
code in the test process. ``tests/test_registered.py`` runs them as subprocesses instead. The marker ``simulate``
marks their 9-minute simulations, which run only with ``PROOFPFN_SIMULATE=1``.
"""
import pytest

collect_ignore_glob = ['registered/*']


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker of the frozen simulations.

    Args:
        config: The pytest configuration.
    """
    config.addinivalue_line('markers', 'simulate: the frozen simulations of test_fv1_analysis.py (about 9 minutes; set PROOFPFN_SIMULATE=1)')
