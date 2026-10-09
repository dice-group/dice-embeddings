"""Paper tables and figures rendered from saved benchmark reports.

    python -m benchmarks.cqa.paper [REPORT.json ...] -o tables.tex [--figures DIR]

Tables need only the standard library; figures need matplotlib. No model,
dataset or rank trace is ever loaded.
"""
