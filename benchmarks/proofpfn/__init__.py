"""The ProofPFN benchmark: zero-shot link prediction on frozen graph sets, by the research protocol (``PROTOCOL.md``).

Modules: ``manifests`` (graph sets and model pins, without PyTorch), ``data`` (loading and query structure),
``systems`` (ProofPFN families and the native references), ``evaluate`` (one graph x systems -> one row file),
``cells`` (the query cells of the registered analysis), ``gate`` (the N validity gate on held-out worlds of the prior),
``adapters`` (the external baselines KGPFN, MERIT and AnyBURL) and ``cli``. ``registered/`` holds the six registered
analysis files, byte-identical to their freeze anchors. Run ``python -m benchmarks.proofpfn --help``.
"""
