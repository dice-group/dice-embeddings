# Registered analysis files (freeze anchors)

The six files of the registered fresh-v1 analysis, byte-identical to the copies in `fresh-v1-runtime/` of the research
snapshot, whose SHA-256 the registration (section 16) fixes. Any edit breaks the anchors, so they are exempt from the
code standard: ruff skips them (`ruff.toml`), pytest does not collect their test scripts (`../conftest.py`), and
nothing imports them in the test process. `../tests/test_registered.py` checks the hashes below, runs the fast checks of
`test_fv1_analysis.py` and `test_fv1_tables.py` as subprocesses, reproduces golden G8 (simulated rows → `fv1_tables.py` →
`fv1_analysis.py`) and, with `PROOFPFN_SIMULATE=1`, runs the simulations of `test_fv1_analysis.py --simulate`.

| File | SHA-256 |
|---|---|
| `fv1_analysis.py` | `eb6af63af900a9967b84f2d3c60ef07e48fd2929a75de90a57ac7c9dfd08388d` |
| `test_fv1_analysis.py` | `e2bcedaccd6f76059c251b2956a0fbda482ef6817300124e5703b971269557a5` |
| `fv1_tables.py` | `4d2eb850e9a5ba5901be4a2d3185c0db427183f4b9edb53035726a1dddb1dcd3` |
| `test_fv1_tables.py` | `43dde76e06b9b19a938fa91c8742e2d5749edafefa4e2c92e70ce00670bf045e` |
| `fv1_drops.py` | `3ca0e50d75af33604e83779b8dc9caa9e91338ab0f28e525555aa257b6eea014` |
| `fv1_devrows.py` | `6af5c62d07dfdd50d948d81f852d23a697caf1ab35e0a20a99adde6440a2538a` |

They import only the standard library, numpy and each other, and put this folder on `sys.path` at import. Run them as
scripts (`python -B benchmarks/proofpfn/registered/<file>`); the paths in their docstrings are those of the research run.
The byte-identical outputs need Python 3.12.2 and numpy 1.26.4 (`fv1_analysis.REGISTERED_ENV`).
