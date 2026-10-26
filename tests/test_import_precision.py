"""Package imports must preserve the caller's PyTorch precision settings."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_import_preserves_float32_matmul_precision():
    # A fresh process exercises the package import even if pytest already loaded it.
    result = subprocess.run(
        [sys.executable, "-c", dedent("""
            import torch

            torch.set_float32_matmul_precision('highest')
            import dicee

            precision = torch.get_float32_matmul_precision()
            assert precision == 'highest', f'import dicee changed precision to {precision}'
        """)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
