#!/usr/bin/env python3
"""Checks compare.py's strict layer-set handling. Run: python -I tools/ernie5-reference/test_compare.py"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

COMPARE = Path(__file__).with_name("compare.py")
TOKENS, HC, HIDDEN = 2, 2, 8


def write_engine(folder: Path, layers: range) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        data = np.full((TOKENS, HC, HIDDEN), layer + 1.0, dtype="<f4")
        data.tofile(folder / f"layer{layer:02d}_pos0_t{TOKENS}_hc{HC}_h{HIDDEN}.f32")


def write_ref(folder: Path, layers: range) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        np.save(folder / f"layer_{layer:02d}_all.npy", np.full((TOKENS, HC, HIDDEN), layer + 1.0, dtype=np.float32))


def run(ref: Path, eng: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-I", str(COMPARE), "--ref", str(ref), "--engine", str(eng), "--strict", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


class StrictLayerSetTests(unittest.TestCase):
    def test_incomplete_reference_fails_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1))
            write_engine(root / "eng", range(3))
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("missing from the reference: [1, 2]", result.stdout)

    def test_complete_dump_passes_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(3))
            write_engine(root / "eng", range(3))
            result = run(root / "ref", root / "eng", "--expect-layers", "3")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_both_short_fails_expect_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(2))
            write_engine(root / "eng", range(2))
            self.assertEqual(run(root / "ref", root / "eng").returncode, 0)
            result = run(root / "ref", root / "eng", "--expect-layers", "3")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("missing [2]", result.stdout)


if __name__ == "__main__":
    unittest.main()
