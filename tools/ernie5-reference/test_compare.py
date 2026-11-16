#!/usr/bin/env python3
"""Checks compare.py's strict layer-set, decode-stitching, token and argmax handling. Run: python -I tools/ernie5-reference/test_compare.py"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

COMPARE = Path(__file__).with_name("compare.py")
TOKENS, HC, HIDDEN = 2, 2, 8


def stream(layer: int, positions: int) -> np.ndarray:
    """A distinct, non-constant stream per layer and position, so per-position cosines are meaningful."""
    rng = np.random.default_rng(layer)
    return rng.standard_normal((positions, HC, HIDDEN)).astype("<f4")


def write_engine(folder: Path, layers: range, parts: list[tuple[int, int]] | None = None, positions: int = TOKENS) -> None:
    """parts: (startPos, tokens) per file; default one prefill file covering every position."""
    folder.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        full = stream(layer, positions)
        for pos, tokens in parts or [(0, positions)]:
            full[pos:pos + tokens].tofile(folder / f"layer{layer:02d}_pos{pos}_t{tokens}_hc{HC}_h{HIDDEN}.f32")


def write_ref(folder: Path, layers: range, positions: int = TOKENS, tokens: list[int] | None = None,
              logits_argmax: int | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        np.save(folder / f"layer_{layer:02d}_all.npy", stream(layer, positions))
    if tokens is not None:
        (folder / "meta.json").write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
    if logits_argmax is not None:
        logits = np.zeros(16, dtype=np.float32)
        logits[logits_argmax] = 1.0
        np.save(folder / "logits_last.npy", logits)


def write_fed(folder: Path, tokens: list[int], argmax: int) -> None:
    (folder / "tokens.json").write_text(json.dumps({"tokens": tokens, "argmax_after_last": argmax}), encoding="utf-8")


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


class DecodeStitchTests(unittest.TestCase):
    # A 3-token prefill and three one-token decode steps: positions 0..5.
    PARTS = [(0, 3), (3, 1), (4, 1), (5, 1)]
    FED = [1, 2, 3, 40, 41, 42]

    def test_prefill_and_decode_steps_stitch_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(2), positions=6, tokens=self.FED, logits_argmax=7)
            write_engine(root / "eng", range(2), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7)
            result = run(root / "ref", root / "eng", "--expect-layers", "2", "--expect-tokens", "6")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("prefill 3, decode 3", result.stdout)
            self.assertIn("argmax after the last position: 7", result.stdout)

    def test_a_missing_decode_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), [(0, 3), (3, 1), (5, 1)], positions=6)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("gap", result.stdout)

    def test_overlapping_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), [(0, 3), (2, 1), (3, 1), (4, 1), (5, 1)], positions=6)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("overlap", result.stdout)

    def test_an_engine_shorter_than_the_reference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), [(0, 3), (3, 1)], positions=6)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("engine covers 4 positions, the reference 6", result.stdout)

    def test_a_prefill_only_reference_fails_expect_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=3)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            self.assertEqual(run(root / "ref", root / "eng").returncode, 0)
            result = run(root / "ref", root / "eng", "--expect-tokens", "6")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_a_token_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6, tokens=[1, 2, 3, 40, 99, 42])
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("token mismatch", result.stdout)

    def test_an_argmax_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6, tokens=self.FED, logits_argmax=5)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("argmax mismatch", result.stdout)

    def test_one_bad_decode_position_is_flagged_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            # Replace decode position 4 with an unrelated stream: the layer is flagged and position 4 is named as the worst.
            np.random.default_rng(99).standard_normal((1, HC, HIDDEN)).astype("<f4").tofile(
                root / "eng" / f"layer00_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            row = next(line for line in result.stdout.splitlines() if line.strip().startswith("0 "))
            self.assertIn("FLAG", row)
            self.assertEqual(row.split()[6], "4", row)


if __name__ == "__main__":
    unittest.main()
