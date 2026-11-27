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


def write_fed(folder: Path, tokens: list[int], argmax: int, choices: list[tuple[int, int, float]] | None = None) -> None:
    fed: dict = {"tokens": tokens, "argmax_after_last": argmax}
    if choices is not None:
        fed["choices"] = [{"position": p, "argmax": a, "margin": m} for p, a, m in choices]
    (folder / "tokens.json").write_text(json.dumps(fed), encoding="utf-8")


def write_choices(folder: Path, argmax: list[int], margin: list[float]) -> None:
    (folder / "choices.json").write_text(json.dumps({"argmax": argmax, "margin": margin}), encoding="utf-8")


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

    def choice_run(self, root: Path, engine_choices: list[tuple[int, int, float]], *extra: str) -> subprocess.CompletedProcess[str]:
        write_ref(root / "ref", range(1), positions=6, tokens=self.FED, logits_argmax=7)
        write_choices(root / "ref", argmax=[1, 2, 3, 40, 41, 7], margin=[1.0, 1.0, 1.0, 0.5, 0.002, 1.0])
        write_engine(root / "eng", range(1), self.PARTS, positions=6)
        write_fed(root / "eng", self.FED, argmax=7, choices=engine_choices)
        return run(root / "ref", root / "eng", *extra)

    def test_per_position_argmax_agreement_passes_and_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.choice_run(Path(tmp), [(3, 40, 0.5), (4, 41, 0.002), (5, 7, 1.0)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("per-position argmax: 3 of 3 compared positions agree", result.stdout)

    def test_a_per_position_argmax_mismatch_fails_and_names_the_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.choice_run(Path(tmp), [(3, 99, 0.4), (4, 41, 0.002), (5, 7, 1.0)])
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("argmax mismatch, argmax after position 3: engine 99 (lead 0.4), reference 40 (lead 0.5)", result.stdout)
            self.assertIn("per-position argmax: 2 of 3 compared positions agree", result.stdout)

    def test_choice_margin_turns_a_near_tie_mismatch_into_a_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            choices = [(3, 40, 0.5), (4, 99, 0.3), (5, 7, 1.0)]
            self.assertEqual(self.choice_run(Path(tmp) / "strict", choices).returncode, 1)
            result = self.choice_run(Path(tmp) / "loose", choices, "--choice-margin", "0.01")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("near tie, argmax after position 4", result.stdout)

    def test_choices_on_one_side_only_say_why_they_were_not_compared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6, tokens=self.FED, logits_argmax=7)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7, choices=[(5, 7, 1.0)])
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("per-position argmax not compared: the reference has no choices.json", result.stdout)

    def test_reference_layers_of_mixed_length_fail(self) -> None:
        # An interrupted shorter rerun into the same folder: layer 0 holds the 3-token prefill, layer 1 the full 6 positions.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=3)
            np.save(root / "ref" / "layer_01_all.npy", stream(1, 6))
            write_engine(root / "eng", range(2), self.PARTS, positions=6)
            result = run(root / "ref", root / "eng", "--expect-tokens", "6")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("reference layers cover different numbers of positions", result.stdout)
            self.assertIn("reference layers [0] do not cover 6 positions", result.stdout)

    def test_extra_fed_tokens_fail_expect_tokens(self) -> None:
        # The engine ran 6 positions; a 3-position reference passes as a prefill check, but not when 3 positions are the declared total.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=3, tokens=self.FED[:3])
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7)
            loose = run(root / "ref", root / "eng")
            self.assertEqual(loose.returncode, 0, loose.stdout + loose.stderr)
            self.assertIn("the engine covers 6 positions; compared the first 3", loose.stdout)
            result = run(root / "ref", root / "eng", "--expect-tokens", "3")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("the engine was fed 6 tokens, the reference ran on 3", result.stdout)

    def test_a_skipped_argmax_check_says_why(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6, tokens=self.FED)   # no --logits
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            write_fed(root / "eng", self.FED, argmax=7)
            result = run(root / "ref", root / "eng")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("argmax not compared: the reference has no logits_last.npy", result.stdout)

    def test_position_threshold_loosens_only_the_per_position_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            # Nudge position 4 so its cosine falls a little below 0.999 while the whole stream stays above it.
            ref4 = stream(0, 6)[4]
            noisy = ref4 + 0.06 * np.random.default_rng(5).standard_normal(ref4.shape).astype("<f4")
            noisy.astype("<f4").tofile(root / "eng" / f"layer00_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            self.assertEqual(run(root / "ref", root / "eng").returncode, 1)
            result = run(root / "ref", root / "eng", "--position-threshold", "0.99")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_tie_margin_leaves_out_only_near_tie_positions_from_their_dsa_layer_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(3), positions=6)
            write_engine(root / "eng", range(3), self.PARTS, positions=6)
            # Layer 1 is a DSA layer whose query 4 kept and dropped keys 1e-4 apart; query 5's gap is wide.
            (root / "ref" / "index_gaps.json").write_text(json.dumps({"1": {"4": 1e-4, "5": 0.2}}), encoding="utf-8")
            for layer in (0, 1, 2):
                np.random.default_rng(99 + layer).standard_normal((1, HC, HIDDEN)).astype("<f4").tofile(
                    root / "eng" / f"layer{layer:02d}_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            # Position 4 is wrong in every layer: layer 0 precedes the DSA layer, so it stays flagged.
            result = run(root / "ref", root / "eng", "--tie-margin", "1e-3", "--threshold", "0.5", "--max-tie-fraction", "1")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("first layer below cosine 0.5: 0", result.stdout)
            self.assertIn("layer 1: left out 1 near-tie positions [4]", result.stdout)
            self.assertNotIn("layer 0: left out", result.stdout)
            # Correct layer 0: layers 1 and 2 then pass because position 4 is a near tie from layer 1 on.
            stream(0, 6)[4:5].tofile(root / "eng" / f"layer00_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            result = run(root / "ref", root / "eng", "--tie-margin", "1e-3", "--threshold", "0.5", "--max-tie-fraction", "1")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("layer 2: left out 1 near-tie positions [4]", result.stdout)
            # A margin below the gap leaves nothing out.
            self.assertEqual(run(root / "ref", root / "eng", "--tie-margin", "1e-5", "--threshold", "0.5", "--max-tie-fraction", "1").returncode, 1)

    def test_a_flag_after_a_near_tie_names_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(3), positions=6)
            write_engine(root / "eng", range(3), self.PARTS, positions=6)
            (root / "ref" / "index_gaps.json").write_text(json.dumps({"1": {"3": 1e-6, "4": 0.2, "5": 0.3}}), encoding="utf-8")
            # Position 4 is wrong in the tie's own DSA layer (1) and in the next layer (2).
            for layer in (1, 2):
                np.random.default_rng(7 + layer).standard_normal((1, HC, HIDDEN)).astype("<f4").tofile(
                    root / "eng" / f"layer{layer:02d}_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            result = run(root / "ref", root / "eng", "--tie-margin", "1e-4", "--threshold", "0.5", "--max-tie-fraction", "1")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            rows = {line.split()[0]: line for line in result.stdout.splitlines() if line.strip()[:1].isdigit() and "FLAG" in line}
            self.assertNotIn("near-tie", rows["1"])   # layer 1's later positions read its inputs, not position 3's output
            self.assertIn("(after near-tie at 3)", rows["2"])

    def test_too_many_near_ties_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            (root / "ref" / "index_gaps.json").write_text(json.dumps({"0": {"4": 1e-4, "5": 0.2}}), encoding="utf-8")
            result = run(root / "ref", root / "eng", "--tie-margin", "1e-3")   # 1 of 2 selecting positions, above 0.05
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("indexer near ties below 0.001: 1 of 2 positions", result.stdout)
            self.assertIn("1 of 2 indexer positions are near ties, more than --max-tie-fraction 0.05", result.stdout)

    def test_a_router_near_tie_is_left_out_like_an_indexer_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(2), positions=6)
            write_engine(root / "eng", range(2), self.PARTS, positions=6)
            # Every position routes; position 4's last kept and first dropped experts are 1e-6 apart in layer 1.
            gaps = {"1": {str(p): (1e-6 if p == 4 else 0.3) for p in range(6)}}
            (root / "ref" / "router_gaps.json").write_text(json.dumps(gaps), encoding="utf-8")
            np.random.default_rng(3).standard_normal((1, HC, HIDDEN)).astype("<f4").tofile(
                root / "eng" / f"layer01_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            self.assertEqual(run(root / "ref", root / "eng", "--threshold", "0.5").returncode, 1)
            # --tie-margin covers the indexer only; the router has its own margin.
            self.assertEqual(run(root / "ref", root / "eng", "--threshold", "0.5", "--tie-margin", "1e-4").returncode, 1)
            result = run(root / "ref", root / "eng", "--threshold", "0.5", "--router-tie-margin", "1e-4", "--max-tie-fraction", "0.2")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("router near ties below 0.0001: 1 of 6 positions", result.stdout)

    def test_explain_names_the_first_drop_and_its_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(3), positions=6)
            write_engine(root / "eng", range(3), self.PARTS, positions=6)
            # Position 4 has the smallest router gap in layer 1, and a smaller one still in layer 0, two layers before its first drop
            # at layer 2 (layer 1 is clean): only layers 1 and 2 can explain that drop, so layer 0's gap is not the one named.
            (root / "ref" / "router_gaps.json").write_text(json.dumps({
                "0": {str(p): (1e-9 if p == 4 else 0.3) for p in range(6)},
                "1": {str(p): (1e-6 if p == 4 else 0.3) for p in range(6)},
                "2": {str(p): 0.3 for p in range(6)}}), encoding="utf-8")
            for layer in (2,):
                np.random.default_rng(11 + layer).standard_normal((1, HC, HIDDEN)).astype("<f4").tofile(
                    root / "eng" / f"layer{layer:02d}_pos4_t1_hc{HC}_h{HIDDEN}.f32")
            result = run(root / "ref", root / "eng", "--threshold", "0.5", "--explain")
            self.assertIn("positions below 0.5: 1", result.stdout)
            line = next(l for l in result.stdout.splitlines() if l.startswith("position 4:"))
            self.assertIn("first below at layer 2", line)
            self.assertIn("router gap 1.00e-06 at layer 1 (smallest 16.7% there)", line)
            self.assertNotIn("1.00e-09", line)

    def test_tie_margin_without_index_gaps_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ref(root / "ref", range(1), positions=6)
            write_engine(root / "eng", range(1), self.PARTS, positions=6)
            result = run(root / "ref", root / "eng", "--tie-margin", "1e-3")
            self.assertEqual(result.returncode, 1)
            self.assertIn("index_gaps.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
