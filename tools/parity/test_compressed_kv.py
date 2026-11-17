"""Tests for compressed_kv.py. Run: python -I tools/parity/test_compressed_kv.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compressed_kv as ckv  # noqa: E402


def decode_steps(kv: torch.Tensor, score: torch.Tensor, ratio: int, ape: torch.Tensor) -> torch.Tensor:
    """The source's decode branch, one token per call: fill slot start_pos % ratio, pool when (start_pos + 1) % ratio == 0."""
    d = kv.shape[1]
    kv_state = torch.zeros(ratio, d)
    score_state = torch.full((ratio, d), float("-inf"))
    out = []
    for p in range(kv.shape[0]):
        slot = p % ratio
        kv_state[slot] = kv[p]
        score_state[slot] = score[p] + ape[slot]
        if (p + 1) % ratio == 0:
            out.append((kv_state * score_state.softmax(dim=0)).sum(dim=0))
    return torch.stack(out) if out else torch.zeros(0, d)


class PoolTests(unittest.TestCase):
    def setUp(self) -> None:
        g = torch.Generator().manual_seed(3)
        self.ratio, self.d = 4, 6
        self.kv = torch.randn(11, self.d, generator=g)
        self.score = torch.randn(11, self.d, generator=g)
        self.ape = torch.randn(self.ratio, self.d, generator=g)

    def test_one_latent_per_completed_group(self) -> None:
        self.assertEqual(tuple(ckv.pool(self.kv, self.score, self.ratio).shape), (2, self.d))
        self.assertEqual(tuple(ckv.pool(self.kv[:3], self.score[:3], self.ratio).shape), (0, self.d))

    def test_softmax_runs_over_the_group_per_channel(self) -> None:
        got = ckv.pool(self.kv, self.score, self.ratio, self.ape)
        for gi in range(2):
            rows = slice(gi * self.ratio, (gi + 1) * self.ratio)
            for c in range(self.d):
                w = torch.softmax(self.score[rows, c] + self.ape[:, c], dim=0)
                self.assertAlmostEqual(float((w * self.kv[rows, c]).sum()), float(got[gi, c]), places=5)

    def test_prefill_equals_the_decode_steps(self) -> None:
        torch.testing.assert_close(ckv.pool(self.kv, self.score, self.ratio, self.ape),
                                   decode_steps(self.kv, self.score, self.ratio, self.ape))

    def test_a_trailing_partial_group_changes_nothing(self) -> None:
        torch.testing.assert_close(ckv.pool(self.kv[:8], self.score[:8], self.ratio, self.ape),
                                   ckv.pool(self.kv, self.score, self.ratio, self.ape))

    def test_ape_shape_and_ratio_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            ckv.pool(self.kv, self.score, self.ratio, self.ape[:, :2])
        with self.assertRaises(ValueError):
            ckv.pool(self.kv, self.score, 1)


class CompressTests(unittest.TestCase):
    def test_projects_pools_and_normalizes(self) -> None:
        g = torch.Generator().manual_seed(5)
        x, wkv, wgate = torch.randn(9, 10, generator=g), torch.randn(6, 10, generator=g), torch.randn(6, 10, generator=g)
        norm, ape = torch.rand(6, generator=g) + 0.5, torch.randn(4, 6, generator=g)
        pooled = ckv.pool(x @ wkv.T, x @ wgate.T, 4, ape)
        want = pooled / torch.sqrt(pooled.pow(2).mean(-1, keepdim=True) + 1e-6) * norm
        torch.testing.assert_close(ckv.compress(x, wkv, wgate, 4, norm, 1e-6, ape), want)

    def test_overlap_is_refused(self) -> None:
        with self.assertRaises(NotImplementedError):
            ckv.compress(torch.zeros(4, 2), torch.zeros(2, 2), torch.zeros(2, 2), 4, torch.ones(2), 1e-6, overlap=True)


class PositionTests(unittest.TestCase):
    def test_group_starts(self) -> None:
        self.assertEqual(ckv.group_starts(3, 128).tolist(), [0, 128, 256])

    def test_a_latent_is_visible_from_its_groups_last_token(self) -> None:
        self.assertEqual([ckv.visible(p, 4) for p in range(9)], [0, 0, 0, 1, 1, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
