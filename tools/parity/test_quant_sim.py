"""Tests for quant_sim.py. Run: python -I tools/parity/test_quant_sim.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_sim as qs  # noqa: E402


class HadamardTests(unittest.TestCase):
    def test_matches_the_sylvester_matrix(self) -> None:
        h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        h8 = torch.kron(torch.kron(h2, h2), h2) / 8 ** 0.5
        x = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(qs.hadamard(x), x @ h8.T)

    def test_is_orthonormal_so_dot_products_survive(self) -> None:
        g = torch.Generator().manual_seed(2)
        q, k = torch.randn(4, 128, generator=g), torch.randn(5, 128, generator=g)
        torch.testing.assert_close(qs.hadamard(q) @ qs.hadamard(k).T, q @ k.T, rtol=1e-5, atol=1e-4)

    def test_width_must_be_a_power_of_two(self) -> None:
        with self.assertRaises(ValueError):
            qs.hadamard(torch.zeros(2, 6))


class Fp8Tests(unittest.TestCase):
    def test_representable_values_with_a_power_of_two_scale_survive(self) -> None:
        # amax 448 * 2^-3 gives scale 2^-3; every value is an E4M3 number times that scale.
        x = torch.tensor([[448.0, -1.0, 0.5, 3.5] * 32]) / 8
        torch.testing.assert_close(qs.fp8_roundtrip_ue8m0(x), x, rtol=0, atol=0)

    def test_error_stays_within_e4m3_rounding(self) -> None:
        x = torch.randn(6, 256, generator=torch.Generator().manual_seed(3)) * 10
        got = qs.fp8_roundtrip_ue8m0(x)
        groups = x.reshape(-1, 128)
        scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True) / 448)))
        # E4M3 keeps 3 mantissa bits: half an ulp is at most 1/16 relative, or 2^-10 * scale in the subnormal range.
        bound = torch.maximum(groups.abs() / 16, scale * 2.0 ** -10).reshape(x.shape)
        self.assertTrue(bool(((got - x).abs() <= bound + 1e-12).all()))

    def test_each_group_has_its_own_scale(self) -> None:
        x = torch.cat([torch.full((1, 128), 1000.0), torch.full((1, 128), 0.001)], dim=1)
        got = qs.fp8_roundtrip_ue8m0(x)
        self.assertGreater(float(got[0, 200]), 0.0)

    def test_width_must_divide_into_blocks(self) -> None:
        with self.assertRaises(ValueError):
            qs.fp8_roundtrip_ue8m0(torch.zeros(1, 100))


if __name__ == "__main__":
    unittest.main()
