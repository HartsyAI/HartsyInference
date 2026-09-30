"""Self-checks for kernel_ports.py against independent scalar implementations. Run: python -m unittest -v"""
import math
import unittest

import torch

import kernel_ports as kp


def scalar_e2m1_rne(v: float) -> int:
    """Independent round-to-nearest-even onto the E2M1 grid, returning the 4-bit code."""
    sign: int = 8 if (v < 0 or (v == 0 and math.copysign(1.0, v) < 0)) else 0
    a: float = min(abs(v), 6.0)
    grid = kp.E2M1_MAGNITUDES
    best: int = 0
    for i in range(1, len(grid)):
        lo, hi = grid[i - 1], grid[i]
        if a >= lo and a <= hi:
            mid = (lo + hi) / 2
            if a < mid:
                best = i - 1
            elif a > mid:
                best = i
            else:
                best = i - 1 if (i - 1) % 2 == 0 else i
            break
    return sign | best


class E2M1Tests(unittest.TestCase):
    def test_ties_and_random_match_scalar(self) -> None:
        ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 0.0, -0.0, 6.0, 7.5, -9.0, 0.24999, 0.25001]
        g = torch.Generator().manual_seed(1)
        rand = (torch.randn(4096, generator=g) * 3).tolist()
        vals = torch.tensor(ties + rand, dtype=torch.float32).clamp(-6, 6)
        got = kp.e2m1_encode(vals).tolist()
        want = [scalar_e2m1_rne(float(v)) for v in vals]
        self.assertEqual(got, want)

    def test_documented_ties(self) -> None:
        expect = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}
        for x, y in expect.items():
            self.assertEqual(float(kp.e2m1_decode(kp.e2m1_encode(torch.tensor([x])))[0]), y, x)

    def test_pack_low_nibble_is_even_element(self) -> None:
        codes = torch.tensor([[1, 2, 15, 8]], dtype=torch.uint8)
        packed = kp.pack_e2m1(codes)
        self.assertEqual(packed.tolist(), [[0x21, 0x8F]])
        self.assertEqual(kp.unpack_e2m1(packed).tolist(), [[0.5, 1.0, -6.0, -0.0]])


class ActQuantTests(unittest.TestCase):
    def test_pow2_scale_is_power_of_two_and_covers_amax(self) -> None:
        g = torch.Generator().manual_seed(2)
        x = torch.randn(8, 256, generator=g) * 10
        q, s = kp.act_quant(x, 128, scale_fmt="ue8m0")
        exp = torch.log2(s)
        self.assertTrue(torch.equal(exp, exp.round()))
        amax = x.reshape(8, 2, 128).abs().amax(-1)
        self.assertTrue(bool((s * 448.0 >= amax - 1e-4).all()))

    def test_zero_block_uses_amax_floor(self) -> None:
        q, s = kp.act_quant(torch.zeros(1, 128), 128)
        self.assertAlmostEqual(float(s[0, 0]), 1e-4 / 448.0, places=12)
        self.assertEqual(float(q.to(torch.float32).abs().max()), 0.0)

    def test_inplace_matches_dequant(self) -> None:
        g = torch.Generator().manual_seed(3)
        x = torch.randn(4, 128, generator=g).to(torch.bfloat16)
        ref = x.clone()
        q, s = kp.act_quant(ref, 128, scale_fmt="ue8m0")
        want = (q.to(torch.float32) * s.repeat_interleave(128, -1)).to(torch.bfloat16)
        kp.act_quant(x, 128, scale_fmt="ue8m0", inplace=True)
        self.assertTrue(torch.equal(x, want))


class Fp4QuantTests(unittest.TestCase):
    def test_roundtrip_error_bounded_e8m0(self) -> None:
        g = torch.Generator().manual_seed(4)
        x = torch.randn(16, 128, generator=g)
        packed, s = kp.fp4_act_quant(x, 32)
        self.assertEqual(packed.shape, (16, 64))
        deq = kp.unpack_e2m1(packed) * s.to(torch.float32).repeat_interleave(32, -1)
        blk = x.reshape(16, 4, 32).abs().amax(-1).repeat_interleave(32, -1)
        self.assertTrue(bool(((deq - x).abs() <= blk * 0.34 + 1e-6).all()))

    def test_e4m3_scale_path(self) -> None:
        g = torch.Generator().manual_seed(5)
        x = torch.randn(4, 64, generator=g)
        packed, s = kp.fp4_act_quant(x, 16, scale_dtype=torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.float8_e4m3fn)
        y = x.clone()
        kp.fp4_act_quant(y, 16, inplace=True, scale_dtype=torch.float8_e4m3fn)
        deq = kp.unpack_e2m1(packed) * s.to(torch.float32).repeat_interleave(16, -1)
        self.assertTrue(torch.equal(deq, y))

    def test_quantized_grid_is_fixed_point(self) -> None:
        g = torch.Generator().manual_seed(6)
        x = torch.randn(2, 64, generator=g)
        once = x.clone(); kp.fp4_act_quant(once, 32, inplace=True)
        twice = once.clone(); kp.fp4_act_quant(twice, 32, inplace=True)
        self.assertTrue(torch.equal(once, twice))


class GemmTests(unittest.TestCase):
    def test_fp8_gemm_matches_dequant_matmul(self) -> None:
        g = torch.Generator().manual_seed(7)
        a = torch.randn(3, 256, generator=g)
        aq, a_s = kp.act_quant(a, 128)
        w = torch.randn(192, 256, generator=g)
        wq = w.to(torch.float8_e4m3fn)
        ws = torch.rand(2, 2, generator=g) + 0.5
        torch.set_default_dtype(torch.float32)
        got = kp.fp8_gemm(aq, a_s, wq, ws)
        a_f = aq.to(torch.float32) * a_s.repeat_interleave(128, -1)
        w_f = wq.to(torch.float32) * ws.repeat_interleave(128, 0)[:192].repeat_interleave(128, 1)
        self.assertTrue(torch.allclose(got, a_f @ w_f.T, rtol=1e-5, atol=1e-5))

    def test_fp4_gemm_matches_dequant_matmul(self) -> None:
        g = torch.Generator().manual_seed(8)
        a = torch.randn(2, 256, generator=g)
        aq, a_s = kp.act_quant(a, 128)
        wf = torch.randn(64, 256, generator=g)
        packed, s = kp.fp4_act_quant(wf, 32)
        torch.set_default_dtype(torch.float32)
        got = kp.fp4_gemm(aq, a_s, packed, s)
        a_f = aq.to(torch.float32) * a_s.repeat_interleave(128, -1)
        w_f = kp.unpack_e2m1(packed) * s.to(torch.float32).repeat_interleave(32, -1)
        self.assertTrue(torch.allclose(got, a_f @ w_f.T, rtol=1e-5, atol=1e-5))


class SparseAttnTests(unittest.TestCase):
    def _case(self, topk: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        b, m, h, d, n = 2, 3, 4, 32, 300
        q = torch.randn(b, m, h, d, generator=g).to(torch.bfloat16)
        kv = torch.randn(b, n, d, generator=g).to(torch.bfloat16)
        sink = torch.randn(h, generator=g)
        idx = torch.randint(0, n, (b, m, topk), generator=g, dtype=torch.int32)
        idx[..., ::5] = -1
        return q, kv, sink, idx, d**-0.5

    def test_blockwise_matches_float64_within_bf16_probability_error(self) -> None:
        for topk in (7, 64, 130):
            q, kv, sink, idx, sc = self._case(topk, topk)
            got = kp.sparse_attn(q, kv, sink, idx, sc).to(torch.float32)
            want = kp.sparse_attn_exact(q, kv, sink, idx, sc).to(torch.float32)
            self.assertLess(float((got - want).abs().max()), 0.05, topk)

    def test_all_invalid_row_is_zero(self) -> None:
        q, kv, sink, idx, sc = self._case(64, 9)
        idx[0, 1] = -1
        out = kp.sparse_attn(q, kv, sink, idx, sc)
        self.assertEqual(float(out[0, 1].abs().max()), 0.0)

    def test_sink_lowers_magnitude(self) -> None:
        q, kv, sink, idx, sc = self._case(64, 10)
        lo = kp.sparse_attn_exact(q, kv, torch.full_like(sink, -30.0), idx, sc).to(torch.float32).abs().mean()
        hi = kp.sparse_attn_exact(q, kv, torch.full_like(sink, 30.0), idx, sc).to(torch.float32).abs().mean()
        self.assertGreater(float(lo), float(hi))


class SinkhornTests(unittest.TestCase):
    def test_comb_is_near_doubly_stochastic_and_ranges(self) -> None:
        g = torch.Generator().manual_seed(11)
        hc = 4
        mixes = torch.randn(2, 5, (2 + hc) * hc, generator=g)
        pre, post, comb = kp.hc_split_sinkhorn(mixes, torch.tensor([1.0, 1.0, 1.0]), torch.zeros((2 + hc) * hc), hc, 20, 1e-6)
        self.assertEqual(comb.shape, (2, 5, hc, hc))
        self.assertTrue(bool((pre > 0).all()) and bool((pre < 1.01).all()))
        self.assertTrue(bool((post > 0).all()) and bool((post < 2).all()))
        self.assertLess(float((comb.sum(-1) - 1).abs().max()), 1e-3)
        self.assertLess(float((comb.sum(-2) - 1).abs().max()), 1e-3)

    def test_matches_scalar_loop(self) -> None:
        g = torch.Generator().manual_seed(12)
        hc, eps = 4, 1e-6
        mixes = torch.randn((2 + hc) * hc, generator=g)
        scale = torch.tensor([0.5, 1.5, 0.75])
        base = torch.randn((2 + hc) * hc, generator=g)
        pre, post, comb = kp.hc_split_sinkhorn(mixes, scale, base, hc, 20, eps)
        m = [[float(mixes[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k]) for k in range(hc)] for j in range(hc)]
        for j in range(hc):
            mx = max(m[j]); e = [math.exp(v - mx) for v in m[j]]; t = sum(e)
            m[j] = [v / t + eps for v in e]
        cs = [sum(m[j][k] for j in range(hc)) + eps for k in range(hc)]
        m = [[m[j][k] / cs[k] for k in range(hc)] for j in range(hc)]
        for _ in range(19):
            m = [[v / (sum(r) + eps) for v in r] for r in m]
            cs = [sum(m[j][k] for j in range(hc)) + eps for k in range(hc)]
            m = [[m[j][k] / cs[k] for k in range(hc)] for j in range(hc)]
        self.assertTrue(torch.allclose(comb, torch.tensor(m, dtype=torch.float32), atol=1e-6))
        sig = lambda v: 1 / (1 + math.exp(-v))
        want_pre = [sig(float(mixes[i] * scale[0] + base[i])) + eps for i in range(hc)]
        want_post = [2 * sig(float(mixes[hc + i] * scale[1] + base[hc + i])) for i in range(hc)]
        self.assertTrue(torch.allclose(pre, torch.tensor(want_pre), atol=1e-6))
        self.assertTrue(torch.allclose(post, torch.tensor(want_post), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
