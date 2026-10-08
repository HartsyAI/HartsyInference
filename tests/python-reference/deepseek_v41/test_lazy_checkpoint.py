"""Checks the lazy shims in lazy_checkpoint.py against the real upstream classes on small synthetic checkpoints.

Run from this directory: ~/dsv41-ref/.venv/bin/python -m unittest -v test_lazy_checkpoint
Needs the upstream `inference/` files (setup_env.sh fetches them); the upstream-comparison tests skip without them.
"""
import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports
import lazy_checkpoint as lc  # noqa: E402

UPSTREAM = os.path.expanduser("~/dsv41-ref/upstream/inference")


def upstream_model():
    if not os.path.isdir(UPSTREAM):
        raise unittest.SkipTest("upstream inference/ not fetched (run setup_env.sh)")
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    import model as mod
    return mod


def write_checkpoint(root: str, shards: dict) -> None:
    """shards: {shard file name: {tensor name: tensor}} -> safetensors shards plus an index."""
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(root, shard))
        weight_map.update({name: shard for name in tensors})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": weight_map}, fh)


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def random_e8m0(shape, generator) -> torch.Tensor:
    return torch.randint(124, 130, shape, generator=generator, dtype=torch.uint8).view(torch.float8_e8m0fnu)


class SafetensorsDirTests(unittest.TestCase):
    def test_reads_every_dtype_and_rows_by_byte_range(self):
        g = torch.Generator().manual_seed(1)
        tensors = {
            "a.bf16": torch.randn(7, 5, generator=g).to(torch.bfloat16),
            "a.f32": torch.randn(3, 4, generator=g),
            "b.fp8": (torch.randn(9, 64, generator=g) * 3).to(torch.float8_e4m3fn),
            "b.e8m0": random_e8m0((9, 2), g),
            "b.i8": torch.randint(-128, 127, (6, 8), generator=g, dtype=torch.int8),
        }
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(root, {"s1.safetensors": {k: v for k, v in tensors.items() if k.startswith("a")},
                                    "s2.safetensors": {k: v for k, v in tensors.items() if k.startswith("b")}})
            src = lc.SafetensorsDir(root)
            for name, expected in tensors.items():
                got = src.read(name)
                self.assertEqual(got.dtype, expected.dtype, name)
                self.assertTrue(torch.equal(raw(got), raw(expected)), name)
            rows = [4, 0, 8, 4]
            got = src.read_rows("b.fp8", rows)
            self.assertTrue(torch.equal(raw(got), raw(tensors["b.fp8"][rows])))
            with self.assertRaises(IndexError):
                src.read_rows("b.fp8", [9])


class LazyShimTests(unittest.TestCase):
    def test_engram_rows_match_upstream_embedding(self):
        mod = upstream_model()
        g = torch.Generator().manual_seed(2)
        rows, dim = 50, 64
        weight = (torch.randn(rows, dim, generator=g) * 2).to(torch.float8_e4m3fn)
        scale = random_e8m0((rows, dim // 32), g)
        upstream = mod.ParallelEngramEmbedding(rows, dim)
        upstream.weight.data, upstream.scale.data = weight, scale
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(root, {"s.safetensors": {"e.weight": weight, "e.scale": scale}})
            lazy = lc.LazyEngramEmbedding(rows, dim)
            lazy.bind(lc.SafetensorsDir(root), "e.weight", "e.scale")
            indices = torch.tensor([[3, 49, 3, 0], [17, 17, 49, 8]])
            self.assertTrue(torch.equal(lazy(indices), upstream(indices)))
            with self.assertRaises(ValueError):
                wrong = lc.LazyEngramEmbedding(rows + 1, dim)
                wrong.bind(lc.SafetensorsDir(root), "e.weight", "e.scale")

    def test_routed_expert_matches_upstream_expert(self):
        mod = upstream_model()
        g = torch.Generator().manual_seed(3)
        dim, inter = 64, 96
        tensors = {}
        reference = mod.Expert(dim, inter, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0)
        for w, (out_f, in_f) in {"w1": (inter, dim), "w2": (dim, inter), "w3": (inter, dim)}.items():
            packed = torch.randint(-128, 127, (out_f, in_f // 2), generator=g, dtype=torch.int8)
            scale = random_e8m0((out_f, in_f // 32), g)
            tensors[f"layers.0.ffn.experts.5.{w}.weight"] = packed
            tensors[f"layers.0.ffn.experts.5.{w}.scale"] = scale
            lin = getattr(reference, w)
            lin.weight.data, lin.scale.data = packed.view(torch.float4_e2m1fn_x2), scale
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(root, {"s.safetensors": tensors})
            lc.LazyRoutedExpert.cache.clear()
            lazy = lc.LazyRoutedExpert(mod.Expert, dim, inter, 10.0)
            lazy.bind(lc.SafetensorsDir(root), 0, 5)
            x = torch.randn(4, dim, generator=g).to(torch.bfloat16)
            weights = torch.rand(4, 1, generator=g)
            torch.set_default_dtype(torch.bfloat16)
            try:
                self.assertTrue(torch.equal(lazy(x, weights), reference(x, weights)))
            finally:
                torch.set_default_dtype(torch.float32)
            with self.assertRaises(KeyError):
                lc.LazyRoutedExpert(mod.Expert, dim, inter, 10.0).bind(lc.SafetensorsDir(root), 0, 6)

    def test_expert_lru_evicts_oldest(self):
        mod = upstream_model()
        g = torch.Generator().manual_seed(4)
        tensors = {}
        for i in range(3):
            for w, (o, n) in {"w1": (32, 32), "w2": (32, 32), "w3": (32, 32)}.items():
                tensors[f"layers.0.ffn.experts.{i}.{w}.weight"] = torch.randint(-128, 127, (o, n // 2), generator=g, dtype=torch.int8)
                tensors[f"layers.0.ffn.experts.{i}.{w}.scale"] = random_e8m0((o, n // 32), g)
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(root, {"s.safetensors": tensors})
            src = lc.SafetensorsDir(root)
            lc.LazyRoutedExpert.cache.clear()
            old_capacity, lc.LazyRoutedExpert.capacity = lc.LazyRoutedExpert.capacity, 2
            try:
                experts = [lc.LazyRoutedExpert(mod.Expert, 32, 32, 0.0) for _ in range(3)]
                for i, e in enumerate(experts):
                    e.bind(src, 0, i)
                    e._inner()
                self.assertEqual(len(lc.LazyRoutedExpert.cache), 2)
                self.assertNotIn((id(src), 0, 0), lc.LazyRoutedExpert.cache)
            finally:
                lc.LazyRoutedExpert.capacity = old_capacity
                lc.LazyRoutedExpert.cache.clear()

    def test_wo_a_dequantizes_like_the_block_codec(self):
        g = torch.Generator().manual_seed(5)
        weight = (torch.randn(64, 96, generator=g) * 4).to(torch.float8_e4m3fn)
        scale = random_e8m0((2, 3), g)
        expected = kernel_ports.dequant_fp8_block(weight, scale, 32, 32)
        self.assertTrue(torch.equal(lc.LazyWoA(weight, scale, torch.float32).weight, expected))
        self.assertTrue(torch.equal(lc.LazyWoA(weight, scale, torch.bfloat16).weight, expected.to(torch.bfloat16)))

    def test_bf16_head_and_embedding_match_widened_float32(self):
        g = torch.Generator().manual_seed(6)
        vocab, dim = 40000, 48
        w = torch.randn(vocab, dim, generator=g).to(torch.bfloat16)
        head = lc.Bf16Head(vocab, dim)
        head.weight.data = w
        x = torch.randn(1, 3, dim, generator=g)
        torch.testing.assert_close(head(x), F.linear(x[:, -1].float(), w.float()), rtol=1e-5, atol=1e-5)
        emb = lc.Bf16Embedding(vocab, dim)
        emb.weight.data = w
        ids = torch.tensor([[0, 39999, 17]])
        self.assertTrue(torch.equal(emb(ids), F.embedding(ids, w.float())))


if __name__ == "__main__":
    unittest.main()
