"""Lazy weight access so the UNMODIFIED upstream model can run on a checkpoint far larger than RAM.

Upstream allocates every parameter with `torch.empty`: 384 routed experts per layer and a 98 GB Engram table per Engram layer cannot exist
as Parameters here. These shims keep the upstream forward arithmetic and read only what a forward pass touches:

  SafetensorsDir          header parse + pread of exact byte ranges (safe_open mmaps whole shards and fails on the 100 GB one)
  LazyRoutedExpert        a routed expert loaded on first use into a small LRU; `MoE.forward` already skips unrouted experts
  LazyEngramEmbedding     gathers only the hashed rows (and their E8M0 scales) of the FP8 table, then upstream's dequantization
  LazyWoA                 `wo_a` kept FP8 + scale and dequantized per call (upstream's convert.py widens it to bf16)
  Bf16Embedding / Bf16Head  embed and head kept bf16 (the checkpoint dtype) instead of widened to float32

`install` patches the upstream module before `Transformer` is built; `replace_wo_a` and `bind_lazy` finish the job afterwards.
"""
import json
import os
import struct
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import kernel_ports

DTYPES = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I8": torch.int8, "U8": torch.uint8,
    "I32": torch.int32, "I64": torch.int64, "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
}
ITEMSIZE = {"BF16": 2, "F32": 4, "F16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1}
CHUNK = 1 << 30  # preadv returns short reads past 2 GiB, so read in 1 GiB pieces


def pread_exact(fd: int, nbytes: int, offset: int) -> bytearray:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    got = 0
    while got < nbytes:
        n = os.preadv(fd, [view[got:got + CHUNK]], offset + got)
        if n <= 0:
            raise EOFError(f"short read at {offset + got} of {offset + nbytes}")
        got += n
    return buf


class SafetensorsDir:
    """A sharded safetensors checkpoint read by byte range. Never maps a shard."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as fh:
            self.weight_map: dict = json.load(fh)["weight_map"]
        self._headers: dict = {}
        self._fds: dict = {}

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._headers.clear()
        for key in [k for k in LazyRoutedExpert.cache if k[0] == self.path]:
            del LazyRoutedExpert.cache[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _shard(self, shard: str):
        if shard not in self._headers:
            fd = os.open(os.path.join(self.path, shard), os.O_RDONLY)
            try:
                prefix = os.pread(fd, 8, 0)
                if len(prefix) != 8:
                    raise EOFError(f"{shard}: file ends inside the 8-byte header length")
                (n,) = struct.unpack("<Q", prefix)
                raw = os.pread(fd, n, 8)
                if len(raw) != n:
                    raise EOFError(f"{shard}: file ends inside its {n}-byte header")
                header = json.loads(raw)
            except BaseException:
                os.close(fd)
                raise
            header.pop("__metadata__", None)
            self._fds[shard] = fd
            self._headers[shard] = (8 + n, header)
        return self._headers[shard]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def info(self, name: str):
        """(shard, dtype string, shape, absolute offset, byte count)."""
        shard = self.weight_map[name]
        data_start, header = self._shard(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        return shard, entry["dtype"], tuple(entry["shape"]), data_start + begin, end - begin

    def read(self, name: str) -> torch.Tensor:
        shard, dtype, shape, offset, nbytes = self.info(name)
        buf = pread_exact(self._fds[shard], nbytes, offset)
        return torch.frombuffer(buf, dtype=torch.uint8).view(DTYPES[dtype]).reshape(shape)

    def read_rows(self, name: str, rows) -> torch.Tensor:
        """Rows `rows` (in the given order) of a tensor whose first dimension is the row index."""
        shard, dtype, shape, offset, _ = self.info(name)
        row_bytes = ITEMSIZE[dtype]
        for d in shape[1:]:
            row_bytes *= d
        fd = self._fds[shard]
        buf = bytearray(len(rows) * row_bytes)
        for i, r in enumerate(rows):
            if not 0 <= r < shape[0]:
                raise IndexError(f"{name}: row {r} outside [0, {shape[0]})")
            piece = os.pread(fd, row_bytes, offset + r * row_bytes)
            # a short read would shift every later row in the buffer, so it must fail here
            if len(piece) != row_bytes:
                raise EOFError(f"{name}: row {r} read {len(piece)} of {row_bytes} bytes")
            buf[i * row_bytes:(i + 1) * row_bytes] = piece
        return torch.frombuffer(buf, dtype=torch.uint8).view(DTYPES[dtype]).reshape(len(rows), *shape[1:])


class LazyRoutedExpert(nn.Module):
    """A routed (FP4) expert whose weights are read from the checkpoint the first time a token is routed to it."""

    cache: "OrderedDict" = OrderedDict()
    capacity: int = 48

    def __init__(self, expert_cls, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self._build = lambda: expert_cls(dim, inter_dim, dtype=torch.float4_e2m1fn_x2, swiglu_limit=swiglu_limit)
        self.source = None
        self.layer = -1
        self.index = -1

    def bind(self, source: SafetensorsDir, layer, index: int) -> None:
        """`layer` is a backbone layer number, or the checkpoint prefix of a DSpark stage such as "mtp.0"."""
        base = layer if isinstance(layer, str) else f"layers.{layer}"
        for w in ("w1", "w2", "w3"):
            for part in ("weight", "scale"):
                key = f"{base}.ffn.experts.{index}.{w}.{part}"
                if not source.has(key):
                    raise KeyError(f"checkpoint has no {key}")
        self.source, self.layer, self.index = source, base, index

    def _inner(self) -> nn.Module:
        key = (self.source.path, self.layer, self.index)
        inner = LazyRoutedExpert.cache.get(key)
        if inner is not None:
            LazyRoutedExpert.cache.move_to_end(key)
            return inner
        inner = self._build()
        for w in ("w1", "w2", "w3"):
            linear = getattr(inner, w)
            base = f"{self.layer}.ffn.experts.{self.index}.{w}"
            weight = self.source.read(f"{base}.weight")
            linear.weight.data = weight.view(torch.float4_e2m1fn_x2) if weight.dtype == torch.int8 else weight
            linear.scale.data = self.source.read(f"{base}.scale")
        LazyRoutedExpert.cache[key] = inner
        while len(LazyRoutedExpert.cache) > LazyRoutedExpert.capacity:
            LazyRoutedExpert.cache.popitem(last=False)
        return inner

    def forward(self, x: torch.Tensor, weights=None) -> torch.Tensor:
        return self._inner()(x, weights)


class LazyEngramEmbedding(nn.Module):
    """The n-gram hash table read by row. Same arithmetic as upstream's `ParallelEngramEmbedding` at world_size 1."""

    block_size = 32

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.source = None
        self.weight_name = self.scale_name = ""

    def bind(self, source: SafetensorsDir, weight_name: str, scale_name: str) -> None:
        _, _, wshape, _, _ = source.info(weight_name)
        _, _, sshape, _, _ = source.info(scale_name)
        if wshape != (self.num_embeddings, self.dim) or sshape != (self.num_embeddings, self.dim // self.block_size):
            raise ValueError(f"{weight_name}: checkpoint shapes {wshape}/{sshape} do not match the model ({self.num_embeddings}, {self.dim})")
        self.source, self.weight_name, self.scale_name = source, weight_name, scale_name

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < 0) | (indices >= self.num_embeddings)
        local = indices.masked_fill(mask, 0)
        unique, inverse = torch.unique(local.flatten(), return_inverse=True)
        rows = unique.tolist()
        weight = self.source.read_rows(self.weight_name, rows)
        scale = self.source.read_rows(self.scale_name, rows)
        values = weight.float().unflatten(-1, (-1, self.block_size)) * scale.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)[inverse].reshape(*indices.shape, self.dim)
        return values.masked_fill(mask.unsqueeze(-1), 0)


class LazyWoA(nn.Module):
    """`wo_a` stored FP8 + block scale, dequantized on access to `out_dtype` (upstream holds it widened)."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
        super().__init__()
        self._w, self._s, self._dtype = weight, scale, out_dtype

    @property
    def weight(self) -> torch.Tensor:
        return kernel_ports.dequant_fp8_block(self._w, self._s, 32, 32).to(self._dtype)


class Bf16Embedding(nn.Module):
    """Embedding kept in the checkpoint's bf16; the lookup is widened to the model's default dtype (exact, bf16 -> float32)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight).to(torch.get_default_dtype())


class Bf16Head(nn.Module):
    """Output head kept bf16; logits are float32 as upstream's, widened in row chunks so the float32 copy is never resident."""

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        x = x.float()
        return torch.cat([F.linear(x, self.weight[i:i + 16384].float()) for i in range(0, self.vocab_size, 16384)], dim=-1)


def install(mod, lru_capacity: int = 48) -> None:
    """Swap the memory-hungry upstream classes for their lazy equivalents. Call before `Transformer(...)`."""
    expert_cls = mod.Expert
    LazyRoutedExpert.capacity = lru_capacity

    def expert_factory(dim, inter_dim, dtype=None, swiglu_limit=0.0):
        if dtype == torch.float4_e2m1fn_x2:
            return LazyRoutedExpert(expert_cls, dim, inter_dim, swiglu_limit)
        return expert_cls(dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)

    mod.Expert = expert_factory
    mod.ParallelEngramEmbedding = LazyEngramEmbedding
    mod.ParallelEmbedding = Bf16Embedding
    mod.ParallelHead = Bf16Head


def replace_wo_a(model, source: SafetensorsDir, out_dtype: torch.dtype) -> set:
    """Swap every layer's `wo_a` for a lazily dequantized one; returns the checkpoint keys it consumed."""
    consumed = set()
    for layer, blk in enumerate(model.layers):
        w_name, s_name = f"layers.{layer}.attn.wo_a.weight", f"layers.{layer}.attn.wo_a.scale"
        weight, scale = source.read(w_name), source.read(s_name)
        want = tuple(blk.attn.wo_a.weight.shape)
        if tuple(weight.shape) != want:
            raise ValueError(f"{w_name}: checkpoint shape {tuple(weight.shape)} != model shape {want}")
        blk.attn.wo_a = LazyWoA(weight, scale, out_dtype)
        consumed.update((w_name, s_name))
    return consumed


def bind_lazy(model, source: SafetensorsDir) -> set:
    """Point every lazy expert and Engram table at the checkpoint; returns the checkpoint keys they may read."""
    keys = set()
    for layer, blk in enumerate(model.layers):
        for index, expert in enumerate(blk.ffn.experts):
            expert.bind(source, layer, index)
            keys.update(f"layers.{layer}.ffn.experts.{index}.{w}.{p}" for w in ("w1", "w2", "w3") for p in ("weight", "scale"))
        if blk.engram is not None:
            w_name, s_name = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
            blk.engram.embed.bind(source, w_name, s_name)
            keys.update((w_name, s_name))
    return keys
