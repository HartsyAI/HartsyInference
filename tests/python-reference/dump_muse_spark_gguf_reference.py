#!/usr/bin/env python3
"""Exact per-quant references for the small Muse Spark GGUFs (see make_muse_spark_small_gguf.py).

For each GGUF, every quantized or 16-bit tensor is dequantized with gguf-py (llama.cpp's reference codecs) and written back
into the transformers model in place of the original weight, undoing the converter's Q/K rope permutation and its gate/up
expert split; the F32 tensors (norms, router, residual gates, router bias) are the converter's exact copies, so the model's
own parameters stand in for them. Types gguf-py cannot decode (Q1_0, Q2_0) go through llama.cpp's own C dequantizer when
--ggml-lib names its libggml-base. The float32 logits of that model are what a forward pass over the GGUF's weights must
produce, so they isolate the engine's codecs and graph from llama.cpp's activation quantization. Writes
<name>.dequant.safetensors (input_ids, logits) beside each GGUF.

--llamacpp-activations instead rounds every matmul input the way llama.cpp's CPU backend does before its dot products
(Q8_0/Q8_1 blocks of 32 or Q8_K blocks of 256, chosen by the weight's type as ggml-cpu's vec_dot_type; F16/BF16 weights
round to their own type) and reports the KL divergence of llama.cpp's saved last-token logits (llamacpp/, from llama-debug
--save-logits) against that simulation and against the exact reference: the first being far smaller shows where llama.cpp's
quantized output departs from the exact one.

Run: python dump_muse_spark_gguf_reference.py --small-dir <dir> [--gguf-py <llama.cpp>/gguf-py] [--ggml-lib <libggml-base>]
         [--llamacpp-activations]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import MuseSparkForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--small-dir", required=True)
parser.add_argument("--gguf-py", default=None)
parser.add_argument("--pattern", default="muse-spark-small-*.gguf")
parser.add_argument("--ggml-lib", default=None, help="libggml-base from a llama.cpp build, for the types gguf-py cannot dequantize")
parser.add_argument("--llamacpp-activations", action="store_true",
                    help="compare llama.cpp's logits with a forward that rounds matmul inputs as its CPU backend does")
args = parser.parse_args()
if args.gguf_py:
    sys.path.insert(0, args.gguf_py)
import gguf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_lib import GgmlLib  # noqa: E402

ggml = GgmlLib(args.ggml_lib) if args.ggml_lib else None
GGUF_PY_MISSING = ("Q1_0", "Q2_0")

model = MuseSparkForConditionalGeneration.from_pretrained(os.path.join(args.small_dir, "hf"), dtype=torch.float32).eval()
text = model.config.text_config
ids = load_file(os.path.join(args.small_dir, "transformers_reference.safetensors"))["input_ids"].long()[None, :]
original = {k: v.clone() for k, v in model.state_dict().items()}
LM = "model.language_model"

# ggml-cpu type_traits_cpu[...].vec_dot_type: what a weight type's dot product rounds its activation row to.
VEC_DOT = {"F16": "F16", "BF16": "BF16", "Q4_0": "Q8_0", "Q5_0": "Q8_0", "Q8_0": "Q8_0", "IQ4_NL": "Q8_0", "MXFP4": "Q8_0",
           "Q1_0": "Q8_0", "Q2_0": "Q8_0", "Q4_1": "Q8_1", "Q5_1": "Q8_1"}
for k_type in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M",
               "IQ4_XS", "TQ1_0", "TQ2_0"):
    VEC_DOT[k_type] = "Q8_K"


def round_q8_0(x: torch.Tensor) -> torch.Tensor:
    """quantize_row_q8_0 (x86): scale max/127 kept as fp16, values x*127/max rounded half to even."""
    b = x.reshape(-1, 32)
    amax = b.abs().amax(dim=1, keepdim=True)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    return (torch.round(b * inv) * (amax / 127.0).half().float()).reshape(x.shape)


def round_q8_k(x: torch.Tensor) -> torch.Tensor:
    """quantize_row_q8_K_ref: iscale -127/max of the largest-magnitude value, nearest_int, clamped to 127, d = 1/iscale (f32)."""
    b = x.reshape(-1, 256)
    pick = b.abs().argmax(dim=1, keepdim=True)
    peak = torch.gather(b, 1, pick)
    iscale = torch.where(peak != 0, -127.0 / peak, torch.zeros_like(peak))
    q = torch.clamp(torch.round(iscale * b), max=127.0)
    return torch.where(peak != 0, q * (1.0 / iscale), torch.zeros_like(b)).reshape(x.shape)


ROUND = {"Q8_0": round_q8_0, "Q8_1": round_q8_0, "Q8_K": round_q8_k,
         "F16": lambda x: x.half().float(), "BF16": lambda x: x.bfloat16().float()}


def repermute(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of llama.cpp's _unpermute_for_rope: interleaved (NORM) rows back to HF rotate_half rows."""
    rows, cols = t.shape
    return t.view(n_heads, rows // n_heads // 2, 2, cols).transpose(1, 2).reshape(rows, cols)


def dense(reader_tensor) -> torch.Tensor:
    if reader_tensor.tensor_type.name in GGUF_PY_MISSING:
        count = int(np.prod(reader_tensor.shape))
        data = ggml.dequantize(int(reader_tensor.tensor_type), reader_tensor.data.reshape(-1), count)
        return torch.from_numpy(data.reshape([int(d) for d in reversed(reader_tensor.shape)]))
    data = gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def load_gguf(reader) -> dict:
    """Writes the GGUF's dequantized weights into the model; returns each written parameter's storage -> ggml type name."""
    model.load_state_dict(original)
    params = dict(model.named_parameters())
    types = {}
    with torch.no_grad():
        for t in reader.tensors:
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                continue
            w = dense(t)
            if t.name == "token_embd.weight":
                params[f"{LM}.embed_tokens.weight"].copy_(w)
                continue
            if t.name == "output.weight":
                target = params["lm_head.weight"]
                target.copy_(w)
            else:
                bid, _, suffix = t.name[len("blk."):].partition(".")
                layer = f"{LM}.layers.{bid}"
                if suffix == "attn_q.weight":
                    target = params[f"{layer}.self_attn.q_proj.weight"]
                    target.copy_(repermute(w, text.num_attention_heads))
                elif suffix == "attn_k.weight":
                    target = params[f"{layer}.self_attn.k_proj.weight"]
                    target.copy_(repermute(w, text.num_key_value_heads))
                elif suffix in SIMPLE:
                    target = params[f"{layer}.{SIMPLE[suffix]}"]
                    target.copy_(w)
                elif suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
                    target = params[f"{layer}.mlp.experts.gate_up_proj"]
                    half = slice(None, text.moe_intermediate_size) if suffix.startswith("ffn_gate") else slice(text.moe_intermediate_size, None)
                    target[:, half, :].copy_(w)
                elif suffix == "ffn_down_exps.weight":
                    target = params[f"{layer}.mlp.experts.down_proj"]
                    target.copy_(w)
                else:
                    raise KeyError(f"unmapped non-F32 tensor {t.name} ({t.tensor_type.name})")
            key = target.untyped_storage().data_ptr()
            if types.setdefault(key, t.tensor_type.name) != t.tensor_type.name and \
                    VEC_DOT[types[key]] != VEC_DOT[t.tensor_type.name]:
                raise ValueError(f"{t.name}: gate and up experts round activations differently")
    return types


SIMPLE = {"attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
          "attn_gate.weight": "self_attn.gate_proj.weight", "ffn_latent_down.weight": "mlp.pre_expert_proj.weight",
          "ffn_latent_up.weight": "mlp.post_expert_proj.weight"}


def forward(rounding: dict | None) -> torch.Tensor:
    """Last-layer logits; with rounding (weight storage -> ggml type), each matmul input is first rounded for that type."""
    exact_linear = F.linear

    def linear(x, weight, bias=None):
        kind = rounding.get(weight.untyped_storage().data_ptr()) if rounding else None
        return exact_linear(ROUND[VEC_DOT[kind]](x) if kind else x, weight, bias)

    F.linear = linear
    try:
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits[0].float().contiguous()
    finally:
        F.linear = exact_linear


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = torch.log_softmax(torch.from_numpy(p_logits).double(), -1)
    q = torch.log_softmax(torch.from_numpy(q_logits).double(), -1)
    return float((p.exp() * (p - q)).sum())


for path in sorted(glob.glob(os.path.join(args.small_dir, args.pattern))):
    name = os.path.basename(path)[:-len(".gguf")]
    reader = gguf.GGUFReader(path)
    if ggml is None and any(t.tensor_type.name in GGUF_PY_MISSING for t in reader.tensors):
        print(f"skip {name}: gguf-py has no dequantizer for {'/'.join(GGUF_PY_MISSING)}; pass --ggml-lib")
        continue
    types = load_gguf(reader)
    exact = forward(None)
    if not args.llamacpp_activations:
        save_file({"input_ids": ids[0].to(torch.int32).contiguous(), "logits": exact},
                  os.path.join(args.small_dir, name + ".dequant.safetensors"))
        print(f"{name}: top-1 {int(exact[-1].argmax())}")
        continue
    saved = os.path.join(args.small_dir, "llamacpp", f"llamacpp-{name}.bin")
    if not os.path.exists(saved):
        print(f"skip {name}: no llama.cpp logits")
        continue
    llama = np.fromfile(saved, dtype=np.float32)
    simulated = forward(types)[-1].numpy()
    print(f"{name}: KL(llama.cpp||exact) {kl(llama, exact[-1].numpy()):.3e}  KL(llama.cpp||rounded) {kl(llama, simulated):.3e}  "
          f"top-1 llama.cpp {int(llama.argmax())} exact {int(exact[-1].argmax())} rounded {int(simulated.argmax())}")
