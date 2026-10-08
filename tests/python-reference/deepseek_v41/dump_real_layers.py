"""Run the UNMODIFIED upstream Transformer on the first N backbone layers of the REAL checkpoint (CPU, pure-torch kernel ports).

Loads embed, layers.0..N-1, norm and head by name from the sharded checkpoint, runs one prefill, and writes the input ids, every
block's output stream, the final normed hidden state of every position and the last position's logits as raw little-endian float32
plus meta.json. The C# side (DeepSeekV41RealWeightsTests.RealLayers_MatchTheUpstreamModel) loads the same layers with MaxLayers=N and compares.

Modes
  exact  float32 compute, GEMM-input activation quantization removed, exact-softmax sparse_attn. The in-place FP8/FP4 latent (KV cache)
         quantization stays, because it is part of the cache format and the host reference models it. Weights decode from the stored
         FP8/MXFP4 bytes. This is the mode the F32 host reference should match tightly; a gap here is a structural difference.
  ports  bf16 default dtype and the pure-torch ports exactly as the unmodified model calls them, FP8 activation quantization included.
         The gap between the host reference and this mode is the precision envelope of the missing activation quantization, not a defect.

Engram layers (1 and 14) need a 98 GB table that upstream allocates whole, so only N == 1 is supported until a row-lazy table exists.
Usage: python dump_real_layers.py <checkpoint dir> <out dir> [--layers 1] [--mode exact|ports] [--ids 0,671,...] [--threads 8]
--ids replaces the default [BOS] + "The capital of France is"; meta.json then records no prompt and the C# tokenizer cross-check is skipped.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports

DEFAULT_PROMPT: str = "The capital of France is"
DEFAULT_IDS: str = "0,671,6102,294,8760,344"  # [BOS] + DEFAULT_PROMPT, checked against tokenizers 0.23.2


def exact_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    # GEMM-input quantization is dropped (the host reference does not model it); the in-place KV latent quantization is part of the
    # model's cache format and the host reference models it, so it stays
    if inplace:
        return kernel_ports.act_quant(x, block_size, scale_fmt, scale_dtype, True)
    return x, None


def exact_fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    return F.linear(a.to(torch.float32), kernel_ports.dequant_fp8_block(b, b_s, block_size, block_size)).to(torch.get_default_dtype())


def exact_fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    return F.linear(a.to(torch.float32), kernel_ports.dequant_mxfp4(b, b_s)).to(torch.get_default_dtype())


def write_f32(path: str, t: torch.Tensor) -> list:
    t = t.detach().to(torch.float32).contiguous().cpu()
    with open(path, "wb") as fh:
        fh.write(t.numpy().tobytes())
    return list(t.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--mode", choices=["exact", "ports"], default="exact")
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    a = p.parse_args()
    if a.layers != 1:
        sys.exit("only --layers 1 is supported: layer 1 carries the 98 GB Engram table")
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    with open(os.path.join(a.checkpoint, "inference", "config.json")) as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(mod.ModelArgs)}
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    cfg.update(n_layers=a.layers, n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=(), vision_n_layers=0,
               engram_layer_ids=(), engram_num_embeddings=(), max_batch_size=1, max_seq_len=64, temperature=0.0)
    args = mod.ModelArgs(**cfg)

    if a.mode == "exact":
        mod.act_quant = exact_act_quant
        mod.fp8_gemm = exact_fp8_gemm
        mod.fp4_gemm = exact_fp4_gemm
        mod.sparse_attn = kernel_ports.sparse_attn_exact
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.bfloat16)

    t0 = time.time()
    model = mod.Transformer(args, None)
    with open(os.path.join(a.checkpoint, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    by_shard: dict = {}
    for name, prm in model.named_parameters():
        if name not in weight_map:
            sys.exit(f"parameter {name} is not in the checkpoint index")
        by_shard.setdefault(weight_map[name], []).append((name, prm))
    consumed: set = set()
    for shard, items in by_shard.items():
        with safe_open(os.path.join(a.checkpoint, shard), framework="pt", device="cpu") as fh:
            for name, prm in items:
                t = fh.get_tensor(name)
                consumed.add(name)
                if prm.dtype == torch.float4_e2m1fn_x2 and t.dtype == torch.int8:
                    t = t.view(torch.float4_e2m1fn_x2)
                if tuple(t.shape) != tuple(prm.shape):
                    sys.exit(f"{name}: checkpoint shape {tuple(t.shape)} != model shape {tuple(prm.shape)}")
                if t.dtype == prm.dtype:
                    prm.data = t
                elif t.dtype == torch.float8_e4m3fn:
                    # stored FP8 with a block-scale companion but declared wider in the model (wo_a): convert.py dequantizes it
                    scale_name = name[: -len("weight")] + "scale"
                    with safe_open(os.path.join(a.checkpoint, weight_map[scale_name]), framework="pt", device="cpu") as sf:
                        scale = sf.get_tensor(scale_name)
                    consumed.add(scale_name)
                    prm.data = kernel_ports.dequant_fp8_block(t, scale, 32, 32).to(prm.dtype)
                else:
                    prm.data = t.to(prm.dtype)
    expected = {k for k in weight_map if k in ("embed.weight", "head.weight", "norm.weight")
                or any(k.startswith(f"layers.{i}.") for i in range(a.layers))}
    # bias_vl is the vision-language routing bias; upstream selects it only for image tokens (image_mask), never for text
    unused = {k for k in expected - consumed if not k.endswith(".bias_vl")}
    if unused:
        sys.exit(f"checkpoint tensors the model did not consume: {sorted(unused)[:10]}")
    if a.mode == "exact":
        for prm in model.parameters():
            if prm.dtype == torch.bfloat16:
                prm.data = prm.data.to(torch.float32)
    model.eval()
    load_s = time.time() - t0

    blocks, finals = [], []
    taps: dict = {}

    def tap(stage: str):
        def hook(_m, _i, out):
            if stage in taps:
                raise RuntimeError(f"tap {stage} fired twice; per-layer taps are not implemented")
            taps[stage] = out
        return hook

    for blk in model.layers:
        blk.register_forward_hook(lambda _m, _i, out: blocks.append(out[0]))
        blk.attn_norm.register_forward_hook(tap("attn_in"))
        blk.attn.register_forward_hook(tap("attn_out"))
        blk.ffn_norm.register_forward_hook(tap("ffn_in"))
        blk.ffn.register_forward_hook(tap("ffn_out"))
    model.norm.register_forward_hook(lambda _m, _i, out: finals.append(out))

    ids = [int(v) for v in a.ids.split(",")]
    t1 = time.time()
    _, logits, _ = model(torch.tensor([ids], dtype=torch.long), 0)
    run_s = time.time() - t1

    os.makedirs(a.out, exist_ok=True)
    # the test cross-checks its tokenizer against the recorded text; custom ids carry none, so that check is skipped for them
    meta = {"mode": a.mode, "layers": a.layers, "ids": ids, "prompt": DEFAULT_PROMPT if a.ids == DEFAULT_IDS else None, "torch": torch.__version__, "default_dtype": str(torch.get_default_dtype()),
            "load_seconds": round(load_s, 1), "forward_seconds": round(run_s, 1), "files": {}}
    meta["files"]["final.f32"] = write_f32(os.path.join(a.out, "final.f32"), finals[0])
    meta["files"]["logits.f32"] = write_f32(os.path.join(a.out, "logits.f32"), logits)
    for stage, t in taps.items():
        meta["files"][f"{stage}.f32"] = write_f32(os.path.join(a.out, f"{stage}.f32"), t)
    for i, b in enumerate(blocks):
        meta["files"][f"block{i}.f32"] = write_f32(os.path.join(a.out, f"block{i}.f32"), b)
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    top = int(torch.argmax(logits[0]))
    print(json.dumps({"mode": a.mode, "load_s": meta["load_seconds"], "forward_s": meta["forward_seconds"], "argmax": top}))


if __name__ == "__main__":
    main()
