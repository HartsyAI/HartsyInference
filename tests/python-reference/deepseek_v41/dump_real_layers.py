"""Run the UNMODIFIED upstream Transformer on the first N backbone layers of the REAL checkpoint (CPU, pure-torch kernel ports).

Weights come from the sharded checkpoint by name through lazy_checkpoint.py, so any depth up to the full 40 layers fits in RAM: routed
experts load when a token is routed to them, the Engram tables are read by row, `wo_a` stays FP8. One prefill and K greedy decode steps run;
every call writes, as raw little-endian float32, the final normed hidden state of each position, the last position's logits and each block's
output stream (plus per-sublayer taps when --stage-taps is on) into <out>/prefill and <out>/step<j>, and meta.json records the token ids,
the oracle's greedy tokens and the per-call timings. The C# side (DeepSeekV41RealWeightsTests.RealLayers_MatchTheUpstreamModel) loads the
same layers with MaxLayers=N, replays the oracle's tokens (teacher forcing) and compares every layer.

Modes
  exact  float32 compute, GEMM-input activation quantization removed, exact-softmax sparse_attn. The in-place FP8/FP4 latent (KV cache)
         quantization stays, because it is part of the cache format and the host reference models it. Weights decode from the stored
         FP8/MXFP4 bytes. This is the mode the F32 host reference should match tightly; a gap here is a structural difference.
  structural  exact, with the FP8/FP4 cache quantization removed too (window KV, compressed latents, indexer queries and keys), against a host
         loaded with QuantizeLatents off. Everything is continuous arithmetic, so a float-noise difference cannot flip an element across a
         quantization boundary and a deep stack must agree to rounding; this is the strict structural check at any depth.
  ports  bf16 default dtype and the pure-torch ports exactly as the unmodified model calls them, FP8 activation quantization included.
         The gap between the host reference and this mode is the precision envelope of the missing activation quantization, not a defect.

Usage: python dump_real_layers.py <checkpoint dir> <out dir> [--layers 1] [--mode exact|ports] [--steps 0]
       [--ids 0,671,... | --prompt TEXT | --prompt-file FILE [--max-prompt-tokens N]] [--stage-taps auto|on|off] [--threads 8]
The default prompt is [BOS] + "The capital of France is". meta.json records the prompt text only when the ids are exactly [BOS] + its
tokens, which is when the C# tokenizer cross-check applies; custom ids and truncated prompts carry none.
"""
import argparse
import dataclasses
import json
import os
import resource
import sys
import time

import torch
import torch.nn.functional as F

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports
import lazy_checkpoint as lc  # noqa: E402

DEFAULT_PROMPT: str = "The capital of France is"
DEFAULT_IDS: str = "0,671,6102,294,8760,344"  # [BOS] + DEFAULT_PROMPT, checked against tokenizers 0.23.2
STAGES = ("attn_in", "attn_out", "ffn_in", "ffn_out")


def exact_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    # GEMM-input quantization is dropped (the host reference does not model it); the in-place KV latent quantization is part of the
    # model's cache format and the host reference models it, so it stays
    if inplace:
        return kernel_ports.act_quant(x, block_size, scale_fmt, scale_dtype, True)
    return x, None


def structural_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    return x if inplace else (x, None)


def structural_fp4_act_quant(x, *args, **kwargs):
    return x


def exact_fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    return F.linear(a.to(torch.float32), kernel_ports.dequant_fp8_block(b, b_s, block_size, block_size)).to(torch.get_default_dtype())


def exact_fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    return F.linear(a.to(torch.float32), kernel_ports.dequant_mxfp4(b, b_s)).to(torch.get_default_dtype())


def chunked_sparse_attn_exact(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
    # queries are independent, so chunking them is exact and bounds the float64 gather of a long prefill
    return torch.cat([kernel_ports.sparse_attn_exact(q[:, i:i + chunk], kv, attn_sink, topk_idxs[:, i:i + chunk], softmax_scale)
                      for i in range(0, q.size(1), chunk)], dim=1)


def write_f32(path: str, t: torch.Tensor) -> list:
    t = t.detach().to(torch.float32).contiguous().cpu()
    with open(path, "wb") as fh:
        fh.write(t.numpy().tobytes())
    return list(t.shape)


def peak_rss_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 1)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss<={peak_rss_gib()}GiB] {message}", file=sys.stderr, flush=True)


def prompt_ids(a) -> tuple:
    """(ids, prompt text or None). The text is returned only when the ids are exactly [BOS] + its tokens."""
    if a.prompt is None and a.prompt_file is None:
        return [int(v) for v in a.ids.split(",")], (DEFAULT_PROMPT if a.ids == DEFAULT_IDS else None)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(a.checkpoint, "tokenizer.json"))
    text = a.prompt if a.prompt is not None else open(a.prompt_file, encoding="utf-8").read()
    ids = [0] + tokenizer.encode(text, add_special_tokens=False).ids
    if a.max_prompt_tokens and len(ids) > a.max_prompt_tokens:
        return ids[:a.max_prompt_tokens], None
    return ids, text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--mode", choices=["exact", "structural", "ports"], default="exact")
    p.add_argument("--steps", type=int, default=0, help="greedy decode steps after the prefill")
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--max-prompt-tokens", type=int, default=0)
    p.add_argument("--stage-taps", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--expert-cache", type=int, default=48, help="routed experts kept resident")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    a = p.parse_args()
    if not 1 <= a.layers <= 40:
        sys.exit("--layers must be 1..40 (the backbone)")
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    ids, prompt = prompt_ids(a)
    stage_taps = a.stage_taps == "on" or (a.stage_taps == "auto" and len(ids) <= 64)
    max_seq_len = max(128, -(-(len(ids) + a.steps + 1) // 128) * 128)

    with open(os.path.join(a.checkpoint, "inference", "config.json")) as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(mod.ModelArgs)}
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    engram_ids = tuple(i for i in cfg["engram_layer_ids"] if i < a.layers)
    cfg.update(n_layers=a.layers, n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=(), vision_n_layers=0,
               engram_layer_ids=engram_ids, engram_num_embeddings=tuple(cfg["engram_num_embeddings"][:len(engram_ids)]),
               max_batch_size=1, max_seq_len=max_seq_len, temperature=0.0)
    args = mod.ModelArgs(**cfg)

    if a.mode in ("exact", "structural"):
        mod.act_quant = structural_act_quant if a.mode == "structural" else exact_act_quant
        if a.mode == "structural":
            mod.fp4_act_quant = structural_fp4_act_quant
        mod.fp8_gemm = exact_fp8_gemm
        mod.fp4_gemm = exact_fp4_gemm
        mod.sparse_attn = chunked_sparse_attn_exact
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.bfloat16)
    lc.install(mod, a.expert_cache)

    t0 = time.time()
    source = lc.SafetensorsDir(a.checkpoint)
    tokenizer = None
    if engram_ids:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.checkpoint)
        log("tokenizer loaded; building the compressed Engram token map")
    model = mod.Transformer(args, tokenizer)
    log(f"model built: {a.layers} layers, Engram layers {engram_ids}")

    consumed = lc.replace_wo_a(model, source, torch.float32 if a.mode in ("exact", "structural") else torch.bfloat16)
    lazy_keys = lc.bind_lazy(model, source)
    for name, prm in model.named_parameters():
        if not source.has(name):
            sys.exit(f"parameter {name} is not in the checkpoint index")
        t = source.read(name)
        consumed.add(name)
        if prm.dtype == torch.float4_e2m1fn_x2 and t.dtype == torch.int8:
            t = t.view(torch.float4_e2m1fn_x2)
        if tuple(t.shape) != tuple(prm.shape):
            sys.exit(f"{name}: checkpoint shape {tuple(t.shape)} != model shape {tuple(prm.shape)}")
        if t.dtype == torch.float8_e4m3fn and prm.dtype != torch.float8_e4m3fn:
            sys.exit(f"{name}: stored FP8 but declared {prm.dtype}; it needs a lazy dequantizing shim")
        prm.data = t if t.dtype == prm.dtype else t.to(prm.dtype)
    expected = {k for k in source.weight_map if k in ("embed.weight", "head.weight", "norm.weight")
                or any(k.startswith(f"layers.{i}.") for i in range(a.layers))}
    # bias_vl is the vision-language routing bias; upstream selects it only for image tokens (image_mask), never for text
    unused = {k for k in expected - consumed - lazy_keys if not k.endswith(".bias_vl")}
    if unused:
        sys.exit(f"checkpoint tensors the model did not consume: {sorted(unused)[:10]}")
    if a.mode in ("exact", "structural"):
        # upstream declares a few small projections bf16 (indexer wk and weights_proj, compressor wkv); the float32 compute needs them widened.
        # embed and head stay bf16 by design (Bf16Embedding / Bf16Head widen on use)
        for name, prm in model.named_parameters():
            if prm.dtype == torch.bfloat16 and name not in ("embed.weight", "head.weight"):
                prm.data = prm.data.to(torch.float32)
    model.eval()
    load_s = time.time() - t0
    log(f"weights bound in {load_s:.1f}s ({len(lazy_keys)} lazy tensors, {len(consumed)} resident)")

    taps: dict = {}

    def tap(stage: str, layer: int):
        def hook(_m, _i, out):
            taps[(stage, layer)] = out[0] if isinstance(out, tuple) else out
        return hook

    for blk in model.layers:
        layer = blk.layer_id
        blk.register_forward_hook(tap("block", layer))
        if stage_taps:
            blk.attn_norm.register_forward_hook(tap("attn_in", layer))
            blk.attn.register_forward_hook(tap("attn_out", layer))
            blk.ffn_norm.register_forward_hook(tap("ffn_in", layer))
            blk.ffn.register_forward_hook(tap("ffn_out", layer))
            if blk.engram is not None:
                blk.engram.register_forward_hook(tap("engram_out", layer))
    model.norm.register_forward_hook(tap("final", -1))

    os.makedirs(a.out, exist_ok=True)
    timings, generated = [], []

    def run(name: str, token_ids: list, start_pos: int) -> int:
        taps.clear()
        t1 = time.time()
        _, logits, _ = model(torch.tensor([token_ids], dtype=torch.long), start_pos)
        timings.append({"call": name, "tokens": len(token_ids), "seconds": round(time.time() - t1, 1)})
        directory = os.path.join(a.out, name)
        os.makedirs(directory, exist_ok=True)
        write_f32(os.path.join(directory, "logits.f32"), logits)
        write_f32(os.path.join(directory, "final.f32"), taps[("final", -1)])
        for (stage, layer), value in taps.items():
            if layer >= 0:
                write_f32(os.path.join(directory, f"{stage}{layer}.f32"), value)
        # cache state after the call: the sliding-window ring of every layer and the compressed KV of the layers that build it
        for blk in model.layers:
            attn = blk.attn
            write_f32(os.path.join(directory, f"cache_window{blk.layer_id}.f32"), attn.window_kv_cache[0])
            if attn.is_kv_source:
                rows = (start_pos + len(token_ids)) // attn.compress_ratio
                write_f32(os.path.join(directory, f"cache_compress{blk.layer_id}.f32"), attn.compress_kv_cache[0, :rows])
        token = int(torch.argmax(logits[0]))
        log(f"{name}: {len(token_ids)} token(s) in {timings[-1]['seconds']}s, argmax {token}")
        return token

    generated.append(run("prefill", ids, 0))
    for j in range(1, a.steps + 1):
        generated.append(run(f"step{j}", [generated[-1]], len(ids) + j - 1))

    meta = {"mode": a.mode, "layers": a.layers, "ids": ids, "prompt": prompt, "generated": generated, "steps": a.steps,
            "stage_taps": stage_taps, "engram_layers": list(engram_ids), "torch": torch.__version__,
            "default_dtype": str(torch.get_default_dtype()), "load_seconds": round(load_s, 1), "timings": timings,
            "peak_rss_gib": peak_rss_gib()}
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    print(json.dumps({"mode": a.mode, "layers": a.layers, "tokens": len(ids), "generated": generated, "peak_rss_gib": meta["peak_rss_gib"]}))


if __name__ == "__main__":
    main()
