"""Run the UNMODIFIED upstream DSpark draft head (mtp.0-2) on the REAL checkpoint in structural mode and dump its forward pass.

The target model runs the prompt (prefill) and one greedy decode step; the DSpark stages are seeded during the prefill (forward_spec with
start_pos 0) and then run once on the decode step's hidden state. The dump holds the target's hidden states feeding DSpark, each DSpark
stage's output stream, attention output and MoE output for the decode call, and the draft ids, logits and confidences. The C# side
(DeepSeekV41DSparkTests) runs its DSpark forward on the same inputs and compares stage by stage.

Usage: python dump_real_dspark.py <checkpoint dir> <out dir> [--max-prompt-tokens 6]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dump_real_layers as d  # noqa: E402  structural shims; also binds sys.modules["kernel"]
import lazy_checkpoint as lc  # noqa: E402


def write_f32(path, t):
    t = t.detach().to(torch.float32).contiguous().cpu()
    with open(path, "wb") as fh:
        fh.write(t.numpy().tobytes())
    return list(t.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod
    from transformers import AutoTokenizer

    mod.act_quant = d.structural_act_quant
    mod.fp8_gemm = d.exact_fp8_gemm
    mod.fp4_gemm = d.exact_fp4_gemm
    mod.fp4_act_quant = d.structural_fp4_act_quant
    mod.sparse_attn = d.chunked_sparse_attn_exact
    torch.set_default_dtype(torch.float32)
    lc.install(mod, 48)

    raw = json.load(open(os.path.join(a.checkpoint, "inference", "config.json")))
    known = {f.name for f in dataclasses.fields(mod.ModelArgs)}
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    cfg.update(vision_n_layers=0, max_batch_size=1, max_seq_len=128, temperature=0.0)
    args = mod.ModelArgs(**cfg)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = mod.Transformer(args, tok)
    source = lc.SafetensorsDir(a.checkpoint)
    lc.replace_wo_a(model, source, torch.float32)
    for s, blk in enumerate(model.mtp):
        blk.attn.wo_a = lc.LazyWoA(source.read(f"mtp.{s}.attn.wo_a.weight"), source.read(f"mtp.{s}.attn.wo_a.scale"), torch.float32)
    lc.bind_lazy(model, source)
    for s, blk in enumerate(model.mtp):
        for index, expert in enumerate(blk.ffn.experts):
            expert.bind(source, f"mtp.{s}", index)
    for name, prm in model.named_parameters():
        if not source.has(name):
            sys.exit(f"parameter {name} is not in the checkpoint index")
        t = source.read(name)
        if prm.dtype == torch.float4_e2m1fn_x2 and t.dtype == torch.int8:
            t = t.view(torch.float4_e2m1fn_x2)
        if tuple(t.shape) != tuple(prm.shape):
            sys.exit(f"{name}: checkpoint shape {tuple(t.shape)} != model shape {tuple(prm.shape)}")
        if t.dtype == torch.float8_e4m3fn and prm.dtype != torch.float8_e4m3fn:
            sys.exit(f"{name}: needs a dequantizing shim")
        prm.data = t if t.dtype == prm.dtype else t.to(prm.dtype)
    for name, prm in model.named_parameters():
        if prm.dtype == torch.bfloat16 and name not in ("embed.weight", "head.weight"):
            prm.data = prm.data.to(torch.float32)
    model.eval()
    print(f"weights bound in {time.time() - t0:.1f}s", flush=True)

    taps = {}
    for s, blk in enumerate(model.mtp):
        blk.register_forward_hook(lambda m, i, out, s=s: taps.__setitem__(("x", s), out[0]))
        blk.attn.register_forward_hook(lambda m, i, out, s=s: taps.__setitem__(("attn", s), out))
        blk.ffn.register_forward_hook(lambda m, i, out, s=s: taps.__setitem__(("ffn", s), out))

    prompt = [0, 671, 6102, 294, 8760, 344]
    os.makedirs(a.out, exist_ok=True)
    with torch.inference_mode():
        _, logits, main_hidden = model(torch.tensor([prompt]), 0)
        seed_token = torch.tensor([prompt[-1]])
        assert model.forward_spec(seed_token, main_hidden, 0) is None
        t1 = int(logits[0].argmax())
        taps.clear()
        _, logits1, main_hidden1 = model(torch.tensor([[t1]]), len(prompt))
        draft_ids, draft_logits, confidence = model.forward_spec(torch.tensor([t1]), main_hidden1, len(prompt))
    files = {"main_hidden_prefill.f32": write_f32(os.path.join(a.out, "main_hidden_prefill.f32"), main_hidden),
             "main_hidden_decode.f32": write_f32(os.path.join(a.out, "main_hidden_decode.f32"), main_hidden1),
             "draft_logits.f32": write_f32(os.path.join(a.out, "draft_logits.f32"), draft_logits),
             "confidence.f32": write_f32(os.path.join(a.out, "confidence.f32"), confidence)}
    for (kind, s), value in taps.items():
        files[f"mtp{s}.{kind}.f32"] = write_f32(os.path.join(a.out, f"mtp{s}.{kind}.f32"), value)
    meta = {"mode": "structural", "prompt": prompt, "decode_token": t1, "start_pos": len(prompt),
            "draft_ids": [int(v) for v in draft_ids[0]], "block_size": args.dspark_block_size,
            "noise_token_id": args.dspark_noise_token_id, "target_layers": list(args.dspark_target_layer_ids),
            "files": files, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    print(json.dumps({"draft_ids": meta["draft_ids"], "seconds": meta["seconds"]}))


if __name__ == "__main__":
    main()
