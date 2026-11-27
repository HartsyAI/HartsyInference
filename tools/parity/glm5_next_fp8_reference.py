"""Tiny glm5_next FP8 reference: block-scales the FFN and DSA projections of the tiny fixture to E4M3 and writes transformers' logits for the
same weights after dequantization.

Offline only, like glm5_next_reference.py; nothing here runs on the engine's inference path. Run it in the CPU reference image with the pinned
transformers (see glm5_next_reference.py for the install line):

    python tools/parity/glm5_next_fp8_reference.py --src tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/tiny \
        --out tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/tiny_fp8 --work /work/glm5_fp8

Layout written to --out (the FP8 checkpoint the C# reader opens with block 16):
    config.json, model-0000{1,2}-of-00002.safetensors, model.safetensors.index.json
        projections as float8_e4m3fn, each with a float32 `weight_scale_inv` grid [ceil(out/B), ceil(in/B)]; dequantized value = fp8 * scale_inv
    input_ids.npy                int64 [S], copied from --src
    fp8_logits.npy               float32 [S, vocab], transformers on the dequantized weights (prefill)
    meta.json                    block size, the quantized names, transformers version, the worst |w - dequant(quant(w))|

Quantization: each block's scale_inv = amax / 448, the E4M3 maximum; values are rounded to E4M3 by torch. The reference and the C# reader
both see exactly fp8 * scale_inv, so the logits differ only in float32 arithmetic.
"""
import argparse
import json
import os
import re
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

BLOCK = 16
FP8_MAX = 448.0
# The published FP8 repo quantizes dense, shared and routed FFN weights and the DSA q_a, q_b, kv_a and o projections; the router, norms,
# indexer, KDA projections, hyper-connection mixes, embedding and head stay as they are.
FFN = re.compile(r"\.mlp\.(?:experts\.\d+\.|shared_experts\.)?(?:gate|up|down)_proj\.weight$")
DSA = re.compile(r"\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$")


def is_fp8(name, keys):
    if FFN.search(name):
        return True
    if DSA.search(name):
        # o_proj is quantized in MLA layers only; a KDA layer's o_proj stays unquantized.
        if name.endswith(".o_proj.weight"):
            return name.replace(".o_proj.weight", ".q_a_proj.weight") in keys
        return True
    return False


def quantize(w, block):
    """Returns (float8_e4m3fn [out, in], float32 scale_inv [ceil(out/B), ceil(in/B)], float32 dequantized [out, in])."""
    out_f, in_f = w.shape
    gr, gc = -(-out_f // block), -(-in_f // block)
    padded = torch.zeros(gr * block, gc * block, dtype=torch.float32)
    padded[:out_f, :in_f] = w
    blocks = padded.view(gr, block, gc, block).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(2, 3))
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (blocks / scale[:, :, None, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = q.float() * scale[:, :, None, None]

    def unblock(t):
        return t.permute(0, 2, 1, 3).reshape(gr * block, gc * block)[:out_f, :in_f].contiguous()

    return unblock(q), scale.contiguous(), unblock(deq)


def read_source(src):
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(load_file(os.path.join(src, shard)))
    return tensors


def write_checkpoint(path, tensors, config_src, extra_config=None):
    os.makedirs(path, exist_ok=True)
    keys = sorted(tensors)
    shards = [keys[0::2], keys[1::2]]
    weight_map, total = {}, 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in names}, os.path.join(path, fname))
        for k in names:
            weight_map[k] = fname
            total += tensors[k].numel() * tensors[k].element_size()
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    config = json.load(open(config_src))
    if extra_config:
        config.update(extra_config)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=1)


def reference_logits(ref_dir, ids):
    # The reference scatters with the indexer's int32 top-k, which torch 2.7.1 rejects; widening changes no value (as in glm5_next_reference.py).
    from transformers.models.glm5_next import modeling_glm5_next as m
    original = m.Glm5NextTextIndexer.forward
    m.Glm5NextTextIndexer.forward = lambda self, *a, **k: original(self, *a, **k).to(torch.int64)
    model = m.Glm5NextForConditionalGeneration.from_pretrained(ref_dir, dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        return model(input_ids=torch.from_numpy(ids)[None], use_cache=False).logits[0].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="scratch directory for the dequantized reference checkpoint")
    ap.add_argument("--block", type=int, default=BLOCK)
    args = ap.parse_args()

    import transformers

    source = read_source(args.src)
    keys = set(source)
    stored, dequant, names = {}, {}, []
    worst = 0.0
    for name in sorted(source):
        w = source[name]
        if is_fp8(name, keys):
            q, scale, deq = quantize(w.float(), args.block)
            stored[name] = q
            stored[name[: -len("weight")] + "weight_scale_inv"] = scale
            dequant[name] = deq
            names.append(name)
            worst = max(worst, float((deq - w.float()).abs().max()))
        else:
            stored[name] = w
            dequant[name] = w

    cfg = os.path.join(args.src, "config.json")
    quant = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [args.block, args.block]}}
    write_checkpoint(args.out, stored, cfg, quant)
    ref_dir = os.path.join(args.work, "dequantized")
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    write_checkpoint(ref_dir, dequant, cfg)

    ids = np.load(os.path.join(args.src, "input_ids.npy"))
    np.save(os.path.join(args.out, "input_ids.npy"), ids)
    logits = reference_logits(ref_dir, ids)
    np.save(os.path.join(args.out, "fp8_logits.npy"), logits)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"transformers": transformers.__version__, "block": args.block, "fp8_tensors": names,
                   "max_abs_quantization_error": worst, "logits_shape": list(logits.shape)}, f, indent=1)
    print(f"quantized {len(names)} tensors; worst quantization error {worst:.3e}; logits {logits.shape}")


if __name__ == "__main__":
    main()
