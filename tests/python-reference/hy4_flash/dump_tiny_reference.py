"""Reference dump for the Hy4 Flash tiny config: the real Hugging Face classes, CPU, float32, saved inputs.

STATUS: written against transformers PR 50903 at eb1d5db464b0df81793f914849142e9b82e056c7 and NOT RUN (no Python with torch on the
authoring machine). Run it on WSL or a pod the user starts, see README.md. Until its output is committed and a C# test compares against
it, the C# side is checked only against a transcription, and no parity row may be written.

    pip install torch numpy safetensors "transformers[torch,vision] @ git+https://github.com/huggingface/transformers.git@eb1d5db464b0df81793f914849142e9b82e056c7"
    python dump_tiny_reference.py --out ./out

The tiny config is tiny_config.json next to this file (shipped legacy key names, read by Hy4FlashConfig too). Two cases are dumped:

  text:   40 token ids (not a multiple of the kpool, longer than index_topk and the sliding window, ids past the 250-long lookup map, the pad id
          mid-sequence), one full forward, then the same ids as a 37-token prefill and three single-token cached steps.
  image:  8 text ids, a 4x4-patch image (2x2 merged grid, 6 placeholder tokens), 10 text ids, one full forward.

Weights are perturbed so every term matters: router bias, sinks, indexer APE, hyper-connection bases and scales, every norm gain, and a many-to-one
Engram lookup table whose hash multipliers are redrawn from max(table)+1 as the reference does after loading.

Outputs: config.json, weights.safetensors (checkpoint key names, so Hy4FlashWeightMap.Check applies to its header), then per case
inputs_<case>.json and <case>__<stage>.npy (float32 unless noted): embed, layer_{i}_out (the [tokens, hc, hidden] stream), layer_{i}_attn_hc_post and
_attn_hc_in, layer_{i}_attn_out, layer_{i}_ffn_hc_post and _ffn_hc_in, layer_{i}_ffn_out, layer_{i}_q_latent, engram_{i}_out, indexer_{i}_token_keys,
indexer_{i}_group_keys, indexer_{i}_keep (bool, kept keys per query), router_{i}_topk_index (int64; compare as sets, the reference top-k is unsorted)
and router_{i}_topk_weight, vision_out, hc_head_out, final_norm_out, logits. Cached steps are logits and layer outputs prefixed cached_prefill__
and cached_step{n}__. manifest.json lists sha256 of every file.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

PINNED_SHA = "eb1d5db464b0df81793f914849142e9b82e056c7"
HERE = Path(__file__).resolve().parent

# Hugging Face module names back to checkpoint names (the inverse of conversion_mapping.py "hy_v4_flash" + "hy_v4_vl").
RENAMES = [
    (r"\.self_attn\.sinks$", ".self_attn.learnable_sink_param"),
    (r"\.self_attn\.gate_proj\.weight$", ".self_attn.linear_gate.weight"),
    (r"\.post_self_attn_layernorm\.weight$", ".self_attn.post_proj_norm.weight"),
    (r"^(model\.language_model\.layers\.\d+)\.post_mlp_layernorm\.weight$", r"\1.mlp.post_fc2_norm.weight"),
    (r"\.mlp\.pre_expert_proj\.weight$", ".mlp.routed_expert_down_proj.weight"),
    (r"\.mlp\.post_expert_proj\.weight$", ".mlp.routed_expert_up_proj.weight"),
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_layer.hc_pre.hc_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_mlp_layer.hc_pre.hc_\1"),
    (r"\.hc_head\.hc_(fn|base|scale)$", r".hc_head.hc_head_\1"),
    (r"^model\.language_model\.", "model."),
    (r"^model\.vision_tower\.layers\.(\d+)\.layer_norm1\.", r"vit.layers.\1.input_layernorm."),
    (r"^model\.vision_tower\.layers\.(\d+)\.layer_norm2\.", r"vit.layers.\1.post_attention_layernorm."),
    (r"\.mlp\.fc1\.", ".mlp.dense_h_to_4h."),
    (r"\.mlp\.fc2\.", ".mlp.dense_4h_to_h."),
    (r"^model\.vision_tower\.patch_merger\.before_rms\.", "vit.perceive.norm."),
    (r"^model\.vision_tower\.patch_merger\.proj_conv\.", "vit.perceive.proj.0."),
    (r"^model\.vision_tower\.patch_merger\.proj_out\.", "vit.perceive.proj.2."),
    (r"^model\.vision_tower\.patch_merger\.image_newline$", "vit.perceive.image_newline"),
    (r"^model\.vision_tower\.", "vit."),
]


def checkpoint_key(name):
    for pattern, replacement in RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


def require_pinned_transformers(allow_unpinned):
    dist = importlib.metadata.distribution("transformers")
    direct = dist.read_text("direct_url.json") or ""
    if PINNED_SHA in direct:
        return
    message = f"transformers {dist.version} is not the pinned checkout {PINNED_SHA} (direct_url.json: {direct[:200]!r})"
    if not allow_unpinned:
        sys.exit(message + "; install the pinned ref or pass --allow-unpinned (the dump then is not the reference)")
    print("WARNING:", message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perturb(model, torch, seed):
    """Make every term of the forward matter: random init leaves gates near 0.5, norm gains at 1 and biases at 0."""
    gen = torch.Generator().manual_seed(seed)

    def noise(t, scale):
        return torch.randn(t.shape, generator=gen) * scale

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".sinks") or "intra_group_ape" in name or "unified_per_hc_scales" in name:
                p.add_(noise(p, 0.5))
            elif re.search(r"\.(attn_hc|ffn_hc)\.base$", name) or name.endswith("hc_head.hc_base"):
                p.add_(noise(p, 0.5))
            elif re.search(r"\.(attn_hc|ffn_hc)\.scale$", name) or name.endswith("hc_head.hc_scale"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            elif p.ndim == 1 and name.endswith("weight") and ("norm" in name or name.endswith("unified_pre_gamma")):
                p.add_(noise(p, 0.3))
            elif name.endswith("unified_pre_gamma"):
                p.add_(noise(p, 0.3))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.add_(noise(b, 0.5))
        for module in model.modules():
            if type(module).__name__ == "HYV4FlashEngramLayer":
                table = (torch.arange(module.lookup_table.shape[0], dtype=torch.int32) // 2).to(torch.int32)
                module.lookup_table.copy_(table)
                module.hash_multipliers.copy_(module.draw_hash_multipliers(int(table.max()) + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    require_pinned_transformers(args.allow_unpinned)

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import HYV4FlashConfig, HYV4FlashForConditionalGeneration

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_dict = json.loads((HERE / "tiny_config.json").read_text())
    (out / "config.json").write_text(json.dumps(config_dict, indent=2))
    text, vision, root = config_dict["text_config"], config_dict["vision_config"], config_dict

    torch.manual_seed(args.seed)
    model = HYV4FlashForConditionalGeneration(HYV4FlashConfig(**json.loads(json.dumps(config_dict)))).to(torch.float32).eval()
    perturb(model, torch, args.seed + 1)

    sd = {checkpoint_key(k): v.detach().to(torch.float32).contiguous() for k, v in model.state_dict().items() if not k.endswith("lookup_table")}
    for k, v in model.state_dict().items():
        if k.endswith("lookup_table"):
            sd[checkpoint_key(k)] = v.detach().to(torch.int32).contiguous()
    save_file(sd, str(out / "weights.safetensors"))

    rng = np.random.default_rng(args.seed)
    ids = rng.integers(3, 250, size=40).astype(np.int64)
    ids[5], ids[11], ids[17], ids[18], ids[30] = 255, 0, 249, 250, 0  # past the map, pad mid-sequence, past the map
    ids_text = ids

    grid = [1, 4, 4]  # t, h, w in patches; patch 4, merge 2 -> a 2x2 merged grid, 2 * (2 + 1) = 6 placeholder tokens
    patch = vision["patch_size"]
    num_patches = grid[1] * grid[2]
    pixels = torch.from_numpy(rng.standard_normal((num_patches, 3 * patch * patch)).astype(np.float32))
    n_image = (grid[1] // 2) * (grid[2] // 2 + 1)
    ids_image = np.concatenate([rng.integers(3, 249, size=8), np.full(n_image, root["image_token_id"]), rng.integers(3, 249, size=10)]).astype(np.int64)

    lm = model.model.language_model
    state = {"prefix": "", "detailed": True, "captured": {}}

    def save(name, tensor):
        if not state["detailed"] and not (name.startswith("layer_") and name.endswith("_out")) and name != "logits":
            return
        t = tensor.detach()
        state["captured"][state["prefix"] + name] = t.float().cpu().numpy() if t.dtype.is_floating_point else t.cpu().numpy()

    handles = [lm.embed_tokens.register_forward_hook(lambda m, i, o: save("embed", o))]
    for n, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(lambda m, i, o, n=n: save(f"layer_{n}_out", o)))
        def hc_hook(prefix, n=n):
            def hook(m, i, o):
                save(f"layer_{n}_{prefix}_post", o[0])
                save(f"layer_{n}_{prefix}_in", o[1])

            return hook

        handles.append(layer.attn_hc.register_forward_hook(hc_hook("attn_hc")))
        handles.append(layer.ffn_hc.register_forward_hook(hc_hook("ffn_hc")))
        handles.append(layer.self_attn.register_forward_hook(lambda m, i, o, n=n: save(f"layer_{n}_attn_out", o[0])))
        handles.append(layer.self_attn.q_a_layernorm.register_forward_hook(lambda m, i, o, n=n: save(f"layer_{n}_q_latent", o)))
        handles.append(layer.mlp.register_forward_hook(lambda m, i, o, n=n: save(f"layer_{n}_ffn_out", o)))
        if layer.engram_layer is not None:
            handles.append(layer.engram_layer.register_forward_hook(lambda m, i, o, n=n: save(f"engram_{n}_out", o)))
        indexer = layer.self_attn.compressed_indexer
        if indexer is not None:
            handles.append(indexer.token_key_norm.register_forward_hook(lambda m, i, o, n=n: save(f"indexer_{n}_token_keys", o)))
            original_pool = indexer.pool_groups

            def pool(token_keys, n=n, original=original_pool):
                groups = original(token_keys)
                save(f"indexer_{n}_group_keys", groups)
                return groups

            indexer.pool_groups = pool
            handles.append(indexer.register_forward_hook(lambda m, i, o, n=n: save(f"indexer_{n}_keep", o > torch.finfo(o.dtype).min / 2)))
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            def router(m, i, o, n=n):
                save(f"router_{n}_topk_weight", o[1])
                save(f"router_{n}_topk_index", o[2])

            handles.append(gate.register_forward_hook(router))
    handles.append(model.model.vision_tower.register_forward_hook(lambda m, i, o: save("vision_out", o.pooler_output[0] if isinstance(o.pooler_output, (list, tuple)) else o.pooler_output)))
    handles.append(lm.hc_head.register_forward_hook(lambda m, i, o: save("hc_head_out", o)))
    handles.append(lm.norm.register_forward_hook(lambda m, i, o: save("final_norm_out", o)))

    with torch.no_grad():
        state["prefix"] = "text__"
        logits = model(input_ids=torch.from_numpy(ids_text)[None], use_cache=False).logits
        save("logits", logits)
        (out / "inputs_text.json").write_text(json.dumps({"input_ids": ids_text.tolist(), "prefill_length": 37}))

        # Cached run: engram n-grams reach back into the history, so the full-history windows are passed explicitly.
        state["detailed"] = False
        full = torch.from_numpy(ids_text)[None]
        ngram, mask = model.model.build_engram_inputs(full, None)
        state["prefix"] = "cached_prefill__"
        first = model(input_ids=full[:, :37], ngram_tokens=ngram[:, :37], engram_mask=mask[:, :37], use_cache=True)
        save("logits", first.logits)
        cache = first.cache
        for step in range(3):
            end = 38 + step
            state["prefix"] = f"cached_step{step}__"
            result = model(input_ids=full[:, end - 1:end], ngram_tokens=ngram[:, end - 1:end], engram_mask=mask[:, end - 1:end], cache=cache, use_cache=True)
            save("logits", result.logits)
            cache = result.cache

        state["detailed"] = True
        state["prefix"] = "image__"
        grid_t = torch.tensor([grid])
        ids_t = torch.from_numpy(ids_image)[None]
        image_logits = model(input_ids=ids_t, pixel_values=pixels, image_grid_thw=grid_t,
                             mm_token_type_ids=(ids_t == root["image_token_id"]).long(), use_cache=False).logits
        save("logits", image_logits)
        (out / "inputs_image.json").write_text(json.dumps({"input_ids": ids_image.tolist(), "image_grid_thw": grid,
                                                           "pixel_values_shape": list(pixels.shape)}))
        np.save(out / "image__pixel_values.npy", pixels.numpy())
    for handle in handles:
        handle.remove()

    manifest = {"transformers_pin": PINNED_SHA, "torch": torch.__version__, "seed": args.seed, "files": {}}
    for name, array in state["captured"].items():
        np.save(out / f"{name}.npy", array)
    for path in sorted(out.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(state['captured'])} stages to {out}")


if __name__ == "__main__":
    main()
