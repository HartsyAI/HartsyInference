"""Reference dump for the Hy4 Flash tiny config: the real Hugging Face classes, CPU, float32, saved inputs.

STATUS: written against transformers PR 50903 at eb1d5db464b0df81793f914849142e9b82e056c7 and NOT RUN (no Python with torch on the
authoring machine). Run it on WSL or a pod the user starts, see tests/python-reference/hy4_flash/README.md. Until its output is
committed and a C# test compares against it, the C# side is checked only against a transcription, and no parity row may be written.

    pip install "transformers[torch,vision] @ git+https://github.com/huggingface/transformers.git@eb1d5db464b0df81793f914849142e9b82e056c7" safetensors numpy
    python dump_tiny_reference.py --out ./out

Outputs (all float32 .npy unless noted): config.json (the shipped legacy key names, readable by Hy4FlashConfig), weights.safetensors (the
checkpoint key names, so Hy4FlashWeightMap.Check applies to its header), inputs.json (token ids and positions), and one .npy per stage:
embed, layer_{i}_out (the [tokens, hc, hidden] stream after each decoder layer), engram_{layerIndex}_out, indexer_{i}_mask (bool, kept keys per
query), router_{i}_topk_index (int64) and router_{i}_topk_weight, hc_head_out, final_norm_out, logits. manifest.json lists sha256 of every file.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

PINNED_SHA = "eb1d5db464b0df81793f914849142e9b82e056c7"

# The tiny config follows the upstream tester (tests/models/hy_v4_flash/test_modeling_hy_v4_flash.py at the pin), spelled in the
# shipped checkpoint's legacy names so one dict feeds both the Hugging Face config and the C# reader. Three layers: two indexed (the
# second carries the Engram memory) and one sliding; layer 0 has the dense FFN. index_topk 16 over kpool 4 makes the indexer select once
# a context passes 16 keys; sliding_window 4 is shorter than the sequences.
TINY_TEXT = {
    "vocab_size": 256, "hidden_size": 48, "num_hidden_layers": 3, "intermediate_size": 96, "moe_intermediate_size": 32,
    "routed_expert_hidden_size": 24, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 1,
    "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 3.998, "swiglu_limit": 10.0,
    "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
    "q_lora_rank": 32, "kv_lora_rank": 16, "swa_num_attention_heads": 8, "swa_qk_nope_head_dim": 8, "swa_qk_rope_head_dim": 8,
    "swa_v_head_dim": 8, "sliding_window": 4, "learnable_sink_init": 0.0,
    "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse"],
    "compressed_indexer_compress_ratio": 4, "index_topk": 16, "index_n_heads": 2, "index_head_dim": 8,
    "hc_mult": 2, "hc_eps": 1e-6, "hc_magnitude": 2.0,
    "engram_layer_ids": [2], "engram_max_ngram_size": 3, "engram_n_head_per_ngram": 2, "engram_n_embed_per_ngram": 16,
    "engram_layer_head_vocab_sizes": {"2": [[101, 103], [107, 109]]}, "engram_pad_id": 0, "engram_seed": 0,
    "engram_fp8_embedding": False, "engram_lookup_table_size": 250,
    "rope_parameters": {
        "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1e7, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    },
    "rms_norm_eps": 1e-5, "max_position_embeddings": 4096, "tie_word_embeddings": False, "hidden_act": "silu",
    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "model_type": "hy_v4_flash",
}
TINY_VISION = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 2, "out_hidden_size": 48,
    "patch_size": 4, "spatial_merge_size": 2, "max_image_size": 64, "rms_norm_eps": 1e-5, "hidden_act": "gelu",
    "model_type": "hy_v4_flash",
}
TINY_ROOT = {
    "architectures": ["HYV4FlashForConditionalGeneration"], "model_type": "hy_v4_flash", "image_token_id": 250, "image_start_token_id": 251,
    "image_end_token_id": 252, "image_newline_token_id": 253, "video_token_id": 254, "video_start_token_id": 255, "video_end_token_id": 249,
    "vision_start_token_id": 251, "spatial_merge_size": 2,
}

# Hugging Face module names back to checkpoint names (the inverse of conversion_mapping.py "hy_v4_flash" + "hy_v4_vl").
RENAMES = [
    (r"\.self_attn\.sinks$", ".self_attn.learnable_sink_param"),
    (r"\.self_attn\.gate_proj\.weight$", ".self_attn.linear_gate.weight"),
    (r"\.post_self_attn_layernorm\.weight$", ".self_attn.post_proj_norm.weight"),
    (r"\.mlp\.post_mlp_layernorm\.weight$", ".mlp.post_fc2_norm.weight"),
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
    config_dict = {**TINY_ROOT, "text_config": TINY_TEXT, "vision_config": TINY_VISION}
    (out / "config.json").write_text(json.dumps(config_dict, indent=2))

    torch.manual_seed(args.seed)
    config = HYV4FlashConfig(**json.loads(json.dumps(config_dict)))
    model = HYV4FlashForConditionalGeneration(config).to(torch.float32).eval()
    # Random init leaves the router bias, sinks, gates and Engram scales near their defaults; perturb them so every term matters.
    gen = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(k in name for k in ("e_score_correction_bias", "sinks", "intra_group_ape", "unified_per_hc_scales", "hc_base")):
                p.add_(torch.randn(p.shape, generator=gen) * 0.5)
            if "lookup_table" in name:
                continue
    sd = {checkpoint_key(k): v.detach().to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    for k, v in model.named_buffers():
        if k.endswith("lookup_table"):
            sd[checkpoint_key(k)] = v.detach().to(torch.int32).contiguous()
    save_file(sd, str(out / "weights.safetensors"))

    # Inputs: 40 tokens, not a multiple of the kpool (the C# side prefills 37 and decodes 3), longer than index_topk and the sliding window,
    # with ids past the 250-long lookup map (250..255 minus the placeholders that carry no n-gram) to exercise the padding rule.
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(3, 250, size=40).astype(np.int64)
    ids[5], ids[17], ids[18] = 255, 249, 249
    (out / "inputs.json").write_text(json.dumps({"input_ids": ids.tolist(), "prefill_length": 37}))
    input_ids = torch.from_numpy(ids)[None]

    captured = {}

    def save(name, tensor):
        captured[name] = tensor.detach().float().cpu().numpy() if tensor.dtype.is_floating_point else tensor.detach().cpu().numpy()

    lm = model.model.language_model
    handles = [lm.embed_tokens.register_forward_hook(lambda m, i, o: save("embed", o))]
    for index, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(lambda m, i, o, n=index: save(f"layer_{n}_out", o)))
        if layer.engram_layer is not None:
            handles.append(layer.engram_layer.register_forward_hook(lambda m, i, o, n=index: save(f"engram_{n}_out", o)))
        if layer.self_attn.compressed_indexer is not None:
            def indexer_hook(m, i, o, n=index):
                save(f"indexer_{n}_mask", o > torch.finfo(o.dtype).min / 2)

            handles.append(layer.self_attn.compressed_indexer.register_forward_hook(indexer_hook))
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            def router_hook(m, i, o, n=index):
                save(f"router_{n}_topk_weight", o[1])
                save(f"router_{n}_topk_index", o[2])

            handles.append(gate.register_forward_hook(router_hook))
    handles.append(lm.hc_head.register_forward_hook(lambda m, i, o: save("hc_head_out", o)))
    handles.append(lm.norm.register_forward_hook(lambda m, i, o: save("final_norm_out", o)))

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    save("logits", logits)
    for handle in handles:
        handle.remove()

    manifest = {"transformers_pin": PINNED_SHA, "torch": torch.__version__, "seed": args.seed, "files": {}}
    for name, array in captured.items():
        np.save(out / f"{name}.npy", array)
    for path in sorted(out.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(captured)} stages to {out}")


if __name__ == "__main__":
    main()
