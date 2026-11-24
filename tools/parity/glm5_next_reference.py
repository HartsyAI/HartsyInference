"""Tiny glm5_next (Sage S2.1 Flash) reference: writes a random checkpoint in the real key layout and the reference outputs for it.

Offline only; nothing here runs on the engine's inference path. Needs CPU torch, safetensors, numpy and transformers installed from the pinned
commit (it holds `models/glm5_next`, not yet in a release):

    pip install "git+https://github.com/huggingface/transformers@52bac5d0e0da56e26289c072f306877ff5870262"
    python tools/parity/glm5_next_reference.py --out tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/tiny

Layout written to --out:
    config.json, model-0000{1,2}-of-00002.safetensors, model.safetensors.index.json   the checkpoint, real key names (per-expert
        gate/up/down_proj, separate q/k/v_conv1d, hc_attn_*, flat f_a_proj, indexer.*, an MTP layer with shared_head, no vision tower)
    input_ids.npy                int64 [S]
    embeds.npy                   float32 [S, hidden]
    layer_L_stream.npy           float32 [S, hc_mult, hidden], the stream after decoder layer L (prefill)
    layer_L_topk.npy             int32 [S, width], the indexer's selection of DSA layer L (prefill), -1 padded
    final_hidden.npy             float32 [S, hidden], after the hc mean and model.norm
    logits.npy                   float32 [S, vocab], prefill
    decode_logits.npy            float32 [S, vocab], the same tokens one at a time through the cache
    meta.json                    transformers commit, seed, shapes, the smallest top-k score margin seen
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

PIN = "52bac5d0e0da56e26289c072f306877ff5870262"

# Four layers cover: KDA + dense MLP, DSA + MoE, KDA + MoE, DSA + MoE. index_topk 16 with pools of 4 keeps 4 pools, so a 41-token
# prompt (not a multiple of 4) selects a strict subset of its keys. swiglu_limit 0.5 makes the clamp bind.
TINY = dict(
    hidden_size=64, vocab_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    layer_types=["linear_attention", "indexed_attention", "linear_attention", "indexed_attention"],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    intermediate_size=96, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
    n_group=1, topk_group=1, routed_scaling_factor=2.5, norm_topk_prob=True, swiglu_limit=0.5,
    q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=0, v_head_dim=16,
    index_n_heads=4, index_head_dim=16, index_topk=16, index_kpool=4, index_kpool_always_select_tail=True,
    hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, rms_norm_eps=1e-5, num_nextn_predict_layers=1,
    linear_attn_config=dict(num_heads=4, head_dim=16, short_conv_kernel_size=4, gate_lower_bound=-5.0,
                            kda_layers=[0, 2], full_attn_layers=[1, 3]),
    indexer_types=["full"] * 4, pad_token_id=0, eos_token_id=[1, 2, 3], tie_word_embeddings=False,
)
SEQ = 41


def randomize(model, gen):
    """The library's init leaves projections near zero; widen everything so each stage moves the output."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.8)
            elif name.endswith(("attn_hc.fn", "ffn_hc.fn")):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.35)
            elif name.endswith(("attn_hc.base", "ffn_hc.base")):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith(("attn_hc.scale", "ffn_hc.scale")):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            elif name.endswith("A_log"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("index_kpool_compress_ape"):
                p.copy_(torch.randn(p.shape, generator=gen))
            elif name.endswith("k_norm.bias") or name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.25)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn(b.shape, generator=gen) * 0.1)


def to_checkpoint(model, cfg, gen):
    """Reverse transformers' glm5_next conversion mapping: the on-disk names are the published checkpoint's."""
    out = {}
    inter = cfg.text_config.moe_intermediate_size
    for key, t in model.state_dict().items():
        t = t.detach().float().contiguous()
        for src, dst in (("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"), ("attn_hc.scale", "hc_attn_scale"),
                         ("ffn_hc.fn", "hc_ffn_fn"), ("ffn_hc.base", "hc_ffn_base"), ("ffn_hc.scale", "hc_ffn_scale")):
            key = key.replace("." + src, "." + dst)
        key = key.replace("self_attn.forget_gate.", "self_attn.")
        if key.endswith("self_attn.conv1d.weight"):
            q, k, v = t.chunk(3, dim=0)
            base = key[: -len("conv1d.weight")]
            out[base + "q_conv1d.weight"], out[base + "k_conv1d.weight"], out[base + "v_conv1d.weight"] = q, k, v
        elif key.endswith("mlp.experts.gate_up_proj"):
            base = key[: -len("gate_up_proj")]
            for e in range(t.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = t[e, :inter].clone()
                out[f"{base}{e}.up_proj.weight"] = t[e, inter:].clone()
        elif key.endswith("mlp.experts.down_proj"):
            base = key[: -len("down_proj")]
            for e in range(t.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = t[e].clone()
        else:
            out[key] = t
    # The checkpoint ships an MTP layer and per-layer shared_head the reference ignores: clone a DSA + MoE layer's shapes into layer N.
    n = cfg.text_config.num_hidden_layers
    donor = f"model.language_model.layers.{n - 1}."
    mtp = f"model.language_model.layers.{n}."
    h = cfg.text_config.hidden_size
    for key, t in list(out.items()):
        if key.startswith(donor) and ".hc_" not in key:
            out[mtp + key[len(donor):]] = torch.randn(t.shape, generator=gen) * 0.2 + (1.0 if t.ndim == 1 else 0.0)
    out[mtp + "enorm.weight"] = 1.0 + 0.1 * torch.randn(h, generator=gen)
    out[mtp + "hnorm.weight"] = 1.0 + 0.1 * torch.randn(h, generator=gen)
    out[mtp + "eh_proj.weight"] = torch.randn(h, 2 * h, generator=gen) * 0.2
    out[mtp + "shared_head.norm.weight"] = 1.0 + 0.1 * torch.randn(h, generator=gen)
    return out


def write_checkpoint(path, tensors, cfg_json):
    keys = sorted(tensors)
    shards = [keys[0::2], keys[1::2]]
    weight_map, total = {}, 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: tensors[k] for k in names}, os.path.join(path, fname))
        for k in names:
            weight_map[k] = fname
            total += tensors[k].numel() * tensors[k].element_size()
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=1)


def config_json(text):
    """The published layout of config.json: the DSA layer type is spelled deepseek_sparse_attention, kda/full schedules sit in linear_attn_config."""
    t = dict(text)
    t["model_type"] = "glm5_next_text"
    t["hidden_act"] = "silu"
    t["scoring_func"] = "sigmoid"
    t["topk_method"] = "noaux_tc"
    t["mla_use_nope"] = True
    t["index_kpool_compress"] = True
    t["qk_head_dim"] = t["qk_nope_head_dim"]
    t["layer_types"] = ["deepseek_sparse_attention" if x == "indexed_attention" else x for x in t["layer_types"]]
    return {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": t,
            "image_token_id": 120, "video_token_id": 121}


def selection_margin(ix, hidden, q_resid, mask):
    """Smallest gap between the last kept and the first dropped pool score over queries that have more candidates than slots (prefill)."""
    import torch.nn.functional as F
    b, s = hidden.shape[:2]
    q = ix.wq_b(q_resid).view(b, s, -1, ix.head_dim)
    k = ix.k_norm(ix.wk(hidden)).view(b, s, -1, ix.head_dim).squeeze(2)
    gate = F.linear(hidden, ix.index_kpool_compress_gate)
    packed = torch.cat([k, gate, mask.to(k.dtype)[..., None]], dim=-1)
    keys, pool_idx, pool_valid = ix.get_pooled_states(packed_states=packed)
    scores = F.relu(torch.matmul(q.float(), keys.transpose(-1, -2).float().unsqueeze(1)) * ix.softmax_scale)
    w = ix.weights_proj(hidden).float() * (ix.n_heads ** -0.5)
    idx = torch.matmul(w.unsqueeze(-2), scores).squeeze(-2)
    visible = ix.get_visible_tokens(valid_keys=mask, q_length=s, current_length=s)
    end = pool_idx[..., -1].clamp(0, s - 1)
    ok = visible.gather(-1, end[:, None, :].expand(b, s, -1)) & pool_valid[:, None]
    sc = idx.masked_fill(~ok, float("-inf"))
    sel = min(ix.index_topk // ix.index_kpool, sc.shape[-1])
    if sc.shape[-1] <= sel:
        return float("inf")
    top = sc.topk(sel + 1, dim=-1).values
    contested = ok.sum(-1) > sel
    gap = (top[..., sel - 1] - top[..., sel])[contested]
    return float(gap.min()) if gap.numel() else float("inf")


def attempt(args, seed):
    import transformers
    from transformers.models.glm5_next import modeling_glm5_next as m
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    text = Glm5NextTextConfig(**TINY)
    vision = Glm5NextVisionConfig(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=TINY["hidden_size"],
                                  projection_intermediate_size=64)
    cfg = Glm5NextConfig(text_config=text, vision_config=vision)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    model = m.Glm5NextForConditionalGeneration(cfg).float().eval()
    randomize(model, gen)
    ids = torch.randint(4, TINY["vocab_size"], (1, SEQ), generator=gen)

    streams, topks, margins, handles = {}, {}, [], []
    for i, layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_hook(lambda mod, inp, out, i=i: streams.__setitem__(i, out[0][0].detach().clone())))
        ix = getattr(layer.self_attn, "indexer", None)
        if ix is not None:
            handles.append(ix.register_forward_hook(lambda mod, inp, out, i=i: topks.__setitem__(i, out[0].detach().clone())))
            handles.append(ix.register_forward_hook(
                lambda mod, a, kw, out: margins.append(selection_margin(mod, kw["hidden_states"], kw["q_resid"], kw["attention_mask"])),
                with_kwargs=True))
    with torch.no_grad():
        embeds = model.model.language_model.embed_tokens(ids)[0]
        logits = model(input_ids=ids, use_cache=False).logits[0]
        for h in handles:
            h.remove()
        hidden = model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[0]
        cache = transformers.DynamicCache(config=cfg.text_config)
        decode = torch.stack([model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits[0, -1] for t in range(SEQ)])
    drift = float((logits - decode).abs().max())
    margin = min(margins)
    return dict(model=model, cfg=cfg, gen=gen, ids=ids, streams=streams, topks=topks, embeds=embeds, logits=logits, hidden=hidden,
                decode=decode, drift=drift, margin=margin, version=transformers.__version__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tries", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers.models.glm5_next import modeling_glm5_next as m
    # The reference scatters with the indexer's int32 top-k, which torch 2.7.1 rejects (it wants int64 indices); widening changes no value.
    original_forward = m.Glm5NextTextIndexer.forward
    m.Glm5NextTextIndexer.forward = lambda self, *a, **k: original_forward(self, *a, **k).to(torch.int64)

    # A near tie between the last kept and first dropped pool lets a different top-k tie order flip the selection (the reference's own
    # prefill and one-token decode disagree then), so a seed is accepted only with a clear margin and prefill == decode.
    for seed in range(args.seed, args.seed + args.tries):
        r = attempt(args, seed)
        print(f"seed {seed}: prefill-vs-decode {r['drift']:.2e}, selection margin {r['margin']:.3e}")
        if r["drift"] < 1e-4 and r["margin"] > 1e-2:
            break
    else:
        raise SystemExit("no seed with an unambiguous selection")

    tensors = {k: v for k, v in to_checkpoint(r["model"], r["cfg"], r["gen"]).items() if ".visual." not in k}
    write_checkpoint(args.out, tensors, config_json(TINY))
    np.save(os.path.join(args.out, "input_ids.npy"), r["ids"][0].numpy())
    np.save(os.path.join(args.out, "embeds.npy"), r["embeds"].numpy())
    for i, s in r["streams"].items():
        np.save(os.path.join(args.out, f"layer_{i}_stream.npy"), s.numpy())
    for i, k in r["topks"].items():
        np.save(os.path.join(args.out, f"layer_{i}_topk.npy"), k.numpy().astype(np.int32))
    np.save(os.path.join(args.out, "final_hidden.npy"), r["hidden"].numpy())
    np.save(os.path.join(args.out, "logits.npy"), r["logits"].numpy())
    np.save(os.path.join(args.out, "decode_logits.npy"), r["decode"].numpy())
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"transformers": r["version"], "pinned_commit": PIN, "seed": seed, "seq": SEQ,
                   "prefill_vs_decode_max_abs": r["drift"], "min_selection_margin": r["margin"]}, f, indent=1)


if __name__ == "__main__":
    main()
