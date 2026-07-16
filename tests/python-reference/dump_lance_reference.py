#!/usr/bin/env python3
"""Reference dump for the Lance (ByteDance) T2I MoT backbone, driven by the REAL upstream code.

Loads the real Lance_3B checkpoint into the upstream `Qwen2ForCausalLM` (qwen2_navit.py from the
bytedance/Lance repo), replays one T2I validation forward (fixed prompt / grid / noise / timestep)
with everything cast to F32 on CPU, and dumps per-layer hidden states + the velocity output so the
C# side (LanceRealWeightParityTests + diff_lance_layers.py) can diff.

The T2I sequence layout replicates `ValidationDataset.t2v_sample` (text_template=True, target
image): ChatML scaffold, caption, `<|vision_start|><|video_pad|>*N<|vision_end|><|im_end|>`,
position ids from the upstream `get_rope_index` (apply_qwen_2_5_vl_pos_emb=true per
inference_lance.sh), and a dense replica of `create_sparse_mask` (cross-checked against the
upstream closure evaluated densely).

Usage:
  python dump_lance_reference.py --ckpt-dir <Lance_3B folder> --ref-repo <bytedance/Lance clone> \
      --out <dump dir> [--prompt "..."] [--size 256] [--timestep 0.75] [--seed 42]
"""
import argparse
import contextlib
import json
import os
import sys
import types

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ref-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default='A cat holds a poster with rainbow text "STOP"')
    ap.add_argument("--size", type=int, default=256, help="square image size, /16 = latent grid")
    ap.add_argument("--timestep", type=float, default=0.75, help="shifted flow time fed to the model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-every", type=int, default=1, help="dump every Nth layer (all by default)")
    return ap.parse_args()


def save_f32(path, t):
    np.asarray(t.detach().to(torch.float32).contiguous().reshape(-1)).astype("<f4").tofile(path)


def main():
    args = parse_args()
    sys.path.insert(0, args.ref_repo)
    # qwen2_navit imports flash_attn at module scope; the varlen kernel is only used by the
    # (unused) KV-cache inference path, so a stub suffices on CPU. A real ModuleSpec keeps
    # transformers' importlib-based availability probe from choking on the stub.
    import importlib.machinery
    fa_stub = types.ModuleType("flash_attn")
    fa_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    fa_stub.flash_attn_varlen_func = None
    sys.modules.setdefault("flash_attn", fa_stub)
    # modeling.lance.__init__ pulls the full training stack (data/, config/, decord); load
    # qwen2_navit standalone — it only uses absolute imports into modeling.qwen2*.
    import importlib.util
    nav_spec = importlib.util.spec_from_file_location(
        "lance_qwen2_navit", os.path.join(args.ref_repo, "modeling/lance/qwen2_navit.py"))
    nav = importlib.util.module_from_spec(nav_spec)
    nav_spec.loader.exec_module(nav)
    mu_spec = importlib.util.spec_from_file_location(
        "lance_modeling_utils", os.path.join(args.ref_repo, "modeling/lance/modeling_utils.py"))
    mu = importlib.util.module_from_spec(mu_spec)
    mu_spec.loader.exec_module(mu)
    du_spec = importlib.util.spec_from_file_location(
        "lance_data_utils", os.path.join(args.ref_repo, "data/data_utils.py"))
    du = importlib.util.module_from_spec(du_spec)
    du_spec.loader.exec_module(du)
    from safetensors.torch import safe_open

    # transformers 5.x dropped the "default" rope-init entry the (4.x-era) upstream modeling files
    # index; restore the plain 1/theta^(2i/d) init with scaling 1.0.
    from transformers import modeling_rope_utils

    if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        def _default_rope(config, device=None, seq_len=None, **kw):
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (config.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device or "cpu") / dim))
            return inv_freq, 1.0
        modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope

    # CPU has no EFFICIENT_ATTENTION backend and the upstream List-mask branch hardcodes bf16;
    # run the math in f32 instead (this is the reference, not the product).
    nav.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def f32_sdpa(q, k, v, attn_mask=None, **kw):
        am = attn_mask.to(torch.float32) if attn_mask is not None else None
        return orig_sdpa(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), am, **kw)

    nav.scaled_dot_product_attention = f32_sdpa

    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)

    # ── config ──
    llm_cfg = json.load(open(os.path.join(args.ckpt_dir, "llm_config.json")))
    from modeling.qwen2.configuration_qwen2 import Qwen2Config
    cfg = Qwen2Config(
        vocab_size=llm_cfg["vocab_size"], hidden_size=llm_cfg["hidden_size"],
        intermediate_size=llm_cfg["intermediate_size"], num_hidden_layers=llm_cfg["num_hidden_layers"],
        num_attention_heads=llm_cfg["num_attention_heads"], num_key_value_heads=llm_cfg["num_key_value_heads"],
        hidden_act=llm_cfg["hidden_act"], max_position_embeddings=llm_cfg["max_position_embeddings"],
        rms_norm_eps=llm_cfg["rms_norm_eps"], rope_theta=llm_cfg["rope_theta"],
        use_sliding_window=False, tie_word_embeddings=False, attention_dropout=0.0,
        qk_norm=True, qk_norm_und=True, qk_norm_gen=True,
        layer_module="Qwen2MoTDecoderLayer", freeze_und=False,
        apply_qwen_2_5_vl_pos_emb=True,
        rope_scaling=llm_cfg["rope_scaling"], vision_config=llm_cfg["vision_config"],
        image_token_id=llm_cfg["image_token_id"], video_token_id=llm_cfg["video_token_id"],
        vision_start_token_id=llm_cfg["vision_start_token_id"],
        pad_token_id=llm_cfg["bos_token_id"], bos_token_id=llm_cfg["bos_token_id"],
        eos_token_id=llm_cfg["eos_token_id"],
    )

    # ── model on meta, real weights assigned in F32 ──
    print("Instantiating Qwen2ForCausalLM (meta) ...", flush=True)
    with torch.device("meta"):
        lm = nav.Qwen2ForCausalLM(cfg)
    ckpt_path = os.path.join(args.ckpt_dir, "model.safetensors")
    print("Loading + F32-casting language_model.* ...", flush=True)
    lm_sd, extra = {}, {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith("language_model."):
                lm_sd[k[len("language_model."):]] = f.get_tensor(k).to(torch.float32)
            else:
                extra[k] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = lm.load_state_dict(lm_sd, strict=False, assign=True)
    real_missing = [m for m in missing if "rotary_emb" not in m]
    assert not real_missing, f"missing keys: {real_missing[:10]}"
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"
    del lm_sd
    from modeling.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
    lm.model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=cfg)  # meta-built buffer → rebuild on cpu
    lm.eval()

    # ── T2I sequence (segment-wise; asserted equal to whole-string tokenization) ──
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
    IM_START, IM_END, VS, VE, VP = 151644, 151645, 151652, 151653, 151656
    system = ("Describe the image by detailing the color, quantity, text, shape, size, texture, "
              "spatial relationships of the objects and background:")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    prefix = [IM_START] + enc("system\n" + system) + [IM_END] + enc("\n") + [IM_START] + enc("user\n")
    caption = enc(args.prompt)
    mid = [IM_END] + enc("\n") + [IM_START] + enc("assistant\n")

    grid_h = grid_w = args.size // 16
    n_vae = grid_h * grid_w
    full_ids = prefix + caption + mid + [VS] + [VP] * n_vae + [VE] + [IM_END]
    rendered = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>")
    whole = enc(rendered.strip())
    vp_pos = whole.index(VP)
    whole_expanded = whole[:vp_pos] + [VP] * n_vae + whole[vp_pos + 1:]
    assert whole_expanded == full_ids, "segment-wise tokenization != whole-string tokenization"

    seq = len(full_ids)
    n_pre = len(prefix) + len(caption) + len(mid) + 1
    gen_idx = torch.arange(n_pre, n_pre + n_vae)
    und_idx = torch.cat([torch.arange(0, n_pre), torch.arange(n_pre + n_vae, seq)])

    # ── position ids from the real get_rope_index ──
    ids_t = torch.tensor(full_ids)[None]
    grid_thw = torch.tensor([[1, grid_h * 2, grid_w * 2]])   # dataset stores h*merge, w*merge
    pos_ids, _ = lm.get_rope_index(
        input_ids=ids_t, image_grid_thw=grid_thw, video_grid_thw=grid_thw,
        second_per_grid_ts=[1.0], attention_mask=torch.ones_like(ids_t))
    save_f32(os.path.join(args.out, "pos_ids.bin"), pos_ids[:, 0].T)   # [seq,3]

    # ── dense mask: my rule, cross-checked vs the upstream create_sparse_mask closure ──
    # The upstream noise split spans <|vision_start|> + pads + <|vision_end|> (len n_vae+2).
    noise = torch.zeros(seq, dtype=torch.bool)
    noise[n_pre - 1:n_pre + n_vae + 1] = True
    q = torch.arange(seq)[:, None]
    kv = torch.arange(seq)[None, :]
    allow = ((q >= kv) & ~noise[None, :].expand(seq, seq)) | (noise[:, None] & noise[None, :])
    split_lens = [n_pre - 1, n_vae + 2, 1]
    attn_modes = ["causal", "noise", "causal"]
    closure = du.create_sparse_mask([seq], split_lens, attn_modes, "cpu")
    upstream_allow = closure(torch.zeros((), dtype=torch.long), torch.zeros((), dtype=torch.long),
                             q.expand(seq, seq), kv.expand(seq, seq))
    assert torch.equal(allow, upstream_allow), "dense mask rule != upstream create_sparse_mask"
    # Production path: flex_attention with a BlockMask (the List/dense branch hardcodes bf16 casts
    # at the call site, which would contaminate an f32 reference). Pad to BLOCK_SIZE like
    # process_attention_mask: extra causal split in its own document so padding is isolated.
    BLOCK_SIZE = 128
    seq_pad_total = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    pad = seq_pad_total - seq
    split_lens_pad = split_lens + ([pad] if pad > 0 else [])
    attn_modes_pad = attn_modes + (["causal"] if pad > 0 else [])
    from torch.nn.attention.flex_attention import create_block_mask
    sparse_pad = du.create_sparse_mask([seq, pad], split_lens_pad, attn_modes_pad, "cpu")
    block_mask = create_block_mask(sparse_pad, B=1, H=cfg.num_attention_heads,
                                   Q_LEN=seq_pad_total, KV_LEN=seq_pad_total,
                                   device="cpu", BLOCK_SIZE=BLOCK_SIZE, _compile=False)

    # ── inputs: noise + timestep + Lance glue ──
    g = torch.Generator().manual_seed(args.seed)
    x_t = torch.randn(n_vae, 48, generator=g, dtype=torch.float32)
    save_f32(os.path.join(args.out, "noise.bin"), x_t)

    time_embedder = mu.TimestepEmbedder(cfg.hidden_size)
    te_sd = {k[len("time_embedder."):]: v for k, v in extra.items() if k.startswith("time_embedder.")}
    time_embedder.load_state_dict(te_sd)
    t_vec = torch.full((n_vae,), args.timestep, dtype=torch.float32)
    t_embed = time_embedder(t_vec)

    lat_pos = (torch.arange(grid_h)[:, None] * 64 + torch.arange(grid_w)[None, :]).flatten()
    pos_embed_rows = extra["latent_pos_embed.pos_embed"][lat_pos]
    vae_embed = x_t @ extra["vae2llm.weight"].T + extra["vae2llm.bias"] + t_embed + pos_embed_rows

    packed = torch.zeros(seq, cfg.hidden_size, dtype=torch.float32)
    packed[und_idx] = lm.model.embed_tokens(torch.tensor(full_ids))[und_idx]
    packed[gen_idx] = vae_embed
    save_f32(os.path.join(args.out, "layers", "packed_in.bin"), packed)

    # ── per-layer hooks ──
    def hook(i):
        def fn(module, inputs, output):
            if i % args.dump_every == 0 or i == cfg.num_hidden_layers - 1:
                save_f32(os.path.join(args.out, "layers", f"layers_{i}.bin"), output)
        return fn
    for i, layer in enumerate(lm.model.layers):
        layer.register_forward_hook(hook(i))

    print(f"Forward: seq={seq} (n_pre={n_pre}, n_vae={n_vae}) ...", flush=True)
    with torch.no_grad():
        hidden = lm(
            packed_sequence=packed.clone(),
            sample_lens=[seq, pad],
            attention_mask=block_mask,
            packed_position_ids=pos_ids,
            mode_forward="validation",
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=gen_idx,
        )
        velocity = hidden[gen_idx] @ extra["llm2vae.weight"].T + extra["llm2vae.bias"]
    save_f32(os.path.join(args.out, "output_velocity.bin"), velocity)

    manifest = {
        "prompt": args.prompt,
        "size": args.size,
        "grid": [1, grid_h, grid_w],
        "timestep": args.timestep,
        "seed": args.seed,
        "seq": seq,
        "n_pre": n_pre,
        "n_vae": n_vae,
        "prefix_tokens": prefix,
        "caption_tokens": caption,
        "mid_tokens": mid,
        "full_tokens": full_ids,
        "num_layers": cfg.num_hidden_layers,
        "dump_every": args.dump_every,
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"velocity: mean={velocity.mean():.6f} std={velocity.std():.6f}")
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
