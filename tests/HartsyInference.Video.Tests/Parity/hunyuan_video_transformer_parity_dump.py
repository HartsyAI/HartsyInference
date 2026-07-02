#!/usr/bin/env python3
"""HunyuanVideo MM-DiT numerical parity harness (Python / reference side).

Builds a TINY diffusers HunyuanVideoTransformer3DModel with guidance_embeds=True (so the embedded-guidance path
is exercised) whose dims are shrunk but structurally identical to the 13B model, seeds deterministic weights,
runs one forward, and dumps:
  * input latent + text hidden states + pooled CLIP vector (raw f32 .bin, row-major, B=1 squeezed)
  * the post-embed states: image after x_embedder ("projin") and text after the token refiner ("refiner")
  * per-block IMAGE hidden states ("double{i}", "single{i}")
  * the final velocity ("out_velocity")
  * the weights remapped to the key names HunyuanVideoDit.LoadWeights reads (the "hybrid" layout), saved as
    weights.safetensors so the C# side loads the SAME weights directly (bypassing the checkpoint converter).

The remap here is the inverse of HunyuanVideoCheckpointConverter's diffusers-folder path: it fuses the refiner's
split to_q/to_k/to_v into self_attn.qkv, swaps the final norm_out [scale,shift]->[shift,scale], and reshapes the
x_embedder Conv3d weight to a 2D patch-embed Linear.

Run:  <ComfyUI-venv python> hunyuan_video_transformer_parity_dump.py [OUT_DIR]   (default: /tmp/hyv_parity)
Use the SwarmUI ComfyUI venv (diffusers 0.38): the hfvenv has an HF_HOME import skew.
"""
import os
import sys
import json
import numpy as np
import torch

from safetensors.torch import save_file
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hyv_parity"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(1234)
torch.use_deterministic_algorithms(True, warn_only=True)

# ------------------------------------------------------------------ tiny config
HEADS, HEAD_DIM = 2, 8            # inner = 16
INNER = HEADS * HEAD_DIM
IN_CH, OUT_CH = 4, 4
TEXT_DIM, POOLED_DIM = 8, 6
NUM_DOUBLE, NUM_SINGLE, NUM_REFINER = 1, 1, 2
CFG = dict(
    in_channels=IN_CH, out_channels=OUT_CH,
    num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
    num_layers=NUM_DOUBLE, num_single_layers=NUM_SINGLE, num_refiner_layers=NUM_REFINER,
    mlp_ratio=4.0, patch_size=2, patch_size_t=1, qk_norm="rms_norm",
    guidance_embeds=True, text_embed_dim=TEXT_DIM, pooled_projection_dim=POOLED_DIM,
    rope_theta=256.0, rope_axes_dim=(4, 2, 2),   # sums to HEAD_DIM=8, all even
)
model = HunyuanVideoTransformer3DModel(**CFG).eval()
with torch.no_grad():
    for _, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * 0.1)

# ------------------------------------------------------------------ tiny inputs
F_LAT, H_LAT, W_LAT = 2, 4, 4     # post-patch grid: T=2, H=2, W=2 -> S_img = 8
L = 5                             # text tokens (unpadded, all valid)
TIMESTEP = 500.0
GUIDANCE = 3000.0                 # = embedded_guidance_scale(3.0) * 1000

vid = torch.randn(1, IN_CH, F_LAT, H_LAT, W_LAT)
txt = torch.randn(1, L, TEXT_DIM)
pooled = torch.randn(1, POOLED_DIM)
mask = torch.ones(1, L, dtype=torch.long)
ts = torch.full((1,), TIMESTEP)
gd = torch.full((1,), GUIDANCE)

# ------------------------------------------------------------------ capture hooks
cap = {}
model.x_embedder.register_forward_hook(lambda m, i, o: cap.__setitem__("projin", o.detach().float().cpu().numpy()))
model.context_embedder.register_forward_hook(lambda m, i, o: cap.__setitem__("refiner", o.detach().float().cpu().numpy()))
def mk(name):
    def hook(m, inp, out):
        cap[name] = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu().numpy()
    return hook
for i, blk in enumerate(model.transformer_blocks):
    blk.register_forward_hook(mk(f"double{i}"))
for i, blk in enumerate(model.single_transformer_blocks):
    blk.register_forward_hook(mk(f"single{i}"))

with torch.no_grad():
    out = model(
        hidden_states=vid, timestep=ts, encoder_hidden_states=txt,
        encoder_attention_mask=mask, pooled_projections=pooled, guidance=gd,
        return_dict=False,
    )[0]

# ------------------------------------------------------------------ remap weights -> hybrid naming
def fuse(sd, pre):
    """Fuse refiner attn.to_q/to_k/to_v -> self_attn.qkv (weight + bias)."""
    for suf in ("weight", "bias"):
        q, k, v = sd.pop(f"{pre}.attn.to_q.{suf}"), sd.pop(f"{pre}.attn.to_k.{suf}"), sd.pop(f"{pre}.attn.to_v.{suf}")
        sd[f"{pre.replace('.attn', '')}.self_attn.qkv.{suf}"] = torch.cat([q, k, v], dim=0)

src = {k: v.detach().contiguous().float().cpu() for k, v in model.state_dict().items()}
out_sd = {}
# collect refiner attn keys to fuse after the main pass
refiner_attn = {}
for k, v in src.items():
    nk = None
    if k == "x_embedder.proj.weight":
        out_sd["img_in.weight"] = v.reshape(v.shape[0], -1).contiguous(); continue
    if k == "x_embedder.proj.bias":
        out_sd["img_in.bias"] = v; continue
    # main time_text_embed (must be checked AFTER context_embedder below via startswith ordering)
    repl = [
        ("context_embedder.proj_in", "txt_in.input_embedder"),
        ("context_embedder.time_text_embed.timestep_embedder.linear_1", "txt_in.t_embedder.in_layer"),
        ("context_embedder.time_text_embed.timestep_embedder.linear_2", "txt_in.t_embedder.out_layer"),
        ("context_embedder.time_text_embed.text_embedder.linear_1", "txt_in.c_embedder.in_layer"),
        ("context_embedder.time_text_embed.text_embedder.linear_2", "txt_in.c_embedder.out_layer"),
        ("time_text_embed.timestep_embedder.linear_1", "time_in.0"),
        ("time_text_embed.timestep_embedder.linear_2", "time_in.2"),
        ("time_text_embed.guidance_embedder.linear_1", "guidance_in.0"),
        ("time_text_embed.guidance_embedder.linear_2", "guidance_in.2"),
        ("time_text_embed.text_embedder.linear_1", "vector_in.0"),
        ("time_text_embed.text_embedder.linear_2", "vector_in.2"),
    ]
    matched = False
    for a, b in repl:
        if k.startswith(a):
            nk = k.replace(a, b, 1); matched = True; break
    if matched:
        out_sd[nk] = v; continue
    if k.startswith("context_embedder.token_refiner.refiner_blocks."):
        rest = k[len("context_embedder.token_refiner.refiner_blocks."):]
        idx, tail = rest.split(".", 1)
        pre = f"txt_in.individual_token_refiner.blocks.{idx}"
        m = {
            "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
            "attn.to_out.0.weight": "self_attn.proj.weight", "attn.to_out.0.bias": "self_attn.proj.bias",
            "ff.net.0.proj.weight": "mlp.0.weight", "ff.net.0.proj.bias": "mlp.0.bias",
            "ff.net.2.weight": "mlp.2.weight", "ff.net.2.bias": "mlp.2.bias",
            "norm_out.linear.weight": "adaLN_modulation.1.weight", "norm_out.linear.bias": "adaLN_modulation.1.bias",
        }
        if tail in m:
            out_sd[f"{pre}.{m[tail]}"] = v
        else:  # attn.to_q/to_k/to_v -> buffer for fuse
            refiner_attn[f"{pre}.attn.{tail.split('attn.')[1]}"] = v
        continue
    if k == "norm_out.linear.weight" or k == "norm_out.linear.bias":
        half = v.shape[0] // 2
        scale, shift = v[:half], v[half:]
        out_sd["final_layer.mod." + k.split(".")[-1]] = torch.cat([shift, scale], dim=0).contiguous(); continue
    if k == "proj_out.weight":
        out_sd["final_layer.proj.weight"] = v; continue
    if k == "proj_out.bias":
        out_sd["final_layer.proj.bias"] = v; continue
    if k.startswith("transformer_blocks."):
        out_sd["double_blocks." + k[len("transformer_blocks."):]] = v; continue
    if k.startswith("single_transformer_blocks."):
        out_sd["single_blocks." + k[len("single_transformer_blocks."):]] = v; continue

# fuse buffered refiner qkv
for i in range(NUM_REFINER):
    pre = f"txt_in.individual_token_refiner.blocks.{i}"
    for suf in ("weight", "bias"):
        q = refiner_attn.pop(f"{pre}.attn.to_q.{suf}")
        k_ = refiner_attn.pop(f"{pre}.attn.to_k.{suf}")
        v_ = refiner_attn.pop(f"{pre}.attn.to_v.{suf}")
        out_sd[f"{pre}.self_attn.qkv.{suf}"] = torch.cat([q, k_, v_], dim=0).contiguous()

save_file(out_sd, os.path.join(OUT, "weights.safetensors"))

# ------------------------------------------------------------------ dump tensors
def dump(name, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(OUT, name + ".bin"))

dump("input_latent", vid.squeeze(0).numpy())
dump("input_text", txt.squeeze(0).numpy())
dump("input_pooled", pooled.squeeze(0).numpy())
dump("projin", cap["projin"].squeeze(0))
dump("refiner", cap["refiner"].squeeze(0))
for i in range(NUM_DOUBLE):
    dump(f"double{i}", cap[f"double{i}"].squeeze(0))
for i in range(NUM_SINGLE):
    dump(f"single{i}", cap[f"single{i}"].squeeze(0))
dump("out_velocity", out.squeeze(0).float().cpu().numpy())

meta = dict(
    fLat=F_LAT, hLat=H_LAT, wLat=W_LAT, sImg=(F_LAT // 1) * (H_LAT // 2) * (W_LAT // 2), L=L,
    inCh=IN_CH, outCh=OUT_CH, inner=INNER, heads=HEADS, headDim=HEAD_DIM,
    textDim=TEXT_DIM, pooledDim=POOLED_DIM,
    numDouble=NUM_DOUBLE, numSingle=NUM_SINGLE, numRefiner=NUM_REFINER,
    timestep=TIMESTEP, guidance=GUIDANCE, ropeAxes=[4, 2, 2], ropeTheta=256.0,
)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(f"[hyv parity] wrote weights + {len(cap)} stage dumps to {OUT}")
print(f"  out_velocity: mean={out.mean().item():.5f} std={out.std().item():.5f} absmax={out.abs().max().item():.5f}")
