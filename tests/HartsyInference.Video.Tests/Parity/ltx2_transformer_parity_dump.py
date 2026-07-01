#!/usr/bin/env python3
"""LTX-2.3 dual-stream DiT numerical parity harness (Python / reference side).

Builds a TINY LTX2VideoTransformer3DModel (diffusers, vendored in hfvenv) whose config matches our
C# LtxVideo2Config field-for-field but shrunk, seeds deterministic random weights, runs one forward
pass, and dumps:
  * the input latent tokens + text-encoder features (raw f32 .bin, row-major, B=1 squeezed)
  * per-block VIDEO hidden states (after proj_in = "projin", and after each transformer block)
  * the final VIDEO velocity ("out_velocity")
  * the weights remapped to the key names our LtxVideo2Transformer.LoadWeights reads
    (norm_q->q_norm, norm_k->k_norm, *a2v_cross_attn_scale_shift_table -> scale_shift_table_a2v_ca_*),
    saved as weights.safetensors so the C# side loads the SAME weights directly (bypassing the
    single-file checkpoint converter).

The C# xUnit test LtxVideo2ParityTests loads these, runs LtxVideo2Transformer.Forward with the same
tiny config on the CPU backend, and prints per-block relL2 vs these .bin dumps to localize the first
divergent block/op.

Run:  python ltx2_transformer_parity_dump.py [OUT_DIR]      (default: /tmp/ltx2_parity)
Uses the diffusers in /home/hartsy/hfvenv (activate it or run its python).

NOTE ON sigma: LTX-2.3 feeds the UNSCALED flow sigma (0..1) to prompt_adaln, while the other modulators
use the SCALED timestep (0..1000). This harness uses distinct values (timestep=500, sigma=0.5) so the C#
side (which now takes an explicit `sigma` arg on Forward) is verified to route sigma to prompt_adaln and
timestep everywhere else — if C# mistakenly fed timestep to prompt_adaln, the text-cross-attn block would
diverge here.
"""
import os
import sys
import json
import numpy as np
import torch

from safetensors.torch import save_file
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ltx2_parity"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(1234)
torch.use_deterministic_algorithms(True, warn_only=True)

# ------------------------------------------------------------------ tiny config
# Mirrors LtxVideo2Config.V23 shape-for-shape but tiny. RoPE/theta/base/scale-factor/causal/timestep
# fields are the EXACT diffusers defaults our C# config hardcodes, so this harness tests the CODE, not
# the (separately-verifiable) checkpoint config values.
INNER = 16          # num_attention_heads(2) * attention_head_dim(8)
AUDIO_INNER = 8     # audio heads(2) * audio head_dim(4)
CFG = dict(
    in_channels=8, out_channels=8, patch_size=1, patch_size_t=1,
    num_attention_heads=2, attention_head_dim=8, cross_attention_dim=INNER,
    vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20, base_height=2048, base_width=2048,
    gated_attn=False, cross_attn_mod=True,
    audio_in_channels=8, audio_out_channels=8, audio_patch_size=1, audio_patch_size_t=1,
    audio_num_attention_heads=2, audio_attention_head_dim=4, audio_cross_attention_dim=AUDIO_INNER,
    audio_scale_factor=4, audio_pos_embed_max_pos=20, audio_sampling_rate=16000, audio_hop_length=160,
    audio_gated_attn=False, audio_cross_attn_mod=True,
    num_layers=2, activation_fn="gelu-approximate", qk_norm="rms_norm_across_heads",
    norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=INNER,
    attention_bias=True, attention_out_bias=True,
    rope_theta=10000.0, rope_double_precision=True, causal_offset=1,
    timestep_scale_multiplier=1000, cross_attn_timestep_scale_multiplier=1000,
    rope_type="split", use_prompt_embeddings=False, perturbed_attn=False,   # LTX-2.3 22B is split
)

model = LTX2VideoTransformer3DModel(**CFG).eval()

with torch.no_grad():
    for _, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * 0.1)

# ------------------------------------------------------------------ tiny inputs
T_LAT, H_LAT, W_LAT = 2, 3, 4          # latent grid -> sv = 24 video tokens (f,h,w order)
SV = T_LAT * H_LAT * W_LAT
SA = 5                                  # audio tokens
LV, LA = 7, 6                           # text seq lengths (video / audio connectors)
FPS = 24.0
TIMESTEP = 500.0
SIGMA = 0.5                             # UNSCALED flow sigma (0..1); distinct from timestep to verify the C# sigma fix

vid = torch.randn(1, SV, 8)
aud = torch.randn(1, SA, 8)
enc_v = torch.randn(1, LV, INNER)
enc_a = torch.randn(1, LA, AUDIO_INNER)
ts = torch.full((1,), TIMESTEP)
sg = torch.full((1,), SIGMA)

# ------------------------------------------------------------------ capture hooks (video stream)
cap = {}
model.proj_in.register_forward_hook(
    lambda m, i, o: cap.__setitem__("projin", o.detach().float().cpu().numpy()))
for i, blk in enumerate(model.transformer_blocks):
    def mk(idx):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            cap[f"block{idx}"] = o.detach().float().cpu().numpy()
        return hook
    blk.register_forward_hook(mk(i))

# ------------------------------------------------------------------ isolated RoPE parity dump
# Dump the video self-attention RoPE cos/sin directly from the reference so the C# test can compare
# LtxVideo2Rope.BuildVideo against it in isolation (rope grid / coordinate normalization is the top
# spatial-periodicity suspect). cos/sin are [1, sv, inner].
with torch.no_grad():
    vcoords = model.rope.prepare_video_coords(1, T_LAT, H_LAT, W_LAT, torch.device("cpu"), fps=FPS)
    vcos, vsin = model.rope(vcoords, device=torch.device("cpu"))

    def rope_to_token_major(t):
        # C# LtxVideo2Rope.BuildVideo produces [T, cosWidth] (token-major, lane = h*(headDim/2)+i).
        # Interleaved rope returns [B, T, dim]; split returns [B, H, T, headDim/2] — fold the latter back to
        # [T, H*(headDim/2)] = [T, dim/2] so the layouts match for the relL2 compare.
        if t.ndim == 4:                          # [B, H, T, r] -> [T, H*r]
            return t.squeeze(0).permute(1, 0, 2).reshape(SV, -1).float().cpu().numpy()
        return t.squeeze(0).float().cpu().numpy()  # [B, T, dim] -> [T, dim]

    np.ascontiguousarray(rope_to_token_major(vcos), dtype=np.float32).tofile(
        os.path.join(OUT, "rope_video_cos.bin"))
    np.ascontiguousarray(rope_to_token_major(vsin), dtype=np.float32).tofile(
        os.path.join(OUT, "rope_video_sin.bin"))

with torch.no_grad():
    out, aout = model(
        hidden_states=vid, audio_hidden_states=aud,
        encoder_hidden_states=enc_v, audio_encoder_hidden_states=enc_a,
        timestep=ts, audio_timestep=ts, sigma=sg, audio_sigma=sg,
        encoder_attention_mask=None, audio_encoder_attention_mask=None,
        num_frames=T_LAT, height=H_LAT, width=W_LAT, fps=FPS, audio_num_frames=SA,
        use_cross_timestep=False, return_dict=False,
    )

# ------------------------------------------------------------------ remap weights -> our key naming
def remap_key(k: str) -> str:
    k = k.replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
    k = k.replace("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video")
    k = k.replace("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio")
    return k

sd = {remap_key(k): v.detach().contiguous().float().cpu() for k, v in model.state_dict().items()}
save_file(sd, os.path.join(OUT, "weights.safetensors"))

# ------------------------------------------------------------------ dump tensors
def dump(name, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(OUT, name + ".bin"))

dump("input_video", vid.squeeze(0).numpy())
dump("input_audio", aud.squeeze(0).numpy())
dump("input_enc_video", enc_v.squeeze(0).numpy())
dump("input_enc_audio", enc_a.squeeze(0).numpy())
dump("projin", cap["projin"].squeeze(0))
for i in range(CFG["num_layers"]):
    dump(f"block{i}", cap[f"block{i}"].squeeze(0))
dump("out_velocity", out.squeeze(0).float().cpu().numpy())

meta = dict(
    tLat=T_LAT, hLat=H_LAT, wLat=W_LAT, sv=SV, sa=SA, Lv=LV, La=LA, fps=FPS,
    timestep=TIMESTEP, sigma=SIGMA,
    in_channels=8, inner=INNER, audio_inner=AUDIO_INNER, out_channels=8,
    num_layers=CFG["num_layers"], num_heads=2, head_dim=8,
    audio_num_heads=2, audio_head_dim=4,
)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)

print(f"[ltx2 parity] wrote weights + {len(cap)} stage dumps to {OUT}")
print(f"  out_velocity: mean={out.mean().item():.5f} std={out.std().item():.5f} "
      f"absmax={out.abs().max().item():.5f}")
for i in range(CFG["num_layers"]):
    b = cap[f"block{i}"]
    print(f"  block{i}: std={b.std():.5f} absmax={np.abs(b).max():.5f}")
