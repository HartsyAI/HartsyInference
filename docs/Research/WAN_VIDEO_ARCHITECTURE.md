# Wan-Video Architecture — Research Notes

> **Status:** Transformer + reuse mapped (ported from diffusers); pipeline scoped | **Last Updated:** 2026-06-10 | **Target:** Wan2.2 **TI2V-5B** (the variant whose VAE we already built)
>
> **Sources (verbatim):** diffusers `models/transformers/transformer_wan.py`, `pipelines/wan/pipeline_wan.py`. Config: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers). License: Apache-2.0.

## Why TI2V-5B = maximum reuse

Wan2.2 TI2V-5B's VAE is **`AutoencoderKLWan`, z_dim 48, scale_factor_spatial 16, scale_factor_temporal 4** — i.e. the **exact Wan2.2 VAE already built as `Wan22VaeDecoder`** (built for Lance: z=48, 16× spatial, 4× temporal, streaming). So a Wan2.2 T2V model reuses that VAE + streaming decode directly. The transformer `in_channels = out_channels = 48` matches the VAE latent. Only the Wan DiT + pipeline are new.

## Transformer (`WanTransformer3DModel`) — TI2V-5B config

`patch_size=(1,2,2), num_attention_heads=24, attention_head_dim=128` (inner_dim **3072**), `in_channels=48, out_channels=48, text_dim=4096` (umT5), `freq_dim=256, ffn_dim=14336, num_layers=30, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6, added_kv_proj_dim=null` (no I2V image path), `pos_embed_seq_len=null` (standard scalar-timestep path — the simpler `temb [6,dim]` branch, NOT the per-token wan2.2-ti2v seq-len path).

**Top-level modules:**
- `patch_embedding: Conv3d(48 → 3072, kernel=stride=(1,2,2))` — non-overlapping patchify = a linear from `48·1·2·2 = 192` → 3072 per patch. Token grid `(T_lat, H_lat/2, W_lat/2) = (T_lat, H/32, W/32)`.
- `condition_embedder` (`WanTimeTextImageEmbedding`): timestep sinusoidal(256, flip_sin_to_cos) → `time_embedder` MLP(256→3072→3072) = **temb [3072]**; `timestep_proj = Linear(3072, 6·3072)(SiLU(temb))` = **modulation [6,3072]**; `text_embedder = PixArtAlphaTextProjection(4096→3072, gelu_tanh)` → encoder [L,3072].
- `rope` (`WanRotaryPosEmbed`): per-head (head_dim 128) interleaved-pair RoPE; head_dim split `t_dim=44, h_dim=w_dim=42` (`h=w=2·(128//6)=42`, `t=128−84=44`); per-axis `get_1d_rotary_pos_embed(dim, max_seq=1024, θ=10000, repeat_interleave_real=True)`; per token (f,h,w) concat the 3 axis freqs → cos/sin [S, 128].
- 30 × `WanTransformerBlock`.
- `norm_out: FP32LayerNorm(3072, no affine)`, `proj_out: Linear(3072 → 48·4=192)`, final `scale_shift_table[1,2,3072]`.

**Block** (`WanTransformerBlock`): all norms are **FP32LayerNorm** (norm1/norm3 no-affine, norm2 affine iff cross_attn_norm). AdaLN modulation = `scale_shift_table[1,6,dim] + timestep_proj` → `(shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate)`.
```
n = norm1(h)·(1+scale_msa)+shift_msa  → attn1(n, rope)            → h += attn·gate_msa     # self-attn + RoPE
n = norm2(h)                          → attn2(n, encoder)          → h += attn              # cross-attn to T5, NO gate, NO rope
n = norm3(h)·(1+c_scale)+c_shift      → ffn(n)                     → h += ff·c_gate         # gelu-approx FFN
```

**Attention** (`WanAttention`/`WanAttnProcessor`): `norm_q/norm_k = RMSNorm(inner_dim, affine, eps)` over full Q/K (across heads) BEFORE head split. RoPE applied **per head** (after split) interleaved-pair: `out[0::2]=x1·cos−x2·sin, out[1::2]=x1·sin+x2·cos`. SDPA (24:24, no GQA). Cross-attn (attn2): K/V from encoder (text_dim already projected to inner). `added_kv_proj` (image KV) = I2V only, skipped.

**Final:** `shift, scale = scale_shift_table[2,dim] + temb` (the [dim] temb, not the [6,dim] one) → `norm_out(h)·(1+scale)+shift` → `proj_out` → unpatchify `(1,2,2)` → latent `[B,48,T_lat,H_lat,W_lat]`.

## VAE — already built ✅

`AutoencoderKLWan` z=48 16×/4× = **`Wan22VaeDecoder`** (+ `DecodeStreaming`). Reuse directly. latents_mean/std (48 each) already in `Wan22VaeLatentNorm`.

## Pipeline (`WanPipeline`) — scoped

umT5 encode (text_dim 4096, max_seq 512) → flow-match (UniPC/FlowMatchEuler, **flow_shift 5.0 720p / 3.0 480p**) over ~50 steps + 2-way text CFG (guidance ~5) → DiT → unpatchify → `Wan22VaeDecoder.Decode` → frames. `num_frames=81`, `T_lat=(num_frames−1)/4+1`, spatial latent `H/16 × W/16`, token grid `H/32 × W/32`. Reuses the Lance/LTX flow-match + frame-streaming.

## Reuse map

| Need | Reuse |
|---|---|
| z=48 16×/4× VAE + streaming | **`Wan22VaeDecoder`** (already built) |
| umT5 / T5 text (4096) | `T5TextEncoder` |
| RMSNorm, SDPA, LayerNorm, Gelu, SiLU | backend ops |
| Flow-match Euler shift + 2-way CFG | `LancePipelineCommon` |
| Frame streaming + encoders | `VideoFrame` / `IVideoEncoder` |

## Net-new (Wan-specific)

- `WanRope` — per-head interleaved 3D RoPE (get_1d_rotary split t/h/w; distinct from `LtxRope` and `Multimodal3DRope`).
- `WanPatchEmbed` — Conv3d-as-linear (1,2,2) patchify + the matching unpatchify.
- `WanVideoBlock` / `WanVideoTransformer` — FP32LayerNorm DiT, 6-param AdaLN, cross-attn to T5.
- `WanVideoPipeline` — flow-match + 2-way CFG, reusing the Wan2.2 VAE.

## Open questions (validation-gated)

- Exact scheduler (UniPCMultistep vs FlowMatchEuler) + the timestep value fed to the DiT (sigma·1000?).
- `get_1d_rotary_pos_embed` exact freq formula (standard `θ^(−2i/dim)`) — confirm vs diffusers.
- umT5 vs T5 tokenizer/encoder specifics (Wan uses umT5-XXL).
- flow_shift per resolution (5.0/3.0) + guidance default.
