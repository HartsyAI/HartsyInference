# LTX-Video Architecture — Research Notes

> **Status:** Transformer + handoff complete (ported from diffusers); VAE + pipeline scoped | **Last Updated:** 2026-06-09 | **Needed Before:** `LtxVideoTransformer`, `LtxVideoVaeDecoder`, `LtxVideoPipeline`
>
> **Sources (verbatim, pulled raw):** diffusers `src/diffusers/models/transformers/transformer_ltx.py` (601 L), `models/autoencoders/autoencoder_kl_ltx.py` (1552 L), `pipelines/ltx/pipeline_ltx.py`. Weights: [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video). License: Apache-2.0 (OpenRAIL-M variant on some checkpoints — verify per weight).

## Summary

LTX-Video (Lightricks) is a fast DiT text-to-video model. A **28-layer single-stream DiT** (inner 2048 = 32 heads × 64) operates on VAE-latent tokens with **self-attention + 3D RoPE**, **cross-attention to a frozen T5-XXL** (caption_channels 4096), **AdaLN-Single** timestep conditioning, and gelu-approximate FFN. The latent comes from a high-compression **3D causal VAE** (~32× spatial, 8× temporal, 128 latent channels) whose **decoder is timestep-conditioned** (a denoising decoder, unusual). Sampling is rectified-flow Euler.

For SharpInference this is the **second video model on the Lance-built foundation**: it reuses `CausalConv3d`, the generic 3D-causal-VAE pattern, `T5TextEncoder` (T5-XXL), backend `RmsNorm`/`ScaledDotProductAttention`, flow-match Euler, and the frame-streaming output (`VideoFrame`/`IVideoEncoder`). The only genuinely new pieces are LTX's own DiT block, its interleaved 3D RoPE, and the timestep-conditioned VAE decoder.

## Transformer (`LTXVideoTransformer3DModel`) — verbatim

**Config:** `in_channels=128, out_channels=128, patch_size=1, patch_size_t=1, num_attention_heads=32, attention_head_dim=64` (inner_dim **2048**), `cross_attention_dim=2048, num_layers=28, activation_fn="gelu-approximate", qk_norm="rms_norm_across_heads", norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=4096`.

**Top-level modules:**
- `proj_in: Linear(128 → 2048)` — VAE-latent tokens → hidden. **No extra patchify** (patch_size 1) — the latent channels ARE the transformer input.
- `scale_shift_table: Parameter[2, 2048]` (final-layer AdaLN), `time_embed: AdaLayerNormSingle(2048, use_additional_conditions=False)`.
- `caption_projection: PixArtAlphaTextProjection(4096 → 2048)` — `Linear(4096→2048) → GELU(tanh) → Linear(2048→2048)` (projects T5 features to the cross-attn dim).
- `rope: LTXVideoRotaryPosEmbed(dim=2048, base_num_frames=20, base_height=2048, base_width=2048, theta=10000)`.
- 28 × `LTXVideoTransformerBlock`.
- `norm_out: LayerNorm(2048, eps=1e-6, affine=False)`, `proj_out: Linear(2048 → 128)`.

**Forward:**
```
image_rotary_emb = rope(hidden, num_frames, height, width, rope_interpolation_scale)   # (cos, sin) [B,S,2048]
hidden = proj_in(hidden)                                                                # [B,S,2048]
temb, embedded_timestep = time_embed(timestep)                                          # temb [B,1,6*2048], emb [B,1,2048]
encoder = caption_projection(encoder_hidden_states)                                     # T5 [B,L,4096] → [B,L,2048]
for block: hidden = block(hidden, encoder, temb, rope, encoder_mask)
shift, scale = (scale_shift_table[None,None] + embedded_timestep[:,:,None]).unbind      # [B,1,2048] each
hidden = norm_out(hidden) * (1 + scale) + shift
out = proj_out(hidden)                                                                  # [B,S,128]
```

**Block forward** (`LTXVideoTransformerBlock`):
```
ada = scale_shift_table[6,dim][None,None] + temb.reshape(B, temb.size(1), 6, dim)       # temb.size(1)=1
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.unbind(2)           # [B,1,dim] each (broadcast over S)
n = norm1(hidden) * (1 + scale_msa) + shift_msa                                          # RMSNorm no-affine eps 1e-6
hidden = hidden + attn1(n, rope) * gate_msa                                              # self-attn + RoPE
hidden = hidden + attn2(hidden, encoder, mask)                                           # cross-attn to T5, NO rope
n = norm2(hidden) * (1 + scale_mlp) + shift_mlp
hidden = hidden + ff(n) * gate_mlp                                                        # gelu-approx FFN
```

**Attention** (`LTXAttention` + `LTXVideoAttnProcessor`): `norm_q`/`norm_k = RMSNorm(2048, eps=1e-5, affine=True)` over the **full** Q/K (across heads) BEFORE the head split. RoPE is applied to the full-dim Q/K (self-attn only), THEN unflatten to `[B,S,32,64]`, SDPA (no GQA — 32:32), flatten, `to_out` Linear(2048→2048, bias). Cross-attn (`attn2`): K/V from `cross_attention_dim=2048` (post-caption-projection T5), with the encoder attention mask, no RoPE. All q/k/v have bias.

**RoPE** (`LTXVideoRotaryPosEmbed`): 3-axis grid `(f, h, w)`; per axis `dim//6` frequencies `theta^linspace(0,1,dim//6) · π/2`; per token `freq · (coord·2 − 1)`; concat 3 axes → `[B,S,dim//2]`; `cos/sin = repeat_interleave(2)` → `[B,S,dim=2048]`. **Interleaved-pair complex** apply (Flux-style): `x.unflatten(2,(-1,2)) → (real, imag); rotated = stack([-imag, real]).flatten; out = x·cos + rotated·sin`. (Distinct from Lance's `Multimodal3DRope` block-repeat form — LTX needs its own `LtxRope`.) `rope_interpolation_scale` defaults to per-axis (1/temporal_compression for t, 1/spatial for h/w) — pins the grid to physical resolution.

**Latent ↔ token handoff:** the VAE latent `[B, 128, F, H, W]` is flattened to tokens `[B, F·H·W, 128]` (patch 1) for the transformer, and unflattened back after `proj_out`. No pixel-shuffle.

## VAE (`AutoencoderKLLTXVideo`) — scoped (port is the next subsystem)

3D causal VAE, **timestep-conditioned decoder**. Config: `in_channels=3 (RGB), out_channels=128 (latent), patch_size=4, patch_size_t=1, block_out_channels=(128,256,512,512), spatio_temporal_scaling=(True,True,True,False), layers_per_block=(4,3,3,3,4), decoder_inject_noise / timestep_conditioning=True`. Net compression ≈ **32× spatial** (patch 4 × spatio-temporal stages) × **8× temporal**, 128 latent channels.

Blocks: `LTXVideoCausalConv3d` (reuse the generic `CausalConv3d`), `LTXVideoResnetBlock3d` (RMSNorm eps 1e-8 + causal conv ×2 + optional 1×1 conv shortcut with `per_channel_scale`; **optional `scale_shift_table[4,dim]` timestep conditioning** when `timestep_conditioning`), `LTXVideoUpsampler3d`/`LTXVideoDownsampler3d` (pixel-shuffle style channel↔space), `LTXVideoMidBlock3D`, up/down blocks. Per-channel `latents_mean`/`latents_std` normalization (read from the checkpoint config — 128 values each).

**Decoder is a denoiser:** `decode(latent, timestep)` — the decoder takes a noise timestep and conditions the resnet blocks via `scale_shift_table` + per-channel scale, with optional injected noise. The pipeline runs the decoder at a fixed small decode timestep. (This is the main structural novelty vs the Wan2.2 VAE, which is a plain decoder.)

## Pipeline (`LTXPipeline`) — scoped

T5-XXL encode (prompt + negative) → flow-match (rectified) Euler over ~`num_inference_steps` (default ~50; LTX distilled variants 8) with a timestep shift → DiT denoise with 2-way text CFG (`guidance_scale` ~3) → VAE timestep-conditioned decode → frames. Latent frames `T_lat = (num_frames − 1)/8 + 1`; spatial `H/32 × W/32`; 128 channels. Reuses the Lance/foundation flow-match + frame-streaming.

## Reuse map (foundation already built)

| Need | Reuse |
|---|---|
| T5-XXL text encoder (4096) | `T5TextEncoder` (existing) |
| RMSNorm, SDPA | backend ops |
| 3D causal conv | `CausalConv3d` (generic) |
| Flow-match Euler shift | `LancePipelineCommon.BuildShiftedTimesteps` pattern / `FlowMatchEulerDiscreteScheduler` |
| Frame streaming + encoders | `VideoFrame`, `IVideoEncoder`, `FfmpegProcessEncoder`, `BmpSequenceEncoder` |
| Streaming VAE decode pattern | `Wan22VaeDecoder.DecodeStreaming` (template; LTX VAE has its own blocks + timestep cond) |

## Net-new (LTX-specific)

- `LtxRope` — interleaved-pair 3D RoPE (distinct from `Multimodal3DRope`).
- `LtxVideoBlock` / `LtxVideoTransformer` — DiT with self-attn(RoPE) + cross-attn(T5) + AdaLN-Single.
- `LtxVideoVaeDecoder` — timestep-conditioned 3D causal VAE decoder (resnet blocks with `scale_shift_table`, pixel-shuffle up-samplers).
- `LtxVideoPipeline` — flow-match + 2-way CFG + timestep-conditioned VAE decode + frame streaming.

## Build order (each compiled + unit-verified)

1. **Transformer** (this subsystem): `LtxVideoConfig`, `LtxRope`, `LtxVideoBlock`, `LtxVideoTransformer` (reuse T5/RMSNorm/SDPA). Validate RoPE invariants + tiny-config forward.
2. **VAE decoder**: port `LTXVideoResnetBlock3d` (+ timestep cond), up-samplers, decoder assembly on `CausalConv3d`. Validate shape/finite + streaming.
3. **Pipeline** + checkpoint converter + frame-streaming entry. Then first-run numeric validation (checkpoint-gated).

## Open questions (validation-gated)

- `PixArtAlphaTextProjection` exact activation (gelu-tanh assumed) — confirm against diffusers.
- `rope_interpolation_scale` defaults per LTX version (0.9.x vs 0.9.5/1.0) — read the pipeline.
- VAE `timestep_conditioning` decode timestep value + injected-noise schedule — read `pipeline_ltx.py` decode call.
- Exact `latents_mean`/`latents_std` (128 each) — from the VAE checkpoint config on download.
- Distilled vs full checkpoints differ in steps/shift/guidance — capture per variant.
</content>
