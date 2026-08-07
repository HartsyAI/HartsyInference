# Wan-Video Architecture — Research Notes

> **Status:** Transformer + reuse mapped (ported from diffusers); pipeline scoped. TI2V-5B/T2V/I2V/A14B/VACE/Animate built. **Wan2.2-S2V researched (not built) — see the S2V section below.** | **Last Updated:** 2026-06-19 | **Target:** Wan2.2 **TI2V-5B** (the variant whose VAE we already built)
>
> **Sources (verbatim):** diffusers `models/transformers/transformer_wan.py`, `pipelines/wan/pipeline_wan.py`. Config: [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers). License: Apache-2.0.

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

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

## What it is

Audio-driven (speech/song) video generation: given a **reference image** (identity/scene), an **audio clip**, and a **text prompt**, it generates a talking/performing character lip-/motion-synced to the audio, with optional **pose-video** driving (like Animate). It generates **long videos by chunks** (autoregressive over the audio), each chunk conditioned on the audio window + the previous chunk's tail frames.

## Backbone = Wan2.1-14B (already built)

S2V is the **Wan2.1 T2V-14B DiT** (umT5 cross-attn, per-head 3D RoPE, 6-param AdaLN, FP32 LN) over the **Wan2.1 16-ch VAE** (8× spatial / 4× temporal — already built as `Wan21VaeDecoder`/`Encoder`). Provisional config: `dim 5120, heads 40, head_dim 128, layers 40, ffn 13824, text_dim 4096, in/out 16` (≈ our `WanVideoConfig.T2V_14B`). So the **base DiT, umT5, and VAE all reuse what we have.** Net-new is the audio path + the motion/reference conditioning + the chunked loop.

## Net-new components (the S2V delta)

1. **Audio feature extractor — Wav2Vec2.** S2V runs a frozen **Wav2Vec2** (the model card uses `TencentGameMate/chinese-wav2vec2-base`, hidden 768, 12 layers; some configs use wav2vec2-large, hidden 1024) over the raw waveform (16 kHz) to get per-timestep audio features. It harvests **multiple hidden layers** (stacked, like our Gemma 49-layer harvest for LTX-2), then resamples to the **video fps** so there is one audio feature group per output frame. *Not currently in the engine* (we have Whisper for STT, not wav2vec2). This is the heaviest net-new dependency.

2. **`CausalAudioEncoder`.** A small **causal Conv1d** stack (over the frame/time axis, like the Animate face encoder) that maps the stacked wav2vec2 features → **per-frame audio tokens at the DiT inner dim** (5120). Possibly with a learnable per-layer weight over the stacked wav2vec2 layers. *(Directly analogous to `WanAnimateFaceEncoder` — reuse that pattern.)*

3. **`AudioInjector` (audio cross-attention).** A set of **extra cross-attention modules inserted at specific DiT block indices** (`audio_inject_layers`, e.g. every Nth of the 40 blocks). At each injection point the latent hidden states **cross-attend to the per-frame audio tokens**, temporally aligned (each latent frame's tokens attend that frame's audio token group — **the exact `WanAnimateFaceBlock` temporal-alignment pattern, T∣S**). Some variants add **AdaIN** modulation (`enable_adain`) — an adaptive instance norm conditioned on a pooled audio/global vector. The injected output is added back to the hidden states (residual), like the face adapter.

4. **Reference + motion-frame conditioning (`motioner`).** The reference image is VAE-encoded; long-video continuity uses **"motion frames"** — the last K frames of the previous chunk (VAE-encoded) prepended/concatenated as clean conditioning (zero-padded for the first chunk). Likely realized as extra latent-channel concat (so `in_channels > 16`) and/or a FramePack-style packing. **This is the fuzziest part** — confirm the exact channel layout + masking from `motioner.py`.

5. **Chunked autoregressive pipeline.** Outer loop over audio windows: per chunk → take the audio segment + previous motion frames + reference → denoise one clip → carry its tail frames as the next chunk's motion conditioning. Optional **FramePack** mode for efficiency.

## Reuse map (S2V)

| Need | Reuse (already built) |
|---|---|
| Base 14B DiT (blocks, RoPE, AdaLN, output) | `WanVideoTransformer` / `WanVideoBlock` / `WanRope` / `WanDitOps` |
| 16-ch VAE decode + **multi-frame encode** | `Wan21VaeDecoder` / `Wan21VaeEncoder.Encode` (Phase 0) |
| umT5 text | `T5TextEncoder` + `T5TextEncoderConfig.Umt5Xxl` |
| Per-frame audio → DiT-dim tokens (causal Conv1d) | **`WanAnimateFaceEncoder` pattern** |
| Temporally-aligned audio cross-attn (T∣S) | **`WanAnimateFaceBlock` pattern** |
| Flow-match Euler + CFG + frame streaming | `LancePipelineCommon` / `VideoFrame` |
| Raw audio I/O, mel/resample DSP | `HartsyInference.Audio/Dsp` (resampler, STFT) |

## S2V Implementation Plan (phased)

> Bar: structural + CPU-tested with synthetic weights, validation-pending (the project standard). Each phase compiles + has a tiny-config test before the next.

- **Phase S0 — Wav2Vec2 audio encoder (the gating prerequisite).** Port a CTC-less Wav2Vec2 feature extractor: the conv feature encoder (7 strided Conv1d + GroupNorm/LayerNorm) + the transformer encoder (N layers, the standard MHSA+FFN our backend already does), exposing **all hidden states** (multi-layer harvest). Config preset from the real checkpoint (base 768/12L vs large 1024/24L — confirm). **Largest net-new piece; ~2 files + test.** *Fallback to de-risk:* first accept **pre-computed audio features** (`[T_audio, layers, dim]`) so S1–S3 can be built/tested before Wav2Vec2 lands (same pattern as pipelines that take pre-encoded embeds).
- **Phase S1 — `WanS2VAudioEncoder`** (CausalAudioEncoder): stacked-layer weighting → causal Conv1d stack → per-frame audio tokens `[T_frames, dim]`. Reuse the `WanAnimateFaceEncoder` Conv1d/causal-pad scaffolding. Test: shape + finite.
- **Phase S2 — `WanS2VTransformer`** = base `WanVideoTransformer` + an `AudioInjector` (a `WanAnimateFaceBlock`-style cross-attn at the `audio_inject_layers`, added residually; + optional AdaIN) + the reference/motion-frame latent concat (config `in_channels` bump). Reuses `WanDitOps`/`WanVideoBlock` like `WanVaceTransformer`/`WanAnimateTransformer`. Test: tiny-config forward with synthetic audio tokens → finite velocity (verify audio influences output).
- **Phase S3 — `WanS2VPipeline`** (chunked): reference VAE-encode → per-chunk { audio window → S1 tokens → S2 denoise with motion frames → decode → carry tail frames }. First a **single-chunk** path (simplest), then the autoregressive multi-chunk loop + optional FramePack. Test: single-chunk E2E (synthetic weights) → frames.
- **Phase S4 — converter + SwarmUI + tests.** `WanVideoCheckpointConverter` additions for the S2V keys (audio_injector / audio_encoder / motioner); SwarmUI loader (audio param input, wav2vec2 side-model) + compat class; checklist + memory.

## S2V Open questions (must confirm against the real repo before/while building)

- **Wav2Vec2 variant + config** (chinese-wav2vec2-base 768/12 vs large 1024/24), which hidden layers are harvested, and the feature→fps resampling rule.
- **`audio_inject_layers`** — exact block indices + count; whether AdaIN is on (`enable_adain`, `adain_mode`) and what conditions it.
- **Motion-frame / reference channel layout** (`motioner.py`): how many motion frames, exact `in_channels`, mask channels, FramePack vs plain concat, first-chunk zero-padding.
- **Chunk scheduling**: clip length, audio-window↔frame mapping, overlap, and how the previous tail conditions the next chunk.
- Provisional 14B config numbers vs the actual `Wan-AI/Wan2.2-S2V-14B` checkpoint header.
