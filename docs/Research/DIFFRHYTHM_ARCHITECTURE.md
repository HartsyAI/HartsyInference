# DiffRhythm — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (DiffRhythm pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

DiffRhythm (ASLP-Lab @ NWPU, 2025, [arXiv:2503.01183](https://arxiv.org/abs/2503.01183)) is the first open-source **end-to-end latent diffusion model** for full-length song generation (synchronized vocals + accompaniment) at 44.1 kHz stereo. The pipeline is two-stage and "embarrassingly simple": (1) a VAE compresses 44.1 kHz stereo waveforms into a 64-dim continuous latent at 21.5 Hz (compression factor 2048), and (2) a 1.1B-parameter Diffusion Transformer (DiT) — built from **16 LLaMA-style decoder layers** at hidden=2048, 32 heads, head_dim=64 — denoises the full-song latent in one shot using **flow matching** (no autoregression, no chunking at the DiT level). Conditioning is: a **MuQ-MuLan** style embedding (or a fine-tuned LSTM over MuQ in DiffRhythm-v1) fed via AdaLN-Zero, plus **G2P phoneme tokens** of the LRC lyrics fed via cross-attention with **sentence-level start-timestamp alignment**. A 4m45s song generates in ~10 s on a single H800 (Apache-2.0).

This file covers the DiffRhythm-1 family (v1.0 and v1.2) plus brief notes on the v2 / "+"-line variants. Pure-C# implementation hints assume reuse of existing HartsyInference DiT blocks (Flux/SD3) and 1-D Conv stacks (Kokoro iSTFTNet / Stable Audio VAE). Flow-matching scheduler details live in **[FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)** (to be authored); shared DiT primitives are in [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md) and [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md); VAE primitives in [VAE_ARCHITECTURE.md](VAE_ARCHITECTURE.md); CFG semantics in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources: paper [arXiv:2503.01183](https://arxiv.org/html/2503.01183v1); follow-ups DiffRhythm+ ([arXiv:2507.12890](https://arxiv.org/abs/2507.12890)) and DiffRhythm 2 ([arXiv:2510.22950](https://arxiv.org/abs/2510.22950)); code [github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm); weights at [ASLP-lab on HF](https://huggingface.co/ASLP-lab); MuLan dependency [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large).

---

## Variants — Released Checkpoints

All weights live under the `ASLP-lab` HuggingFace org and ship as `transformer/diffusion_pytorch_model.safetensors` plus a config. License is Apache-2.0 (model + code).

| Model | HF path | Max audio length | Approx params (DiT only) | Notes |
|---|---|---|---|---|
| **DiffRhythm-base** | `ASLP-lab/DiffRhythm-base` | 1 m 35 s (95 s) | ~1.1 B | Original release (Mar 2025). Phoneme + style conditioning. Trained on the 60 kh / 1 M song corpus. |
| **DiffRhythm-full** | `ASLP-lab/DiffRhythm-full` | 4 m 45 s (285 s) | ~1.1 B | Same architecture as base, fine-tuned at full length. The "headline" 10-second-song-in-10-seconds checkpoint. |
| **DiffRhythm-v1.2** | `ASLP-lab/DiffRhythm-1_2` | 1 m 35 s | ~1.1 B | v1.2 (mid-2025): better quality, fewer repetition/omission artifacts, richer instrumentation, supports editing and continuation. |
| **DiffRhythm-v1.2-full** | `ASLP-lab/DiffRhythm-1_2-full` | 4 m 45 s | ~1.1 B | Full-length v1.2. Current default for production use. |
| **DiffRhythm-vae** | `ASLP-lab/DiffRhythm-vae` | — | ~157 M | Shared VAE for all DiT variants. TorchScript `vae_model.pt` (a `ScriptModule`). Latent space is identical to Stable Audio 2's VAE so it is **plug-compatible** with Stable Audio's latent diffusion. |
| **(downstream) DiffRhythm+** | `ASLP-lab/DiffRhythm` (Spaces v1.2 commit) | up to 4m45s | ~1.1 B | Replaces the v1 LSTM-over-MuQ style adapter with **MuQ-MuLan** for unified text+audio style prompts; adds DPO-style preference optimization. Paper: [arXiv:2507.12890](https://arxiv.org/abs/2507.12890). |
| **(downstream) DiffRhythm2** | `ASLP-lab/DiffRhythm2` | full songs | larger | Switches to **Block Flow Matching** for better long-range structure. Out of scope for this doc; reference only. |

**External dependencies** (all required at inference):

| Component | HF path | Size | Role |
|---|---|---|---|
| MuQ-MuLan style encoder | `OpenMuQ/MuQ-MuLan-large` | ~600 MB | Audio↔text joint embedding (512-dim). Replaces v1's LSTM-on-MuQ adapter in v1.2 / DiffRhythm+. |
| MuQ music SSL backbone | `OpenMuQ/MuQ-large-msd-iter` | ~700 MB | Mel-residual-VQ music encoder used inside MuQ-MuLan. |
| espeak-ng | system binary | — | G2P fallback for phoneme extraction (multilingual). |
| MERT / FLAP (optional) | — | — | Reference embedding for evaluation only. |

VAE file size on disk: ~600 MB (FP32 ScriptModule). DiT file size: ~2.1 GB FP32 / ~1.05 GB FP16. Total weight footprint with MuLan: ~4.3 GB (FP16) / ~7 GB (FP32).

---

### 5 Conditioning Summary

| Conditioning | How fed | Shape |
|---|---|---|
| **Timestep `t`** | Sinusoidal → 2 × MLP → add to style → AdaLN-Zero | [B, 2048] |
| **Style** (text or audio) | MuQ-MuLan → 512-d → Linear(512→2048) → add to timestep | [B, 2048] |
| **Lyric phonemes** | G2P → `nn.Embedding(vocab, 2048)` → optional small ConvNeXt blocks (see F5-TTS lineage) → cross-attention K/V at every block | [B, L_phoneme, 2048] |
| **Sentence-level alignment** | Boolean mask `[L_lat, L_phoneme]` built from LRC start times: phonemes of sentence *k* can be attended only from latent positions in `[start_k, start_{k+1})` | mask passed to cross-attn |
| **CFG dropouts at training** | independent 20 % dropout on style, on lyrics, and on both — enables 3-way / 2-way CFG at inference | — |

## Sampling — Flow Matching

DiffRhythm is trained with **conditional flow matching** (CFM), the same family used by F5-TTS, Matcha-TTS, Stable Diffusion 3, Flux, and AuraFlow. There is **no DDPM/DDIM noise schedule**. See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) for the math; here are the DiffRhythm-specific knobs.

| Knob | Value |
|---|---|
| Training objective | predict velocity `v_t = z_1 − z_0` where `z_t = (1−t) z_0 + t z_1`, `z_1 ~ N(0,I)`, `z_0 = VAE.encode(audio)` |
| Inference solver | first-order **Euler** (default) over a uniform `t ∈ [0,1]` grid; second-order midpoint optional |
| Default # inference steps | **32** (NFE = 32 per song; user-configurable 8–50) |
| CFG enabled | yes (joint CFG over both style and lyrics — they trained with **independent 20 % dropout** on each so all 4 corners are well-defined) |
| Default CFG scale | **4.0** in the released `infer.py` (range 1.0–10.0 useful) |
| CFG formula (one-direction) | `v̂ = v_uncond + w · (v_cond − v_uncond)` applied at every Euler step |
| Sway sampling | **yes** — F5-TTS-style "sway sampling": warp `t_i ∈ [0,1]` by `t' = t + s·(cos(π t/2) − 1 + t)` with `s ≈ -1.0` (denser steps near `t=0` for better lyric fidelity); enabled by default |
| Reflow | **not** used in v1; DiffRhythm 2 uses Block Flow Matching, separate work |

End-to-end song generation: **~10 s** for a 4m45s song on H800 with FP16 + Flash-Attn at 32 NFE; **~30 s** on RTX 3060 with `--chunked` + FP16. With 16 NFE the quality is still acceptable and inference halves.

---

## Features

| Feature | Supported? | How |
|---|---|---|
| **Lyrics-to-song** | Yes (primary use case) | LRC file + style prompt |
| **Style transfer / style prompt** | Yes | Reference WAV (v1) or text (v1.2+ via MuLan) |
| **Length control** | Yes | Set `duration_s` → `L_lat = round(duration_s × 21.5)`. Bounded by training: ≤ 95 s for `-base`, ≤ 285 s for `-full`. Beyond 285 s, RoPE extrapolation degrades. |
| **Pure instrumental** | Yes | Empty lyric input (the 20 % training dropout teaches this) |
| **Multi-language vocals** | Yes (en / zh primary; ja / ko / fr / de via espeak-ng) | G2P backend handles language; training corpus was ~30 % zh / ~60 % en / ~10 % instrumental |
| **Vocal-only / accompaniment-only output** | No native stem separation — output is always a single mixed stereo file |
| **Song editing (v1.2)** | Yes | Mask-and-fill in the latent: provide reference WAV + edit interval, the DiT denoises only the masked latent frames |
| **Song continuation (v1.2)** | Yes | Same mechanism, mask the tail |
| **ControlNet-style chord/melody control** | No (not in v1; experimental in DiffRhythm+) |
| **Vocal-cloning from reference singer** | No (style is timbre+genre, not speaker identity) |
| **Variable BPM / time signature control** | No (implicit via style prompt only) |

---

## Memory and Performance

| Configuration | VRAM | Time for 285 s song (32 NFE) |
|---|---|---|
| H800 80 GB, FP16, Flash-Attn, no chunking | ~14 GB | **~10 s** |
| A100 40 GB, FP16, Flash-Attn | ~14 GB | ~12 s |
| RTX 4090, FP16, Flash-Attn | ~12 GB | ~18 s |
| RTX 3060 12 GB, FP16, SDPA, `--chunked` VAE | ~6 GB | ~30 s |
| RTX 3060 8 GB, FP16, SDPA, `--chunked` + 16 NFE | ~7.5 GB | ~20 s |
| FP32 (debug only) | ~24 GB | ~25 s |

**Where the VRAM goes (FP16, full 6128-token attention):**
- DiT weights: ~2.1 GB (FP16)
- VAE weights: ~300 MB (FP16)
- MuLan weights: ~600 MB (FP16, often kept resident)
- Latent buffer `[2, 64, 6128]` (CFG batch): ~6 MB
- Attention scores `[2, 32, 6128, 6128]`: 9.5 GB in FP16 dense → **this is why Flash-Attention is required**; FlashAttn-2 stores no full score matrix, dropping this to ~200 MB activation working set
- AdaLN/SwiGLU activations across 16 blocks: ~1 GB

**Practical thresholds:** without Flash-Attention you cannot fit a 285 s song on anything smaller than A100-40GB. With Flash-Attention, 8 GB is the floor. The C# port **must** use the existing Flash-Attention CUDA kernel (see [FLASH_ATTENTION.md](FLASH_ATTENTION.md)) for the self-attention path.

---

## Comparison to ACE-Step and YuE

All three target full-length (multi-minute) song generation with synchronized vocals + accompaniment, released within ~6 months of each other.

| Aspect | **DiffRhythm** (Mar 2025) | **YuE** (early 2025) | **ACE-Step** (June 2025) |
|---|---|---|---|
| Family | Latent diffusion (DiT + CFM) | LLM (autoregressive) | Latent diffusion + flow matching |
| DiT / LM size | 1.1 B | ~7 B (multi-stage codec LM) | ~3.5 B |
| Audio tokenizer | Continuous VAE (Stable-Audio-2 latent, 64-ch @ 21.5 Hz) | Discrete neural codec (XCodec, ~50 Hz) | Continuous VAE + DAC-like + diffusion-decoder hybrid |
| Sampling | Flow matching, ~10–32 Euler steps | Autoregressive token-by-token (slow) | Flow matching, ~30 steps |
| Lyrics conditioning | G2P phonemes + sentence-level cross-attn alignment | Token-prefixed in the LM context | "Representation alignment" loss against a phoneme encoder |
| Style conditioning | MuQ-MuLan embedding (v1.2+); text or audio | Text prompt in LM context | CLAP / text embedding |
| Output length | Up to 285 s (4m45s) | Up to ~5 min (memory-bounded) | Up to ~4 min |
| Inference time (~4 min song, A100) | **~10 s** | **~5–10 min** (AR) | ~15 s |
| Open weights | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Strengths | **Fastest** (50× MusicGen, 18× faster than YuE-class), simple recipe, plug-compat with Stable Audio VAE, multilingual | Best lyric adherence (LM nature), more controllable | Best vocal quality, strong long-range structure |
| Weaknesses | Vocal articulation slightly weaker than ACE-Step; long-range structure can drift past training length | Slow AR inference, structural artifacts at length | Slightly slower than DiffRhythm; heavier model |

**Position**: DiffRhythm is the **speed-and-simplicity** choice. Its single 1.1 B DiT + 157 M VAE is the lightest full-song model that produces complete songs in seconds. ACE-Step beats it on vocal fidelity but is 50 % heavier and slightly slower; YuE wins on lyric fidelity but is **orders of magnitude** slower due to AR. For a real-time / low-VRAM C# inference engine, **DiffRhythm-v1.2-full is the obvious first target**; ACE-Step is the natural follow-up.

---

## C# Implementation Notes (HartsyInference.Audio)

### 1 VAE (1-D Conv stack)

- **Reuse**: this is the **same 1-D Conv + Snake activation architecture** as the Stable Audio 2 VAE and shares structural DNA with iSTFTNet's decoder. The C# `Conv1D` + `ConvTranspose1D` + `SnakeActivation` ops needed here are the same ones [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) describes for Kokoro and that ACE-Step / Stable Audio will both reuse.
- **Weights**: TorchScript `vae_model.pt` cannot be loaded directly — add a one-time Python pre-processing tool (`tools/diffrhythm_vae_to_safetensors.py`) that loads the `ScriptModule` and dumps a flat `vae.safetensors` keyed by submodule name. C# loads via the existing safetensors loader ([SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md)).
- **Snake activation** kernel: implement as a fused PTX op `y = x + (1/β) · sin²(βx)` with per-channel learned `β`. No CPU fallback needed on hot path (decoder is GPU-only).
- **Chunked decode**: implement window-overlap-and-add at the input of the decoder to support 8 GB GPUs. Window = 32 latent frames (≈ 1.5 s of audio), overlap = 4 frames, raised-cosine crossfade.

### 2 DiT

- **Reuse**: closest existing block in HartsyInference is the **Flux / SD3 DiT block** ([FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md)). Differences:
  - 1-D RoPE instead of 2-D (simpler: one axis, head_dim/2 frequencies).
  - Cross-attention block per DiT block (vs Flux's joint MMDiT). Add a `DiTCrossAttnBlock` variant alongside `DiTSelfAttnBlock`.
  - SwiGLU FFN instead of GELU-MLP (already used by F-Lite per [F_LITE_ARCHITECTURE.md](F_LITE_ARCHITECTURE.md); the same kernel applies).
  - Standard LLaMA RMSNorm (with learned scale), **not** F-Lite's no-scale variant.
- **Attention**: use existing Flash-Attention CUDA path; mandatory for 6 k tokens.
- **Sentence-window cross-attn mask**: build once per song on CPU (`bool[L_lat, L_phoneme]`), upload to GPU as a bias `(0 / -inf)` tensor added pre-softmax.
- **Weight loading**: standard `.safetensors`; expect `transformer.blocks.{i}.{self_attn,cross_attn,ffn,norm1,norm2,norm3,adaLN_modulation}.{weight,bias}`. Confirm exact key names against the released `diffusion_pytorch_model.safetensors` index.

### 3 Text Encoders

- **G2P (espeak-ng)**: C# wrapper via P/Invoke around `libespeak-ng.so` / `libespeak-ng.dll`. Already a known dependency for Kokoro ([KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md)) — share the binding. Need IPA mode + language switching per LRC line if multilingual.
- **LRC parser**: trivial regex `\[(\d+):(\d+\.\d+)\](.*)` per line; sort by timestamp; produce `(start_seconds, raw_text)` tuples.
- **MuQ-MuLan**: **larger lift.** Two sub-models:
  - *Text tower*: XLM-RoBERTa-base (~278 M) + 8 Transformer layers. XLM-R is a BERT-family model; reuse HartsyInference's existing BERT/RoBERTa primitives. SentencePiece tokenizer ([TOKENIZERS.md](TOKENIZERS.md)).
  - *Audio tower*: MuQ encoder (mel → ConvNeXt → Transformer with Mel-Residual VQ heads). New. Plan as its own implementation task before DiffRhythm v1.2 ships; for v1.0 the LSTM-on-MuQ adapter is even smaller but still requires MuQ.
  - For an initial MVP, support **audio-prompt only** (skip MuLan text tower) — this matches DiffRhythm-base/full v1.0 behavior and gets the pipeline to "first sound" with much less work.

### 4 Flow-Matching Scheduler

- See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) (to-be-written) for the shared scheduler used by DiffRhythm, F5-TTS, Stable Audio Open, and audio Flux variants.
- Required ops: uniform `t` grid, sway-sample warp (one-line), Euler step `z ← z − v·dt`, CFG mixing (`v = v_u + w·(v_c − v_u)`).
- Schedule type identifier for the existing scheduler registry ([DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md)): `FlowMatchEulerSway` or similar.

### 5 Suggested Package Layout

Package placement (one folder per package under `src/`, GPU behind `IBackend`):

```
HartsyInference.Audio
  ├─ Vae/StableAudioVae.cs           (shared with Stable Audio path)
  ├─ DiffRhythm/
  │   ├─ DiffRhythmPipeline.cs       (the orchestrator)
  │   ├─ DiffRhythmDiT.cs            (the 1.1 B DiT)
  │   ├─ LrcParser.cs
  │   └─ SentenceAlignmentMask.cs
  ├─ TextEncoders/
  │   ├─ MuQEncoder.cs               (audio side)
  │   ├─ MuLanProjection.cs
  │   └─ XlmRobertaEncoder.cs        (text side, reused from Kokoro/whisper)
  └─ Schedulers/
      └─ FlowMatchEulerSway.cs
```

### 6 First-Slice MVP

For a smoke test before all sub-models are ported:

1. Hard-code phoneme tokens from a pre-computed file (skip espeak-ng for the first run).
2. Hard-code style embedding to a saved-out tensor from the reference Python (skip MuLan).
3. Implement VAE decode + DiT forward + Euler loop only.
4. Verify against a reference Python run with matching seed/CFG/NFE — bit-exact latent at step 0, MAE < 1e-3 on the decoded waveform spectrogram.

This isolates the two biggest pure-C# lifts (DiT and VAE) from the auxiliary encoders.

---
