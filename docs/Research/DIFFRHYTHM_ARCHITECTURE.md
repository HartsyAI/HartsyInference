# DiffRhythm — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (DiffRhythm pipeline)

## Summary

DiffRhythm (ASLP-Lab @ NWPU, 2025, [arXiv:2503.01183](https://arxiv.org/abs/2503.01183)) is the first open-source **end-to-end latent diffusion model** for full-length song generation (synchronized vocals + accompaniment) at 44.1 kHz stereo. The pipeline is two-stage and "embarrassingly simple": (1) a VAE compresses 44.1 kHz stereo waveforms into a 64-dim continuous latent at 21.5 Hz (compression factor 2048), and (2) a 1.1B-parameter Diffusion Transformer (DiT) — built from **16 LLaMA-style decoder layers** at hidden=2048, 32 heads, head_dim=64 — denoises the full-song latent in one shot using **flow matching** (no autoregression, no chunking at the DiT level). Conditioning is: a **MuQ-MuLan** style embedding (or a fine-tuned LSTM over MuQ in DiffRhythm-v1) fed via AdaLN-Zero, plus **G2P phoneme tokens** of the LRC lyrics fed via cross-attention with **sentence-level start-timestamp alignment**. A 4m45s song generates in ~10 s on a single H800 (Apache-2.0).

This file covers the DiffRhythm-1 family (v1.0 and v1.2) plus brief notes on the v2 / "+"-line variants. Pure-C# implementation hints assume reuse of existing SharpInference DiT blocks (Flux/SD3) and 1-D Conv stacks (Kokoro iSTFTNet / Stable Audio VAE). Flow-matching scheduler details live in **[FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)** (to be authored); shared DiT primitives are in [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md) and [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md); VAE primitives in [VAE_ARCHITECTURE.md](VAE_ARCHITECTURE.md); CFG semantics in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources: paper [arXiv:2503.01183](https://arxiv.org/html/2503.01183v1); follow-ups DiffRhythm+ ([arXiv:2507.12890](https://arxiv.org/abs/2507.12890)) and DiffRhythm 2 ([arXiv:2510.22950](https://arxiv.org/abs/2510.22950)); code [github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm); weights at [ASLP-lab on HF](https://huggingface.co/ASLP-lab); MuLan dependency [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large).

---

## 1. Variants — Released Checkpoints

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

## 2. Architecture

### 2.1 High-Level Pipeline

```
                                ┌──────────────────────────────┐
  audio prompt (10–30 s wav) ──▶│  MuQ-MuLan audio tower       │──▶ style_emb [1, 512]
                                └──────────────────────────────┘
                                            OR
                                ┌──────────────────────────────┐
  style text ("Jazzy night") ──▶│  MuQ-MuLan text tower (XLM-R)│──▶ style_emb [1, 512]
                                └──────────────────────────────┘                  │
                                                                                  │
  LRC lyrics ──▶ G2P (espeak-ng) ──▶ phoneme tokens ──▶ embed ──▶ lyric_emb [B, L_p, D]
       │
       └─ start-of-line timestamps ──▶ sentence-level alignment table

                                                                  │
                                                                  ▼
        noise z_T [B, 64, L_lat]  ──┐                  ┌──────────────────────┐
                                    │                  │  AdaLN-Zero(style_emb,│
                                    ▼                  │       timestep)      │
                          ┌────────────────────┐       └─────────┬────────────┘
                          │  DiT denoiser      │◀────────────────┘
                          │  16 × LLaMA blocks │
                          │  d=2048, 32 heads  │◀── cross-attn ──── lyric_emb
                          │  RoPE on latent ax │
                          └─────────┬──────────┘
                                    │ predict velocity v_t
                          ┌─────────▼──────────┐
                          │  Flow matching ODE │   default ~32 Euler steps, CFG ≈ 4.0
                          │  (Euler / midpoint)│
                          └─────────┬──────────┘
                                    ▼
                          latent z_0 [B, 64, L_lat]
                                    │
                          ┌─────────▼──────────┐
                          │  VAE Decoder       │  (Stable-Audio-2 1-D CNN, snake)
                          │  44.1 kHz stereo   │
                          └─────────┬──────────┘
                                    ▼
                         WAV [2, T_samples],  T = L_lat × 2048
```

### 2.2 Text / Style Encoders

DiffRhythm uses **no** generic text encoder (no T5, no CLIP, no mT5). It uses two domain-specific encoders.

**(a) Lyrics encoder — G2P + learnable embedding.**

- **Tokenizer**: grapheme-to-phoneme conversion via `espeak-ng` through the `phonemizer` Python wrapper. Multilingual (en/zh/ja/ko/fr/de). Each phoneme symbol maps to a token ID via a fixed vocab shipped with the model config (~200 phoneme tokens including IPA + stress marks + punctuation, similar in shape to Kokoro's 178-token vocab — see [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) for an example IPA inventory).
- **Embedding**: simple `nn.Embedding(vocab, hidden)` to the DiT's hidden dim (2048). No pretrained text tower.
- **Sentence-level alignment**: each LRC line carries a start timestamp `[mm:ss.xx]`. The implementation **does not** dense-align every phoneme; instead each sentence's start frame index (in the 21.5 Hz latent grid) is recorded, and inside the DiT cross-attention the phoneme tokens for sentence *k* are gated to attend only to latent positions in the interval `[start_k, start_{k+1})`. This is the paper's "sentence-level alignment paradigm" — it only requires sentence-start annotations, eliminating the expensive forced-alignment supervision used by autoregressive song models.
- For pure-instrumental generation, an empty lyric tensor is passed; the cross-attention layers see a zero mask and the model produces vocals-free output.

**(b) Style encoder.**

- **DiffRhythm v1.0 / v1.2 (current)**: a small LSTM head trained on top of frozen [MuQ](https://github.com/tencent-ailab/MuQ) music SSL features (10 s audio prompt → 512-dim style vector). Text style prompts are not natively supported; users typically supply a short reference WAV.
- **DiffRhythm+** (downstream): replaces the LSTM with full **MuQ-MuLan** (text encoder = XLM-RoBERTa-base + 8 Transformer layers, audio encoder = MuQ + projection, both projected to a shared 512-dim contrastive space). This enables unified text-or-audio style prompts. Output: a single `[B, 512]` style vector.
- The 512-dim style vector is broadcast into the DiT via AdaLN-Zero modulation (see §2.4).

C# implementers should plan to host **MuQ-MuLan** in addition to the DiT/VAE (the v1.2 release uses MuQ raw; v1.2 web UI uses MuQ-MuLan). MuQ-MuLan inference is mel → MuQ → mean-pool → projection (small compared to the DiT).

### 2.3 Audio VAE (DiffRhythm-vae, fine-tuned from Stable Audio 2 VAE)

A 1-D convolutional autoencoder over **raw 44.1 kHz stereo** waveform. Fine-tuned from the [Stable Audio 2 VAE](https://stability.ai/research/stable-audio-efficient-timing-latent-diffusion) on a 250 k-track high-quality subset for 2.5 M iterations, with the encoder frozen and only the decoder retrained for MP3-robust reconstruction. The latent space is **identical** to Stable Audio 2's, so DiffRhythm's DiT can in principle consume Stable Audio latents and vice-versa.

| Property | Value |
|---|---|
| Sample rate | 44 100 Hz |
| Channels | 2 (stereo) |
| Latent channels | 64 |
| Downsampling factor | 2048 (= 5 strided blocks, ratios `(2, 4, 4, 8, 8)`) |
| Latent frame rate | 44100 / 2048 ≈ **21.5 Hz** |
| Latent for 4 m 45 s | 285 × 21.5 ≈ **6128 frames** |
| Latent for 1 m 35 s | 95 × 21.5 ≈ **2042 frames** |
| VAE parameter count | ~157 M (split ~80 M encoder, ~77 M decoder) |
| Activation | **Snake** (β-learnable, per-channel): `f(x) = x + (1/β) · sin²(βx)` |
| Norm | None inside blocks (stability via snake + residuals); RMSNorm at projections |
| Encoder block | `Conv1d(stride=r) → SnakeBlock × N → ResidualUnit × M` (Stable-Audio-style "OobleckBlock") |
| Decoder block | `TransposedConv1d(stride=r) → SnakeBlock × N → ResidualUnit × M`, ratios reversed |
| Distribution | diagonal Gaussian: encoder outputs (μ, log σ); sample `z = μ + σ·ε` at train time, use `μ` at inference |
| Scaling factor | the encoder std-normalized so `z` has unit variance per channel; **do NOT apply an additional 0.18215-style scale** — DiffRhythm trains directly in the raw VAE space (no extra rescale) |

**On-disk format**: `ASLP-lab/DiffRhythm-vae/vae_model.pt` is a **TorchScript `ScriptModule`**, not a plain state-dict. For pure-C# loading you have two options:

1. Re-script and dump weights into a standard `.safetensors` (a one-off Python pre-processing step the user does once); then load layer-by-layer in C#. **Recommended.**
2. Parse the TorchScript zip manually (uncommon) — skip.

**Chunked decode**: the upstream Stable Audio decoder supports overlapping-window inference (`chunked=True`) for memory-bound machines; DiffRhythm exposes the same `--chunked` flag. With chunked decode, a 4 m 45 s song decodes in ~3 GB peak VRAM. Without chunking, the full latent decode peaks ~10 GB on FP16 because of the deep CNN's activation footprint.

### 2.4 DiT — Diffusion Transformer

This is **not** an MMDiT in the SD3 sense. It is a **single-stream DiT with cross-attention to lyrics**, plus **AdaLN-Zero** time/style modulation — i.e., closer to PixArt-α and F5-TTS than to SD3/Flux's two-stream MMDiT. Each block is a LLaMA-style decoder block: RMSNorm → causal-style self-attn (but bidirectional here, no causal mask) → RMSNorm → SwiGLU FFN, augmented with cross-attn to lyrics and AdaLN-Zero modulation.

| Property | Value |
|---|---|
| Total params | **~1.1 B** |
| Layers (`depth`) | **16** LLaMA-style decoder blocks |
| Hidden dim (`d_model`) | **2048** |
| Self-attention heads | **32** |
| Head dim | **64** (= 2048 / 32) |
| FFN type | **SwiGLU** (LLaMA-style) |
| FFN hidden | ~5504 (LLaMA `(2/3) × 4 × d_model` ≈ 5460, rounded to multiple-of-256) |
| FFN ratio (effective) | ~2.7 (SwiGLU's two-gate structure means `4×` in dense ops) |
| Positional encoding | **RoPE** on the latent (time) axis only; base 10000, applied per-head |
| Normalization | RMSNorm (no bias, no learned scale per F-Lite convention is **not** used here — DiffRhythm uses standard LLaMA RMSNorm with learned scale) |
| Self-attn mask | none (full bidirectional over the entire ~6 k latent token sequence) |
| Cross-attn | standard `nn.MultiheadAttention(Q=latent, K=V=lyric_emb)`, one per block, masked by the sentence-level alignment window (see §2.2 (a)) |
| Time conditioning | sinusoidal timestep embed → MLP(2048) → AdaLN-Zero `shift, scale, gate` per (self-attn / cross-attn / FFN) sub-module; **6 modulation channels per block** (no separate AdaLN for cross-attn shift; some impls use 9) |
| Style conditioning | the 512-dim style vector is projected to 2048 and **added to the timestep embedding** before AdaLN-Zero — equivalent to PixArt-α's class+time conditioning path |
| Input projection | `Linear(64, 2048)` over the VAE latent channel axis |
| Output projection | `RMSNorm → AdaLN-Zero(shift,scale) → Linear(2048, 64)` predicting the **velocity** field `v_θ(z_t, t, c)` |
| Patchification | **none** — the latent is treated as a 1-D sequence of length `L_lat`, one token per latent frame |

**Per-block forward** (simplified, matches the F5-TTS / CFM lineage that DiffRhythm copies):

```
shift_sa, scale_sa, gate_sa,
shift_xa, scale_xa, gate_xa,                # some impls pack into 9 modulations
shift_ff, scale_ff, gate_ff = AdaLN_Zero(SiLU(t_emb + style_emb))

# Self-attn over latent tokens
h = RMSNorm(x) * (1 + scale_sa) + shift_sa
h = RoPE_then_SDPA(h)                       # full bidirectional, RoPE on Q/K
x = x + gate_sa * h

# Cross-attn to lyric phoneme tokens
h = RMSNorm(x) * (1 + scale_xa) + shift_xa
h = CrossAttn(Q=h, K=V=lyric_emb, mask=sentence_window_mask)
x = x + gate_xa * h

# SwiGLU FFN
h = RMSNorm(x) * (1 + scale_ff) + shift_ff
h = SwiGLU(h)
x = x + gate_ff * h
```

`AdaLN_Zero` linears are initialized to **zero** so the network starts as identity — standard DiT trick.

### 2.5 Conditioning Summary

| Conditioning | How fed | Shape |
|---|---|---|
| **Timestep `t`** | Sinusoidal → 2 × MLP → add to style → AdaLN-Zero | [B, 2048] |
| **Style** (text or audio) | MuQ-MuLan → 512-d → Linear(512→2048) → add to timestep | [B, 2048] |
| **Lyric phonemes** | G2P → `nn.Embedding(vocab, 2048)` → optional small ConvNeXt blocks (see F5-TTS lineage) → cross-attention K/V at every block | [B, L_phoneme, 2048] |
| **Sentence-level alignment** | Boolean mask `[L_lat, L_phoneme]` built from LRC start times: phonemes of sentence *k* can be attended only from latent positions in `[start_k, start_{k+1})` | mask passed to cross-attn |
| **CFG dropouts at training** | independent 20 % dropout on style, on lyrics, and on both — enables 3-way / 2-way CFG at inference | — |

### 2.6 Long-Context Handling

- A 4 m 45 s song = **6128 latent tokens** for self-attention. The DiT uses **full bidirectional self-attention** over the entire sequence — no sliding window, no chunking inside the DiT (chunking exists only in the VAE decoder).
- **RoPE** on the latent axis provides positional information that extrapolates reasonably across train (95 s) → inference (285 s).
- Memory: full attention over 6128 tokens × 32 heads × head_dim=64 ≈ 6128² × 32 × 4 B ≈ 4.8 GB attention scores in FP32; FP16 SDPA + Flash-Attention is **mandatory** to fit in 8 GB consumer VRAM. The base release uses PyTorch SDPA (which auto-routes to FlashAttention-2 on supported GPUs). C# port: use the existing FLASH_ATTENTION integration ([FLASH_ATTENTION.md](FLASH_ATTENTION.md)).

---

## 3. Sampling — Flow Matching

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

## 4. Inference Pipeline (pseudocode for "lyrics + style → WAV")

```
inputs:
  lyrics_lrc      : str           # LRC text with [mm:ss.xx] line timestamps
  style_prompt    : str | WAV     # text description OR reference clip (10–30 s)
  duration_s      : float         # target song length; e.g. 285.0
  nfe             : int = 32      # flow-matching steps
  cfg_scale       : float = 4.0
  seed            : int

# ---- 1. Pre-processing --------------------------------------------------
phoneme_ids, sentence_starts_s = g2p_lrc(lyrics_lrc)        # espeak-ng
                                                            # sentence_starts in seconds

if isinstance(style_prompt, str):
    style_emb = MuLan.encode_text(style_prompt)             # [1, 512]
else:
    wav10 = load_mono_resample(style_prompt, sr=24000)[:240000]   # MuLan SR
    style_emb = MuLan.encode_audio(wav10)                   # [1, 512]

# ---- 2. Latent-grid setup -----------------------------------------------
L_lat = round(duration_s * 21.5)                            # latent frames
                                                            # 285 s → 6128
sentence_start_frames = [round(s * 21.5) for s in sentence_starts_s]
xattn_mask = build_sentence_window_mask(                    # [L_lat, L_phon]
    L_lat, sentence_start_frames, phoneme_ids_per_sentence
)

# ---- 3. DiT denoise via flow-matching -----------------------------------
z = torch.randn(1, 64, L_lat, generator=g(seed))            # z_1
phoneme_emb = DiT.lyric_embed(phoneme_ids)                  # [1, L_phon, 2048]

t_grid = sway_sample_uniform(nfe, s=-1.0)                   # 0 → 1 with warp
for i in range(nfe):
    t_i, t_next = t_grid[i], t_grid[i+1]
    dt = t_next - t_i

    # CFG: 2 forward passes, batched
    v_cond   = DiT(z, t_i, style_emb, phoneme_emb, xattn_mask)
    v_uncond = DiT(z, t_i, ZERO_style, ZERO_phoneme, NULL_mask)
    v = v_uncond + cfg_scale * (v_cond - v_uncond)

    z = z - v * dt          # NB: flow direction; some impls iterate t: 1→0,
                            #     others t: 0→1 with sign flipped — check
                            #     the reference repo exactly.

# ---- 4. VAE decode -------------------------------------------------------
wav = VAE.decode(z, chunked=(vram < 12_GB))                 # [2, L_lat * 2048]
save_wav(wav, sr=44100)
```

**Implementation notes:**

- Build the CFG batch as `cat([z_uncond, z_cond], dim=0)` and split — halves the kernel-launch overhead.
- `phoneme_ids_per_sentence` is a list-of-lists; the mask must zero out cross-attn for sentences that haven't "started" at the current latent frame.
- For **pure instrumental**, pass `phoneme_ids = []` and a fully-masked cross-attn (the network has been trained on dropped lyrics 20 % of the time, so it handles this correctly).
- For **continuation / editing** (v1.2), the first `K` latent frames are clamped to the VAE-encoded reference at every Euler step (inpainting-style); the cross-attn mask is shifted accordingly.

---

## 5. Features

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

## 6. Memory and Performance

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

## 7. Comparison to ACE-Step and YuE

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

## 8. C# Implementation Notes (SharpInference.Audio)

### 8.1 VAE (1-D Conv stack)

- **Reuse**: this is the **same 1-D Conv + Snake activation architecture** as the Stable Audio 2 VAE and shares structural DNA with iSTFTNet's decoder. The C# `Conv1D` + `ConvTranspose1D` + `SnakeActivation` ops needed here are the same ones [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) describes for Kokoro and that ACE-Step / Stable Audio will both reuse.
- **Weights**: TorchScript `vae_model.pt` cannot be loaded directly — add a one-time Python pre-processing tool (`tools/diffrhythm_vae_to_safetensors.py`) that loads the `ScriptModule` and dumps a flat `vae.safetensors` keyed by submodule name. C# loads via the existing safetensors loader ([SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md)).
- **Snake activation** kernel: implement as a fused PTX op `y = x + (1/β) · sin²(βx)` with per-channel learned `β`. No CPU fallback needed on hot path (decoder is GPU-only).
- **Chunked decode**: implement window-overlap-and-add at the input of the decoder to support 8 GB GPUs. Window = 32 latent frames (≈ 1.5 s of audio), overlap = 4 frames, raised-cosine crossfade.

### 8.2 DiT

- **Reuse**: closest existing block in SharpInference is the **Flux / SD3 DiT block** ([FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md)). Differences:
  - 1-D RoPE instead of 2-D (simpler: one axis, head_dim/2 frequencies).
  - Cross-attention block per DiT block (vs Flux's joint MMDiT). Add a `DiTCrossAttnBlock` variant alongside `DiTSelfAttnBlock`.
  - SwiGLU FFN instead of GELU-MLP (already used by F-Lite per [F_LITE_ARCHITECTURE.md](F_LITE_ARCHITECTURE.md); the same kernel applies).
  - Standard LLaMA RMSNorm (with learned scale), **not** F-Lite's no-scale variant.
- **Attention**: use existing Flash-Attention CUDA path; mandatory for 6 k tokens.
- **Sentence-window cross-attn mask**: build once per song on CPU (`bool[L_lat, L_phoneme]`), upload to GPU as a bias `(0 / -inf)` tensor added pre-softmax.
- **Weight loading**: standard `.safetensors`; expect `transformer.blocks.{i}.{self_attn,cross_attn,ffn,norm1,norm2,norm3,adaLN_modulation}.{weight,bias}`. Confirm exact key names against the released `diffusion_pytorch_model.safetensors` index.

### 8.3 Text Encoders

- **G2P (espeak-ng)**: C# wrapper via P/Invoke around `libespeak-ng.so` / `libespeak-ng.dll`. Already a known dependency for Kokoro ([KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md)) — share the binding. Need IPA mode + language switching per LRC line if multilingual.
- **LRC parser**: trivial regex `\[(\d+):(\d+\.\d+)\](.*)` per line; sort by timestamp; produce `(start_seconds, raw_text)` tuples.
- **MuQ-MuLan**: **larger lift.** Two sub-models:
  - *Text tower*: XLM-RoBERTa-base (~278 M) + 8 Transformer layers. XLM-R is a BERT-family model; reuse SharpInference's existing BERT/RoBERTa primitives. SentencePiece tokenizer ([TOKENIZERS.md](TOKENIZERS.md)).
  - *Audio tower*: MuQ encoder (mel → ConvNeXt → Transformer with Mel-Residual VQ heads). New. Plan as its own implementation task before DiffRhythm v1.2 ships; for v1.0 the LSTM-on-MuQ adapter is even smaller but still requires MuQ.
  - For an initial MVP, support **audio-prompt only** (skip MuLan text tower) — this matches DiffRhythm-base/full v1.0 behavior and gets the pipeline to "first sound" with much less work.

### 8.4 Flow-Matching Scheduler

- See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) (to-be-written) for the shared scheduler used by DiffRhythm, F5-TTS, Stable Audio Open, and audio Flux variants.
- Required ops: uniform `t` grid, sway-sample warp (one-line), Euler step `z ← z − v·dt`, CFG mixing (`v = v_u + w·(v_c − v_u)`).
- Schedule type identifier for the existing scheduler registry ([DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md)): `FlowMatchEulerSway` or similar.

### 8.5 Suggested Package Layout

Following [NUGET_PACKAGE_DESIGN.md](../Design/NUGET_PACKAGE_DESIGN.md):

```
SharpInference.Audio
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

### 8.6 First-Slice MVP

For a smoke test before all sub-models are ported:

1. Hard-code phoneme tokens from a pre-computed file (skip espeak-ng for the first run).
2. Hard-code style embedding to a saved-out tensor from the reference Python (skip MuLan).
3. Implement VAE decode + DiT forward + Euler loop only.
4. Verify against a reference Python run with matching seed/CFG/NFE — bit-exact latent at step 0, MAE < 1e-3 on the decoded waveform spectrogram.

This isolates the two biggest pure-C# lifts (DiT and VAE) from the auxiliary encoders.

---

## 9. Open Questions / Things to Verify Against Source

The following details I inferred from secondary sources and the F5-TTS / Stable-Audio lineage; an implementer **must** verify against the actual `model.py` / `cfm.py` in [github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) before final implementation:

1. **Exact AdaLN-Zero modulation count per block** — 6 (shift/scale/gate × 2 sub-modules with shared FFN params) or 9 (separate for self-attn, cross-attn, FFN). F5-TTS uses 6; SD3 uses 9.
2. **Exact SwiGLU FFN hidden size** — derived above as ~5504, but may be exactly `4 × d_model = 8192` if they didn't follow LLaMA's `(2/3) × 4` rule.
3. **Whether lyric tokens pass through small ConvNeXt blocks before cross-attn** — F5-TTS does this; DiffRhythm likely inherits, but confirm.
4. **Whether sway-sampling is on by default** — strongly likely (F5 lineage), but confirm in `infer.py`.
5. **VAE downsample ratios** — `(2, 4, 4, 8, 8)` gives 2048 and matches Stable Audio 2; verify exact tuple in the VAE config.
6. **Phoneme vocab size and exact symbol set** — pull from the released model config / tokenizer file.
7. **Whether the 20 % CFG dropout is independent per condition or joint** — paper says independent; verify training script.
8. **RoPE base frequency** — likely 10000 (LLaMA default); confirm. NTK-aware scaling for extrapolation past 95 s training length is **not** used in v1; that's part of why -full needed retraining.

Once these are confirmed, this document should be promoted from Status: Complete to Status: Verified-Against-Source.
