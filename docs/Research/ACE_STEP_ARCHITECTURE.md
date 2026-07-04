# ACE-Step — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (ACE-Step pipeline)

## Table of Contents

1. [Summary](#1-summary)
2. [Detailed Findings](#2-detailed-findings)
3. [Key Numbers / Constants](#3-key-numbers--constants)
4. [Data Layouts / Formats](#4-data-layouts--formats)
5. [Algorithm Steps](#5-algorithm-steps)
6. [Reference Implementations](#6-reference-implementations)
7. [Differences Between Implementations](#7-differences-between-implementations)
8. [Open Questions](#8-open-questions)
9. [Implementation Notes for HartsyInference](#9-implementation-notes-for-hartsyinference)

---

## 1. Summary

**ACE-Step** is an open-source music generation foundation model from ACE Studio + StepFun. As of mid-2026 two distinct generations have been released:

- **ACE-Step v1 (3.5B, May 2025)** — the original Diffusion Transformer that put the project on the map. Pure flow-matching DiT operating on a 2-D mel-latent produced by a custom "Music-DCAE" autoencoder, conditioned by UMT5-base text embeddings, a Conformer lyric encoder, and a 512-d global speaker/timbre vector. Trained with auxiliary REPA cosine-alignment losses to MERT and m-HuBERT embeddings. Apache 2.0. Generates 4 min of stereo 48 kHz music in ~20 s on an A100. This is the variant the public ecosystem (ComfyUI, Replicate, Diffusers) standardized on.
- **ACE-Step v1.5 (Jan 2026 → April 2026 XL)** — a major architectural rewrite. Replaces the diffusers DiT with a **Qwen3-style decoder LM** (alternating sliding-window / full attention, RMSNorm, SwiGLU, GQA) whose **output is FSQ-quantized audio tokens** rather than continuous latents. Three size tiers: **2B base/sft/turbo** (hidden 2048, 24 layers, 16 heads), **5B XL base/sft/turbo** (hidden 2560, 32 layers, 32 heads), each tier paired with a separate **Music DCAE / 1D Stable-Audio-format VAE** for waveform reconstruction. Adds optional **Qwen3-based "planner" LMs** (0.6B / 1.7B / 4B) that act as omni-capable lyric/style writers. MIT license. <4 GB VRAM for the 2B turbo, ~9 GB for the XL.

Both generations share: UMT5 text encoder, 32k-context RoPE (θ=1e6), in/out 8-channel latents (v1) or 192-channel pre-quant features (v1.5), structured-lyric tags (`[verse]`, `[chorus]`, `[bridge]`, `[instrumental]`, …), 50+ language support, and a flow-matching scheduler with shift=3.0 (v1) or learned `(μ=-0.4, σ=1.0)` timestep distribution (v1.5).

For HartsyInference, the **v1 3.5B model is the implementation target first**: its component boundaries are clean (UMT5 text encoder + Conformer lyric encoder + DiT + DCAE + HiFiGAN vocoder), it reuses UMT5 (already in HartsyInference for AuraFlow) and a flow-match Euler scheduler (already in HartsyInference for Flux/SD3). v1.5 should follow once Qwen3-style decoder LM + FSQ vocoder are scoped — those are HartsyInference.LLM territory plus a new audio codec component.

This file covers ACE-Step architecture, weights, conditioning, and inference. The shared flow-matching mathematics live in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md). The HiFiGAN vocoder family used by both generations is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). UMT5 details live with the AuraFlow text-encoder notes ([TEXT_ENCODERS.md](TEXT_ENCODERS.md)).

---

## 2. Detailed Findings

### 2.1 Released Model Variants

#### v1 family (3.5B DiT — May 2025)

| Repo (HF) | Size | Use | Notes |
|---|---|---|---|
| `ACE-Step/ACE-Step-v1-3.5B` | 8.28 GB total | General text+lyrics → song, 17 languages | Flagship. Diffusion-pipeline distribution (diffusers 0.32.2 format) |
| `ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA` | small LoRA | Style fine-tune | LoRA on top of v1 |
| `ACE-Step/RapMachine` (LoRA) | small LoRA | Rap-specialized vocal | Released May 2025 |

The v1 base archive contains four sub-folders, each with its own `config.json` + safetensors:

```
ACE-Step-v1-3.5B/
├── ace_step_transformer/      # The DiT (24 layers × inner 2560)
├── music_dcae_f8c8/           # Mel → 8-ch latent autoencoder (Sana-style DCAE)
├── music_vocoder/             # HiFi-GAN variant ADaMoSHiFiGANV1 (mel → 44.1 kHz wav)
├── umt5-base/                 # Standard google/umt5-base text encoder
├── config.json                # Top-level pipeline glue (639 B)
└── README.md
```

#### v1.5 family (2B/5B Qwen3-style LM with FSQ tokens — Jan-April 2026)

| Repo (HF) | Params | Hidden / Layers / Heads | Use |
|---|---|---|---|
| `ACE-Step/acestep-v15-base` | ~2B | 2048 / 24 / 16 | Pre-train, 50 steps CFG, full feature set |
| `ACE-Step/acestep-v15-sft` | ~2B | 2048 / 24 / 16 | SFT polish over base, 50 steps |
| `ACE-Step/Ace-Step1.5` (turbo) | ~2B | 2048 / 24 / 16 | 8-step turbo, ~752 likes on HF, most-downloaded variant |
| `ACE-Step/acestep-v15-turbo-continuous` | ~2B | 2048 / 24 / 16 | Continuous (non-FSQ) variant for streaming |
| `ACE-Step/acestep-v15-xl-base` | ~5B | 2560 / 32 / 32 | Pre-train, higher fidelity |
| `ACE-Step/acestep-v15-xl-sft` | ~5B | 2560 / 32 / 32 | SFT polish |
| `ACE-Step/acestep-v15-xl-turbo` | ~5B | 2560 / 32 / 32 | 8-step turbo |
| `ACE-Step/acestep-v15-xl-turbo-diffusers` | ~5B | 2560 / 32 / 32 | Same weights re-exported in diffusers format |
| `ACE-Step/acestep-v15-turbo-rl` | ~2B | 2048 / 24 / 16 | RL-aligned (announced, not all weights public yet) |
| `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format` | small | – | 1-D VAE in Stable-Audio-Open latent format (alternative codec) |

#### v1.5 LM planners (Qwen3 base, optional)

| Repo (HF) | Params | Use |
|---|---|---|
| `acestep-5Hz-lm-0.6B` | 0.6B | Lyric / style CoT + query rewrite (lightweight) |
| `acestep-5Hz-lm-1.7B` | 1.7B | Balanced |
| `acestep-5Hz-lm-4B` | 4B | Strong composition + audio understanding |

These LMs take a free-form user request, emit a structured song blueprint (genre tags, lyrics with structure markers, BPM, key), and hand it to the DiT/decoder. They are pure HartsyInference.LLM work — the audio pipeline only needs to consume the resulting structured prompt.

#### GGUF quantizations (community)

- `Serveurperso/ACE-Step-1.5-GGUF` — Q4/Q5/Q8 quantizations of v1.5 variants for the `acestep.cpp` (GGML) project. ACE-Step has a working VST3 plugin built on this stack.

License: **Apache 2.0** for v1, **MIT** for v1.5.

### 2.2 v1 3.5B Architecture (the implementation target)

#### Top-level pipeline

```
text prompt ──► UMT5-base ─► text_embeds   (B, T_text, 768)
                            ─► attn_mask
lyrics ──► lang detect ─► VoiceBpeTokenizer ─► token IDs
                            ─► ConformerEncoder ─► lyric_embeds  (B, T_lyric, 1024)
speaker (opt) ──► precomputed 512-d vec   (B, 512)
genre / style tags ──► UMT5 ─► (folded into text_embeds path) ─► 768-d

noise z_t  (B, 8, 16, F_lat)  ─► ACEStepTransformer2DModel ─► velocity v
                                   └── FlowMatchEuler step ─► z_{t-1}
                                                              ...
                                                              ─► z_0  (B, 8, 16, F_lat)

z_0 ─► Music-DCAE decoder ─► mel  (B, 2, 128, F_mel)        # 2 = stereo, per-channel
mel ─► ADaMoSHiFiGANV1 vocoder ─► wav  (B, 2, T_samples@44.1kHz)
wav ─► optional resample to user rate
```

The latent grid is **2-D**: `(channels=8, height=16, width=F_lat)`. The 16-tall "height" axis is the mel-frequency axis after DCAE 8× downsampling (128 mel bins / 8 = 16). The "width" axis is time, downsampled 8× from mel frames (`F_lat = mel_frames / 8`).

#### ACEStepTransformer2DModel (`ace_step_transformer/config.json`, verbatim)

```json
{
  "_class_name": "ACEStepTransformer2DModel",
  "_diffusers_version": "0.32.2",
  "attention_head_dim": 128,
  "in_channels": 8,
  "inner_dim": 2560,
  "lyric_encoder_vocab_size": 6693,
  "lyric_hidden_size": 1024,
  "max_height": 16,
  "max_position": 32768,
  "max_width": 32768,
  "mlp_ratio": 2.5,
  "num_attention_heads": 20,
  "num_layers": 24,
  "out_channels": 8,
  "patch_size": [16, 1],
  "rope_theta": 1000000.0,
  "speaker_embedding_dim": 512,
  "ssl_encoder_depths": [8, 8],
  "ssl_latent_dims": [1024, 768],
  "ssl_names": ["mert", "m-hubert"],
  "text_embedding_dim": 768
}
```

Note that **inner_dim = 2560 = num_attention_heads (20) × attention_head_dim (128)** — the published model is the 2560-wide / 24-layer / 20-head / 128-head-dim variant. (An older incorrectly-documented spec circulated naming 1536 / 24 / 64; that does not match `config.json`.) Total params ≈ 3.5B once you include the lyric Conformer (8 layers × 1024), SSL projection MLPs, time/text/speaker projections, and the patch in / final out heads.

#### LinearTransformerBlock (one of 24)

Despite the name "Linear", the attention is **standard scaled-dot-product** with rotary positional embeddings. The "Linear" refers to the FFN being a gated convolution (`GLUMBConv`) rather than a plain MLP — borrowed from Sana / EfficientViT design. Layout:

```
x ── RMSNorm(elementwise_affine=False, eps=1e-6) (norm1)
  │      │
  │      └─► scale_msa, shift_msa  (from adaLN scale_shift_table[temb] → 6 chunks)
  │
  ├─► Self-Attn (RoPE on q/k; QK-norm if present)
  │      └─► gate_msa
  ├─► Cross-Attn → text_embeds + speaker_token (1) + lyric_embeds (concatenated)
  │
  ├─► RMSNorm (norm2)
  │      └─► scale_mlp, shift_mlp
  ├─► GLUMBConv FFN (1×1 → 2× → DW conv 3×3 → SiLU gate chunk → 1×1)
  │      └─► gate_mlp
  └─► residual add
```

AdaLN modulation = **6 chunks per block** (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), produced by `self.scale_shift_table` indexed by the timestep embedding. This is the same pattern as PixArt-α/Sana/Lumina, so HartsyInference can reuse its existing `AdaLNZero6` modulation kernel.

RoPE: `Qwen2RotaryEmbedding(dim=128, max_position_embeddings=32768, base=1_000_000.0)`. Same RoPE flavor as Qwen2 / Lumina-Next. Independent RoPE caches for self-attention (audio-latent positions) and cross-attention (text + lyric positions).

#### Patch in / out

```
PatchEmbed:
  Conv2d(8 → 64, k=3, s=1)
  GroupNorm(32, 64)
  Conv2d(64 → 2560, k=patch_size=[16,1], s=[16,1])   # collapses height-16 → height-1
  flatten → (B, F_lat, 2560)
```

After patchification the sequence is purely **1-D along time** with `F_lat` tokens. Effective audio token rate = `44100 / 512 / 8 ≈ 10.77 Hz` (one DiT token ≈ 92.9 ms). A 4-minute song = `240 × 10.77 ≈ 2585` tokens. Plenty of headroom under the 32k RoPE max.

```
T2IFinalLayer:
  RMSNorm + adaLN (shift, scale) from temb
  Linear(2560 → 16 × 1 × 8)        # = patch[0]*patch[1]*out_channels
  reshape/einsum to (B, 8, 16, F_lat)
```

#### Cross-attention conditioning sequence

The cross-attention K/V sequence is the concatenation of three sources:

```
[ speaker_token(1)  |  text_embeds(T_text, 2560)  |  lyric_embeds(T_lyric, 2560) ]
```

with corresponding key-padding mask. The projections:

- `speaker_embedder = nn.Linear(512, 2560)` — speaker_token is 1 token after projection
- `genre_embedder = nn.Linear(768, 2560)` — applied to UMT5 last_hidden_state before concat (genre/style tags travel through the same UMT5 path as the natural-language prompt; there is no separate "genre" encoder)
- `lyric_proj = nn.Linear(1024, 2560)` — applied after the Conformer

#### Lyric encoder

```python
lyric_embs = nn.Embedding(vocab_size=6681, dim=1024)   # 6681 (code) vs 6693 (config); see Open Questions
ConformerEncoder(
    hidden_size = 1024,
    num_layers = 8,          # (likely; from related deepwiki + v1.5 num_lyric_encoder_hidden_layers=8)
    decoding_chunk_size = 1,
)
proj = Linear(1024, 2560)
```

Conformer = self-attn + depthwise-conv + macaron FFN — the standard ESPnet/wenet Conformer block. HartsyInference does not have one yet (Whisper uses a plain transformer); this is new code.

#### SSL alignment heads (training only — safe to omit at inference)

```python
projectors = nn.ModuleList([
    MLP(2560 → 2×2560 → 1024, SiLU),  # MERT alignment (24-layer wav2vec-style)
    MLP(2560 → 2×2560 → 768),         # m-HuBERT alignment
])
# Hooked at ssl_encoder_depths = [8, 8] → both projectors read from DiT layer 8 hidden state.
# Cosine-embedding loss against frozen MERT / m-HuBERT features at training time.
```

These exist in the safetensors but contribute no compute at inference; HartsyInference can skip projecting and drop the weights.

### 2.3 Music DCAE (`music_dcae_f8c8/config.json`, verbatim)

```json
{
  "_class_name": "AutoencoderDC",
  "_diffusers_version": "0.32.2",
  "in_channels": 2,
  "latent_channels": 8,
  "attention_head_dim": 32,
  "encoder_block_out_channels": [128, 256, 512, 1024],
  "encoder_block_types":       ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
  "encoder_layers_per_block":  [2, 2, 3, 3],
  "encoder_qkv_multiscales":   [[], [], [5], [5]],
  "decoder_block_out_channels": [128, 256, 512, 1024],
  "decoder_block_types":       ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
  "decoder_layers_per_block":  [3, 3, 3, 3],
  "decoder_qkv_multiscales":   [[], [], [5], [5]],
  "decoder_act_fns": "silu",
  "decoder_norm_types": "rms_norm",
  "downsample_block_type": "Conv",
  "upsample_block_type": "interpolate",
  "scaling_factor": 0.41407
}
```

This is the **Sana DCAE architecture** from NVIDIA's "Deep Compression Autoencoder" paper applied to **2-D mel spectrograms, not raw waveforms**. Key facts:

- **Input**: stereo mel spectrogram of shape `(B, 2, 128, T_mel)`. `in_channels=2` is the **stereo channel pair** (left + right packed as input channels of the 2-D conv).
- **Output (encode)**: `(B, 8, 16, T_mel/8)` — both H (mel bins 128→16) and W (time) downsampled by 8.
- **Channels schedule**: 128 → 256 → 512 → 1024 across 4 stages of `ResBlock × N` with the last stage adding `EfficientViTBlock` self-attention (head_dim 32, 5-scale multi-scale linear attention).
- **Down/up**: strided 2-D Conv down, nearest-or-bilinear interpolate up (no transposed conv).
- **Norm**: RMSNorm; **Act**: SiLU (no snake activation — that's Stable Audio's VAE, not this one).
- **Latent scaling for the DiT**: `latent_for_dit = (encoded - shift) * scale`. Two scale conventions appear in the code:
  - the DCAE intrinsic `scaling_factor = 0.41407` (diffusers convention)
  - the pipeline-level `scale_factor = 0.1786`, `shift_factor = -1.9091` applied around the DCAE call.
  Inspection of `music_dcae_pipeline.py` shows the pipeline-level `(0.1786, -1.9091)` numbers are what flow into the DiT; the 0.41407 is left as the AutoencoderDC's own internal scaling that the pipeline overrides.
- **Mel normalisation** before the encoder: clip log-mel to `[-11.0, +3.0]` then rescale to `mean=0.5, std=0.5` (i.e. `(x + 11)/14 * 2 - 1` then standardise).
- **Mel chunking**: encoder processes `mel_chunk_size = 1024` frames per chunk → `latent_chunk_size = 128` latent frames per chunk. Used for overlapping windowed decode of long songs.

### 2.4 Music Vocoder (`music_vocoder/config.json`, verbatim)

```json
{
  "_class_name": "ADaMoSHiFiGANV1",
  "sampling_rate": 44100,
  "n_mels": 128, "num_mels": 512,
  "n_fft": 2048, "win_length": 2048, "hop_length": 512,
  "f_min": 40, "f_max": 16000,
  "depths": [3, 3, 9, 3],
  "dims":   [128, 256, 384, 512],
  "drop_path_rate": 0.0,
  "input_channels": 128,
  "kernel_sizes": [7],
  "pre_conv_kernel_size": 13,
  "post_conv_kernel_size": 13,
  "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5],[1,3,5]],
  "resblock_kernel_sizes":   [3, 7, 11, 13],
  "upsample_initial_channel": 1024,
  "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4, 4],
  "upsample_rates":        [4, 4, 2, 2, 2, 2, 2],
  "use_template": false
}
```

ADaMoSHiFiGANV1 is a **HiFi-GAN-family vocoder** (multi-period + multi-scale discriminator at training time; only the generator at inference). At inference: mel `(B, 128, T_mel)` → 44.1 kHz mono waveform `(B, T_mel × 512)`. Stereo is handled by **running the vocoder twice**, once per channel:

```python
wav_ch1 = vocoder.decode(mels[:, 0])
wav_ch2 = vocoder.decode(mels[:, 1])
wav = torch.stack([wav_ch1, wav_ch2], dim=1)   # (B, 2, T)
```

Upsample factors `4·4·2·2·2·2·2 = 512` × `hop_length=512` is **not** how this works; the 512-factor is the upsample stack only — `hop_length=512` is just the mel STFT hop, and `upsample_rates` reverse it. ConvNeXt-style stages (`depths=[3,3,9,3]`) appear in the trunk before the upsampling, which is unusual vs. textbook HiFi-GAN. Full block-level layout is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) under the "ADaMoSHiFiGANV1 (ACE-Step)" section.

### 2.5 v1.5 Architecture (Qwen3-style FSQ-token decoder)

v1.5 is a fundamentally different model class — it is **not** a continuous-latent DiT. Instead it is a Qwen3-style causal-ish transformer that emits **discrete audio tokens** (Finite Scalar Quantization indices), trained with a flow-matching objective in token logit space. The exported configs (`acestep-v15-base/config.json`, `acestep-v15-xl-base/config.json`) confirm:

#### v1.5 2B base (verbatim selected keys)

```json
"model_type": "acestep",
"hidden_size": 2048,
"intermediate_size": 6144,
"num_hidden_layers": 24,
"num_attention_heads": 16,
"num_key_value_heads": 8,             # GQA, 2:1 ratio
"head_dim": 128,
"hidden_act": "silu",
"rms_norm_eps": 1e-06,
"rope_theta": 1000000,
"max_position_embeddings": 32768,
"use_sliding_window": true,
"sliding_window": 128,
"layer_types": [ "sliding_attention", "full_attention", … ],   # alternating per layer
"vocab_size": 64003,                  # token vocabulary (text + audio tokens)
"in_channels": 192,                   # input feature channels (pre-quant)
"fsq_dim": 2048,
"fsq_input_levels": [8, 8, 8, 5, 5, 5],          # 6-d FSQ, 8·8·8·5·5·5 = 64000 codes
"fsq_input_num_quantizers": 1,
"audio_acoustic_hidden_dim": 64,
"num_audio_decoder_hidden_layers": 24,
"num_lyric_encoder_hidden_layers": 8,
"num_timbre_encoder_hidden_layers": 4,
"text_hidden_dim": 1024,
"timbre_hidden_dim": 64,
"timbre_fix_frame": 750,
"patch_size": 2,
"pool_window_size": 5,
"timestep_mu": -0.4,
"timestep_sigma": 1.0,
"data_proportion": 0.5,
"dtype": "bfloat16"
```

#### v1.5 XL (5B)

Same fields, but: `hidden_size=2560`, `intermediate_size=9728`, `num_hidden_layers=32`, `num_attention_heads=32` (still `num_key_value_heads=8` → 4:1 GQA), plus separate `encoder_hidden_size=2048`, `encoder_intermediate_size=6144`, `encoder_num_attention_heads=16`, `encoder_num_key_value_heads=8`. So the **XL is asymmetric**: a 2B-class encoder feeding a 5B-class decoder. The `lyric_alignment_layers_config` maps lyric tokens to specific decoder layers (e.g. layer 3 alignment hits depths [18, 27], etc.).

#### Key v1.5 architectural elements

- **FSQ codec** — Finite Scalar Quantization with levels `[8, 8, 8, 5, 5, 5]` → 64 000 codes per token (matches `vocab_size=64003` minus 3 special tokens). Replaces continuous latents with a flat discrete vocabulary the decoder can model with cross-entropy. Audio tokens flow through the *same* embedding table as text tokens.
- **Sliding + full attention** alternating per layer (`sliding_window=128`) — classic Qwen2/3 long-context pattern. Lets the model attend across the whole song without quadratic memory.
- **GQA** (16:8 in 2B, 32:8 in XL).
- **Continuous-time flow matching in logit space** — `timestep_mu=-0.4`, `timestep_sigma=1.0` indicates the logit-normal timestep distribution from SD3, with shift baked into the sampling distribution rather than into the scheduler (see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)).
- **Separate timbre / lyric encoders**: `num_timbre_encoder_hidden_layers=4`, `num_lyric_encoder_hidden_layers=8`, distinct from the main decoder.
- **Sliding pool of 5 frames** (`pool_window_size=5`) — explicit temporal pooling to keep the audio sequence manageable; rate of decoder tokens ≈ 5 Hz (matches the "5Hz" branding of the planner LMs).
- **Turbo variants** flip `is_turbo: true`. CFG is disabled in turbo (matches v1 turbo behaviour); only 8 sampling steps.

For HartsyInference v1.5 work, treat the decoder as **a HartsyInference.LLM-class causal transformer with a custom audio-token output head** rather than as a diffusion model. The FSQ decode step (token → continuous → vocoder → waveform) is the only piece that needs HartsyInference.Audio code; the rest is HartsyInference.LLM.

### 2.6 UMT5-base text encoder (v1 and v1.5)

Both generations use **`google/umt5-base`** as the text encoder — the multilingual T5 variant with per-layer relative-attention bias (the same model AuraFlow uses; see SwarmUI memory note). 580M params, hidden 768, 12 layers, 12 heads, 16k SentencePiece vocab. ACE-Step uses a custom **language-ID-prefix-token** convention from `SUPPORT_LANGUAGES`:

```python
SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285,
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293,
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}
```

These IDs are passed to the **VoiceBpeTokenizer** (the *lyric* tokenizer — see §2.7), not to UMT5. UMT5 still gets the raw multilingual prompt as text. The language IDs are an XTTS-style language-prefix convention on top of VoiceBpe.

### 2.7 Lyric tokenization (VoiceBpeTokenizer)

ACE-Step v1 uses the **XTTS-v2 VoiceBpeTokenizer** for lyrics (`acestep.models.lyrics_utils.lyric_tokenizer.VoiceBpeTokenizer`). Vocabulary size **6 681** (code) or **6 693** (config) — discrepancy noted in Open Questions; the larger figure likely reserves extra special tokens. This is **not** UMT5's vocabulary.

#### Tokenization procedure (from `pipeline_ace_step.py`)

```python
def tokenize_lyrics(self, lyrics):
    lines = lyrics.split("\n")
    token_ids = [261]                       # opening token
    for line in lines:
        if not line.strip():
            token_ids.append(2)             # blank-line separator
            continue
        lang = self.get_lang(line)          # auto-detect language per line
        if structure_pattern.match(line):   # line is "[verse]", "[chorus]", etc.
            t = self.lyric_tokenizer.encode(line, "en")    # tags are always tokenized as English
        else:
            t = self.lyric_tokenizer.encode(line, lang)
        token_ids.extend(t)
        token_ids.append(2)                 # end-of-line separator
    return token_ids
```

#### Structure tags

The structure tags are **plain bracketed strings tokenized via BPE** — there are no dedicated special token IDs for `[verse]`/`[chorus]`. The model has learned to react to those byte patterns. The canonical tag set documented in the v1.5 musician's guide:

| Tag | Meaning |
|---|---|
| `[Intro]` | atmospheric setup, usually instrumental |
| `[Verse]` | main storytelling, moderate energy |
| `[Pre-Chorus]` | tension-build before chorus |
| `[Chorus]` | emotional peak, max energy |
| `[Bridge]` | shift with different melody/feel |
| `[Instrumental]` | vocals absent, instruments only |
| `[Outro]` | wind-down / fade |
| `[instrumental]` / `[inst]` | (alias) generates a vocal-free track |

Modifiers can be inline: `[Chorus - anthemic]`, `[Verse - whispered]`. Treated as part of the tag string.

For phonemic conversion of non-Latin scripts (Chinese, Japanese, Korean) the official pipeline runs **G2P preprocessing** before VoiceBpe — same convention as SongGen. HartsyInference can either bundle pyphonemes-equivalent G2P or expect callers to pre-phonemize.

### 2.8 Conditioning inputs summary

| Input | Encoding path | Final tensor going into DiT |
|---|---|---|
| **Style / genre / mood prompt** ("epic orchestral, 120 BPM, C major, …") | UMT5-base → `Linear(768 → 2560)` | `(B, T_text, 2560)` |
| **Lyrics** with `[section]` tags, multi-language per line | VoiceBpeTokenizer → Conformer (8 × 1024) → `Linear(1024 → 2560)` | `(B, T_lyric, 2560)` |
| **Speaker / timbre** (optional, for voice cloning) | Precomputed 512-d embedding (from an external speaker encoder; identity not specified in v1 code) → `Linear(512 → 2560)` | `(B, 1, 2560)` |
| **BPM, key, time signature** | Free-text in the style prompt — no dedicated encoder | (via UMT5) |
| **Instrumental flag** | `[instrumental]` / `[inst]` literal in lyrics input | (via VoiceBpe) |
| **Reference audio** (for cover / style transfer) | DCAE-encode the reference to a latent, splice into the noise schedule as initial state (see §2.10 edit/repaint) | `(B, 8, 16, F_lat)` |
| **Duration** | User-supplied seconds → `F_lat = int(duration × 44100 / 512 / 8)` controls the noise tensor's time dimension | sets `F_lat` |
| **Voice gender / language** | Tokenized as words in the style prompt or controlled implicitly via language-id of lyric lines | (via UMT5 + VoiceBpe lang) |

`(B, T_text, 2560)`, `(B, 1, 2560)`, `(B, T_lyric, 2560)` are **concatenated along the sequence axis** into one combined K/V tensor for cross-attention, with a key-padding mask that masks-out empty regions per condition.

### 2.9 Flow-matching sampler

See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) for the underlying math.

ACE-Step v1 ships three concrete schedulers (all in `acestep/schedulers/`):

- **`FlowMatchEulerDiscreteScheduler`** (default) — diffusers-compatible Euler with `num_train_timesteps=1000`, `shift=3.0`. Sigma schedule: `sigmas = t/1000` then `sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`. This is the SD3-style **resolution-dependent shift**; for audio the shift is constant 3.0.
- **`FlowMatchHeunDiscreteScheduler`** — 2nd-order Heun, same shift schedule, ~2× compute per step.
- **`FlowMatchPingPongScheduler`** — Euler with injected noise at every step: `prev = (1 - σ_{next}) * denoised + σ_{next} * gaussian_noise`. Adds stochasticity for variation; quality-vs-determinism tradeoff.

No "sway sampling" (Stable Audio Open's trick) is implemented in v1.

Default inference steps: **27** (fast preset) or **60** (quality). The RTF table on the model card uses these two columns. Some workflows go up to 100 for max quality.

#### Guidance

Three guidance modes (`acestep/apg_guidance.py`):

1. **Standard CFG** — `pred = pred_uncond + g * (pred_cond - pred_uncond)`. Default `g = 7.0`.
2. **APG (Adaptive Projected Guidance)** — decompose `diff = pred_cond - pred_uncond` into parallel + orthogonal components relative to `pred_cond`, apply momentum buffer (default momentum = -0.75), norm-threshold (default 2.5), then `pred = pred_cond + (g - 1) * (diff_orthog + eta * diff_parallel)`. Defaults: `eta = 0.0`, so only the orthogonal component is used.
3. **CFG-Zero★** — dynamically rescales `g` via `alpha = (cond · uncond) / |uncond|²`, optionally zeros out early-step predictions when `use_zero_init=True`.

The pipeline can also do **dual conditioning** (source + target predictions for editing): `apg_forward` is called twice, once each for src and tgt, and the velocity delta is what drives the latent. This is how the **flow-edit** lyric-modification mode works (see §2.10).

Turbo variants disable CFG and run 8 steps only.

### 2.10 Supported features per generation

| Feature | v1 (3.5B) | v1.5 (2B/XL) |
|---|---|---|
| Text-to-music (style prompt only) | yes | yes |
| Lyrics-to-song (with vocals) | yes | yes |
| Instrumental mode (`[inst]`) | yes | yes |
| Style transfer / reference audio | partial (edit mode) | yes (cover mode) |
| Continuation / extension | yes (via repaint loop) | yes (native + repaint) |
| Inpainting / repaint | yes (`flow-edit` masked ODE) | yes (built-in) |
| Lyric-localised edit | yes (`flow-edit` with token-aligned mask) | yes (improved alignment) |
| Variations / strength control | yes (trigFlow noise reformulation) | yes |
| Multi-language vocal synth | 17 languages (top-10 quality) | 50+ languages |
| Voice cloning (timbre transfer) | partial (512-d speaker vector exists but no reference encoder shipped) | yes (timbre encoder built-in, 4-layer) |
| Lego mode (multi-track composition) | no | yes |
| Vocal-to-BGM / singing-to-accompaniment | no | yes |
| LoRA training | yes | yes (one-click in v1.5 UI) |
| ControlNet-style conditioning | no | no (still no released controlnet for either) |

### 2.11 Memory footprint

#### v1 3.5B

- FP32: ~14 GB (DiT ~14 GB + UMT5 ~2.3 GB + DCAE ~700 MB + vocoder ~150 MB)
- BF16/FP16 (recommended): ~7 GB DiT + 1.2 GB UMT5 ≈ **~9 GB total weights**
- Activation memory at 4-minute song generation: ~3-5 GB extra → **~12-14 GB total VRAM for FP16 single-batch**
- With CPU offload of UMT5 + DCAE + vocoder: **8 GB VRAM** achievable on consumer GPUs (as advertised)
- GGUF Q4_K_M (community): DiT ~2.5 GB → fits in 6 GB cards

#### v1.5 2B (turbo)

- BF16: <4 GB VRAM total, on-card

#### v1.5 XL (5B)

- BF16: ~9 GB weights, **≥12 GB with offload + quant**, **≥20 GB recommended**
- 4-bit GGUF: ~3 GB

#### Inference latency

- A100: 4 min @ 27 steps in ~20 s (RTF 12×–27×)
- RTX 4090: same in ~15 s (RTF 34.48× at 27 steps)
- RTX 3090: ~32 s
- M2 Max (MLX backend): ~2 min generation
- v1.5 turbo: <2 s on A100, <10 s on RTX 3090

### 2.12 Comparison vs other open music models

| Model | Type | Vocals? | Length | Codec | Notes |
|---|---|---|---|---|---|
| **ACE-Step v1 3.5B** | DiT + flow-match on mel latent | yes (with lyrics) | up to 4 min coherent | Music DCAE (continuous) | Strongest open vocals + structured-lyric support; multi-language |
| **ACE-Step v1.5 XL 5B** | Qwen3-style decoder + FSQ tokens | yes | 10 s – 10 min | FSQ-quantized | Higher fidelity, 50+ languages, MIT |
| **MusicGen Large 3.3B** | Autoregressive decoder over EnCodec tokens | no (instrumental only) | up to ~30 s | EnCodec 32 kHz | Older (2023), instrumental, single-shot |
| **Stable Audio Open 1.0** (1.2B) | DiT + flow-match on 1-D Stable-Audio latent | no | 47 s | Oobleck VAE (44.1 kHz mono → 64-d @ 21.5 Hz) | Stereo SFX/loops; permissive license; no vocals |
| **YuE-7B / YuE-Anneal** | LLaMA-style decoder over hybrid audio tokens | yes (with lyrics) | up to 5 min | Custom semantic+acoustic codec | Comparable vocals quality to v1 but slower (LM autoregression) |
| **Mustango** (UNet) | Latent diffusion w/ music-theory control | no | ~10 s | – | Specialised in chord/beat conditioning |
| **Suno v3** / **Udio** | Closed | yes | up to 4 min | – | Quality benchmark for closed-source |

ACE-Step's distinguishing wins: (1) full vocals + structured lyrics with section markers, (2) flow-matching speed (much faster than autoregressive YuE/MusicGen), (3) 4-minute coherent generation, (4) open weights with practical 8 GB VRAM floor.

---

## 3. Key Numbers / Constants

### v1 DiT (`ACEStepTransformer2DModel`)

```
num_layers            24
num_attention_heads   20
attention_head_dim    128
inner_dim             2560                   # = 20 * 128
mlp_ratio             2.5                    # FFN hidden = 2560 * 2.5 = 6400
in_channels           8
out_channels          8
patch_size            [16, 1]                # height-collapse only
max_position          32768
rope_theta            1_000_000.0
text_embedding_dim    768                    # UMT5 hidden
speaker_embedding_dim 512
lyric_hidden_size     1024                   # Conformer hidden
lyric_encoder_vocab_size 6693 (config) / 6681 (code)
ssl_names             ["mert", "m-hubert"]
ssl_encoder_depths    [8, 8]                 # DiT layer to tap for SSL alignment
ssl_latent_dims       [1024, 768]            # MERT 1024-d, m-HuBERT 768-d
```

### v1 Music DCAE

```
in_channels                  2              # stereo (left, right)
latent_channels              8
encoder_block_out_channels   [128, 256, 512, 1024]
encoder_block_types          [ResBlock×3, EfficientViTBlock]
encoder_layers_per_block     [2, 2, 3, 3]
decoder_layers_per_block     [3, 3, 3, 3]
attention_head_dim           32             # in EfficientViT blocks
encoder_qkv_multiscales      [[],[],[5],[5]]
scaling_factor               0.41407        # diffusers-style
pipeline scale_factor        0.1786         # actually applied around DCAE in ACE-Step
pipeline shift_factor        -1.9091
mel_chunk_size               1024 frames
latent_chunk_size            128 frames     # = 1024 / 8
time_downsample_factor       8
mel range clip               [-11.0, +3.0]
mel post-clip standardise    mean=0.5, std=0.5
```

### v1 Vocoder (ADaMoSHiFiGANV1)

```
sampling_rate         44100
n_mels                128
num_mels              512                    # internal? see Open Questions
n_fft                 2048
win_length            2048
hop_length            512                    # → 44100/512 ≈ 86.13 mel Hz
f_min                 40 Hz
f_max                 16000 Hz
depths                [3, 3, 9, 3]           # ConvNeXt-style stages
dims                  [128, 256, 384, 512]
upsample_initial_channel 1024
upsample_rates        [4, 4, 2, 2, 2, 2, 2]  # product = 512 (matches hop_length)
upsample_kernel_sizes [8, 8, 4, 4, 4, 4, 4]  # 2× upsample_rates
resblock_kernel_sizes [3, 7, 11, 13]
resblock_dilations    [[1,3,5],[1,3,5],[1,3,5],[1,3,5]]
pre_conv_kernel_size  13
post_conv_kernel_size 13
input_channels        128
```

### Sampler (v1, default scheduler)

```
num_train_timesteps   1000
shift                 3.0
default infer_steps   27 (fast) / 60 (quality)
default guidance      7.0
APG momentum          -0.75
APG norm threshold    2.5
APG eta               0.0
```

### Latent / time math

```
F_lat = int(duration_seconds × 44100 / 512 / 8)
      = duration_seconds × ~10.77 tokens/s
      → 4 min (240 s) ≈ 2585 latent frames
mel_frames        = duration_seconds × 44100 / 512  ≈ 86.13 / s
samples           = duration_seconds × 44100
latent tensor     (B, 8, 16, F_lat)                    # H=16 = 128 mel / 8
mel tensor        (B, 2, 128, mel_frames)              # 2 = stereo
wav tensor        (B, 2, samples)                      # stereo 44.1 kHz
```

### v1.5 base (2B)

```
hidden_size               2048
intermediate_size         6144         # FFN
num_hidden_layers         24
num_attention_heads       16
num_key_value_heads       8            # GQA 2:1
head_dim                  128
sliding_window            128
layer_types               alternating sliding / full
vocab_size                64003
in_channels (features)    192
fsq_dim                   2048
fsq_input_levels          [8,8,8,5,5,5]   # 64000 codes
pool_window_size          5             # ≈ 5 Hz token rate
timestep_mu / sigma       (-0.4, 1.0)   # logit-normal timestep dist
text_hidden_dim           1024
timbre_hidden_dim         64
timbre_fix_frame          750
patch_size                2
rope_theta                1_000_000
max_position_embeddings   32768
dtype                     bfloat16
```

### v1.5 XL (5B)

Same as 2B except:

```
hidden_size               2560
intermediate_size         9728
num_hidden_layers         32
num_attention_heads       32
num_key_value_heads       8            # GQA 4:1
encoder_hidden_size       2048         # asymmetric: 2B-class encoder
encoder_intermediate_size 6144
encoder_num_attention_heads     16
encoder_num_key_value_heads     8
num_audio_decoder_hidden_layers 24
num_lyric_encoder_hidden_layers 8
num_timbre_encoder_hidden_layers 4
```

---

## 4. Data Layouts / Formats

### 4.1 Safetensors keys (v1 3.5B DiT)

Indicative prefix tree (extracted from class definitions, not from a key dump):

```
proj_in.weight, proj_in.bias                  # patch embed conv stack
caption_projection.linear.weight,bias         # 768 → 2560 (text)
lyric_embs.weight                             # (6693, 1024)
lyric_encoder.*                               # Conformer 8 layers
lyric_proj.linear.weight,bias                 # 1024 → 2560
speaker_embedder.linear.weight,bias           # 512 → 2560
time_embed.timestep_embedder.linear_1.weight,bias
time_embed.timestep_embedder.linear_2.weight,bias
transformer_blocks.{0..23}.scale_shift_table  # (6, 2560)
transformer_blocks.{0..23}.norm1, norm2       # RMSNorm (no params if elementwise_affine=False)
transformer_blocks.{0..23}.attn1.{q,k,v,out}_proj.weight,bias    # self-attn
transformer_blocks.{0..23}.attn2.{q,k,v,out}_proj.weight,bias    # cross-attn
transformer_blocks.{0..23}.ff.glumb.*         # GLUMBConv FFN
projectors.{0,1}.*                            # SSL alignment heads (trainable only)
proj_out.norm_final.scale_shift_table
proj_out.linear.weight,bias                   # 2560 → 16*1*8 = 128
```

### 4.2 Tensor shapes through the v1 forward pass

```
text_ids                 (B, T_text)
text_embeds = UMT5(text_ids)             (B, T_text, 768)
text_attn_mask                           (B, T_text)

lyric_ids                                (B, T_lyric)         # VoiceBpe
lyric_mask                               (B, T_lyric)
lyric_hidden = Conformer(lyric_ids)      (B, T_lyric, 1024)

speaker_vec                              (B, 512)             # optional, else None

t (timestep)                             (B,)                 # scalar per batch
temb = TimestepEmbed(t)                  (B, 256) → MLP → (B, 2560)

# noise / latent
z_t                                      (B, 8, 16, F_lat)

# patch embed
h = PatchEmbed(z_t)                      (B, F_lat, 2560)

# build cross-attn K/V
spk_tok = speaker_embedder(speaker_vec)  (B, 1, 2560)         # zero-padded if speaker_vec is None
txt_tok = caption_projection(text_embeds)(B, T_text, 2560)
lyr_tok = lyric_proj(lyric_hidden)       (B, T_lyric, 2560)
ctx = concat([spk_tok, txt_tok, lyr_tok], dim=1)              # (B, 1+T_text+T_lyric, 2560)
ctx_mask = concat([ones(1), text_mask, lyric_mask], dim=1)

# 24 blocks
for blk in transformer_blocks:
    h = blk(h, ctx, ctx_mask, temb)      # self-attn + cross-attn + GLUMB

# unpatch
v_pred = ProjOut(h, temb)                (B, 8, 16, F_lat)

# Euler step
sigma, sigma_next = scheduler.sigmas[i], scheduler.sigmas[i+1]
z_{t-1} = z_t + (sigma_next - sigma) * v_pred       # straight-line interpolation
```

### 4.3 Vocoder data flow

```
latent z_0                  (B, 8, 16, F_lat)
mel_pred = DCAE.decode(z_0) (B, 2, 128, mel_frames)
mel_pred = mel_pred * 14 - 11   # de-normalise back to log-mel range [-11, +3]
for ch in 0,1:
    wav_ch = vocoder(mel_pred[:, ch])    (B, mel_frames * 512)
wav = stack([wav_0, wav_1], dim=1)        (B, 2, T_samples)
```

### 4.4 Lyric token stream layout

```
[ 261 | line1_tok... | 2 | line2_tok... | 2 | ... | 2 ]
       ^ start tag              ^ EOL    ^ blank line → bare 2
```

With language-id-prefix from `SUPPORT_LANGUAGES` injected by the BPE tokenizer per line. Structure tags like `[verse]` are *byte-encoded* into the same stream — no dedicated special-token IDs.

---

## 5. Algorithm Steps

### 5.1 End-to-end v1 inference (pseudocode)

```python
def generate_ace_step_v1(
    style_prompt: str,           # "epic orchestral, 120 BPM, C major"
    lyrics: str,                 # multiline with [verse]/[chorus] markers; can be empty
    duration_seconds: float,     # 30..240 typical
    speaker_vec: Optional[np.ndarray] = None,   # (512,)
    infer_steps: int = 27,
    guidance: float = 7.0,
    guidance_mode: str = "apg",  # "cfg" | "apg" | "cfg_zero_star"
    scheduler: str = "euler",    # "euler" | "heun" | "pingpong"
    seed: int = 0,
    out_sample_rate: int = 44100,
) -> np.ndarray:                 # (2, T_samples_out)

    # ---- 1. Tokenise inputs ----
    text_ids, text_mask = umt5_tokenizer(style_prompt, max_len=256, return_mask=True)
    lyric_ids = tokenize_lyrics(lyrics)            # VoiceBpe procedure §2.7
    lyric_ids, lyric_mask = pad_to_max(lyric_ids, max_len=4096)

    # ---- 2. Run text encoders ONCE ----
    with no_grad():
        text_embeds = umt5_encoder(text_ids, text_mask)        # (1, T, 768)
        lyric_hidden = conformer_lyric_encoder(lyric_ids,
                                               lyric_mask)     # (1, T_l, 1024)

    if speaker_vec is None:
        speaker_vec = zeros(512)
    speaker_tok = speaker_embedder(speaker_vec)[None, None, :] # (1, 1, 2560)
    txt_tok   = caption_projection(text_embeds)                # (1, T, 2560)
    lyr_tok   = lyric_proj(lyric_hidden)                       # (1, T_l, 2560)
    ctx       = cat([speaker_tok, txt_tok, lyr_tok], dim=1)
    ctx_mask  = cat([ones(1,1), text_mask, lyric_mask], dim=1)

    # ---- 3. Initial noise ----
    F_lat = int(duration_seconds * 44100 / 512 / 8)
    z = randn((1, 8, 16, F_lat), seed=seed)

    # ---- 4. Scheduler init ----
    sch = make_scheduler(scheduler, num_train_timesteps=1000, shift=3.0)
    timesteps, sigmas = sch.set_timesteps(infer_steps)
    momentum_cond = MomentumBuffer(momentum=-0.75)
    momentum_uncond = MomentumBuffer(momentum=-0.75)

    # ---- 5. Denoising loop ----
    for i, t in enumerate(timesteps):
        # conditional pass
        v_cond = dit(z, t, ctx, ctx_mask)
        # unconditional pass (zero text, zero lyrics, zero speaker)
        v_uncond = dit(z, t, zeros_like(ctx), zeros_mask)
        # guidance
        if guidance_mode == "cfg":
            v = v_uncond + guidance * (v_cond - v_uncond)
        elif guidance_mode == "apg":
            v = apg_forward(v_cond, v_uncond, guidance, momentum_cond,
                            norm_threshold=2.5, eta=0.0)
        elif guidance_mode == "cfg_zero_star":
            v = cfg_zero_star(v_cond, v_uncond, guidance, use_zero_init=(i==0))
        # Euler step
        z = z + (sigmas[i+1] - sigmas[i]) * v

    # ---- 6. Decode ----
    z0 = z / scale_factor + shift_factor  # invert pipeline-level scaling
    mel = dcae.decode(z0)                  # (1, 2, 128, F_mel)
    mel = mel * 14 - 11                    # invert normalisation
    wav_l = vocoder(mel[:, 0])             # (1, F_mel * 512)
    wav_r = vocoder(mel[:, 1])
    wav = stack([wav_l, wav_r], dim=1)     # (1, 2, T)

    if out_sample_rate != 44100:
        wav = resample(wav, 44100, out_sample_rate)
    return wav[0].cpu().numpy()
```

### 5.2 Flow-edit / repaint algorithm (lyric-localised modification)

```python
def flow_edit(
    src_latent,           # (1, 8, 16, F_lat) — DCAE-encoded existing song
    src_lyrics, tgt_lyrics, src_text, tgt_text,
    edit_mask,            # (1, 1, 1, F_lat) — 1 where to edit, 0 to preserve
    n_avg=1,              # # source velocity averages
    n_min=0, n_max=infer_steps,  # which sampling steps to edit on
):
    z = src_latent
    for i, t in enumerate(timesteps):
        # always compute the SOURCE velocity (preserves what we keep)
        v_src_cond   = dit(z, t, ctx_src,   mask_src)
        v_src_uncond = dit(z, t, ctx_zero, mask_zero)
        v_src = apg_forward(v_src_cond, v_src_uncond, g, mom_src, …)
        # only compute target velocity inside the edit window
        if n_min <= i < n_max:
            v_tgt_cond   = dit(z, t, ctx_tgt,   mask_tgt)
            v_tgt_uncond = dit(z, t, ctx_zero, mask_zero)
            v_tgt = apg_forward(v_tgt_cond, v_tgt_uncond, g, mom_tgt, …)
            v = v_src + edit_mask * (v_tgt - v_src)   # masked velocity delta
        else:
            v = v_src
        z = z + (sigmas[i+1] - sigmas[i]) * v
    return decode(z)
```

The mask is built from the lyric-line → latent-frame alignment (which is implicit — there is no per-phoneme aligner in v1; the user supplies the time window). v1.5 has built-in lyric-to-frame alignment via `lyric_alignment_layers_config`.

### 5.3 APG step

```python
def apg_forward(pred_cond, pred_uncond, scale, momentum_buf,
                norm_threshold=2.5, eta=0.0):
    diff = pred_cond - pred_uncond
    if momentum_buf is not None:
        momentum_buf.update(diff)        # running_avg = m * prev + diff,  m=-0.75
        diff = momentum_buf.value
    # norm threshold
    n = diff.norm(dim=dims, keepdim=True)
    diff = diff * (norm_threshold / n.clamp_min(norm_threshold))
    # parallel + orthogonal decomposition along pred_cond
    p_hat = pred_cond / pred_cond.norm(dim=dims, keepdim=True).clamp_min(1e-8)
    diff_par  = (diff * p_hat).sum(dim=dims, keepdim=True) * p_hat
    diff_orth = diff - diff_par
    update    = diff_orth + eta * diff_par   # default eta=0 → orth only
    return pred_cond + (scale - 1.0) * update
```

---

## 6. Reference Implementations

### Primary

- **`ace-step/ACE-Step`** (v1) — https://github.com/ace-step/ACE-Step
  - `acestep/models/ace_step_transformer.py` — DiT class, patch embed, final layer
  - `acestep/models/attention.py` — `LinearTransformerBlock`, `Attention`, `GLUMBConv`
  - `acestep/models/customer_attention_processor.py` — attention processors (RoPE application)
  - `acestep/models/lyrics_utils/lyric_tokenizer.py` — `VoiceBpeTokenizer`
  - `acestep/models/lyrics_utils/conformer.py` — Conformer encoder
  - `acestep/music_dcae/music_dcae_pipeline.py` — DCAE encode/decode + mel transform
  - `acestep/music_dcae/music_vocoder.py` — `ADaMoSHiFiGANV1`
  - `acestep/pipeline_ace_step.py` — top-level orchestration, `tokenize_lyrics`, `calc_v`, schedulers
  - `acestep/apg_guidance.py` — APG + CFG-Zero★ + MomentumBuffer
  - `acestep/schedulers/scheduling_flow_match_*.py` — Euler / Heun / PingPong

- **`ace-step/ACE-Step-1.5`** — https://github.com/ace-step/ACE-Step-1.5 — v1.5 inference and training code; ships HF `auto_map` with `modeling_acestep_v15_*` files inside each checkpoint repo.

### HuggingFace

- `ACE-Step/ACE-Step-v1-3.5B` — primary v1 weights
- `ACE-Step/Ace-Step1.5` — turbo 2B
- `ACE-Step/acestep-v15-base`, `-sft`, `-turbo`, `-turbo-continuous`
- `ACE-Step/acestep-v15-xl-{base,sft,turbo,turbo-diffusers}` — 5B XL family
- `ACE-Step/acestep-5Hz-lm-{0.6B,1.7B,4B}` — Qwen3-based planner LMs
- `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format` — alternative Stable-Audio-format codec
- `Serveurperso/ACE-Step-1.5-GGUF` — community GGUF quantizations

### Paper

- **arXiv 2506.00045** — *"ACE-Step: A Step Towards Music Generation Foundation Model"* (June 2025). v1 architecture, training recipe, REPA loss formulation.
- **arXiv 2602.00744** — *"ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation"* (Feb 2026). v1.5 architecture, FSQ codec, planner LM, 50-language extension.

### Third-party / ecosystem

- **ComfyUI** native ACE-Step nodes (since May 2025) — https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1
- **acestep.cpp** + **acestep.vst3** — C++17 / GGML port + VST3 plugin for v1.5 (CPU/CUDA/Metal/Vulkan)
- **Replicate** `lucataco/ace-step` — hosted inference
- **WaveSpeedAI** Ace-Step 1.5 — hosted, ~70 % speedup over reference
- **DeepWiki** — code-explanation index at https://deepwiki.com/ace-step/ACE-Step (useful but not authoritative)

### Underlying components

- **DCAE** (Sana) — NVIDIA's Deep Compression Autoencoder; HF `mit-han-lab/dc-ae-f32c32-sana-1.0` etc.
- **UMT5** — `google/umt5-base` (already used by AuraFlow in HartsyInference)
- **MERT** — `m-a-p/MERT-v1-330M` (training-only target)
- **m-HuBERT** — `utter-project/mHuBERT-147` (training-only target)
- **VoiceBpeTokenizer** — vendored from XTTS-v2 (`coqui-ai/TTS`)
- **HiFi-GAN** family — base reference https://github.com/jik876/hifi-gan, see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)
- **APG** — Sadat et al. 2024 "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models" (arXiv 2410.02416)

---

## 7. Differences Between Implementations

### Inner-dim discrepancy (incorrect doc circulating)

Some third-party docs (including auto-generated wikis) describe the DiT as `num_heads=24, head_dim=64, inner_dim=1536`. The **actual published `config.json`** is `num_heads=20, head_dim=128, inner_dim=2560`. Implementers must use the config-file numbers.

### Lyric vocab size 6681 vs 6693

The Embedding in `ACEStepTransformer2DModel.__init__` uses `vocab_size=6681` (literal in code), but `config.json` declares `lyric_encoder_vocab_size=6693`. The most likely explanation is that the on-disk safetensors weight has shape `(6693, 1024)` (allowing for extra special tokens) and the constructor is overridden by the from-pretrained loader. **Use 6693 when allocating the embedding to match safetensors.**

### `scaling_factor` 0.41407 vs `scale_factor` 0.1786

`AutoencoderDC.config.scaling_factor = 0.41407` is the diffusers-style intrinsic scaling. The ACE-Step pipeline ignores it and applies `scale_factor=0.1786`, `shift_factor=-1.9091` itself in `music_dcae_pipeline.py`. **Implementers must apply the pipeline-level numbers, not the config-level one.**

### Vocoder `num_mels=512` mystery

The vocoder config has both `n_mels=128` (real number of mel bins, matches the DCAE input) and `num_mels=512` (unused at inference; possibly an artifact of the upstream `nemo`/`fish-speech` config schema). **Use 128 as the spectrogram bin count.**

### Three schedulers, no consensus

Different community workflows default to different schedulers. The official pipeline uses Euler by default; ComfyUI exposes all three; PingPong is preferred for "variations" mode because it injects fresh noise per step.

### v1 vs v1.5 — fundamentally different inference

v1 is a continuous-latent flow-matching DiT (diffusers-style). v1.5 is a Qwen3-style decoder over FSQ tokens. They share UMT5, structured lyrics, and the general "music generation" UX, but the inference code is not interchangeable. HartsyInference will need two distinct pipelines.

### Stereo handling

The DCAE config says `in_channels=2`, suggesting joint stereo at the latent level. But the vocoder is mono — at decode time the pipeline runs the vocoder **twice** (once per channel), then stacks. So stereo is preserved through the DCAE but not through the vocoder.

---

## 8. Open Questions

- **Lyric embedding vocab size** — 6681 (code) vs 6693 (config). Verify by inspecting `lyric_embs.weight` shape in `diffusion_pytorch_model.safetensors`. Implementation must match the actual tensor shape.
- **Speaker encoder identity** — v1 takes a 512-d pre-computed speaker vector but the repo does not ship the encoder that produces it. Is it RawNet3? ECAPA-TDNN? WavLM-based? For v1 HartsyInference can support either "no speaker" (zero vector) or "pre-supplied vector"; producing speaker vectors from a reference audio file is **out of scope** until the upstream encoder is identified.
- **Exact APG parameters** in production — defaults are `momentum=-0.75, threshold=2.5, eta=0.0`, but the README's recommended preset and ComfyUI's "quality" preset may differ. Replicate's hosted version uses `momentum=-0.5` per some forks.
- **DCAE pipeline-scale provenance** — `0.1786` and `-1.9091` look like empirical stats of a held-out audio distribution; confirm whether they were computed once over a calibration set or learned. Either way, treat them as constants.
- **Conformer layer count** — code is parametric but only `num_lyric_encoder_hidden_layers=8` from v1.5 confirms; the v1 default is unstated. Inspect the safetensors keys to count blocks directly.
- **REPA layer choice** at training time — `ssl_encoder_depths=[8, 8]` indicates DiT layer 8 hidden state is the alignment tap. Training-only; safe to drop at inference but verify when porting trainer code (out of scope for inference).
- **Vocoder `num_mels=512`** — confirm by reading the actual `ADaMoSHiFiGANV1` source to determine whether 512 is the input convolution width, an intermediate channel count, or truly unused.
- **v1.5 FSQ decoding path** — exact mapping from FSQ indices → continuous latent → waveform is implied by `audio_acoustic_hidden_dim=64`, `fsq_dim=2048`, `num_audio_decoder_hidden_layers=24`, but the chain (token → embed → 24-layer decoder → DCAE/1D-VAE → mel → vocoder?) is not fully specified in the public configs. Needs a deeper read of `modeling_acestep_v15_*.py` after first 2B implementation lands.
- **`timestep_mu=-0.4, timestep_sigma=1.0`** in v1.5 — these match SD3's logit-normal timestep distribution but flipped sign on μ. Confirm in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md).
- **`lyric_alignment_layers_config`** in v1.5 — `{3: [18, 27], 4: [22], 5: [5, 6, 7], 6: [2, 12, 13], 7: [20, 21]}` looks like a lyric-encoder-layer → decoder-layer cross-attention routing map, but the indexing convention (encoder-layer? decoder-layer? both?) needs source inspection.

---

## 9. Implementation Notes for HartsyInference

### What we already have

- **UMT5-base encoder** — implemented for AuraFlow. ACE-Step uses the same `google/umt5-base` weights, same tokenizer (SentencePiece, 256k vocab), same per-layer relative-attention-bias path. **Reuse directly.** The only ACE-Step-specific addition is the `SUPPORT_LANGUAGES` ID list, which only the lyric tokenizer cares about — UMT5 is fed plain text.
- **Flow-match Euler scheduler** — implemented for Flux/SD3. Add a `shift=3.0` config knob if not already there, plus PingPong (trivial extension: inject noise per step) and Heun (already present for some samplers).
- **AdaLN-6 modulation, RoPE, RMSNorm, GroupNorm** — all in the kernel library already from Flux/SD3/Lumina work. ACE-Step's `Qwen2RotaryEmbedding(dim=128, base=1e6, max_pos=32768)` is the same RoPE we use for Lumina-Next.
- **GGUF backend** — for loading the community Q4/Q5/Q8 quantizations of the DiT and UMT5.
- **HiFi-GAN vocoder skeleton** — partially built for Kokoro (iSTFTNet, which is a HiFi-GAN cousin). The trunk needs extension to the ConvNeXt-style `depths=[3,3,9,3]` stages used by ADaMoSHiFiGANV1, but ResBlocks + transposed-conv upsamplers + multi-receptive-field fusion are all already implemented.

### What we need to build

1. **GLUMBConv FFN** — gated MBConv-style FFN with depthwise-separable conv + SiLU gate. New op for HartsyInference. Sana / EfficientViT use the same block, so it's worth landing as a reusable `GLUMBConv` layer rather than ACE-Step-specific code.
2. **EfficientViTBlock** — multi-scale linear-attention block used in the deepest DCAE stages. Standard for Sana-family. Implementable on top of our existing scaled-dot-product attention with a lightweight multi-scale projection wrapper.
3. **Conformer encoder** — 8-layer Conformer block (macaron FFN + multi-head self-attn + depthwise conv + macaron FFN). Doesn't exist yet (Whisper uses a plain transformer). Reusable elsewhere (FunASR, SenseVoice, future ASR work).
4. **VoiceBpeTokenizer** — port the XTTS-v2 BPE merge table + multilingual support. Should live in a shared text-tokenization module; can be reused by SongGen, XTTS, or any other vocoder-conditioning model that adopts it.
5. **ADaMoSHiFiGANV1 vocoder** — new vocoder variant; biggest new piece in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Stereo-via-two-passes pattern is a 6-line wrapper around the mono vocoder.
6. **Music-DCAE** — Sana-style 2-D AutoencoderDC over mel spectrograms. New code but shares ResBlock + EfficientViTBlock with the DCAE in Sana-1.0/2.0 (future image work). Land as a generic `AutoencoderDC` module configurable via the diffusers-format `config.json`.
7. **STFT + mel filter bank** — for the vocoder's `mel_transform` step (and the inverse if we ever need to mel-encode reference audio). cuFFT-via-PTX path; matches Whisper / Kokoro mel preprocessing. We have STFT already for Whisper.
8. **APG + CFG-Zero★ guidance helpers** — small kernels: parallel/orthogonal decomposition, momentum buffer, norm thresholding. Useful beyond ACE-Step (Flux, SD3 community workflows are adopting APG).
9. **FlowMatchPingPongScheduler** — Euler with injected noise; trivial extension of the existing flow-match scheduler.
10. **Lyric / structure-tag preprocessor** — utility that runs G2P for CJK lines, language detection per line, line splitting, and the `[261]…[2]…[2]` token-list construction. Pure C# string work; no GPU.

### Suggested implementation order

1. **Scheduler + APG kernels** — small, hot-path-relevant, validates against Python reference quickly with no model weights needed.
2. **GLUMBConv + EfficientViTBlock** — kernel-level building blocks. Validate against AutoencoderDC reference outputs.
3. **Music-DCAE encode/decode** — wire up `AutoencoderDC` + mel transform + scaling. Validate against a known waveform → latent → waveform roundtrip (tolerance: PSNR > 30 dB).
4. **ADaMoSHiFiGANV1 vocoder** — slot into the existing HiFi-GAN code path. Validate per-channel decode against Python reference (numerical match to 1e-3 in waveform L1).
5. **Conformer + VoiceBpe + lyric preprocessor** — text-side pipeline. Validate token-ID stream byte-equal to Python output.
6. **ACEStepTransformer2DModel forward** — the DiT itself. Reuse RMSNorm/RoPE/AdaLN-6/cross-attention from existing DiT code. Validate one forward pass against Python reference at FP32 (tolerance: relative error < 1e-3 for hidden states, < 1e-2 for final velocity).
7. **End-to-end pipeline** — chain it all together. Validate against a published seed/prompt → known waveform (tolerance: PESQ > 3.5 vs reference).
8. **Quantization** — wire up GGUF Q4/Q8 loading paths through the existing GGUF backend. Verify perceptual quality.
9. **Edit / repaint / cover modes** — flow-edit mask-aware loop on top of the basic generation path.
10. **v1.5 2B turbo** — second pipeline. Largely orthogonal; touches HartsyInference.LLM for the decoder and HartsyInference.Audio for the FSQ → mel → wav path.

### Validation strategy

For v1 3.5B, the gold reference is the official PyTorch pipeline with `seed=0`, `prompt="lo-fi hip hop, mellow piano"`, `lyrics=""` (instrumental), `duration=30`, `infer_steps=27`, `guidance=7.0`, `scheduler="euler"`, `guidance_mode="cfg"` (avoid APG for first validation — APG has momentum state that drifts numerically). Verify:

- UMT5 last_hidden_state matches our existing AuraFlow UMT5 implementation byte-for-byte (already a regression test).
- DCAE latent after encoding the reference output matches Python reference (relative error < 1e-4).
- DiT velocity at step 0 matches (relative error < 5e-3 in BF16).
- Final waveform matches Python output in PESQ (>4.0) and short-term spectral L1 (<0.02).

### Memory layout note

The DiT's hidden activation at peak is roughly `(1, F_lat, 2560)` for the audio stream plus `(1, 1 + T_text + T_lyric, 2560)` for the cross-attention context. For a 4-minute song, `F_lat ≈ 2585`, so the audio token sequence dominates. Single-stream allocation in BF16 = `2585 × 2560 × 2 = 13 MB` per layer-residual. Activation checkpointing across 24 layers is unnecessary even on 12 GB cards. Standard `NativeMemory.AlignedAlloc` arenas (one per layer-pair) suffice.

### Package routing

- `HartsyInference.Audio.AceStep` — new sub-namespace for the v1 pipeline (DiT, DCAE, ADaMoS vocoder, lyric preprocessor, edit modes).
- `HartsyInference.Diffusion.Schedulers` — extend with `FlowMatchPingPongScheduler` (already has Euler / Heun for v1).
- `HartsyInference.TextEncoders.UMT5` — reuse (already present for AuraFlow).
- `HartsyInference.Audio.Vocoders` — extend with ADaMoSHiFiGANV1; shared with future Stable-Audio-Open / Suno-style vocoders.
- `HartsyInference.Audio.Codecs.DCAE` — new module; shared with future Sana image work.
- `HartsyInference.Text.VoiceBpe` — new tokenizer module; shared with XTTS / SongGen if/when added.

v1.5 should land in a separate `HartsyInference.Audio.AceStepV15` sub-namespace once HartsyInference.LLM Qwen3 decoder work catches up, because the inference loop is causal-LM-shaped rather than diffusion-shaped.
