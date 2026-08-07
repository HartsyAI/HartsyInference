# F5-TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (F5-TTS pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

F5-TTS (Shanghai Jiao Tong University X-LANCE Lab / SWivid, 2024) is a **fully non-autoregressive, zero-shot voice-cloning TTS** built on **Conditional Flow Matching (CFM)** over mel spectrograms. It takes a short reference audio clip (3-15 s, 12 s hard cap) plus its transcript and a target text, and produces target speech in the reference speaker's voice. The whole utterance is generated **jointly** in a single forward integration pass — not autoregressively — which is what distinguishes it from XTTS/Bark/Sesame-style decoder TTS. At ~336 M params (DiT) + ~14 M (Vocos) it is currently the leading open-weight zero-shot voice cloning model for English and Mandarin Chinese; community fine-tunes cover ~10 additional languages.

The pipeline is `(ref_audio, ref_text, target_text) → mel-prep + char-tokenize → Flow-Matching DiT (32 NFE, CFG=2.0, Sway Sampling s=-1.0) → Vocos vocoder → 24 kHz waveform`. There is **no G2P, no phonemizer, no learned duration predictor** — characters go straight into a 256-token byte-level embedding and duration is a closed-form ratio of reference and target character counts. The DiT uses standard SD3/Flux-style AdaLN-Zero blocks (`dim=1024, depth=22, heads=16, head_dim=64, ff_mult=2`) preceded by a **ConvNeXt V2** text stem (4 blocks, depthwise-Conv1D kernel=7 + GRN). The vocoder is `charactr/vocos-mel-24khz`, **but fine-tuned by the F5-TTS team to 100 mel bins** (the public charactr checkpoint is 100-bin in F5's mel parameterization — `n_fft=1024, hop=256, win=1024`).

This file covers the model architecture and inference pipeline. The Sway-Sampling scheduler math is in [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section. Mel preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Vocos vocoder implementation details are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) Vocos section.

**Sources**:
- Paper: ["F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"](https://arxiv.org/abs/2410.06885) (Chen et al., ACL 2025)
- Repo: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) (`src/f5_tts/model/cfm.py`, `dit.py`, `modules.py`, `infer/utils_infer.py`)
- Weights: [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS), [SWivid/E2-TTS](https://huggingface.co/SWivid/E2-TTS)
- Vocoder: [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)
- Community variants index: [`src/f5_tts/infer/SHARED.md`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md)

## Model Variants

All official models share the same DiT shape (`dim=1024, depth=22, heads=16, ff_mult=2, conv_layers=4`) — they differ only in training data, training steps, vocab, and zero-init policy.

| Variant | Params | Languages | HF path | Checkpoint | File size | License |
|---|---|---|---|---|---|---|
| **F5TTS_v1_Base** *(current default, 2024-12)* | ~335.8 M | EN + ZH (code-switch) | `SWivid/F5-TTS` | `F5TTS_v1_Base/model_1250000.safetensors` | ~1.34 GB FP32 / ~672 MB FP16 | CC-BY-NC-4.0 |
| **F5TTS_v1_Base_no_zero_init** | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_v1_Base_no_zero_init/model_1250000.safetensors` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **F5TTS_Base** *(legacy v0)* | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_Base/model_1200000.safetensors` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **F5TTS_Base_bigvgan** | ~335.8 M | EN + ZH | `SWivid/F5-TTS` | `F5TTS_Base_bigvgan/...` | ~1.34 GB FP32 | CC-BY-NC-4.0 |
| **E2TTS_Base** *(paper predecessor)* | ~333 M | EN + ZH | `SWivid/E2-TTS` | `E2TTS_Base/model_1200000.safetensors` | ~1.33 GB FP32 | CC-BY-NC-4.0 |

Training data for the base models: **Emilia 95k h ZH + EN** ([amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)). Repo total (all variants) is **6.74 GB**.

**v1 vs v0 differences**: v1 fixes the AdaLN-Zero init (the `_no_zero_init` debug variant was published to demonstrate the difference), uses a slightly cleaner Emilia split, and trained 50 k more steps. Both have identical architecture — checkpoint swap only.

**Community language variants** (from [`SHARED.md`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/SHARED.md), 2025):

| Language | Base | Training data | Notable changes |
|---|---|---|---|
| Arabic | F5-TTS-**Small** | EN + AR mixed (tens of thousands h) | smaller DiT |
| Finnish | F5-TTS-Base | Common Voice + VoxPopuli | vocab.txt extended |
| French | F5-TTS-Base | LibriVox FR | |
| German | F5-TTS-Base | Mozilla CV 19.0 + 800 h crowdsourced | |
| Hindi | F5-TTS-**Small** | IndicTTS + IndicVoices-R | smaller DiT, Devanagari chars |
| Italian | F5-TTS-Base | `ylacombe/cml-tts` | |
| Japanese | F5-TTS-Base | Emilia JA 1.7 k + Galgame 5.4 k | |
| Latvian | F5-TTS-Base | Common Voice LV | |
| Russian | F5-TTS-Base | Common Voice RU | Cyrillic chars |
| Spanish | F5-TTS-Base | VoxPopuli ES + TEDx (218 h) | |

There is also **Cross-Lingual F5-TTS** ([arXiv:2509.14579](https://arxiv.org/abs/2509.14579)) — a 2025 follow-up adding language-agnostic voice cloning, and **Fast F5-TTS** ([fast-f5-tts.github.io](https://fast-f5-tts.github.io/)) — a 7-NFE distilled variant.

The **"F5-TTS-Small"** topology (used by Arabic, Hindi) is approximately `dim=768, depth=18, heads=12, ff_mult=2, conv_layers=4` (~155 M params). The official repo carries the YAML; planned variants in the HartsyInference loader must read `dim/depth/heads/ff_mult/conv_layers/text_dim/text_num_embeds` from the YAML next to the safetensors and dispatch accordingly.

## Flow matching — Sway Sampling

F5-TTS uses **rectified flow / Conditional Flow Matching** with **Euler integration** in time `t: 0 → 1` (data at `t=0`, noise at `t=1` — opposite sign convention from SD3 sigmas, but the per-step update is identical).

**Defaults** (from `src/f5_tts/infer/utils_infer.py`):

| Knob | Value | Notes |
|---|---|---|
| `nfe_step` | **32** | ablation flat above ~16 NFE |
| `cfg_strength` | **2.0** | both `cond_mel` and `text_emb` zeroed on uncond branch |
| `sway_sampling_coef` | **−1.0** | enables Sway; `None`/`0` = uniform |
| ODE solver | **Euler** | `midpoint` available as fallback |
| `target_rms` | **0.1** | RMS of ref audio normalized to this before mel |
| `cross_fade_duration` | **0.15 s** | between text chunks |
| `speed` | **1.0** | multiplies target duration (smaller = faster speech) |

**Sway Sampling** is a one-shot remap of the uniform timestep grid applied **before** the ODE loop:

```python
# from cfm.py — exact code
t = torch.linspace(0, 1, steps)            # uniform NFE grid
if sway_sampling_coef is not None:
    t = t + sway_sampling_coef * (torch.cos(torch.pi/2 * t) - 1 + t)
# t still starts at 0 and ends at 1; only interior density shifts
```

With `s = -1.0` more samples cluster near `t = 0` (data end), so the solver spends more NFE polishing fine detail and fewer NFE on coarse structure near noise. Paper ablation: `s = -1.0` reduces WER ~10 % and raises SIM-O ~0.02 vs. uniform at NFE=32. **See [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section** for the full derivation, the sigma-space relationship, and a per-step value table — that doc is canonical for the scheduler implementation.

**CFG combination** (per step):
```
v_cond  = DiT(x_t, t, cond_mel, text_emb)
v_uncond = DiT(x_t, t, zeros_like(cond_mel), zeros_like(text_emb))
v = v_uncond + cfg_strength * (v_cond - v_uncond)
x_{t-dt} = x_t - v * dt          # Euler step (negative because we integrate 1→0)
# at every step, overwrite the ref region of x with ref_mel
```

E2-TTS (predecessor) is mathematically the same model with no ConvNeXt text stem and `sway_sampling_coef = 0`. For HartsyInference: **E2-TTS = F5-TTS scheduler with sway off, plus drop the ConvNeXt blocks from the text path.**

## Memory and performance

| Resource | F5TTS_v1_Base |
|---|---|
| Disk (FP32 safetensors) | ~1.34 GB |
| Disk (FP16) | ~672 MB |
| VRAM @ FP16 inference | ~2.5 GB peak (model 0.7 GB + activations + Vocos 0.05 GB) |
| VRAM @ FP32 inference | ~5 GB peak |
| Min recommended GPU | 8 GB VRAM (FP16); 16 GB comfortable; 24 GB for training |
| **RTF @ 32 NFE, FP16, RTX 4090** | **~0.15** (≈ 6.5× realtime) |
| RTF @ 16 NFE, FP16, RTX 4090 | ~0.075 (≈ 13× realtime) |
| RTF @ 7 NFE distilled (Fast F5-TTS, 3090) | ~0.030 |
| Quantized (community Q8) | ~400 MB VRAM |
| Latency per 1 s of audio @ 32 NFE | ~150 ms on 4090 |

The bottleneck is the **DiT** (22 layers × 32 NFE × 2 CFG branches × ~T forward passes), specifically attention over `T` tokens where `T ≈ 94 * audio_seconds`. For a 10 s output `T ≈ 940`, so the attention is well-suited to flash-attention / our own fused-attention kernel; this is where the bulk of optimization payoff lives. Vocos is negligible cost (one feed-forward conv stack + one iSTFT per call).

## C# implementation notes (HartsyInference)

| Component | Reuse / new | Source of truth |
|---|---|---|
| **DiT block (AdaLN-Zero, RoPE, GELU FFN)** | **Reuse** Flux/SD3 block; F5 differs only in `dim_head=64`, `ff_mult=2`, no GeGLU | [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md) |
| **Sinusoidal + MLP time embedding** | **Reuse** from image diffusion stack | [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md) |
| **Mel preprocessing (24 kHz / 100 bin / 1024 FFT / 256 hop / Hann / log-clamp)** | **Reuse** existing mel module; add the 100-bin variant | [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) |
| **ConvNeXt V2 text stem (4 blocks, dwconv k=7, GRN, intermediate=1024)** | **New** — implement `ConvNeXtV2Block` + `GRN` for 1D temporal data. Depthwise Conv1D groups=channels. | this doc § 2.4 |
| **Char/byte/pinyin tokenizer** | **New trivial component**. Start with byte mode (a `byte[] → int[]` map plus a filler token). Defer pinyin (it would pull in a Jieba-style segmenter + pinyin tables). | this doc § 2.1 |
| **Sway-sampling scheduler** | **New** — small scheduler class extending the existing `FlowMatchEulerDiscreteScheduler`: just apply the cosine remap to the timestep grid before the loop. | [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) F5-TTS section |
| **CFG combiner for velocity** | **Reuse** existing CFG helper from image flow matching (identical formula) | [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md) |
| **In-context infilling loop (concat ref+target, overwrite ref region each step)** | **New** wrapper around the Euler loop | this doc § 8 |
| **Vocos vocoder (8 ConvNeXt blocks → magnitude+phase → iSTFT)** | **New** — implement once, shared with Kokoro-vocos variant + EnCodec decode | [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) "Vocos Architecture (Alternative)" |
| **Safetensors loader** | **Reuse** | [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md) |
| **GGUF quantized variants (community Q8)** | **Reuse** GGUF backend; verify tensor naming matches `f5_tts.*` convention | [GGUF_BACKEND.md](GGUF_BACKEND.md) |

**Package placement**: `HartsyInference.Audio` owns the F5-TTS pipeline, Vocos vocoder, mel preprocessor, and Sway scheduler. The DiT block kernels live in `HartsyInference.Core` (already shared with image diffusion). No new native code required — pure C# + PTX.

**Recommended first-milestone scope**:
1. Load `F5TTS_v1_Base/model_1250000.safetensors` (FP32 → FP16) into a typed weight dictionary.
2. Implement `ByteTokenizer` (vocab size 257, filler at 256).
3. Implement `ConvNeXtV2Block1D` + `GRN1D`.
4. Wire 22-layer DiT using existing AdaLN-Zero blocks, with the 200-channel input projection (concat noisy + cond).
5. Implement Sway-sampled Euler loop with the ref-region overwrite anchor.
6. Implement Vocos (mel-24kHz config) and validate against `vocos.decode(mel)` from Python reference within 1e-3 L2 on a held-out mel.
7. End-to-end test: `(reference 5 s wav, ref text, target text) → 24 kHz wav`, compare against Python F5-TTS output on the same inputs at same seed/NFE/CFG/Sway; SIM-O should match within 0.005, WER within 1 %.

**Reference fidelity targets**: validate the DiT block-by-block (mel → first block output, then mid-network, then final velocity) against the Python reference with all stochasticity removed (fixed noise tensor, NFE=32, deterministic). Tolerances: 1e-3 atol FP16, 1e-5 atol FP32, both relative to torch reference.

## Cross-References

- [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) — F5-TTS scheduler, Sway Sampling derivation, exact CFM math.
- [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — 24 kHz / 100-bin mel parameters.
- [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) — Vocos architecture and implementation.
- [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) — alternative TTS pipeline (autoregressive-style with G2P) for comparison.
- [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md) — reusable AdaLN-Zero DiT blocks.
- [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md) — CFG combination for velocity fields.
- [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md), [GGUF_BACKEND.md](GGUF_BACKEND.md) — weight loading.
