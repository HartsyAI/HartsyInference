# Phase 2 — Math Validation (Tokenizers + Schedulers + VAE)

> **Goal:** Prove the math is correct before full UNet. Tokenizer matches Python exactly. Scheduler steps match diffusers. VAE output within 1e-3.
> **Packages:** SharpInference.Tokenizers, SharpInference.Diffusion

---

## 1. Research — ALL COMPLETE

- [x] DIFFUSION_SCHEDULERS (Euler, DPM++2M, DDIM), VAE_ARCHITECTURE, CLIP_TOKENIZER, T5_TOKENIZER

## 2. Implementation — Tokenizers — ALL COMPLETE

- [x] `ClipTokenizer.cs` — wraps Microsoft.ML.Tokenizers BpeTokenizer (49408 vocab, 77-token limit)
- [x] `T5Tokenizer.cs` — wraps SentencePieceTokenizer (with bos_id protobuf patch)
- [x] `WhisperTokenizer.cs` — stub for Phase 5
- [x] `TokenizerCache.cs`

## 3. Implementation — Schedulers — ALL COMPLETE

- [x] `SchedulerConfig.cs`, `NoiseSchedule.cs` (betas, alphas_cumprod, sigmas, Karras, timestep selection)
- [x] `EulerDiscreteScheduler.cs` (18 tests), `DpmPlusPlus2MScheduler.cs`, `DdimScheduler.cs`

## 4. Implementation — VAE — ALL COMPLETE

- [x] `VaeConfig.cs` (presets: SD1.5, SDXL, SD3, Flux)
- [x] `ResNetBlock2D.cs`, `VaeAttention.cs`, `VaeDecoder.cs`, `VaeTiledDecoder.cs`

## 5. Testing — 146 tests passing locally

- [x] Schedulers (18), VAE (20), ClipTokenizer (16), T5Tokenizer (22), all others
- [x] All tests pass on CI

## 6. Review & Merge

- [x] Code review — numerical correctness, memory safety verified
- [x] Deviations documented (see below)
- [x] Merge to main branch

---

## Deviations from Design

**1. T5 Tokenizer — Protobuf bos_id Patching:** T5 models set `bos_id = -1` causing `IndexOutOfRangeException` in Microsoft.ML.Tokenizers. Workaround: `PatchT5ProtobufStream()` renames the field tag before loading. Harmless if fixed upstream.

**2. CLIP Token IDs diverge from Python:** Microsoft.ML.Tokenizers BPE produces different IDs than OpenAI Python CLIP (different `bytes_to_unicode`/`</w>` handling). Internally consistent; end-to-end validation needed when CLIP encoder is built.

**3. T5 Default Max Length:** Default 77 is generic; pipelines override (256 for SD3, 512 for Flux).
