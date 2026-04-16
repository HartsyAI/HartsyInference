# Phase 2 — Math Validation (Tokenizers + Schedulers + VAE)

> **Goal:** Prove the math is correct before tackling the full UNet. Tokenizer output matches Python references exactly. Scheduler step sequences match diffusers. VAE output matches diffusers within 1e-3.
> **Packages:** SharpInference.Tokenizers, SharpInference.Diffusion

---

## 1. Research

- [ ] Complete [CLIP_TOKENIZER.md](../Research/CLIP_TOKENIZER.md) — BPE algorithm, vocab, regex, bytes_to_unicode
- [ ] Complete [T5_TOKENIZER.md](../Research/T5_TOKENIZER.md) — SentencePiece unigram, protobuf format, byte fallback
- [ ] Complete [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) — Euler, DPM++2M, DDIM exact formulas
- [ ] Complete [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) — decoder layer structure, tiled decoding, constants

## 2. Implementation — SharpInference.Tokenizers

- [ ] `ClipTokenizer.cs` — BPE tokenizer matching OpenAI CLIP exactly (49408 vocab, 77-token limit)
- [ ] `T5Tokenizer.cs` — SentencePiece unigram tokenizer for SD3/Flux (32128 vocab)
- [ ] `WhisperTokenizer.cs` — Whisper multilingual BPE + special tokens (stub for Phase 5)
- [ ] `TokenizerCache.cs` — Reuse tokenizers across pipeline instances

## 3. Implementation — SharpInference.Diffusion (Schedulers)

- [ ] `EulerDiscreteScheduler.cs` — Euler discrete scheduler with Karras sigmas support
- [ ] `DpmPlusPlus2MScheduler.cs` — DPM++ 2M multistep scheduler
- [ ] `DdimScheduler.cs` — DDIM scheduler with configurable eta
- [ ] Shared scheduler utilities (beta schedule, alphas_cumprod, sigma computation)

## 4. Implementation — SharpInference.Diffusion (VAE)

- [ ] `VaeDecoder.cs` — Full VAE decoder (latents → pixels) matching AutoencoderKL
- [ ] `VaeTiledDecoder.cs` — Tiled decoding with overlap blending for large images
- [ ] VAE helper types (ResNet blocks, mid block, upsample layers)

## 5. Testing

- [ ] `ClipTokenizerTests.cs` — tokenize test strings, compare against Python CLIP output (exact match)
- [ ] `T5TokenizerTests.cs` — tokenize test strings, compare against HuggingFace T5Tokenizer (exact match)
- [ ] `EulerSchedulerTests.cs` — compare 20-step sequence against diffusers (within 1e-4)
- [ ] `DpmPlusPlus2MSchedulerTests.cs` — compare step sequence against diffusers (within 1e-4)
- [ ] `DdimSchedulerTests.cs` — compare step sequence against diffusers (within 1e-4)
- [ ] `VaeDecoderTests.cs` — decode synthetic latents, verify output shape and value ranges
- [ ] All tests pass on CI

## 6. Review & Merge

- [ ] Code review — numerical correctness (scheduler math, tokenizer edge cases)
- [ ] Code review — memory safety (proper disposal, no leaks)
- [ ] Document any deviations from design plan
- [ ] Merge to main branch
