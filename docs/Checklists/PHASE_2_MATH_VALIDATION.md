# Phase 2 — Math Validation (Tokenizers + Schedulers + VAE)

> **Goal:** Prove the math is correct before tackling the full UNet. Tokenizer output matches Python references exactly. Scheduler step sequences match diffusers. VAE output matches diffusers within 1e-3.
> **Packages:** SharpInference.Tokenizers, SharpInference.Diffusion

---

## 1. Research

- [x] Complete [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) — Euler, DPM++2M, DDIM exact formulas — done, verified against diffusers source
- [x] Complete [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) — decoder layer structure, tiled decoding, constants — done, all model variants documented
- [x] Complete [CLIP_TOKENIZER.md](../Research/CLIP_TOKENIZER.md) — BPE algorithm, vocab, regex, bytes_to_unicode — done
- [x] Complete [T5_TOKENIZER.md](../Research/T5_TOKENIZER.md) — SentencePiece unigram, protobuf format, byte fallback

## 2. Implementation — SharpInference.Tokenizers

- [x] `ClipTokenizer.cs` — BPE tokenizer wrapping Microsoft.ML.Tokenizers BpeTokenizer (49408 vocab, 77-token limit) — done
- [x] `T5Tokenizer.cs` — SentencePiece tokenizer wrapping Microsoft.ML.Tokenizers SentencePieceTokenizer — done
- [x] `WhisperTokenizer.cs` — Whisper multilingual BPE + special tokens (stub for Phase 5) — done
- [x] `TokenizerCache.cs` — Reuse tokenizers across pipeline instances — done

## 3. Implementation — SharpInference.Diffusion (Schedulers)

- [x] `SchedulerConfig.cs` — Shared config (beta schedule, prediction type, timestep spacing) — done
- [x] `NoiseSchedule.cs` — Static utilities (betas, alphas_cumprod, sigmas, Karras sigmas, timestep selection) — done
- [x] `EulerDiscreteScheduler.cs` — Euler discrete scheduler with Karras sigmas support — done (18 tests passing)
- [x] `DpmPlusPlus2MScheduler.cs` — DPM++ 2M multistep scheduler — done
- [x] `DdimScheduler.cs` — DDIM scheduler with configurable eta — done

## 4. Implementation — SharpInference.Diffusion (VAE)

- [x] `VaeConfig.cs` — Configuration with presets for SD1.5, SDXL, SD3, Flux — done
- [x] `ResNetBlock2D.cs` — GroupNorm→SiLU→Conv3x3→GroupNorm→SiLU→Conv3x3 + skip connection — done
- [x] `VaeAttention.cs` — Mid-block single-head self-attention with GroupNorm and residual — done
- [x] `VaeDecoder.cs` — Full decoder (post_quant_conv → conv_in → mid_block → up_blocks → norm → conv_out) — done
- [x] `VaeTiledDecoder.cs` — Tiled decoding with overlap blending for large images — done

## 5. Testing

- [x] `SchedulerTests.cs` — 18 tests covering noise schedule, Euler/DDIM/DPM++ step, timesteps, add_noise — all passing
- [x] `VaeDecoderTests.cs` — 20 tests covering config presets, scaling math, channel progression, tiled params, blending — all passing
- [x] `ClipTokenizerTests.cs` — 16 tests covering encode/decode, SOT/EOT, padding, truncation, lowercasing, dispose — all passing with real CLIP vocab/merges
- [x] `T5TokenizerTests.cs` — 22 tests covering encode/decode, EOS, padding, attention mask, SD3/Flux max lengths, bos_id patch — all passing with real T5 model
- [x] All 146 tests pass locally (Core: 37, CPU: 19, ModelHandler: 14, Tokenizers: 38, Diffusion: 38)
- [ ] All tests pass on CI

## 6. Review & Merge

- [x] Code review — numerical correctness (scheduler math, tokenizer edge cases) — reviewed all schedulers and VAE components, fixed confusing variable names in EulerDiscreteScheduler.SigmaToTimestep, fixed TokenizerCache key ignoring maxLength
- [x] Code review — memory safety (proper disposal, no leaks) — all tensor disposal patterns verified correct, unsafe pointer code bounds-safe via clamping, thread-safe Interlocked patterns used throughout
- [x] Document any deviations from design plan — see [PHASE_2_DEVIATIONS.md](PHASE_2_DEVIATIONS.md) (T5 protobuf bos_id patch, CLIP token ID divergence, T5 default max length)
- [ ] Merge to main branch
