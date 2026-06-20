# PocketTTS (Kyutai pocket-tts) — Architecture

> Spec for Kyutai's Pocket-TTS (~100M CPU TTS, CC-BY-4.0). Sources: PyPI `pocket-tts` v2.1.0,
> `github.com/kyutai-labs/pocket-tts` (`pocket_tts/models/tts_model.py` verified), HF `kyutai/pocket-tts`
> (gated), arXiv 2509.06926 (CALM — Continuous Audio Language Models). Fetched 2026-06-20. **Confirmed real.**

## What it is — IMPORTANT
**NOT a discrete SNAC/DAC/Mimi-RVQ codec-LM.** It is an autoregressive LM over **continuous** audio latents:
each 12.5 Hz frame is a continuous latent produced by a **flow-matching / Lagrangian-self-distillation (LSD)**
head with latent CFG. `FlowLMModel` (streaming transformer + KV cache) over time; `_sample_next_latent(...,
lsd_decode_steps)`; a `DummyQuantizer` (no codebooks). Codec = a **modified Mimi VAE** producing a continuous
Gaussian latent (WavLM-distilled), 24 kHz mono, 12.5 Hz, methods `encode_to_latent`/`decode_from_latent`.

## Verified API / behavior
- `TTSModel.load_model()`; `get_state_for_audio_prompt(name_or_path)` — built-in voice name OR a .wav
  path/URL (resample → Mimi-encode → prime AR state; `@lru_cache(2)`); `generate_audio(state, text)` → 1D PCM at
  `model.sample_rate` (= mimi 24 kHz). 26 built-in EN voices (alba, giovanni, …, vera) + per-language 24-layer
  variants (fr/de/pt/it/es). Distilled released model ≈ **6 transformer layers, ~100M params**.
- Weights `tts_b6369a24.safetensors` (~236 MB) + SentencePiece `tokenizer.model` + `embeddings*/`, `languages/`
  voice dirs. Config sub-trees `flow_lm.*` (`flow_lm.transformer.*`, `flow_lm.conditioner.tokenizer`) + `mimi.*`.

## NOT FOUND (config-gated)
The released variant's exact `d_model`, layer count, vocab size, and latent dim are in the `b6369a24` YAML /
safetensors header — **not in public docs**. Backbone family (Llama/Qwen/GPT) is described generically, not
branded. Verbatim safetensors tensor keys: NOT FOUND.

## C# build status (`Models/PocketTts/`)
- [x] [`PocketTtsConfig`](../../src/HartsyInference.Audio/Models/PocketTts/PocketTtsConfig.cs) — documented
  skeleton: 24 kHz / 12.5 Hz, the 26 voices, LSD steps, with `DModel`/`LatentDim` placeholders flagged for
  reconcile. **Config tested.**
- [ ] **Deferred (config-gated, intentionally not coded against guessed dims):** the continuous-latent AR loop
  (a Qwen/GPT streaming transformer with a **regression** head, not token logits), the per-frame **flow-matching
  / LSD** sampler (reuse `ConditionalCfm`), the **continuous-Mimi** path (reuse the built `Mimi`, bypass the RVQ
  quantizer → `encode_to_latent`/`decode_from_latent`), SentencePiece tokenizer, and voice-prompt priming. All
  reuse plans are recorded; implementation waits on reading the real config from the checkpoint.
