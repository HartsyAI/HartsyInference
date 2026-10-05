# Breeze TTS 2 — architecture research

Source snapshot: 2026-10-05. Primary sources: BreezeBlue's inference repository and checkpoint configuration.
This note records the implementation boundary; it does not establish runtime verification.

## Confirmed contract

- Breeze TTS 2 is a bilingual English/Chinese TTS model with voice design, voice cloning, voice direction,
  inline vocal events, and streaming output.
- The released checkpoint uses a T5Gemma2 text encoder, a Qwen3-derived autoregressive backbone, a 12-layer
  depth decoder, and 16 audio codebooks.
- The bundled audio tokenizer is based on Qwen3-TTS 12 Hz and produces 24 kHz audio.
- The eager path is the first implementation target. CUDA graph and fast-path modes are performance work,
  not part of the initial correctness contract.

## Reuse boundary

The existing `QwenTts` package already contains the Qwen3 talker, MTP/depth code predictor, and 12 Hz codec
decoder. Breeze must reuse those components only where tensor keys, conditioning layout, and codec configuration
match the Breeze checkpoint. Its top-level model and text encoder remain separate from `Qwen3TtsModel`.

## Open questions before loader wiring

- Exact Breeze checkpoint shard names and tensor prefixes.
- T5Gemma2 configuration, tokenizer assets, and text-to-backbone projection dimensions.
- Backbone hidden size and attention layout for every published checkpoint variant.
- CFG batching and prompt splice rules for design, direction, and clone modes.
- Inline event tokenization for English parentheses and Chinese brackets.
- Exact streaming codec state and first-audio chunk contract.

## Validation plan

1. Inspect `config.json`, shard indexes, and tokenizer metadata without executing arbitrary checkpoint code.
2. Save text encoder states, backbone logits, depth logits, and codec tensors from the official reference.
3. Add synthetic shape/key-map tests before loading real weights.
4. Verify voice-design, voice-direction, and voice-clone modes independently.
5. Compare monolithic and streamed decode output and record transcript plus listening evidence.
