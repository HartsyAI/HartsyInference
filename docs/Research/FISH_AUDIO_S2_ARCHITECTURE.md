# Fish Audio S2 — architecture research

Source snapshot: 2026-10-05. Primary sources: `fishaudio/s2-pro` model card and Fish Audio S2 technical
report. This note records the implementation contract; it does not claim engine verification.

## Confirmed contract

- The model is a decoder-only text/audio model with Dual-AR generation.
- The slow AR is a 36-layer Qwen3-derived transformer of approximately 4.13B parameters.
- The slow transformer uses 32 attention heads, 8 KV heads, QK normalization, and RoPE base 1,000,000.
- The fast AR is a four-layer transformer of approximately 0.42B parameters and emits the remaining
  acoustic codebooks for each semantic frame.
- The codec uses 10 RVQ codebooks with 4096 entries each and an approximately 21 Hz frame rate.
- S2 supports multilingual, multi-speaker, multi-turn generation and inline natural-language controls.
- The public tokenizer is Qwen3 BPE with ByteLevel pre-tokenization.
- The S2 model weights use the Fish Audio Research License; this is separate from the code license of
  community ports and must be surfaced in model metadata before catalog registration.

## Implementation boundary

The existing `FishSpeech` namespace implements Fish-Speech 1.5 and its firefly FSQ codec. S2 must use a
separate `FishAudio` namespace and configuration. Shared transformer, sampling, streaming, and codec
primitives may be reused only after checkpoint-key and intermediate-tensor parity confirms their contracts.

## Open questions before generation wiring

- Exact S2 safetensor key layout and fused QKV ordering.
- Exact slow-model vocabulary and tokenizer special-token ids. The current configuration's vocabulary size is
  provisional until `config.json` and tokenizer metadata are inspected together.
- Fast-model input projection and codebook embedding layout.
- Codec decoder architecture, tensor prefixes, causal padding, and output scaling.
- Prompt format for reference audio, multi-turn context, and inline controls.
- Differences between S2 Pro, S2.1 Pro, OpenAudio S1, and OpenAudio S1-mini checkpoints.
- Whether the released Hugging Face checkpoint is safetensors/PyTorch-native or requires a conversion step;
  community GGUF ports are useful behavioral references but are not the engine's weight source.

## Validation plan

1. Inspect the checkpoint index and tensor headers without executing pickle files.
2. Save tokenizer ids, slow hidden states, slow logits, fast logits, and codec latent tensors from the
   official reference implementation.
3. Match those tensors with deterministic CPU tests before adding backend-specific optimization.
4. Validate generated audio with waveform/codec comparisons and an independent transcription oracle.
5. Keep each checkpoint variant as a separate catalog entry until shared behavior is demonstrated.
