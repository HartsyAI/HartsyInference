# Sesame CSM — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Sesame CSM pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

**CSM (Conversational Speech Model)** is SesameAI's open-weight, real-time conversational TTS released March 13 2025 as `sesame/csm-1b` (Apache 2.0). It is the architecture behind the viral "Maya / Miles" demo at [sesame.com](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) and is purpose-built for low-latency, full-duplex spoken dialogue. The design is a **dual-transformer**: a Llama-3.2-style **backbone** (1B params, 16 layers, 2048 hidden, GQA 32 heads / 8 KV heads) consumes interleaved text + audio frames and predicts the **semantic codebook (codebook 0)** of the next 80 ms Mimi frame; a much smaller **audio decoder** (~100 M params, 4 layers, 1024 hidden, GQA 8 / 2) auto-regressively predicts the remaining 7 acoustic codebooks of that same frame conditioned on the backbone's hidden state and the semantic token. The 8 codebooks are then fed to the **Mimi codec** ([Kyutai, see AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section), which decodes them to 24 kHz PCM in a single causal pass — yielding one frame = 1920 samples = 80 ms of audio. End-to-end first-audio latency is ~150–250 ms on a consumer GPU. Speaker identity, style, and prosody come from a **conversation history prefix**: a list of `Segment(speaker_id, text, audio)` objects that are tokenized (text via the Llama-3.2 BPE, audio via Mimi.encode) and concatenated as context before the prompt — there are no learned speaker embeddings.

This file covers the dual-transformer LM, the conversation-format prompt assembly, and the streaming generation loop. The Mimi codec internals (causal SEANet + bottleneck transformer + split-RVQ decoder) are in [AUDIO_CODECS.md](AUDIO_CODECS.md) Mimi section. The Llama-3.2 backbone follows the native `HartsyInference.LLM` patterns (RoPE, GQA, RMSNorm, SwiGLU). General TTS context-prompt patterns appear in [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources: [SesameAILabs/csm](https://github.com/SesameAILabs/csm) (`models.py`, `generator.py`, `run_csm.py`), [HuggingFace sesame/csm-1b](https://huggingface.co/sesame/csm-1b), [Sesame research blog — Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice), [HF Transformers `CsmForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/csm) (added in v4.52.1, May 20 2025), [torchtune llama3_2 builder](https://github.com/pytorch/torchtune), [kyutai-labs/moshi `loaders.get_mimi`](https://github.com/kyutai-labs/moshi).

## Key Numbers / Constants

| Name                              | Value          | Source                                      |
|-----------------------------------|---------------:|---------------------------------------------|
| Sample rate                       | 24 000 Hz      | Mimi codec                                  |
| Frame rate                        | 12.5 Hz        | Mimi codec                                  |
| Samples per frame                 | 1920           | 24000 / 12.5                                |
| Frame duration                    | 80 ms          | 1 / 12.5                                    |
| Backbone hidden dim               | 2048           | `embed_dim` in `llama3_2_1B`                |
| Backbone layers                   | 16             | `num_layers` in `llama3_2_1B`               |
| Backbone heads / KV heads         | 32 / 8         | `num_heads`, `num_kv_heads`                 |
| Backbone head dim                 | 64             | 2048 / 32                                   |
| Backbone FFN dim                  | 8192           | `intermediate_dim`                          |
| Backbone RoPE base / scale_factor | 500 000 / 32   | Llama-3.2 long-context scaling              |
| Decoder hidden dim                | 1024           | `embed_dim` in `llama3_2_100M`              |
| Decoder layers                    | 4              | `num_layers`                                |
| Decoder heads / KV heads          | 8 / 2          |                                             |
| Decoder head dim                  | 128            | 1024 / 8                                    |
| Decoder FFN dim                   | 8192           | `intermediate_dim` (8× hidden — unusually wide for a 4-layer model) |
| Max sequence length (backbone)    | 2048           | `max_seq_len`; hard cap for context+gen     |
| Decoder max seq len               | 8              | `audio_num_codebooks`; reset every frame    |
| `audio_num_codebooks` (config)    | 32             | All 32 trained; only first 8 sampled        |
| `audio_num_codebooks` (inference) | 8              | 1 semantic + 7 acoustic                     |
| `audio_vocab_size`                | 2048           | Mimi codebook size                          |
| `text_vocab_size`                 | 128 256        | Llama-3.2 vocab                             |
| Default temperature               | 0.9            | `Generator.generate` default                |
| Default top-k                     | 50             | `Generator.generate` default                |
| Default max_audio_length_ms       | 90 000         | = 1125 frames                               |
| Total LM params (weights only)    | ~1.32 B        | embeddings (397 M) + backbone (~750 M) + decoder (~100 M) + heads (~70 M) |
| LM VRAM (bf16, weights only)      | ~2.6 GB        |                                             |
| Mimi VRAM                         | ~200 MB        |                                             |
| Backbone KV cache (bf16, 2048 seq)| 67 MB          |                                             |
| Per-frame latency (RTX 4090 bf16) | ~15–25 ms      | RTF ≈ 0.2–0.3                              |
| First-audio p50 (RTX 4090)        | ~150–250 ms    | Prefill + 1 frame                           |

## Data Layouts / Formats

### Wide frame tensor

`tokens` shape `(B, S, 33)`, dtype int64:

```
column index: 0   1   2   ...   31      32
content:      cb0 cb1 cb2 ...   cb31    text_token
```

`tokens_mask` shape `(B, S, 33)`, dtype bool. Exactly one of the two groups is set per row at construction:

- **Text row**: `mask[s, 32] = True`, `mask[s, 0:32] = False`.
- **Audio row**: `mask[s, 0:32] = True`, `mask[s, 32] = False`.

After embedding, all 33 vectors are masked then summed → `(B, S, D=2048)`.

### Backbone input position

`input_pos` shape `(B, S)`, dtype int64. Continuous 0..S-1 for prefill; advanced by 1 per generated frame in incremental mode (just `curr_pos[:, -1:] + 1`).

### Codebook sample tensor

Output of `generate_frame`: shape `(B, 8)` int64. Concatenated to form `(B, 8, T_gen)` after permute, then passed to `mimi.decode`.

### Mimi codes tensor (encoder output / decoder input)

Shape `(B, n_codebooks, T_frames)`:

- Encoder: `mimi.encode(wav).shape == (B, 32, T_frames)` (at `set_num_codebooks(32)`).
- Decoder: accepts `(B, n_q, T_frames)` for any `n_q ≤ 32`; CSM generation passes `(B, 8, T_gen_frames)`.

### Audio output tensor

`mimi.decode(codes)` returns `(B, 1, T_frames * 1920)` float32 in `[-1, 1]`. The reference `Generator.generate` then runs SilentCipher watermarking and a sample-rate-identity resample (the resample appears to be a no-op safety net for cases where the watermarker changes sample rate). For HartsyInference v1, skip both — return the raw PCM.

### Llama tokenizer

Standard Llama-3.2 BPE (`tokenizer.json` 17.2 MB) plus a post-processor that wraps each input with BOS/EOS. Reuse HartsyInference.LLM's Llama-3.2 tokenizer; just set the post-processor template.

## Reference Implementations

| Implementation                                    | Repo / path                                                                 | Notes                                                                              |
|---------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Original Sesame reference                         | [SesameAILabs/csm](https://github.com/SesameAILabs/csm)                     | `models.py`, `generator.py`, `run_csm.py`, `watermarking.py`. Uses torchtune Llama. |
| HuggingFace Transformers port                     | `transformers.models.csm.CsmForConditionalGeneration` (v4.52.1+, 2025-05-20) | Native HF API. Bundles Mimi. Supports static cache + CUDA graphs. Best reference for HF-format checkpoint shape. |
| Mimi codec (Kyutai)                               | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) `moshi/models/loaders.py`, `moshi/modules/seanet.py`, `moshi/quantization/{vq,core_vq}.py` | The codec implementation. Use as reference for the Mimi C# port. |
| Llama-3.2 builder used by Sesame                  | [pytorch/torchtune](https://github.com/pytorch/torchtune) `torchtune/models/llama3_2/` | The exact `llama3_2.llama3_2(...)` factory called by `models.py`. Verifies hyperparameters match HartsyInference.LLM. |

## Differences Between Implementations

- **torchtune Llama vs HF Llama vs HartsyInference.LLM Llama.** All three implement identical math (RoPE + GQA + RMSNorm + SwiGLU) but with different code paths and tensor naming. The CSM checkpoint is **torchtune-named** (`backbone.layers.0.attn.q_proj.weight`, etc.). The HF-Transformers `transformers-*.safetensors` rename to the HF convention (`model.layers.0.self_attn.q_proj.weight`). When loading from safetensors in C#, support both naming schemes or document which one we target.

- **Mimi `num_codebooks` at load.** The reference uses 32 (encodes context fully); HF Transformers CSM defaults to 8 (only what's needed for generation). The 8-codebook path is faster and saves memory, but **context audio loses ~75% of its acoustic information**, weakening voice cloning. We should default to 32 to match the reference behavior.

- **Watermarking.** Reference applies SilentCipher watermarking on every generation. HF version does not. We follow HF — no watermark in v1.

- **Resampling step at end of generate().** Reference does `torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)` after watermarking. Without the watermarker, `wm_sample_rate == self.sample_rate` so this is a no-op. Skip it.

## Implementation Notes for HartsyInference

### What HartsyInference already has (or will, from HartsyInference.LLM)

- Llama-3.2 transformer block (RoPE + GQA + RMSNorm + SwiGLU) — both backbone (16 layers) and decoder (4 layers) use this directly.
- Llama-3 BPE tokenizer.
- KV-cache infrastructure (incremental decode).
- Top-k sampling.
- Safetensors loader.
- bf16 PTX kernels for matmul, softmax, RMSNorm, RoPE.

### What is new and must be built

1. **Mimi codec** (`HartsyInference.Audio.Codecs.Mimi`):
   - Causal SEANet encoder + decoder with streaming ring buffers (see [AUDIO_CODECS.md](AUDIO_CODECS.md)).
   - Bottleneck Transformer (RoPE + GELU, 8 layers × 8 heads × 512 dim, causal with 250-frame finite context).
   - Split-RVQ: standalone semantic VQ + 7-step residual VQ on top.
   - Both `encode(wav) → codes[B, 32, T]` and `decode(codes) → wav[B, 1, T*1920]`.
   - **Streaming `decode_one_frame(codes[B, 8]) → pcm[B, 1920]`** — this is the critical path for low-latency.

2. **CSM `Model`** (`HartsyInference.Audio.Tts.Sesame.SesameCsmModel`):
   - Two `LlamaTransformer` instances (1B backbone, 100M decoder).
   - Three parameters / heads: `projection (2048→1024)`, `codebook0_head (2048→2048)`, `audio_head (31, 1024, 2048)`.
   - Two embedding tables: `text_embeddings (128256, 2048)`, `audio_embeddings (65536, 2048)`.
   - `_embed_tokens(wide_frame) → (B, S, 33, D)`: 33 lookups per row.
   - `generate_frame(...)` implementing the dual-loop above. Decoder KV cache reset per frame.

3. **CSM `Generator`** (`HartsyInference.Audio.Tts.Sesame.SesameCsmGenerator`):
   - Conversation-context tokenizer (text + Mimi encode).
   - Outer frame loop with EOS detection.
   - Both `GenerateAsync` (non-streaming) and `GenerateStreamingAsync` (showcase path).

### Suggested implementation order

1. **Mimi codec** (encoder + decoder + streaming decoder). Validate against `transformers.MimiModel` to ≤1e-3 RMS.
2. **CSM Model class** with greedy (T=0) sampling. Validate `generate_frame` on a fixed prompt + seed against the Python reference at the codebook level (8 ints per frame should match).
3. **Non-streaming Generator**. Validate end-to-end PCM is perceptually equivalent to the Python reference (no bit-match expected due to multinomial sampling).
4. **Streaming Generator** with `IAsyncEnumerable<AudioChunk>`. Validate per-frame decode produces same PCM as one-shot decode on the same codes.
5. **Performance pass**: fuse wide-frame embed-sum kernel; pre-allocate all per-frame buffers; aim for ≤25 ms per-frame on RTX 4090 bf16.
6. **(Optional) CUDA-graph capture** of the steady-state inner loop for sub-15 ms per-frame.

### Performance targets

| Metric                                            | Target           | Stretch          |
|---------------------------------------------------|------------------|------------------|
| First-audio latency (RTX 4090 bf16, ~30 s ctx)    | ≤ 250 ms         | ≤ 150 ms         |
| Steady-state per-frame                            | ≤ 25 ms          | ≤ 15 ms          |
| Steady-state RTF (lower is better)                | ≤ 0.3            | ≤ 0.2            |
| VRAM (model + KV cache, batch=1)                  | ≤ 3.5 GB         | ≤ 3.0 GB         |
| Numerical agreement with Python ref (codebook 0)  | ≥ 95% top-1 match at T=0 | 100% bit-match at T=0 with same RNG seed |

This is the showcase real-time TTS pipeline for HartsyInference — getting the streaming path right is more important than matching every last ms of single-frame throughput.
