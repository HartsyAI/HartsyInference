# Bark (Suno) — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Bark pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Bark is a text-prompted generative audio model released by Suno in April 2023. Unlike conventional TTS systems, Bark produces not just speech but also music, ambient noise, sound effects, laughter, sighs, gasps, and other non-verbal vocalizations from a single text prompt. The pipeline is a three-stage cascade of GPT-2-style Transformers followed by an EnCodec 24 kHz decoder:

1. **Semantic Transformer** — causal LM. Takes BERT-tokenized text (offset into a shared 129,600-entry vocab) plus an optional speaker-prompt prefix and predicts a stream of semantic tokens at 49.9 Hz from a custom 10,000-entry HuBERT-derived semantic codebook.
2. **Coarse Transformer** — causal LM. Takes semantic tokens (offset) plus optional history and predicts the first two EnCodec acoustic codebooks at 75 Hz, interleaved as a single flat sequence (book 0, book 1, book 0, book 1, ...).
3. **Fine Transformer** — non-causal masked LM. Takes the 2 coarse codebooks and iteratively fills in EnCodec codebooks 2..7 in six refinement passes (codebook 0 is given by the coarse output and acts as input embedding base; the architecture supports `n_codes_total=8` with `n_codes_given=1`, leaving 7 trainable output heads).
4. **EnCodec 24 kHz, 8-codebook decoder** — converts the 8 acoustic codebook streams back to 24 kHz mono PCM.

Two variants ship: **Bark** (~1.0B params per stage HF-counted, ~900M trainable per stage) using hidden=1024 / 24 layers / 16 heads, and **Bark-Small** (~80M per stage trainable, ~300M per HF param count) using hidden=768 / 12 layers / 12 heads (a vanilla GPT-2-small footprint). Text is tokenized with `bert-base-multilingual-cased` (~120k WordPiece vocab) and shifted by `TEXT_ENCODING_OFFSET=10048` to live in the upper region of the 129,600-entry semantic input vocab. Voice control is via pre-extracted "speaker prompts" — triples of `(semantic, coarse, fine)` token streams from ~10s reference clips, prepended as in-context history at each of the three stages. ~100+ official voice prompts ship for 13 languages (English, German, Spanish, French, Hindi, Italian, Japanese, Korean, Polish, Portuguese, Russian, Turkish, Chinese).

The EnCodec component is covered in detail in [AUDIO_CODECS.md](AUDIO_CODECS.md) under "EnCodec (Meta / Défossez 2022)". The streaming considerations are in [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) under TTS section 3.1. The Kokoro-style TTS pipeline is in [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md). For phonemizer-free TTS like Bark there is no G2P step.

Sources: [suno-ai/bark](https://github.com/suno-ai/bark), [suno/bark](https://huggingface.co/suno/bark), [suno/bark-small](https://huggingface.co/suno/bark-small), [HF transformers Bark](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bark).

## Key Numbers / Constants

| Constant | Value | Notes |
|---|---|---|
| `SAMPLE_RATE` | 24,000 Hz | EnCodec / output audio |
| `SEMANTIC_RATE_HZ` | 49.9 Hz | Custom HuBERT-derived semantic codebook |
| `COARSE_RATE_HZ` | 75 Hz | EnCodec frame rate (24000/320) |
| `SEMANTIC_VOCAB_SIZE` | 10,000 | Semantic codebook size |
| `SEMANTIC_PAD_TOKEN` | 10,000 | Doubles as EOS |
| `CODEBOOK_SIZE` | 1,024 | EnCodec per-codebook size |
| `N_COARSE_CODEBOOKS` | 2 | Coarse stage outputs |
| `N_FINE_CODEBOOKS` | 8 | Total EnCodec codebooks |
| `TEXT_ENCODING_OFFSET` | 10,048 | BERT token id shift |
| `TEXT_PAD_TOKEN` | 129,595 | Text-side pad |
| `SEMANTIC_INFER_TOKEN` | 129,599 | "Begin semantic generation" marker |
| `COARSE_SEMANTIC_PAD_TOKEN` | 12,048 | Pad in coarse input vocab |
| `COARSE_INFER_TOKEN` | 12,050 | "Begin coarse generation" marker |
| Semantic input length (text+history+infer) | 513 | 256 + 256 + 1 |
| `max_coarse_history` | 630 | Coarse history truncation |
| `sliding_window_len` (coarse) | 60 | Semantic tokens consumed per window |
| `max_fine_history_length` | 512 | Fine model context overlap |
| `block_size` (all three GPTs) | 1024 | Max sequence length |
| `n_codes_total` (fine) | 8 | EnCodec codebooks |
| `n_codes_given` (fine) | 1 | Codebook 0 is provided input |
| `min_eos_p` (semantic) | 0.2 | Early-stop threshold on EOS probability |
| Default temperatures | sem=0.7, coarse=0.7, fine=0.5 | Fine is greedy (`do_sample=false`) |
| Default top_k | 50 (all stages) | |
| Hidden dim (full / small) | 1024 / 768 | All three GPTs |
| Layers (full / small) | 24 / 12 | All three GPTs |
| Heads (full / small) | 16 / 12 | `d_head=64` both |
| EnCodec upsample ratios | [8, 5, 4, 2] | Product = 320 |

## Data Layouts / Formats

### Semantic Transformer Input (513 tokens)
```
Position:   [0 ......... 255]      [256 ......... 511]     [512]
Content:    text tokens             semantic history        SEMANTIC_INFER_TOKEN
            (BERT IDs + 10048,      (prompt tokens,         (129599)
             right-pad with         right-pad with
             TEXT_PAD_TOKEN=129595) SEMANTIC_PAD_TOKEN=10000)
Dtype:      int64
Shape:      (B=1, 513)
```

### Semantic Transformer Output
```
A 1-D sequence of int IDs in [0, 9999], length up to 768.
EOS = 10000 (SEMANTIC_PAD_TOKEN).
Frame rate: 49.9 Hz → 768 tokens ≈ 15.4 seconds.
```

### Coarse Transformer Input Window (per sliding step)
```
Position:   [0 .... 59]                  [60]                [61 ... 691]
Content:    semantic window               COARSE_INFER_TOKEN  flat coarse history
            (60 semantic tokens, no       (12050)             ([c0,c1,c0,c1,...]
             offset — they live in the                        with codebook-1
             upper region of vocab=12096                      tokens offset by +1024,
             alongside coarse tokens)                         length ≤ 630)
Dtype:      int64
Shape:      (B=1, 1 + 60 + ≤630) = (1, ≤691)
```

### Coarse Transformer Output (per window)
```
~180 flattened tokens per window (60 semantic × ~3 coarse-per-semantic).
Even positions: codebook 0 in [0, 1023].
Odd positions:  codebook 1 in [1024, 2047] (subtract 1024 after sampling).
After full run: reshape to (2, T_coarse) of EnCodec indices in [0, 1023].
```

### Fine Transformer Input/Output (one 1024-window pass)
```
Input:  int64 tensor (B=1, 1024, 8) where columns 0..i-1 are filled (i = pred_idx),
        columns i..7 contain placeholder zeros.
        First fine_pred_idx columns may include the speaker fine_prompt prepended.
Forward: embeddings of columns 0..i summed → 24-block bidirectional transformer →
         head[i - 1] → logits (B, 1024, 1056).
Sample (greedy by default), write column i, increment i. Repeat for i=1..7.

For sequences longer than 1024: slide forward 512 tokens (overlap 512),
re-run all 7 passes, keep the last 512 of each subsequent window.
```

### Speaker Prompt File Set
```
Three .npy files per voice (numpy save format):
  <voice>_semantic_prompt.npy  shape (~256,)        int64
  <voice>_coarse_prompt.npy    shape (2, ~384)      int64
  <voice>_fine_prompt.npy      shape (8, ~384)      int64
Total size per voice: ~10-20 kB
```

### EnCodec Decoder Input/Output
```
Input:  int64 tensor (B=1, 8, T) — 8 codebook indices in [0, 1023]
        T = coarse_T = (semantic_T × 75 / 49.9) ≈ semantic_T × 1.503
Output: float32 tensor (B=1, 1, T*320) — mono 24kHz PCM, range ~[-1, 1]
```

### HF Consolidated Weight File (`pytorch_model.bin`)
```
PyTorch pickle, single state_dict with hierarchical keys:
  "semantic.*"       (BarkSemanticModel weights)
  "coarse_acoustics.*" (BarkCoarseModel weights)
  "fine_acoustics.*"   (BarkFineModel weights)
  "codec_model.*"      (EncodecModel weights)
Total: 4.49 GB (full) / 1.68 GB (small), FP32.
```

### Original Suno Per-Stage `.pt` Files
```
Each file is a pickled dict:
{
  "model_args": { "n_layer": int, "n_head": int, "n_embd": int,
                  "block_size": int, "input_vocab_size": int,
                  "output_vocab_size": int, "bias": bool, ... },
  "model": OrderedDict of "<prefix>module.<key>": Tensor
}
The "module." prefix is from nn.DataParallel training and must be stripped on load.
Key naming follows nanoGPT: transformer.wte, transformer.wpe,
  transformer.h.<n>.{ln_1,attn,ln_2,mlp}.*, transformer.ln_f, lm_head.
```

## Implementation Notes for HartsyInference

1. **Three independent GPT-2-style models**: all three stages reuse the same Transformer block. Implement one `GptBlock` (LayerNorm + CausalSelfAttention/NonCausalSelfAttention + LayerNorm + MLP) and one `GptModel` shell parametrized by `(hidden, layers, heads, vocab, block_size, attention_kind)`. Then instantiate three times. Estimated ~600 lines of C# total for the model definitions.

2. **`bias=False` everywhere except LayerNorms** in the released weights (despite the original nanoGPT defaulting to `bias=True`). Mirror the HF behavior: Linear modules omit bias, LayerNorms keep bias. Saves ~3% memory and is required to load the released checkpoints.

3. **BERT WordPiece tokenizer** (mBERT, ~120k vocab). Cross-reference [TOKENIZERS.md](TOKENIZERS.md) for the WordPiece implementation. We need:
   - UTF-8 NFC normalization, no lowercasing (cased model).
   - Whitespace splitting + Chinese character segmentation (a single CJK character is its own word).
   - WordPiece greedy longest-match-first with `##` continuation prefix.
   - The vocab can be loaded from `vocab.txt` (one token per line, line number = id) — much simpler than parsing the full `tokenizer.json`.
   - We do NOT need `[CLS]`/`[SEP]`/`[MASK]` insertion — Bark drops them.

4. **EnCodec decoder reuse**: implement once per [AUDIO_CODECS.md](AUDIO_CODECS.md). The exact same decoder is shared with MusicGen, AudioGen, and many other models — high reuse value. Bark only needs the **decoder path**; we can skip the encoder unless we want a "voice cloning from raw audio" feature (which would require also re-implementing the HuBERT semantic clusterer — much harder).

5. **Speaker-prompt format**: simplest representation is three flat `int32` arrays per voice. Convert from `.npy` to our packed format once at packaging time. Voice file size is <20 kB, so we can ship all ~120 official voices for a few MB.

6. **`AlternatingCodebooksLogitsProcessor`**: implement as a single boolean toggle in the coarse sampler. Before the softmax: if `step_count % 2 == 0` set logits indices 1024..2047 to `float.MinValue`; else set 0..1023 and 2048.. to MinValue.

7. **Fine-stage non-causal attention**: most-easy if our `GptBlock` accepts an `IsCausal` flag and we generate a different attention mask (all-ones vs lower-triangular). Both kernels can be unified.

8. **Fine-stage embedding sum**: 8 separate `Embedding(1056, hidden)` modules per `BarkFineModel`. At step `i`, compute `Σ_{k=0..i} W_emb_k[tokens[:,:,k]]`. This is `i+1` parallel gathers + a sum reduction along a 9th axis. A naive loop of i+1 gathers each producing `(B, T, H)` and adding to an accumulator is fine.

9. **Sampling**: temperature + top-k is straightforward — sort logits, take top 50, softmax, multinomial. Same code as for any GPT-2 generation. Fine stage is greedy: just argmax.

10. **KV cache**: only needed for the two causal stages (semantic, coarse). Fine is one-shot per pass, no cache needed. The KV cache size for full Bark semantic = `24 layers × 2 (k,v) × 1024 max_seq × 1024 hidden × 2 (fp16) = 100 MB` worst case. Tractable.

11. **Memory plan** (full Bark, FP16, single GPU):
    - Semantic weights: ~900 MB (mostly the 132M embedding table)
    - Coarse weights: ~600 MB
    - Fine weights: ~700 MB (8 embedding tables × 1056 × 1024 × 2 + heads + transformer)
    - EnCodec weights: ~40 MB
    - KV caches: ~200 MB peak
    - Activations: ~500 MB peak
    - **Total ≈ 3 GB FP16, 6 GB FP32**. Comfortably fits a single 8 GB card.
    Plan: ship FP16 weights by default; allow opt-in BF16 / FP32.

12. **Streaming (cross-reference [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) §3.1)**: real streaming is awkward because the fine stage requires a full coarse window. Two paths:
    - **No streaming**: simplest, ~1.5-3 s first-audio latency on a fast GPU.
    - **Coarse-only streaming**: emit raw EnCodec-decoded coarse-only audio as the coarse stage progresses (audio quality is degraded — missing 6 fine codebooks ≈ 1.5 kbps quality), then re-emit the fully-fine version when it finishes. Probably not worth the code complexity for v1.
    - Defer streaming to v2.

13. **Determinism**: with the default sampling, runs are non-deterministic unless we set the RNG seed. The HF code respects PyTorch's RNG. For HartsyInference we need a deterministic seedable sampler — `Random(seed)` for the multinomial draws, parameterized via a constructor argument.

14. **Loading the original Suno `.pt` files vs HF consolidated `pytorch_model.bin`**: both are PyTorch pickle. Convert offline to safetensors as a packaging step. Cross-reference [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md). Suno per-stage files have a `model_args` dict + a `model` state_dict with `"module."` prefix; HF consolidated has hierarchical keys (`semantic.*`, `coarse_acoustics.*`, `fine_acoustics.*`, `codec_model.*`) with no `module.` prefix. Both convert cleanly.

15. **`SUNO_USE_SMALL_MODELS` / `SUNO_OFFLOAD_CPU`**: not needed in our design — we'll expose a `BarkVariant` enum (`Full | Small`) and an explicit `OffloadPolicy` API that pages each GPT into VRAM only when running.

16. **Validation harness**: end-to-end golden tests should compare our output to Python's `bark.generate_audio()` output bit-for-bit at fixed seed. EnCodec decode is deterministic; the three GPT forward passes are deterministic; only the sampling step introduces variance. With seed control we can match Python exactly.

## Reference Implementations

- [suno-ai/bark](https://github.com/suno-ai/bark) — Official reference. Nanagpt-derived, ~2000 LoC total.
  - `bark/model.py` — Semantic + Coarse GPT (causal).
  - `bark/model_fine.py` — Fine GPT (non-causal masked LM).
  - `bark/generation.py` — All inference orchestration, the constants table, sampling defaults.
  - `bark/api.py` — High-level `generate_audio()` wrapper.
- [huggingface/transformers `models/bark/`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bark) — HF port with KV cache, batched generation, FlashAttention support.
  - `modeling_bark.py` — `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, `BarkModel` wrapper.
  - `configuration_bark.py` — Config dataclasses (good source for default hyperparams).
  - `generation_configuration_bark.py` — Per-stage generation defaults.
  - `processing_bark.py` — Text + voice-prompt preprocessing.
- [suno/bark](https://huggingface.co/suno/bark) — Full-size weights.
- [suno/bark-small](https://huggingface.co/suno/bark-small) — Small variant.
- [ylacombe/bark-large](https://huggingface.co/ylacombe/bark-large) — Speaker-embedding registry (~550 voices).
- [serp-ai/bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone) — Community fork that demonstrates speaker-prompt extraction (HuBERT + EnCodec). Useful only if we want to expose voice cloning.

## Differences Between Implementations

| Aspect | Original Suno | HF Transformers |
|---|---|---|
| Weight format | 3 × `.pt` per variant | 1 × `pytorch_model.bin` |
| Class structure | Plain `GPT` + `FineGPT` | `BarkSemanticModel`, `BarkCoarseModel`, `BarkFineModel`, `BarkModel` |
| Sampling | Manual top-k loop in `generation.py` | `transformers.generation.utils.GenerationMixin` |
| Coarse alternation | Hard-coded `if step%2` masking | `AlternatingCodebooksLogitsProcessor` (subclass of `LogitsProcessor`) |
| KV cache | Manual list-of-tuples cache | HF's `Cache` abstraction |
| Speaker prompts | Bundled `.npz` per voice in `assets/prompts/` | External `ylacombe/bark-large` repo with per-prompt `.npy` files |
| Tokenizer | Direct call to `BertTokenizer` | `BarkProcessor` wraps it |
| EnCodec | Imported from `encodec` pip package | Internal `EncodecModel` from `transformers` |
| Bias on Linears | nanoGPT default `True` (but weights have none) | Honors config (`bias=false`) |
| Attention impl | Manual + optional `F.scaled_dot_product_attention` | SDPA / FlashAttention2 / eager all supported |

For HartsyInference we model after the HF version (cleaner separation, config-driven), but use the original Suno per-stage `.pt` only if it's smaller for download.
