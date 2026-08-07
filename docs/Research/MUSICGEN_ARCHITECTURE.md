# MusicGen / AudioGen / AudioCraft — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (MusicGen pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

MusicGen (Meta FAIR, Jun 2023, [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)) and AudioGen (Meta FAIR, Sep 2022, [arXiv:2209.15352](https://arxiv.org/abs/2209.15352), re-released Jun 2023 under AudioCraft) are autoregressive Transformer language models that generate discrete audio codec tokens conditioned on text. The two models share a single architectural recipe (the "AudioCraft" recipe): a frozen T5-base text encoder, a single-stage causal decoder LM, and a frozen EnCodec audio tokenizer; they differ only in the codec configuration (32 kHz / 4 codebooks for music, 16 kHz / 4 codebooks for sound effects) and training data. The key insight versus prior work (AudioLM, VALL-E) is the **delay pattern**: a single transformer can predict K parallel residual-vector-quantized codebooks per step by offsetting each codebook by k timesteps, collapsing the K-pass hierarchy into one pass at the cost of K extra steps of latency at the end.

This file covers MusicGen (mono, melody, stereo), AudioGen, and the shared transformer / pattern / CFG machinery. The EnCodec audio codec (encoder, decoder, RVQ, training-time discriminators) is documented in [AUDIO_CODECS.md](AUDIO_CODECS.md) — refer to its EnCodec section for codec internals. T5 encoder details are in [TEXT_ENCODERS.md](TEXT_ENCODERS.md). General LLM transformer patterns (KV-cache, RoPE vs sinusoidal, attention) follow the native `HartsyInference.LLM` package. CFG sampling math is in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft), [MusicGen paper (arXiv:2306.05284)](https://arxiv.org/abs/2306.05284), [AudioGen paper (arXiv:2209.15352)](https://arxiv.org/abs/2209.15352), HuggingFace configs for `facebook/musicgen-{small,medium,large,melody,melody-large}` and `facebook/musicgen-stereo-*` and `facebook/audiogen-medium`, [HF Transformers MusicGen docs](https://huggingface.co/docs/transformers/model_doc/musicgen), `audiocraft/modules/codebooks_patterns.py`, `audiocraft/models/musicgen.py`, `audiocraft/modules/conditioners.py`.

## Key Numbers and Constants

| Constant | Value | Source |
|----------|-------|--------|
| MusicGen codec sample rate | 32000 Hz | `facebook/encodec_32khz` |
| MusicGen codec frame rate | 50 Hz | EnCodec config |
| MusicGen downsample factor | 640× | 32000 / 50 |
| AudioGen codec sample rate | 16000 Hz | `facebook/encodec_16khz` |
| AudioGen codec frame rate | 50 Hz | EnCodec config |
| AudioGen downsample factor | 320× | 16000 / 50 |
| Codebook size | 2048 | MusicgenDecoderConfig.vocab_size |
| Number of codebooks (mono) | 4 | MusicgenDecoderConfig.num_codebooks |
| Number of codebooks (stereo) | 8 | 2 channels × 4 |
| Special token id (pad/BOS/null) | 2048 | MusicgenDecoderConfig.pad_token_id |
| Delay pattern (mono) | [0, 1, 2, 3] | DelayedPatternProvider default |
| Delay pattern (stereo) | [0, 0, 1, 1, 2, 2, 3, 3] | `interleave_stereo_codebooks` |
| Max LM context | 2048 tokens | MusicgenDecoderConfig.max_position_embeddings |
| Max audio duration (single shot) | 30 s | 1500 codec frames + delay tail |
| Chunked extend_stride | 18 s | musicgen.set_generation_params default |
| Default CFG scale (cfg_coef) | 3.0 | musicgen.set_generation_params default |
| Default top_k | 250 | musicgen.set_generation_params default |
| Default top_p | 0.0 (off) | musicgen.set_generation_params default |
| Default temperature | 1.0 | musicgen.set_generation_params default |
| Default duration | 15 s | musicgen.set_generation_params default |
| Text encoder hidden | 768 | t5-base |
| Text encoder layers | 12 | t5-base |
| Text encoder vocab | 32128 | t5-base SentencePiece |
| Text encoder max length | 512 | t5-base |
| Chroma bins | 12 | ChromaStemConditioner.n_chroma |
| Chroma frame rate | 50 Hz (aligned to codec) | ChromaStemConditioner.winhop |
| Head dim (all sizes) | 64 | hidden_size / num_heads |
| FFN ratio (all sizes) | 4.0 | ffn_dim / hidden_size |

## Data Layouts and Formats

### Audio codes tensor in the LM

Shape `[B, K, T_lm]` where `T_lm = T_codec + K - 1` after applying delay pattern. dtype `int32` (or `int64` for HF). Value range `[0, codebook_size]` with `codebook_size` (= 2048) reserved as the special token.

### LM input embedding sum

```
inp[B, 1, D] = sum_{k=0..K-1} embed_k( codes[B, k, s - delays[k]] )
```

When `s < delays[k]`, the token is the special token 2048, which has its own learned embedding row.

### LM output logits per step

Shape `[B, K, codebook_size]` = `[B, 4, 2048]` for mono. Each codebook head is an independent `nn.Linear(D, 2048)`. No shared parameters across heads.

### Cross-attention inputs

Shape `[B, L_text + L_chroma, D_cross]` where `D_cross` equals the decoder hidden_size (the T5-768-to-D bridge happens in a `Linear(768, D)` projection inside the conditioner). `L_chroma = 0` for non-melody models. Cross-attention mask is the OR of the text mask and a chroma mask (chroma is fully valid).

### Checkpoint format (HF Transformers)

`facebook/musicgen-*` ships as **safetensors**. Composite model layout:
- `text_encoder.*` — full T5-base.
- `audio_encoder.*` — full EnCodec (encoder + decoder + quantizer).
- `decoder.*` — the MusicGen LM.
- `enc_to_dec_proj.*` — Linear(768, decoder_hidden) used to project T5 output into cross-attention dim.

For inference only, the `audio_encoder` weights for the EnCodec *encoder* are unused (we never re-encode audio unless doing audio-prompted continuation); we only need the EnCodec *decoder* weights. HartsyInference can split these.

## Reference Implementations

1. **facebookresearch/audiocraft** (original, PyTorch, MIT code / CC-BY-NC weights): the canonical source. `audiocraft/models/musicgen.py`, `audiocraft/models/lm.py`, `audiocraft/modules/codebooks_patterns.py`, `audiocraft/modules/conditioners.py`. Validate everything against this.
2. **HuggingFace Transformers** (PyTorch): `src/transformers/models/musicgen/modeling_musicgen.py`. Slightly different code path (uses the generic `GenerationMixin`, batches CFG by concat-doubling the batch). Easier to read; matches AudioCraft outputs bit-for-bit when given the same seed.
3. **musicgen.cpp** (community ggml port): C++ port of MusicGen used by some llama.cpp-derived audio tools. Useful as a cross-check for quantized inference. Not all variants supported.
4. **MLX-audio** (Apple, Swift/Python on Apple Silicon): another reference port.

For validation, the AudioCraft official and HF Transformers implementations should produce identical waveforms when given the same RNG seed, same text, same sampling params, and the same starting EnCodec codes (if audio-prompted).

## Differences Between Implementations

- **HF Transformers batches CFG by stacking** the conditional and unconditional inputs into one 2B-batch forward pass; AudioCraft has both `two_step_cfg=True` (separate forwards) and `False` (also batched). They produce numerically equivalent logits but differ in memory pattern.
- **HF stores the delay pattern in `decoder.build_delay_pattern_mask`**; AudioCraft uses a `PatternProvider` object and emits a "Pattern" with explicit `LayoutCoord` for each LM step. Functionally equivalent.
- **HF EnCodec wrapper** has a slightly different audio normalization step (reflection padding to multiples of stride); for short audio prompts (<<1s) you can get off-by-a-few-frames behavior versus AudioCraft. Negligible for inference.
- **Stereo handling** in HF Transformers required a v4.34+ patch; older versions silently truncated to mono.
- **EnCodec int weight format**: HF safetensors store residual quantizer codebooks as float32 `[num_codebooks=4, codebook_size=2048, codebook_dim=128]`. AudioCraft stores them in nested `nn.Embedding` layers. Same numerical weights, different addressing.

## Implementation Notes for HartsyInference

This is the first **autoregressive** model in HartsyInference (everything to date is diffusion or feed-forward — diffusion samplers loop, but each step is feed-forward). Many of the patterns needed already exist in HartsyInference.LLM; MusicGen reuses those patterns.

### What HartsyInference already has

- **T5 encoder** — exists in `HartsyInference.Diffusion` (used by SD3, Flux, AuraFlow). T5-base is smaller than the T5-XXL used by diffusion but uses the same code path. Reuse it; just allow a different size config.
- **SentencePiece tokenizer** — exists for T5.
- **Safetensors loader** — exists.
- **CUDA/Vulkan compute** — exists.
- **CFG** — pattern documented in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md). The MusicGen CFG is structurally identical to SD/Flux CFG (one cond pass, one uncond pass, combine with `uncond + s*(cond - uncond)`). The only new aspect is that it is applied per-step in an autoregressive loop instead of per-denoise-step.

### What is new and must be built

1. **KV-cache management**. Required for autoregressive decode to be tractable. Design:
   - `KvCache` struct per layer, holding two `NativeBuffer` of shape `[B, H, T_max, D_head]` for K and V.
   - `T_max` allocated up front to the maximum sequence length (2048 for MusicGen). No re-allocation during decode.
   - Append-only writes: at step s, write into slot `T_used[s]`, then `T_used += 1`.
   - Self-attn read: K and V from `[0..T_used]`. Cross-attn read: K and V are filled once from the encoder output and then read-only across all steps.
   - For batched CFG, the cache is 2× the batch dimension. Easier: have two separate caches, one per branch.
   - Memory pre-allocation is critical because GC pressure in a 1500-step loop is the difference between RTF 2× and RTF 0.5×.

2. **Causal Transformer decoder block**. Different from the diffusion transformer blocks we have:
   - Standard pre-norm LayerNorm (we have `Layernorm`; reuse).
   - Causal self-attention (we have full attention; need to add a causal mask path; for KV-cache decode, the mask is "query attends to all cached + self", which is trivial).
   - Cross-attention (we have this for SD3-style joint attention; the encoder-decoder variant is simpler — no shared projection).
   - GELU FFN with 4× ratio (we have GELU; standard).
   - No RoPE — sinusoidal positions added to embeddings, plain absolute. Trivial.
   - Architecture is close enough to HartsyInference.LLM that we can reuse the attention/FFN kernels and add cross-attention.

3. **EnCodec decoder** — pure C# reimplementation per [AUDIO_CODECS.md](AUDIO_CODECS.md). The encoder is only needed for audio-prompted continuation; ship the decoder first.

4. **Delay-pattern state machine**. New code. Inputs: a `[B, K, T_codec]` codes buffer. Operations:
   - `BuildInputForStep(int s) -> int[K]`: returns the K token ids to embed and sum at LM step s.
   - `WriteOutputForStep(int s, int[K] sampled)`: writes the sampled tokens at the correct delayed positions in the codes buffer.
   - `Undelay() -> [B, K, T_codec]`: at end of decode, slice out the actual audio codes.
   - This is a ~50-line class. Worth a dedicated `DelayPattern` type to keep the LM loop clean.

5. **Sampling kernel** (top-k + temperature). Already in HartsyInference.LLM; reuse it. Per-codebook (vectorize across K).

6. **Chromagram extractor** (melody-only). STFT-based, 12 bins. We have STFT code from Whisper preprocessor. Reuse it; add a chroma binning step. Or skip melody for v1 and ship text-only first.

7. **Stereo channel handling**. Add as a config flag; the decoder is wider (8 input embeddings, 8 output heads). Same loop, just K=8 and a different delay pattern.

### Suggested implementation order

1. EnCodec decoder (validate against `facebook/encodec_32khz` decode of saved test codes).
2. KV-cache infrastructure (with a tiny GPT-2 toy model as a unit test).
3. MusicGen decoder block (without cross-attention) — validate self-attention output against HF Transformers on a fixed input.
4. Add cross-attention; load `musicgen-small`; validate one decode step against HF.
5. Delay pattern + sampling; do 10-step decode and compare to HF.
6. Full 500-step decode → EnCodec → audio; validate FAD or just audible quality.
7. CFG; validate guidance changes the output as expected.
8. Chunked generation for >30 s.
9. Stereo (add 8-codebook path).
10. Melody conditioning (chromagram + concat to cross-attn).
11. AudioGen (just swap codec and reload — should otherwise just work).

### Performance targets

- Match HF Transformers fp16 RTF within 1.2× on RTX 4090 for `musicgen-medium` after KV-cache and a sane attention kernel.
- Stretch goal: beat HF Transformers RTF by using batched CFG + Flash-Attention-style fused QKV (see [FLASH_ATTENTION.md](FLASH_ATTENTION.md)).
- Memory: stay under HF Transformers fp16 footprint for `musicgen-large` (~8.5 GB for 10 s). KV-cache pre-allocation should be a single contiguous block per layer.
