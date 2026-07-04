# MusicGen / AudioGen / AudioCraft — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (MusicGen pipeline)

## Summary

MusicGen (Meta FAIR, Jun 2023, [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)) and AudioGen (Meta FAIR, Sep 2022, [arXiv:2209.15352](https://arxiv.org/abs/2209.15352), re-released Jun 2023 under AudioCraft) are autoregressive Transformer language models that generate discrete audio codec tokens conditioned on text. The two models share a single architectural recipe (the "AudioCraft" recipe): a frozen T5-base text encoder, a single-stage causal decoder LM, and a frozen EnCodec audio tokenizer; they differ only in the codec configuration (32 kHz / 4 codebooks for music, 16 kHz / 4 codebooks for sound effects) and training data. The key insight versus prior work (AudioLM, VALL-E) is the **delay pattern**: a single transformer can predict K parallel residual-vector-quantized codebooks per step by offsetting each codebook by k timesteps, collapsing the K-pass hierarchy into one pass at the cost of K extra steps of latency at the end.

This file covers MusicGen (mono, melody, stereo), AudioGen, and the shared transformer / pattern / CFG machinery. The EnCodec audio codec (encoder, decoder, RVQ, training-time discriminators) is documented in [AUDIO_CODECS.md](AUDIO_CODECS.md) — refer to its EnCodec section for codec internals. T5 encoder details are in [TEXT_ENCODERS.md](TEXT_ENCODERS.md). General LLM transformer patterns (KV-cache, RoPE vs sinusoidal, attention) follow the dotLLM patterns in [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md). CFG sampling math is in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft), [MusicGen paper (arXiv:2306.05284)](https://arxiv.org/abs/2306.05284), [AudioGen paper (arXiv:2209.15352)](https://arxiv.org/abs/2209.15352), HuggingFace configs for `facebook/musicgen-{small,medium,large,melody,melody-large}` and `facebook/musicgen-stereo-*` and `facebook/audiogen-medium`, [HF Transformers MusicGen docs](https://huggingface.co/docs/transformers/model_doc/musicgen), `audiocraft/modules/codebooks_patterns.py`, `audiocraft/models/musicgen.py`, `audiocraft/modules/conditioners.py`.

## Detailed Findings

### Model Family Overview

There are three product lines on top of one architecture:

- **MusicGen** — music generation at 32 kHz, mono. Sizes 300M / 1.5B / 3.3B. Also a "melody" variant that adds chromagram conditioning. Text-only conditioning via T5-base.
- **MusicGen-Stereo** — stereo (2-channel) fine-tunes of the mono MusicGen models. Trained for 200k additional steps from the mono checkpoints. Same transformer dims, but operates on 8 codebooks (2 channels × 4) with a paired delay pattern.
- **AudioGen** — sound effect / environmental audio generation at 16 kHz, mono. Originally proposed Sep 2022 with a different design; the released `facebook/audiogen-medium` (1.5B) was retrained to follow the MusicGen recipe (delay pattern, single-stage LM). No melody mode.

Architecture is invariant across these three. Differences live in:

1. The EnCodec configuration the LM consumes (32 kHz vs 16 kHz; 4 vs 4 codebooks; channel count).
2. The conditioning attribute set (text only, text + chroma, text only respectively).
3. The training data and license.

### Variants Table

All sizes confirmed from `config.json` on the HuggingFace model repos.

| Model | Params | Decoder hidden_size | Layers | Heads | FFN dim | Head dim | Channels | Codebooks (n_q) | Codec sample rate | Codec frame rate | Codebook size | Text encoder | Max LM ctx (tokens) | Max audio duration |
|-------|-------:|--------------------:|-------:|------:|--------:|---------:|---------:|----------------:|------------------:|-----------------:|--------------:|--------------|--------------------:|-------------------:|
| `musicgen-small` | 300M | 1024 | 24 | 16 | 4096 | 64 | 1 | 4 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s (1503 codec frames) |
| `musicgen-medium` | 1.5B | 1536 | 48 | 24 | 6144 | 64 | 1 | 4 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s |
| `musicgen-large` | 3.3B | 2048 | 48 | 32 | 8192 | 64 | 1 | 4 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s |
| `musicgen-melody` | 1.5B | 1536 | 48 | 24 | 6144 | 64 | 1 | 4 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) + chroma | 2048 | 30 s |
| `musicgen-melody-large` | 3.3B | 2048 | 48 | 32 | 8192 | 64 | 1 | 4 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) + chroma | 2048 | 30 s |
| `musicgen-stereo-small` | 300M | 1024 | 24 | 16 | 4096 | 64 | 2 | 8 (=2×4) | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s |
| `musicgen-stereo-medium` | 1.5B | 1536 | 48 | 24 | 6144 | 64 | 2 | 8 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s |
| `musicgen-stereo-large` | 3.3B | 2048 | 48 | 32 | 8192 | 64 | 2 | 8 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | 30 s |
| `musicgen-stereo-melody` | 1.5B | 1536 | 48 | 24 | 6144 | 64 | 2 | 8 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) + chroma | 2048 | 30 s |
| `musicgen-stereo-melody-large` | 3.3B | 2048 | 48 | 32 | 8192 | 64 | 2 | 8 | 32 kHz | 50 Hz | 2048 | T5-base (frozen) + chroma | 2048 | 30 s |
| `audiogen-medium` | 1.5B | 1536 | 48 | 24 | 6144 | 64 | 1 | 4 | 16 kHz | 50 Hz | 2048 | T5-base (frozen) | 2048 | ~10 s (training); generates longer at inference but quality degrades |

Notes:
- FFN dim is always `4 * hidden_size` (standard GPT/LLaMA ratio). Activation is GELU.
- Head dim is always 64 (`hidden_size / num_heads`).
- The `large-melody` checkpoint was added after the original paper (it is not in Table 1 of arXiv:2306.05284 but is published on HF).
- "T5-base (frozen)" means the original `t5-base` (220M params, 12 encoder layers, hidden 768, 12 heads, FFN 3072, vocab 32128, max position 512). Text encoder weights are frozen during MusicGen training. Some HF checkpoints package Flan-T5-base instead of vanilla T5-base, but the dimensions are identical and the AudioCraft codebase loads with `t5-base` by default.
- "Max audio duration" of 30 s is the design limit: the decoder uses sinusoidal positional embeddings with `max_position_embeddings = 2048`, and 30 s × 50 Hz × (1 step per codec frame with delay overhead) = ~1503 LM steps. Longer audio is generated via overlapping chunked generation with an `extend_stride` (default 18 s) — see Inference Loop below.
- Codec frame rate of 50 Hz means the EnCodec encoder downsamples 32 kHz audio by exactly 640× (32000 / 640 = 50). For AudioGen, EnCodec downsamples 16 kHz audio by 320×.

### Architecture: Decoder LM

The decoder is a standard pre-norm causal Transformer with these properties (per `MusicgenDecoderConfig` in HF Transformers and `audiocraft/modules/transformer.py`):

- **Embeddings**: K parallel `nn.Embedding(2048+1, hidden_size)` tables, one per codebook. The "+1" reserves token id `2048` as the special token (used for BOS / padding / "no token yet" in the delay pattern). Per-step input to the transformer is the **sum** of the K embedding lookups, not a concatenation.
- **Position encoding**: Sinusoidal absolute position embeddings (`SinusoidsPositionalEmbedding`), added to the embedded input. Not learned, not RoPE. This is why max duration is capped — there is no extrapolation.
- **Layer count**: 24 (small) / 48 (medium, large, audiogen-medium).
- **Hidden size**: 1024 / 1536 / 2048 for small / medium / large.
- **Heads**: 16 / 24 / 32 (head dim always 64).
- **FFN**: 4× hidden, GELU activation, no gating (not SwiGLU).
- **Normalization**: Standard LayerNorm, pre-norm (norm-then-attention, norm-then-FFN, residual after).
- **Causal mask**: Lower-triangular self-attention mask.
- **Cross-attention**: Each layer has a cross-attention sublayer attending from the decoder positions to the T5 encoder hidden states. Standard `enc-dec` transformer layout: `self_attn -> cross_attn -> ffn`. Cross-attention K/V come from the (text-encoder-output, projected to decoder hidden size if needed).
- **Output heads**: K parallel `nn.Linear(hidden_size, 2048)` heads, one per codebook. At step t, the same decoder hidden state is fed into all K heads, producing K independent distributions over 2048 codebook entries.
- **No tied embeddings** (`tie_word_embeddings=False`).
- **Dropout**: 0.1 during training; 0 at inference.

This is functionally a "GPT-2-ish" decoder with an extra cross-attention sublayer and K input embedding tables + K output heads. It is straightforwardly a "T5-style decoder" (encoder-decoder LM with cross-attention) but with the input/output multiplied across the K codebooks.

### Audio Tokenizer: EnCodec

**MusicGen uses `facebook/encodec_32khz`**: 1-channel input, 32 kHz, 50 Hz token rate, 4 codebooks of size 2048, target bandwidth 2.2 kbps. The encoder downsamples by 640× through strided convolutions; the RVQ quantizer produces 4 codebook indices per frame.

**AudioGen uses `facebook/encodec_16khz`** (or an equivalent re-trained 16 kHz codec): 1-channel input, 16 kHz, 50 Hz token rate, 4 codebooks of size 2048. Downsamples by 320×.

For codec internals (encoder conv stack, RVQ algorithm, decoder ConvTranspose stack, bandwidth control, dequantizer math, weight format), see **AUDIO_CODECS.md EnCodec section**. The MusicGen decoder LM treats the codec purely as a black-box token producer/consumer.

### Codebook Interleaving Patterns

This is the central novelty of MusicGen and the main piece of logic that does not exist in HartsyInference.LLM or any existing HartsyInference component. Reference implementation: `audiocraft/modules/codebooks_patterns.py`.

Given a tensor of audio codes with shape `[B, K, T]` (B = batch, K = codebooks, T = codec frames), a "pattern" specifies how those K×T tokens are arranged into a flat sequence of LM steps. The paper compares four patterns; MusicGen ships with **delay**.

#### 1. Parallel pattern (`ParallelPatternProvider`, `delays=[0,0,0,0]`)

All K codebooks at the same timestep are predicted simultaneously and consumed simultaneously. LM step `s` corresponds to codec timestep `s`, and produces all K tokens in parallel.

```
LM steps:    s=0       s=1       s=2       s=3
codebook 0:  c0[0]     c0[1]     c0[2]     c0[3]
codebook 1:  c1[0]     c1[1]     c1[2]     c1[3]
codebook 2:  c2[0]     c2[1]     c2[2]     c2[3]
codebook 3:  c3[0]     c3[1]     c3[2]     c3[3]
```

Pro: fewest LM steps (= T). Con: at step s, the model predicts c1[s] without seeing c0[s], so quality is worse than autoregressive-over-codebook designs.

#### 2. Delay pattern (`DelayedPatternProvider`, `delays=[0,1,2,3]`) — DEFAULT for MusicGen

Each codebook k is shifted by k timesteps. Codebook 0 is "in phase"; codebook 1 is one step late; codebook 2 is two steps late; codebook 3 is three steps late. The LM sequence length is `T + K - 1`.

```
LM step:     s=0       s=1       s=2       s=3       s=4       s=5       s=6
codebook 0:  c0[0]     c0[1]     c0[2]     c0[3]     c0[4]     ----      ----      ----
codebook 1:  ----      c1[0]     c1[1]     c1[2]     c1[3]     c1[4]     ----      ----
codebook 2:  ----      ----      c2[0]     c2[1]     c2[2]     c2[3]     c2[4]     ----
codebook 3:  ----      ----      ----      c3[0]     c3[1]     c3[2]     c3[3]     c3[4]
```

At LM step `s`, the input column is `[c0[s], c1[s-1], c2[s-2], c3[s-3]]`. Positions before each codebook starts (the `----`) are filled with the special token `2048`. The trailing tail of length `K-1` is also filled with the special token before being passed in, and discarded after decoding.

Why this works: when the LM is predicting `[c0[s+1], c1[s], c2[s-1], c3[s-2]]` at the next step, codebook 1's prediction sees the just-predicted codebook 0 of the same codec frame as context (because c0[s] is at step s and c1[s] is at step s+1). So the model implicitly models the "coarse-to-fine" dependency between codebooks without doing K sequential passes.

To generate T codec frames you need `T + K - 1` LM steps (3 extra steps for K=4). After generation, the columns are "un-delayed" (shifted back) and the first K-1 and trailing K-1 columns are dropped.

#### 3. Coarse-first / VALL-E pattern (`CoarseFirstPattern`, `delays=[0]`, fine codebooks generated in a second pass)

Generate codebook 0 fully for the entire T frames first (T steps). Then in a second model pass, generate codebooks 1..K-1 (also T steps each, but these can be non-causal in time). Used by AudioLM and VALL-E. MusicGen does not ship this; it is in the codebase for comparison only.

#### 4. Unrolled / fully flattened pattern (`UnrolledPatternProvider`)

K codebooks at the same timestep are predicted strictly sequentially: c0[0], c1[0], c2[0], c3[0], c0[1], c1[1], .... Sequence length is K×T (highest quality, slowest). Used as an upper-bound baseline in the paper.

The paper shows delay pattern within ~0.3 FAD of the fully unrolled pattern at 4× the speed.

### Classifier-Free Guidance (CFG)

MusicGen uses standard text-conditional CFG (see [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md) for the general theory).

Two-stream prediction at each LM step:

1. **Conditional pass**: feed text encoder hidden states from the actual prompt as cross-attention K/V.
2. **Unconditional pass**: feed cross-attention K/V from a null prompt — for MusicGen this is the embedding of an empty string passed through T5, masked to length zero (so attention has nothing to attend to and the cross-attention output is zero/learned-null). The decoder hidden state is shared; only the cross-attention input differs.
3. **Combine logits per codebook head**:

```
logits = uncond + cfg_coef * (cond - uncond)
```

Then sample.

There are two CFG modes in `audiocraft/models/musicgen.py`:

- `two_step_cfg=False` (default): batch the conditional and unconditional inputs as a 2B batch through one transformer call. Cheaper if you have memory.
- `two_step_cfg=True`: run two separate forward passes. Used when memory-constrained or when batch shape varies.

The "double CFG" mode (`cfg_coef_beta`, used by MAGNeT-style models) is also supported in the code but not the standard MusicGen recipe.

CFG is applied **independently per codebook head** at the same LM step — same `cfg_coef` scales all four head logits with the same formula.

### Melody Conditioning (MusicGen-Melody)

Reference: `ChromaStemConditioner` in `audiocraft/modules/conditioners.py`.

1. Input: reference audio (mono or stereo), arbitrary sample rate.
2. **Stem separation** (training only): a Demucs source-separation model removes the drum and bass stems from the reference, keeping only the harmonic/melodic content. At inference, you can either supply pre-separated melody or skip separation.
3. **Resample** the (separated) audio to the codec sample rate (32 kHz).
4. **Chromagram extraction**: STFT-based 12-bin chromagram (constant-Q-like). `n_chroma = 12`. The chromagram window/hop is configured to match a target temporal resolution of ~50 Hz (so the chroma sequence length aligns with the codec frame rate; `chroma_len = 235` in the published melody config, corresponding to about 4.7 s of chroma per training example at 50 Hz).
5. **Argmax-and-zero** (optional, used in the released model): for each frame, keep only the dominant chroma bin (set the others to zero). This forces the conditioner to encode pitch class identity rather than chord color, which empirically gives sharper melody following. The full 12-D vector is still passed downstream — the non-max entries are just zeroed.
6. **Projection**: `nn.Linear(12, output_dim)` where `output_dim` matches the cross-attention K/V dim of the decoder. This projected chroma sequence is then **prepended** to the T5 encoder hidden states along the sequence axis, and the concatenated `[chroma; text]` sequence becomes the cross-attention K/V for every decoder layer.
7. At inference time with no melody supplied, the chroma is set to all zeros and behaves as a null condition.

So melody conditioning does NOT add a new cross-attention block; it concatenates new tokens onto the existing text cross-attention stream. The decoder is unchanged.

### Stereo Variants

Reference: `interleave_stereo_codebooks` in `audiocraft/solvers/musicgen.py` and the stereo model card.

1. EnCodec processes left and right channels **independently** (it is a mono codec). Each channel produces its own 4-codebook stream at 50 Hz. Total: 8 codebooks per frame.
2. The LM treats these as K=8 codebooks (4 left + 4 right). The decoder input/output heads are widened from 4 to 8 (the stereo checkpoints have 8 input embedding tables and 8 output heads).
3. The delay pattern is doubled. Instead of `[0,1,2,3]`, the stereo delay is `[0, 0, 1, 1, 2, 2, 3, 3]` — the L and R streams share a delay at each codec level. Sequence length is still `T + K_eff - 1` = `T + 3` LM steps (K_eff = 4 effective delay levels, not 8, because L and R at the same depth share a step).
4. Training: 200k updates fine-tuning from the matching mono checkpoint. All other hyperparameters identical.
5. Inference: identical loop, but the LM emits 8 tokens per step, the un-delay produces an `[B, 8, T]` token tensor, which is split into `[B, 4, T]` left and `[B, 4, T]` right, each fed independently through the EnCodec decoder, then stacked into 2-channel waveform.

### Sampling Parameters (defaults from `audiocraft/models/musicgen.py:set_generation_params`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_sampling` | `True` | If False, greedy argmax. |
| `top_k` | `250` | Set to 0 to disable. |
| `top_p` | `0.0` | Disabled by default; if >0, supersedes top_k. |
| `temperature` | `1.0` | Standard softmax temperature. |
| `cfg_coef` | `3.0` | The "guidance_scale" in HF Transformers. |
| `cfg_coef_beta` | `None` | Double-CFG, off by default. |
| `two_step_cfg` | `False` | Use batched single forward by default. |
| `extend_stride` | `18 s` | When duration > max_duration (30 s), generate in 30 s windows that overlap by `max_duration - extend_stride = 12 s`. |
| `duration` | `15 s` | Default if not set explicitly. |

The HuggingFace Transformers wrapper exposes these as `guidance_scale=3.0`, `do_sample=True`, `top_k=250`, `temperature=1.0`, `max_new_tokens=...` on `model.generate()`.

### Inference Loop (shapes assume B=1, MusicGen-large, mono, K=4)

```text
INPUTS
  text: string (or list for batch)
  duration_seconds: float, e.g. 10.0

STAGE 1 — Text encode (once)
  tokens: int32[1, L_text]                 via T5 tokenizer (SentencePiece, max 512)
  enc_hidden: float32[1, L_text, 768]      via T5-base encoder (frozen)
  enc_hidden_proj: float32[1, L_text, 2048] via Linear(768->2048) bridge (in the LM config)
  enc_mask: int32[1, L_text]               1 for valid, 0 for pad
  // For CFG: also build a null version
  null_hidden: float32[1, 0, 2048]         empty sequence (or learned-zero K/V)
  null_mask:   int32[1, 0]

STAGE 2 — Set up LM state
  T_codec  = round(duration_seconds * 50)              // e.g. 500 for 10 s
  T_lm     = T_codec + K - 1                           // = T_codec + 3 for K=4
  codes    = int32[1, K, T_codec + K - 1]              // initialized with special-token id 2048
  // BOS column: codes[:, :, 0..K-1] receive the delay-pattern "fill" — left-zone of the staircase
  kv_cache_self  = [ (K_l, V_l) for l in 1..48 ]       // self-attn KV cache, grows by 1 per step
  kv_cache_cross = [ (K_l, V_l) for l in 1..48 ]       // cross-attn KV cache, fixed at L_text

STAGE 3 — Autoregressive decode
  for s in 0 .. T_lm - 1:
    // Build input embedding for step s by summing K codebook embeddings.
    // For each codebook k, the "current" input is the token that the delay pattern says belongs
    // at LM position s for codebook k, i.e. codes[:, k, s - delays[k]] (= 2048 if s < delays[k]).
    inp_tokens: int32[1, K]
    inp_embed:  float32[1, 1, 2048]  = sum_k embed_k(inp_tokens[:, k])
    inp_embed += sinusoidal_pos(s)

    // Forward one step through 48 decoder layers (uses & updates kv_cache_self).
    // Cross-attention reads from enc_hidden_proj using kv_cache_cross.
    h: float32[1, 1, 2048] = decoder_layers(inp_embed, kv_cache_self, kv_cache_cross, enc_mask)

    // CFG: run twice (or batched 2x) — once with enc_hidden, once with null_hidden.
    h_cond, h_uncond = h_under_each_cond_pass
    logits_cond:   float32[1, K, 2048] = stack_k(head_k(h_cond))
    logits_uncond: float32[1, K, 2048] = stack_k(head_k(h_uncond))
    logits = logits_uncond + cfg_coef * (logits_cond - logits_uncond)

    // Sample per codebook independently.
    for k in 0..K-1:
      probs = softmax(logits[:, k, :] / temperature)
      probs = top_k_top_p_filter(probs, top_k=250, top_p=0.0)
      next_tok = multinomial(probs)
      // Place into the delayed codes tensor.
      target_t = s - delays[k]    // skip if target_t < 0 (still in the staircase pre-roll)
      if 0 <= target_t < T_codec:
        codes[:, k, target_t + delays[k]] = next_tok
    // (some implementations just write codes[:, k, s] = next_tok and un-delay later — equivalent)

STAGE 4 — Un-delay and decode audio
  // Drop the K-1 prefix columns (where some codebooks have no real token)
  // and the K-1 suffix columns. Realign so codes[:, k, t] is the true token for codec frame t.
  audio_codes: int32[1, K, T_codec] = undelay(codes)

  // EnCodec decoder produces waveform.
  // See AUDIO_CODECS.md EnCodec section for the decoder details.
  waveform: float32[1, 1, T_codec * 640] = encodec_decoder(audio_codes)  // 32 kHz
  // For stereo: split into [1, 4, T] left and [1, 4, T] right, decode each, stack to [1, 2, T*640].
```

For durations longer than 30 s, MusicGen does **chunked generation**:

```
generated = []
remaining = duration - 30
generated.append( generate_chunk(0, 30, prompt=None) )  // 30 s
while remaining > 0:
  // Use the last `extend_stride` seconds of the previous chunk as audio prompt.
  prompt_codes = encodec_encode( last_18s_of(generated[-1]) )
  next_chunk = generate_chunk(prompt_codes, 30, prompt=prompt_codes)
  generated.append( next_chunk_without_prompt_portion )
  remaining -= (30 - extend_stride)
```

This keeps continuity because the audio prompt re-seeds the LM with the prior context. The overlap is silently dropped.

### AudioGen Differences

AudioGen (the released `facebook/audiogen-medium`) is architecturally **identical** to `musicgen-medium` except:

1. **Codec**: 16 kHz EnCodec (`facebook/encodec_16khz`) instead of 32 kHz. Still 4 codebooks, 50 Hz, codebook size 2048. Audio bandwidth and quality are lower (it is a foley/SFX model, not a music model).
2. **No chromagram conditioning**, no melody mode. Text-only via T5-base.
3. **Training data**: ~4000 hours of sound effects (AudioSet, BBC SFX library, FSD50K, etc.) instead of 20k hours of music.
4. **No stereo variant.** All AudioGen outputs are mono.
5. **No formal max-duration limit** documented, but the sinusoidal positional embeddings cap things at ~30 s for the same reason. In practice it is used for 5–10 s clips.
6. The original 2022 AudioGen paper (arXiv:2209.15352) describes a different multi-stream architecture with no delay pattern; **that design was discarded** in favor of the MusicGen recipe for the public release. Cite the 2306.05284 paper for architecture, the 2209.15352 paper only for training data and task framing.
7. **Checkpoint format**: unlike `facebook/musicgen-*` (combined safetensors), `facebook/audiogen-medium` ships PyTorch **`.bin` pickles** and pairs with a separately-fetched `google/t5-base`. `MusicGenCheckpointConverter` therefore exposes format-agnostic loaders — `LoadDecoderAny` / `LoadEnCodecAny` / `LoadTextEncoderAny` dispatch by extension to `SafeTensorsLoader` or `PytorchPickleLoader` and apply the same key mapping. Fetch t5-base with `AudioModelCache.Get("google/t5-base", "pytorch_model.bin")`. Covered by `AudioGenLoaderTests`.

### Memory and Performance (consumer GPU, fp16 inference)

Approximate numbers from community reports and AudioCraft README.

| Model | Params | fp16 VRAM (weights) | fp16 VRAM (with KV cache for 10 s) | RTF on RTX 4090 |
|-------|-------:|--------------------:|-----------------------------------:|----------------:|
| small | 300M | ~0.6 GB | ~1.0 GB | ~10× real-time (1 s audio per ~0.1 s wall) |
| medium | 1.5B | ~3.0 GB | ~4.0 GB | ~3× real-time |
| large | 3.3B | ~6.6 GB | ~8.5 GB | ~1.5× real-time |
| melody (1.5B) | 1.5B | ~3.0 GB | ~4.0 GB | ~3× real-time |
| audiogen-medium | 1.5B | ~3.0 GB | ~3.5 GB | ~4× real-time |

Stereo variants are ~10% slower and use ~20% more KV-cache memory because of the doubled codebook count (input/output widening).

KV cache size for the decoder self-attention:
- per layer per token: `2 * num_heads * head_dim * dtype_bytes` bytes (K and V).
- large model: `2 * 32 * 64 * 2 = 8192` bytes per token per layer × 48 layers = 393 KB per LM step.
- 10 s of audio = 500 codec frames = 503 LM steps → ~197 MB self-attention cache. Cross-attention cache is fixed (`L_text * num_heads * head_dim * 2 * dtype_bytes` per layer) and small.

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

## Algorithm Steps

### Delay-pattern encode (training and audio-prompted inference)

```text
input:  codes shape [B, K, T]
output: delayed shape [B, K, T + K - 1]

for k in 0..K-1:
  delayed[:, k, 0 .. delays[k] - 1] = SPECIAL_TOKEN
  delayed[:, k, delays[k] .. delays[k] + T - 1] = codes[:, k, 0..T-1]
  delayed[:, k, delays[k] + T .. T + K - 2] = SPECIAL_TOKEN
```

### Delay-pattern decode (after generation)

```text
input:  delayed shape [B, K, T + K - 1]   (with K-1 leading and trailing pre-pad of special tokens)
output: codes shape [B, K, T]

for k in 0..K-1:
  codes[:, k, 0..T-1] = delayed[:, k, delays[k] .. delays[k] + T - 1]
// Drop the special-token columns from start and end.
```

### Single decode step

```text
1. Build inp_tokens[B, K] from `codes` using the delay pattern (current LM step s).
2. inp_embed = sum_k Embedding_k(inp_tokens[:, k])     # [B, 1, D]
3. inp_embed += sinusoidal_pos(s)                       # [B, 1, D]
4. For each decoder layer l in 1..L:
     h = LayerNorm(h)
     h = h + SelfAttention(h, cache=kv_self[l])         # uses & updates KV cache
     h = LayerNorm(h)
     h = h + CrossAttention(h, kv=enc_hidden_proj, cache=kv_cross[l], mask=enc_mask)
     h = LayerNorm(h)
     h = h + GELU(h @ W_ffn_up) @ W_ffn_down
5. logits[B, K, V] = stack_k( head_k(h) )
6. Apply CFG combine: logits = logits_uncond + cfg_coef * (logits_cond - logits_uncond)
7. Sample per codebook: next_tok[B, K] = sample(logits, top_k, top_p, temperature)
8. Write next_tok back into `codes` at the appropriate delayed position.
```

### Sampling (top_k + temperature)

```text
1. scaled = logits / temperature                       # [B, V]
2. probs = softmax(scaled)
3. If top_k > 0:
     keep top-k entries of probs, set rest to 0, renormalize.
4. If top_p > 0:
     sort probs descending, find smallest prefix whose cumulative sum >= top_p,
     keep that prefix, set rest to 0, renormalize.
5. multinomial sample one index.
```

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

## Open Questions

1. The HF `audiogen-medium` config is not on the HF hub at the same URL pattern as MusicGen — the model is loaded via AudioCraft's own loader, which suggests it is hosted in `facebook/audiogen-medium` under a different config schema. Verify the exact decoder hidden size for AudioGen by loading the checkpoint and inspecting; the table above assumes it equals `musicgen-medium` based on paper-claimed "1.5B" param count and "MusicGen architecture" claim, but we have not byte-verified.
2. The exact `delays` array for stereo (whether `[0,0,1,1,2,2,3,3]` or `[0,1,2,3,0,1,2,3]`) needs to be confirmed against the actual `audiocraft/solvers/musicgen.py` source — both forms appear in community discussions. The interleaving order (L-R-L-R per depth, vs all-L-then-all-R) matters for un-delay logic.
3. Whether AudioGen uses Flan-T5 or vanilla T5: AudioCraft's `text_conditioner.py` defaults to `t5-base`, but the released checkpoint may have been trained on either. Both have identical architecture so loading works for either, but the tokenizer vocabulary should be checked (Flan-T5 uses the same SentencePiece).
4. Chroma extraction details (window size, hop length, STFT n_fft) for melody conditioning are not fully documented in the model card. The chroma_len=235 in the melody config implies a specific hop, but the conditioner code reads it from the audio sample rate at runtime. Need to read `ChromaExtractor` source for exact STFT params for a C# port.
5. RTF numbers above are community-reported; we should run our own benchmarks once HartsyInference.Audio compiles.

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
