# Moonshine — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Moonshine pipeline)

Moonshine (Useful Sensors, 2024) is a tiny encoder-decoder ASR family explicitly designed for edge devices and live transcription. Unlike Whisper, it operates **directly on the raw 16 kHz waveform** (no mel spectrogram), uses **RoPE** instead of learned absolute positional embeddings, and processes **variable-length** audio without zero-padding to 30 s.

Pure C# implementation is straightforward: a 3-layer Conv1D front-end, then a standard pre-existing pre-LN Transformer encoder/decoder pattern (we already have this for Whisper and the HartsyInference.LLM decoder), reusing the RoPE we built for Flux / Hunyuan / Z-Image and the BPE tokenizer infrastructure from HartsyInference.Tokenizers.

---

## 1. Variants

The "Moonshine v1" paper (arXiv:2410.15608, Oct 2024) shipped two English-only models. A v2 streaming family followed in late 2025 / early 2026.

| Variant | Params | hidden | enc layers | dec layers | heads | partial_rotary_factor | safetensors size | License |
|---|---|---|---|---|---|---|---|---|
| `UsefulSensors/moonshine-tiny`            | 27.1M | 288 | 6  | 6  | 8 | **0.9**   | ~108 MB (F32) | MIT |
| `UsefulSensors/moonshine-base`            | 61.5M | 416 | 8  | 8  | 8 | **0.62**  | ~246 MB (F32) | MIT |
| `UsefulSensors/moonshine-streaming-tiny`   | ~34M  | (v2: ergodic encoder with sliding-window attention) |  |  |  |  |  | MIT |
| `UsefulSensors/moonshine-streaming-base`   | ~58M  | (v2)  |  |  |  |  |  | MIT |
| `UsefulSensors/moonshine-streaming-small`  | ~123M | (v2)  |  |  |  |  |  | MIT |
| `UsefulSensors/moonshine-streaming-medium` | ~245M | (v2)  |  |  |  |  |  | MIT |

> **Initial pure-C# target: `moonshine-tiny` and `moonshine-base` (v1).** The v2 ergodic streaming models swap the conv front-end for an 80-sample-window + CMVN + asinh + 2x stride-2 conv preprocessor, and use a position-free sliding-window encoder. That is a separate architecture; implement v1 first.

Newer related: "Flavors of Moonshine" (arXiv:2509.02523, Sep 2025) — tiny specialized ASR for specific accents/dialects, same backbone.

---

## 2. Architecture overview

```
raw waveform (16 kHz, mono, float32)  [N samples]
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Audio Preprocessor (3x Conv1D, learned)        │
│   - replaces the mel spectrogram entirely        │
│   - downsamples by 384x                          │
└─────────────────────────────────────────────────┘
   │  [N/384, hidden_size]   (≈41.67 frames/sec)
   ▼
┌─────────────────────────────────────────────────┐
│  Transformer Encoder (pre-LN, RoPE)             │
│   - 6 layers (tiny) / 8 layers (base)            │
│   - MHA (no GQA in released checkpoints)         │
│   - FFN = Linear → GELU → Linear (NOT gated)     │
└─────────────────────────────────────────────────┘
   │  encoder_hidden_states  [N/384, hidden_size]
   ▼
┌─────────────────────────────────────────────────┐
│  Transformer Decoder (pre-LN, RoPE, causal)     │
│   - 6 layers (tiny) / 8 layers (base)            │
│   - self-attn (causal, RoPE) + cross-attn        │
│   - FFN = Gated SiLU (SwiGLU-style)              │
└─────────────────────────────────────────────────┘
   │  decoder_hidden_states
   ▼
   tied lm_head → vocab logits (32768)
```

Both encoder and decoder are **pre-LayerNorm** transformers (the conventional modern arrangement: `x = x + Attn(LN(x))`). The HuggingFace `modeling_moonshine.py` puts `input_layernorm` before the attention, `post_attention_layernorm` before the MLP, and a final `layer_norm` after the last encoder/decoder layer.

---

## 3. Audio front-end (Conv1D, no mel)

The single biggest differentiator vs Whisper. Defined inside `MoonshineEncoder.__init__` in HF transformers:

```python
embed_dim = config.hidden_size
self.conv1     = nn.Conv1d(1,           embed_dim,    kernel_size=127, stride=64, bias=False)
self.conv2     = nn.Conv1d(embed_dim,   2 * embed_dim, kernel_size=7,   stride=3)
self.conv3     = nn.Conv1d(2 * embed_dim, embed_dim,   kernel_size=3,   stride=2)
self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)
```

Forward pipeline (raw mono waveform → encoder input):

```
x = input_values.unsqueeze(1)            # [B, 1, N]
x = tanh(conv1(x))                       # [B, hidden,   N1]   N1 ≈ (N-127)/64 + 1
x = groupnorm(x)
x = gelu(conv2(x))                       # [B, 2*hidden, N2]   N2 ≈ (N1-7)/3 + 1
x = gelu(conv3(x))                       # [B, hidden,   N3]   N3 ≈ (N2-3)/2 + 1
x = x.permute(0, 2, 1)                   # [B, N3, hidden]   → transformer input
```

**Per-layer details:**

| Layer | in_ch | out_ch | kernel | stride | bias | activation     | norm before |
|-------|-------|--------|--------|--------|------|----------------|-------------|
| conv1 | 1     | H      | 127    | 64     | no   | tanh           | —           |
| conv2 | H     | 2H     | 7      | 3      | yes  | GELU           | GroupNorm(1, H) on conv1 output |
| conv3 | 2H    | H      | 3      | 2      | yes  | GELU           | —           |

H = `hidden_size` (288 tiny, 416 base).

**Downsampling factor: 64 × 3 × 2 = 384x.**
At 16 kHz this yields **≈41.67 encoder frames per second** (1 frame = 24 ms of audio).
Examples:
- 1 second of audio (16,000 samples) → ~41 encoder tokens
- 10 seconds → ~416 encoder tokens
- 30 seconds → ~1250 encoder tokens (well beyond the `max_position_embeddings=194` decoder cap, but encoder is **not** capped — RoPE doesn't need a pre-set table; see §5).

**Padding semantics:** PyTorch `nn.Conv1d` defaults to `padding=0` ("valid"). Implementations must use valid (no-pad) convolution to match. Output length for valid conv1d is `floor((L_in - kernel) / stride) + 1`.

**Input format:**
- 16 kHz mono PCM, float32 in roughly `[-1, +1]`.
- `preprocessor_config.json` declares `feature_extractor_type: Wav2Vec2FeatureExtractor`, `do_normalize: false`, `padding_value: 0.0`, `sampling_rate: 16000`. We are **not** running the Wav2Vec2 mean/variance norm path — `do_normalize=false` means the raw float waveform is passed through unchanged.
- Minimum input length: long enough that all three convs produce ≥1 output sample. Conv1 needs ≥127 samples (8 ms). After all three: minimum useful audio is roughly 127 + 64·(7 + 3·(3-1)) ≈ a few hundred samples, but in practice the model is trained on speech ≥0.5 s. Keep an asserted minimum of e.g. 0.1 s to avoid degenerate shapes.

---

## 4. Encoder

Standard pre-LN Transformer encoder operating on Conv1D outputs.

| Param                          | tiny  | base  |
|--------------------------------|-------|-------|
| `hidden_size`                  | 288   | 416   |
| `encoder_num_hidden_layers`    | 6     | 8     |
| `encoder_num_attention_heads`  | 8     | 8     |
| `encoder_num_key_value_heads`  | 8     | 8     | (MHA; no GQA in released checkpoints)
| `head_dim` (= hidden/heads)    | 36    | 52    |
| `intermediate_size` (FFN)      | 1152  | 1664  |
| `encoder_hidden_act`           | gelu  | gelu  |

> `head_dim=36` (tiny) and `52` (base) are not multiples of 8. `config.json` sets `pad_head_dim_to_multiple_of: 8` so the projections are padded internally for fast attention kernels. For a pure-C# implementation that runs its own MHA kernel, we can ignore the pad and use the native head_dim — but make sure RoPE is applied to the **unpadded** head dim (or to the rotary subset of it).

**Per-layer structure (encoder, pre-LN):**

```
y = x + SelfAttn(LN(x), rope=rope_emb)        # MHA, no bias (attention_bias=False)
y = y + MLP(LN(y))                            # MLP = Linear → GELU → Linear
```

Encoder self-attention is **bidirectional** (no causal mask). Padding mask is only needed if batching variable-length audio in one forward (we will run batch=1 on inference; effectively no mask needed).

**Encoder MLP (linear, NOT gated):**

```python
self.fc1 = nn.Linear(hidden_size, intermediate_size)   # 288 → 1152 (tiny)
self.fc2 = nn.Linear(intermediate_size, hidden_size)
# forward: fc2(gelu(fc1(x)))
```

A final `self.layer_norm = nn.LayerNorm(hidden_size)` is applied after the last encoder layer.

---

## 5. Decoder

Causal Transformer decoder with cross-attention to encoder.

| Param                          | tiny  | base  |
|--------------------------------|-------|-------|
| `hidden_size`                  | 288   | 416   |
| `decoder_num_hidden_layers`    | 6     | 8     |
| `decoder_num_attention_heads`  | 8     | 8     |
| `decoder_num_key_value_heads`  | 8     | 8     |
| `intermediate_size`            | 1152  | 1664  |
| `decoder_hidden_act`           | silu  | silu  |
| `tie_word_embeddings`          | true  | true  |

**Per-layer structure (decoder, pre-LN):**

```
y = x + SelfAttn(LN(x), causal=True, rope=rope_emb, kv_cache=...)
y = y + CrossAttn(LN(y), key_value_states=encoder_hidden_states,
                  cross_kv_cache=... )          # NO RoPE on cross-attn
y = y + MLP(LN(y))                              # gated SiLU
```

A final `LayerNorm` follows the last decoder layer; logits = `lm_head(y)`, with `lm_head.weight` tied to `embed_tokens.weight`.

**Decoder MLP (gated SiLU / SwiGLU-style):**

```python
self.fc1 = nn.Linear(hidden_size, intermediate_size * 2)   # 288 → 2304 (tiny)
self.fc2 = nn.Linear(intermediate_size, hidden_size)
# forward:
#   h, gate = fc1(x).chunk(2, dim=-1)
#   y = fc2(silu(gate) * h)
```

Note this is a single fused `fc1` with `2*intermediate_size` output (not two separate `gate_proj` / `up_proj` linears like Llama). When loading safetensors the single `mlp.fc1.weight` tensor must be split into the two halves at forward time, or pre-split at load time.

**Cross-attention:** queries come from the decoder hidden state, keys/values come from the encoder output. No RoPE is applied to cross-attention K/Q (no positional rotation across modalities). The encoder K/V tensors are precomputed once per utterance and cached for the entire decode loop.

---

## 6. Variable-length input (the killer feature)

Whisper hard-pads every input to 30 s of mel frames (3000 frames). Moonshine does not pad at all:

1. The conv front-end is a pure local operation — output length is determined by input length.
2. The encoder uses **RoPE**, which generates rotary embeddings on the fly per `(seq_len, head_dim)`. There is no learned positional embedding table to set a maximum length.
3. The encoder has **no length-dependent normalisation or pooling** — every operation is per-token MLP or attention.
4. `max_position_embeddings=194` in `config.json` is the **decoder** cap (max output tokens per call). The encoder is **uncapped** in principle, though attention is O(N²) so very long audio still gets expensive.

Practical implications for our C# pipeline:
- Allocate the encoder-side input tensor sized to `ceil((N-127)/64+1)`-pipelined-through-the-convs each call. No fixed 30-s buffer.
- The decoder KV-cache for cross-attention is sized to whatever the encoder produced this call.
- Decoder `max_length` should be derived from audio duration to avoid hallucination loops (see §9 below): the README recommends `max_length = audio_seconds * 6.5` (a hard cap on English word rate × tokens-per-word).

---

## 7. RoPE positional encoding

Standard RoPE, partial rotary, applied to query and key in self-attention only (encoder self-attn + decoder self-attn). **Cross-attention is RoPE-free.**

**Config:**
- `rope_theta = 10000.0`
- `partial_rotary_factor` = **0.9** (tiny), **0.62** (base)
- `rope_scaling = null` (no NTK / linear scaling)
- Rotary dim = `floor(head_dim * partial_rotary_factor)`, rounded to an even number
  - tiny: head_dim=36, 0.9 → rotary_dim=32 (last 4 channels of each head untouched)
  - base: head_dim=52, 0.62 → rotary_dim=32 (last 20 channels untouched)
- `max_position_embeddings = 194` (decoder cache cap; encoder positions just keep counting from 0..N3-1)

**Inverse-frequency table (per head, reused across layers):**

```
dim = rotary_dim
inv_freq[i] = 1 / (10000 ** (2*i / dim))    for i in 0 .. dim/2 - 1
```

**Per-position cos / sin:**

```
freqs[pos, i]   = pos * inv_freq[i]
cos[pos, :]     = repeat_interleave(cos(freqs[pos, :]), 2)    # length = rotary_dim
sin[pos, :]     = repeat_interleave(sin(freqs[pos, :]), 2)    # length = rotary_dim
```

> HF Moonshine uses the **interleaved** layout: `cos`/`sin` are `repeat_interleave(2, dim=-1)` (not the GPT-NeoX-style `[cos, cos, sin, sin]` concat). The corresponding rotation pairs adjacent dims `(2i, 2i+1)` rather than `(i, i+rotary_dim/2)`. Our existing RoPE implementations probably use the concat layout — **double-check the layout when wiring this up**; if mismatched, swap to interleaved by adjusting how cos/sin are tiled or how Q/K are reshaped before the rotation. This is the single most common bug when porting RoPE.

**Apply rotation** (interleaved variant):

```
# x: [B, heads, seq, head_dim]; split last dim into (rotary_dim, head_dim - rotary_dim)
x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

# rotate halves on interleaved pairs:
x1 = x_rot[..., 0::2]       # even indices
x2 = x_rot[..., 1::2]       # odd indices
# new pairs after rotation:
y1 = x1 * cos_pairs - x2 * sin_pairs
y2 = x1 * sin_pairs + x2 * cos_pairs
# re-interleave back into x_rot's shape, then concat x_pass

return concat([x_rot_rotated, x_pass], dim=-1)
```

**Positions:** encoder uses positions `0..N3-1` where N3 = encoder seq length. Decoder uses positions `0..L-1` over generated tokens (and during cached decoding, position = current_length - 1 for the new token).

---

## 8. Tokenizer

- Base: **Llama 1 / 2 byte-level BPE** (same merges + base vocab).
- Base vocab: **32 000** tokens. Plus **768** reserved special tokens → **`vocab_size = 32768`**.
- Encoded as a HuggingFace `tokenizer.json` (~1.99 MB) — same JSON schema HartsyInference.Tokenizers already supports.
- Special tokens:
  - `bos_token_id = 1`
  - `eos_token_id = 2`
  - `pad_token_id = 2` (same as EOS — only used for label masking, not at inference)
  - `decoder_start_token_id = 1` (= BOS)
  - Other 766 reserved IDs are unused at inference (kept for future expansion / fine-tuning).

**Decode loop input/output:**
- Start the decoder with `[BOS]` (id 1).
- Stop on `EOS` (id 2) or when `max_length` reached.
- Detokenise with the standard byte-level BPE decoder (same code path as Llama in HartsyInference.LLM).

If we want to share code with the HartsyInference Llama tokenizer: yes, this works directly — Moonshine ships the same `tokenizer.json` format and the same base merges. Only difference is the upper 768 IDs.

---

## 9. Greedy decoding

The released checkpoints assume greedy / argmax decoding (with optional beam search to mitigate hallucination loops, but the C# reference path can ship greedy first).

**Decode loop:**

```
encoder_out = encode(waveform)                    # [N3, hidden] ; cache cross-attn K/V per layer
tokens = [BOS]
kv_self = empty per-layer self-attn cache
max_len = min(194, ceil(audio_seconds * 6.5))     # hallucination guard
for step in range(max_len):
    logits = decoder_step(tokens[-1], encoder_out, kv_self, kv_cross_cached)
    next_id = argmax(logits[-1])
    if next_id == EOS: break
    tokens.append(next_id)
return tokenize.decode(tokens[1:])
```

**KV caches:**
- **Self-attn cache** grows by 1 entry per step (K and V tensors `[heads, t, head_dim]`).
- **Cross-attn cache** is computed once from `encoder_out` and reused for every step.
- For RoPE: cache the **post-rotation** K (apply rotation when K is written into the cache). This means each new token's K gets rotated once at position `t` and never again.

Hallucination mitigation: clip max_length using the 6.5 tokens/sec heuristic from the model card (English ~3.5 wpm × ~2 tokens/word). For non-streaming use, simple temperature=0 greedy + EOS detection is fine; beam search is the next improvement.

---

## 10. Streaming inference

> Cross-reference: future `STREAMING_AUDIO_INFERENCE.md` (not yet authored in this repo). When that doc exists, link the chunking / VAD / overlap policies here.

**v1 models (`moonshine-tiny`, `moonshine-base`) are not natively streaming** — they are full-utterance models. To "stream" them in production, the typical pattern (and the one Moonshine's own Python wrapper uses) is:

1. **VAD-gated re-transcription:** A small VAD (Silero, or RMS-threshold) detects speech segments. When a segment ends (or every N seconds), feed the accumulated audio to the model and re-run encoder + greedy decode. Display the new transcript.
2. **Sliding window with overlap:** Run on `[t-W, t]` every `S` seconds (W = window, e.g. 8 s; S = stride, e.g. 1 s). Diff-merge transcripts to commit stable prefix tokens. Because the encoder is variable-length, partial windows work fine.
3. **Re-use encoder K/V cache across re-runs:** The conv front-end is local, so for an audio buffer that grew from N to N+ΔN samples, we can in principle re-run conv1/2/3 only on the new chunk plus a small left-context (kernel sizes give ≈ 254 samples = 15 ms of left context needed at the waveform). For v1 this is a manual optimisation; for v2 ("streaming-tiny" etc.) the model itself bakes this in.

**v2 streaming models** (separate checkpoints) replace the conv front-end with `80-sample windows + per-frame CMVN + asinh + 2x stride-2 conv` (≈ 50 Hz frame rate) and use a **position-free ergodic encoder with sliding-window attention** so the encoder can be applied incrementally without recomputing. They also cache partial decoder state. Reported latency: Pi 5 = 237 ms (tiny), 527 ms (small), 802 ms (medium). **v2 is a separate implementation task — do not couple to v1.**

**Recommended C# streaming policy for v1:**
- Run a small VAD (we already need this for general STT).
- Buffer raw audio. On each VAD endpoint (or every 1 s during continuous speech), encode the last K seconds (K ≤ 30 s in practice) and decode.
- Render new tokens as they appear. On final endpoint, run one last full-buffer pass for the committed transcript.

---

## 11. Memory and performance

### File sizes (FP32 safetensors)
- `moonshine-tiny`: **108 MB** (`model.safetensors`) + 1.99 MB tokenizer + a few KB configs.
- `moonshine-base`: **246 MB** + 1.99 MB tokenizer.

### Runtime memory (rough — FP16)
- tiny: ~55 MB weights + ~10 MB activations + a few MB KV cache = **~70 MB total** in FP16, fits easily on a Pi.
- base: ~125 MB weights + ~20 MB activations + a few MB KV cache = **~150 MB total** in FP16.

### Reported latency / RTF (from model card + v2 paper)
- `moonshine-tiny` mean WER 12.65 on Open ASR Leaderboard; **RTFx = 753** (i.e. 753× real-time on the leaderboard reference GPU).
- `moonshine-base` mean WER 9.99; RTFx = 566.
- v2 streaming latency on Raspberry Pi 5: tiny 237 ms, small 527 ms, medium 802 ms (end-to-end per chunk). Whisper-large-v3 will not run on Pi 5 at all.
- Paper headline: ~5× compute reduction vs `whisper-tiny-en` on a 10 s clip with no WER regression.

(Exact RTF on RTX 3060 / Pixel 8 / Coral isn't published in the v1 paper; v1 was published before the streaming-focused v2 benchmarks.)

---

## 12. Comparison to Whisper

(Open ASR Leaderboard, lower WER is better.)

| Model                | params | LS clean | LS other | TEDLium | Mean WER |
|----------------------|-------:|---------:|---------:|--------:|---------:|
| whisper-tiny-en      | 39M    | 5.66     | 15.45    | 5.97    | 12.81    |
| **moonshine-tiny**   | **27M**| 4.55     | 11.68    | (n/a)   | **12.65**|
| whisper-base-en      | 74M    | 4.25     | 10.35    | 4.87    | 10.32    |
| **moonshine-base**   | **61M**| 3.38     | 8.15     | (n/a)   | **9.99** |

Headlines:
- Moonshine matches or beats Whisper at smaller param counts.
- ~5× less compute per 10 s clip vs Whisper-tiny (because of no 30 s pad).
- Moonshine's variable-length encoder is the single biggest source of the speedup on real-world short utterances (commands, dictation).
- Moonshine is **English-only** in v1. (v2 added 7 more languages: Arabic, Japanese, Korean, Mandarin, Spanish, Ukrainian, Vietnamese.)

---

## 13. C# implementation notes

### What's new vs what's reuse
| Component                         | Status                                                                    |
|-----------------------------------|---------------------------------------------------------------------------|
| Conv1D front-end (3 layers + GN)  | **New, trivial.** Three valid-padding Conv1Ds + tanh + GroupNorm(1) + GELU. No mel, no FFT. |
| Encoder transformer (pre-LN)      | Reuse the Whisper encoder pattern; swap absolute-pos for RoPE.            |
| Decoder transformer (pre-LN)      | Reuse the Whisper decoder pattern; swap absolute-pos for RoPE; FFN is gated SiLU (different from encoder). |
| RoPE                              | Reuse Flux / Hunyuan / Z-Image RoPE. **Watch the interleaved vs concat layout** (see §7). |
| Llama BPE tokenizer               | Reuse the HartsyInference Llama tokenizer (32k merges + 768 reserved); same `tokenizer.json` format. |
| MHA self-attn / cross-attn        | Reuse Whisper attention kernels. KV cache layout same as Whisper.         |
| GroupNorm(num_groups=1)           | Equivalent to LayerNorm over the channel dim of a `[B, C, T]` tensor. We may already have this from Conv2D path. |
| Greedy decode loop                | Reuse Whisper greedy + EOS logic; swap in the `max_length = 6.5 × seconds` guard. |
| Feature extractor                 | **None.** Just resample mono to 16 kHz f32, hand the array to the conv front-end. |

### Layer-by-layer C# work breakdown
1. **Audio loader:** decode WAV/FLAC/OGG → mono float32 at 16 kHz (existing audio I/O).
2. **Conv1D op:** valid-padding (no pad), kernel up to 127. The conv1 case is `in_channels=1, out_channels=288, kernel=127, stride=64` — small kernel, small output channels, easy CUDA/CPU kernels. Conv2 and conv3 are even smaller.
3. **GroupNorm(1):** identical to LayerNorm over channels.
4. **RoPE precompute:** one `(max_seq_len, rotary_dim)` cos/sin table per model load. Encoder grows dynamically; expand on demand.
5. **MHA + cross-MHA:** existing.
6. **Gated SiLU FFN for decoder:** new compared to Whisper (Whisper uses plain GELU). Single `fc1` → chunk into (h, gate) → `silu(gate) * h` → `fc2`.
7. **KV cache:** standard. Store post-rotation K for self-attn; cross-attn K/V cached once after encode.
8. **Greedy decode loop:** standard.

### Numerical tolerances
Per project rule, validate against the HuggingFace transformers reference:
- Match logits to within `1e-3` (FP32) / `5e-2` (FP16) on a fixed test utterance.
- Match top-1 token IDs at every decode step on at least 5 LibriSpeech-clean clips.
- WER on `librispeech_asr_dummy/validation` clean split should be within 0.2 absolute WER of the HF reference.

---

## 14. HuggingFace safetensors layout

The HF transformers port uses the following module names (and therefore safetensors keys). Confirmed by inspecting `transformers/models/moonshine/modeling_moonshine.py`.

### Top-level
| Key prefix                     | Notes |
|--------------------------------|-------|
| `model.encoder.…`              | encoder + conv front-end |
| `model.decoder.…`              | decoder |
| `proj_out.weight`              | lm head; **tied** to `model.decoder.embed_tokens.weight` (may or may not be present in the safetensors file — if missing, alias to the embedding) |

### Encoder front-end (conv + groupnorm)
```
model.encoder.conv1.weight                      [hidden, 1, 127]              (bias=False, no key)
model.encoder.conv2.weight                      [2*hidden, hidden, 7]
model.encoder.conv2.bias                        [2*hidden]
model.encoder.conv3.weight                      [hidden, 2*hidden, 3]
model.encoder.conv3.bias                        [hidden]
model.encoder.groupnorm.weight                  [hidden]
model.encoder.groupnorm.bias                    [hidden]
```

### Encoder layers (i = 0 .. encoder_num_hidden_layers - 1)
```
model.encoder.layers.{i}.self_attn.q_proj.weight        [hidden, hidden]      (no bias)
model.encoder.layers.{i}.self_attn.k_proj.weight        [hidden, hidden]
model.encoder.layers.{i}.self_attn.v_proj.weight        [hidden, hidden]
model.encoder.layers.{i}.self_attn.o_proj.weight        [hidden, hidden]
model.encoder.layers.{i}.input_layernorm.weight         [hidden]
model.encoder.layers.{i}.input_layernorm.bias           [hidden]
model.encoder.layers.{i}.post_attention_layernorm.weight [hidden]
model.encoder.layers.{i}.post_attention_layernorm.bias   [hidden]
model.encoder.layers.{i}.mlp.fc1.weight                 [intermediate, hidden]
model.encoder.layers.{i}.mlp.fc1.bias                   [intermediate]
model.encoder.layers.{i}.mlp.fc2.weight                 [hidden, intermediate]
model.encoder.layers.{i}.mlp.fc2.bias                   [hidden]
```

### Encoder final norm
```
model.encoder.layer_norm.weight                 [hidden]
model.encoder.layer_norm.bias                   [hidden]
```

### Decoder embeddings + layers (j = 0 .. decoder_num_hidden_layers - 1)
```
model.decoder.embed_tokens.weight                       [vocab_size=32768, hidden]

model.decoder.layers.{j}.self_attn.q_proj.weight        [hidden, hidden]      (no bias; attention_bias=False)
model.decoder.layers.{j}.self_attn.k_proj.weight        [hidden, hidden]
model.decoder.layers.{j}.self_attn.v_proj.weight        [hidden, hidden]
model.decoder.layers.{j}.self_attn.o_proj.weight        [hidden, hidden]
model.decoder.layers.{j}.input_layernorm.weight         [hidden]
model.decoder.layers.{j}.input_layernorm.bias           [hidden]

model.decoder.layers.{j}.encoder_attn.q_proj.weight     [hidden, hidden]
model.decoder.layers.{j}.encoder_attn.k_proj.weight     [hidden, hidden]
model.decoder.layers.{j}.encoder_attn.v_proj.weight     [hidden, hidden]
model.decoder.layers.{j}.encoder_attn.o_proj.weight     [hidden, hidden]
model.decoder.layers.{j}.post_attention_layernorm.weight [hidden]
model.decoder.layers.{j}.post_attention_layernorm.bias   [hidden]

model.decoder.layers.{j}.mlp.fc1.weight                 [2*intermediate, hidden]  ← gated SiLU, fused gate+up
model.decoder.layers.{j}.mlp.fc1.bias                   [2*intermediate]
model.decoder.layers.{j}.mlp.fc2.weight                 [hidden, intermediate]
model.decoder.layers.{j}.mlp.fc2.bias                   [hidden]
model.decoder.layers.{j}.final_layernorm.weight         [hidden]
model.decoder.layers.{j}.final_layernorm.bias           [hidden]
```

### Decoder final norm + LM head
```
model.decoder.layer_norm.weight                 [hidden]
model.decoder.layer_norm.bias                   [hidden]
proj_out.weight                                 [vocab_size, hidden]    ← tied to embed_tokens.weight
```

### Conv1D tensor layout reminder
PyTorch `Conv1d.weight` is stored as `[out_channels, in_channels, kernel_size]`. Our SafeTensors loader keeps the on-disk layout; the conv kernel can read this layout directly without transpose.

### Loader hints
- If `proj_out.weight` is absent from `model.safetensors` (it may be tied and stripped), alias `proj_out.weight = model.decoder.embed_tokens.weight`.
- All Linear weights are stored as `[out_features, in_features]` (PyTorch convention) — apply as `y = x @ W.T + b`.
- `attention_bias=False` means no `q/k/v/o_proj.bias` keys exist. Encoder conv1 also has `bias=False`. All other Linears and convs do have biases.
- `pad_head_dim_to_multiple_of=8` in config is a runtime padding hint for kernel-friendly head_dim; weights are stored at the natural `head_dim` (36 / 52) and do **not** need to be re-padded on load.

---

## Sources

- [arXiv:2410.15608 — Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)
- [arXiv:2602.12241 — Moonshine v2: Ergodic Streaming Encoder ASR](https://arxiv.org/abs/2602.12241)
- [arXiv:2509.02523 — Flavors of Moonshine: Tiny Specialized ASR Models](https://arxiv.org/abs/2509.02523)
- [HF model — UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)
- [HF model — UsefulSensors/moonshine-base](https://huggingface.co/UsefulSensors/moonshine-base)
- [HF docs — Moonshine model_doc](https://huggingface.co/docs/transformers/model_doc/moonshine)
- [HF transformers — configuration_moonshine.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/moonshine/configuration_moonshine.py)
- [HF transformers — modeling_moonshine.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/moonshine/modeling_moonshine.py)
- [GitHub — moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)
- [Pete Warden's blog — Introducing Moonshine](https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/)
