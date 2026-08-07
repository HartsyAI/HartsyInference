# Moonshine — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Moonshine pipeline)

> **Stub.** The narrative walkthrough and restated pseudocode were removed on 2026-08-06 — this model
> is built and verified, so the C# is the source of truth for *how it works*. What remains is what the
> code cannot tell you: upstream provenance, reference constants, and bring-up traps. History is in git.

Moonshine (Useful Sensors, 2024) is a tiny encoder-decoder ASR family explicitly designed for edge devices and live transcription. Unlike Whisper, it operates **directly on the raw 16 kHz waveform** (no mel spectrogram), uses **RoPE** instead of learned absolute positional embeddings, and processes **variable-length** audio without zero-padding to 30 s.

Pure C# implementation is straightforward: a 3-layer Conv1D front-end, then a standard pre-existing pre-LN Transformer encoder/decoder pattern (we already have this for Whisper and the HartsyInference.LLM decoder), reusing the RoPE we built for Flux / Hunyuan / Z-Image and the BPE tokenizer infrastructure from HartsyInference.ModelAssets.Tokenizers.

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

## 8. Tokenizer

- Base: **Llama 1 / 2 byte-level BPE** (same merges + base vocab).
- Base vocab: **32 000** tokens. Plus **768** reserved special tokens → **`vocab_size = 32768`**.
- Encoded as a HuggingFace `tokenizer.json` (~1.99 MB) — same JSON schema HartsyInference.ModelAssets.Tokenizers already supports.
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
