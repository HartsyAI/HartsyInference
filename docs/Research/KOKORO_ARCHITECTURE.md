# Kokoro TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Kokoro pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Kokoro is an open-weight text-to-speech model with 82M parameters, Apache 2.0 licensed, based on the **StyleTTS 2** architecture ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) with an **iSTFTNet** vocoder ([arXiv:2203.02395](https://arxiv.org/abs/2203.02395)). The model is decoder-only (no diffusion, no encoder release) and produces 24 kHz audio. The full pipeline is: text → G2P (misaki + espeak-ng fallback) → phoneme token IDs → PLBERT contextual encoding → prosody prediction (duration, F0, energy) → text encoding → length regulation → iSTFTNet decoder → raw waveform. Voice identity is controlled by a 256-dim style vector split into two 128-dim halves: one for the decoder, one for the prosody predictor. Trained on ~hundreds of hours of permissive audio with IPA phoneme labels, ~1000 A100-80GB hours total.

This file covers the model architecture. The vocoder (iSTFTNet) is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The G2P phonemization step is in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md).

Sources: [hexgrad/kokoro](https://github.com/hexgrad/kokoro), [hexgrad/Kokoro-82M HF](https://huggingface.co/hexgrad/Kokoro-82M), [hexgrad/misaki](https://github.com/hexgrad/misaki), [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2).

## Key Numbers / Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Total parameters | 82M | ~327 MB in FP32 |
| Sample rate | 24,000 Hz | Fixed, not configurable |
| Mel channels (n_mels) | 80 | Standard mel spectrogram bins |
| Hidden dimension | 512 | Main model hidden size |
| Style dimension | 128 | Per-half (256 total in voice pack) |
| PLBERT hidden size | 768 | ALBERT encoder output dim |
| PLBERT layers | 12 | Transformer layers in ALBERT (with weight sharing!) |
| PLBERT attention heads | 12 | Multi-head attention |
| PLBERT intermediate size | 2048 | Feed-forward inner dim |
| Max position embeddings | 512 | Context window (token limit) |
| Vocabulary size (n_token) | 178 | Phoneme + punctuation tokens |
| Decoder layers (n_layer) | 3 | Used in TextEncoder, ProsodyPredictor |
| Text encoder kernel size | 5 | Conv1d kernel for TextEncoder |
| Max duration (max_dur) | 50 | Duration prediction bins |
| Dropout | 0.2 | Model-wide dropout rate |
| PLBERT dropout | 0.1 | Separate BERT dropout |
| Effective hop size (mel→audio) | 300 | 10 * 6 * 5 = upsampling stages × iSTFT hop |
| Voice embedding shape | (511, 1, 256) | Per voice .pt file |

## Data Layouts / Formats

### Phoneme Token Encoding
```
Input:  "həlˈO wˈɜɹld"  (IPA phoneme string from misaki)
Mapped: [50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46]
Padded: [0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 0]
Shape:  (1, 14)  as LongTensor
```

Token 0 is the padding/BOS/EOS token. Characters not in the vocab are silently dropped.

### Voice Pack (.pt file)
```
Shape: (511, 1, 256)  float32
Index: voicepack[len(tokens)] -> (1, 256)
Split: ref_s[:, :128] for decoder, ref_s[:, 128:] for predictor
File size: ~500 KB per voice
```

### ONNX Voice Pack (.bin file)
```
Shape: (num_voices * 512, 1, 256)  float32
Access: voices[voice_index * 512 + len(tokens)] -> (1, 256)
```

### Audio Output
```
Shape: (num_samples,) float32, range approximately [-1, 1]
Sample rate: 24,000 Hz
Format: raw PCM, typically written to WAV via soundfile
```

### Model Weight File (.pth)
```
Top-level dict with 4 keys:
  "bert"         -> CustomAlbert state_dict
  "text_encoder" -> TextEncoder state_dict
  "predictor"    -> ProsodyPredictor state_dict
  "decoder"      -> Decoder (iSTFTNet) state_dict
Total size: 327 MB (FP32)
```

## Implementation Notes for HartsyInference

1. **ALBERT weight sharing**: PLBERT is ALBERT with `n_layer=12` but only ONE set of weights shared across all layers. This affects how we instantiate the encoder: load the weights once, loop 12 times during forward. Standard transformer code that creates 12 distinct `LayerBlock` instances would 12x the memory and silently miscompute.

2. **G2P is the hardest part**: see [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). For Kokoro specifically: the misaki library is Python-only with spaCy + espeak-ng deps. Pure-C# options:
   - Port the misaki English dictionary (a flat text file) and the homograph disambiguation rules (~1000 entries).
   - For non-English languages, ship a per-language phoneme dictionary derived from espeak-ng's lexicon (one-time offline conversion).
   - For OOV English words, fall back to a learned G2P model — or ship a Hash → IPA dictionary covering top 100k words.

3. **Voice pack loading**: The .pt files are PyTorch pickle format. Convert offline to a simple flat float32 binary (or safetensors) at our model packaging step. Don't parse pickle at load time.

4. **Key tensor operations needed**:
   - Matrix multiplication (for alignment expansion — this is just a sparse-to-dense expand, can be done with a Gather op).
   - 1D convolution and transposed convolution.
   - Bidirectional LSTM — see implementation notes below.
   - ALBERT transformer (with parameter sharing).
   - Instance normalization (`AdaIN1d`).
   - Inverse STFT (FFT-based) — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).
   - Snake activation: `x + (1/alpha) * sin^2(alpha * x)`.
   - SineGen: cumulative sum of phase increments, sine wave generation.

5. **BiLSTM implementation**: PyTorch's BiLSTM is fused. For HartsyInference we need:
   - Forward LSTM over input
   - Backward LSTM over reversed input
   - Concat along feature dim
   LSTM cell: standard `i, f, g, o = chunk(W*[x; h] + b, 4)`, `c = f*c_prev + i*tanh(g)`, `h = o*tanh(c)`. Plan to add to `HartsyInference.Core/Modules` as `LstmCell` / `BiLstm`.

6. **Alignment expansion**: The alignment matrix `pred_aln_trg` is `(1, seq_len, total_frames)` with each row a one-hot indicator. The matmul `en = d.transpose @ pred_aln_trg` is equivalent to a `Repeat` op (each `d[:, t]` repeated `duration[t]` times). Implement directly as a Repeat — faster, less memory.

7. **Duration rounding determinism**: `round(sum_of_sigmoid / speed)` must match Python's `round()` — Python uses banker's rounding (round-half-to-even). C#'s `Math.Round` defaults to ToEven so this is fine, but `(int)Math.Round(x)` for negatives behaves differently. Force `MidpointRounding.ToEven`.

8. **Memory considerations**: The full FP32 model is 327 MB. The INT8 ONNX variant is 92.4 MB. Voice packs are ~500 KB each. Plan to ship FP16 by default and downcast at load time.

9. **No streaming in v1**: Implement synchronous "generate whole utterance" first. Streaming requires partial-alignment decoding which the reference does not support cleanly. Defer to v2.

10. **iSTFTNet decoder validation**: validate the decoder in isolation by injecting hand-crafted (alignment, F0, energy, style) tuples and comparing waveform output to the reference Python implementation within 1e-3 PCM tolerance.

## Reference Implementations

- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) — Official Python/PyTorch reference, pip installable.
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — Official model card and weights (.pth).
- [hexgrad/misaki](https://github.com/hexgrad/misaki) — G2P library for phoneme conversion.
- [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) — ONNX conversion with multiple quantization levels.
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — Standalone ONNX Runtime wrapper.
- [kokoro.js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) — Official JS implementation.
- [Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) — Apple MLX port.
- [mlalma/MisakiSwift](https://github.com/mlalma/MisakiSwift) — Swift port of misaki G2P.
- [misaki-rs](https://lib.rs/crates/misaki-rs) — Rust port of misaki G2P.
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — Base architecture paper implementation.
