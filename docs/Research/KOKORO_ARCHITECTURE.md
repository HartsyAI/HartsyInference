# Kokoro TTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Kokoro pipeline)

## Summary

Kokoro is an open-weight text-to-speech model with 82M parameters, Apache 2.0 licensed, based on the **StyleTTS 2** architecture ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) with an **iSTFTNet** vocoder ([arXiv:2203.02395](https://arxiv.org/abs/2203.02395)). The model is decoder-only (no diffusion, no encoder release) and produces 24 kHz audio. The full pipeline is: text → G2P (misaki + espeak-ng fallback) → phoneme token IDs → PLBERT contextual encoding → prosody prediction (duration, F0, energy) → text encoding → length regulation → iSTFTNet decoder → raw waveform. Voice identity is controlled by a 256-dim style vector split into two 128-dim halves: one for the decoder, one for the prosody predictor. Trained on ~hundreds of hours of permissive audio with IPA phoneme labels, ~1000 A100-80GB hours total.

This file covers the model architecture. The vocoder (iSTFTNet) is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The G2P phonemization step is in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md).

Sources: [hexgrad/kokoro](https://github.com/hexgrad/kokoro), [hexgrad/Kokoro-82M HF](https://huggingface.co/hexgrad/Kokoro-82M), [hexgrad/misaki](https://github.com/hexgrad/misaki), [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2).

## Detailed Findings

### Overall Architecture

Kokoro follows the StyleTTS 2 design with five primary neural network components:

1. **PLBERT (Phoneme-Level BERT)** — A `CustomAlbert` (modified ALBERT from HuggingFace Transformers) that produces contextual phoneme embeddings. Configured with hidden_size=768, 12 attention heads, 12 transformer layers, intermediate_size=2048, max_position_embeddings=512. ALBERT shares parameters across all 12 layers (so 12 layers, 1 set of weights — that's why the model is small despite the 12-layer depth).

2. **BERT Encoder** — A single `nn.Linear(768, 512)` projection layer that maps PLBERT outputs from 768 dimensions down to the model's hidden_dim of 512.

3. **Text Encoder** — Embedding layer (178 tokens -> 512 dim) followed by multiple 1D CNN layers (kernel_size=5) with LayerNorm, LeakyReLU, and 0.2 dropout, then a bidirectional LSTM (512 -> 256 per direction = 512 output). Depth is 3 layers (matching `n_layer`).

4. **Prosody Predictor** — Predicts duration, F0 (pitch), and energy (N). Contains:
   - **DurationEncoder**: 3 layers of bidirectional LSTM with AdaLayerNorm, processing text embeddings concatenated with the style vector.
   - **Shared LSTM**: Input dimension `(hidden_dim + style_dim)` = 640, hidden `hidden_dim//2` = 256 per direction.
   - **Duration projection**: Linear(512, 50) where 50 = max_dur. Duration is predicted as sigmoid probabilities summed across the 50 bins.
   - **F0 pathway**: 3 `AdainResBlk1d` modules with upsampling.
   - **Energy (N) pathway**: 3 `AdainResBlk1d` modules.

5. **Decoder (iSTFTNet)** — Converts aligned text features + F0 + energy + style into a waveform. Uses Adaptive Instance Normalization (AdaIN) throughout for style conditioning, Snake activation functions, a harmonic+noise source module (SourceModuleHnNSF), multi-stage upsampling with transposed convolutions, and a final inverse STFT to produce the waveform. **Full design in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)** under the "Kokoro/StyleTTS2 iSTFTNet" section.

### Inference Pipeline (Forward Pass)

The `KModel.forward` method in [`kokoro/model.py`](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py) implements the following steps:

1. **Phoneme string -> token IDs**: Each character in the phoneme string is mapped to an integer via the `vocab` dict from config.json. Unknown characters are dropped. A padding token `0` is prepended and appended: `[0, *token_ids, 0]`.

2. **PLBERT encoding**: `bert_dur = self.bert(input_ids, attention_mask)` produces shape `(1, seq_len, 768)`.

3. **Linear projection**: `d_en = self.bert_encoder(bert_dur).transpose(-1, -2)` produces shape `(1, 512, seq_len)`.

4. **Style split**: `s = ref_s[:, 128:]` extracts the prosody-predictor half of the style vector.

5. **Duration prediction**: The DurationEncoder processes the projected BERT output with the style vector, then an LSTM + Linear produces duration logits of shape `(1, seq_len, 50)`. Sigmoid + sum + division by speed gives per-phoneme durations. Rounded to integers with min=1.

6. **Alignment matrix**: An alignment matrix `pred_aln_trg` of shape `(1, seq_len, total_frames)` is constructed by repeating each phoneme index by its predicted duration.

7. **Length regulation**: The encoded features are expanded: `en = d.transpose(-1, -2) @ pred_aln_trg` producing frame-level features.

8. **F0 and energy prediction**: `F0_pred, N_pred = self.predictor.F0Ntrain(en, s)` produces frame-level pitch and energy contours.

9. **Text encoding**: `t_en = self.text_encoder(input_ids, input_lengths, text_mask)` produces another phoneme-level encoding, then expanded: `asr = t_en @ pred_aln_trg`.

10. **Decoding**: `audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])` where only the first 128 dims of the style vector are used for the decoder. The decoder produces a 24 kHz waveform.

### Phoneme Set and Encoding

Kokoro uses a vocabulary of **178 IPA-based tokens** (indices 0-177, where 0 is the padding token). The complete mapping is defined in `config.json` under the `vocab` key.

**English phoneme set (from [EN_PHONES.md](https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md))**:
- 49 total English phonemes: 41 shared (US+UK), 4 US-only, 4 UK-only
- 2 stress marks: `ˈ` (primary, token 156), `ˌ` (secondary, token 157)
- 22 IPA consonants: b, d, f, h, j, k, l, m, n, p, s, t, v, w, z + ɡ, ŋ, ɹ, ʃ, ʒ, ð, θ
- 2 consonant clusters: ʤ (dʒ merged), ʧ (tʃ merged)
- 10 IPA vowels: ə, i, u, ɑ, ɔ, ɛ, ɜ, ɪ, ʊ, ʌ
- 4 diphthongs encoded as single uppercase letters: A=eɪ, I=aɪ, W=aʊ, Y=ɔɪ
- 1 custom vowel: ᵊ (muted schwa)
- US-only: æ, O (=oʊ), ᵻ (schwa-to-ɪ blend), ɾ (flapped t/d)
- UK-only: a (ash vowel), Q (=əʊ), ɒ, ː (vowel lengthener)

**Additional tokens** beyond English include symbols for Japanese, Chinese, and other languages: ʣ(18), ʥ(19), ʦ(20), ʨ(21), ᵝ(22), pitch accent arrows ↓(169), →(171), ↗(172), ↘(173), and many more IPA symbols up to index 177.

**Punctuation tokens**: ; (1), : (2), , (3), . (4), ! (5), ? (6), — (9), ... (10), " (11), ( (12), ) (13), curly quotes (14, 15), space (16).

### Voice/Speaker Embeddings

**Format**: Each voice is stored as a PyTorch `.pt` file (e.g., `af_heart.pt`) containing a tensor of shape `(511, 1, 256)`.

- The first dimension (511) corresponds to different input sequence lengths (1 to 511 tokens, matching max_position_embeddings=512 minus 1).
- At inference time, the style vector is selected by input length: `ref_s = voicepack[len(tokens)]` giving a tensor of shape `(1, 256)`.
- The 256-dimensional style vector is functionally split:
  - `ref_s[:, :128]` — fed to the **decoder** (controls voice timbre, speaker identity)
  - `ref_s[:, 128:]` — fed to the **prosody predictor** (controls speaking style, rhythm, intonation)

**ONNX voice format**: In the ONNX community model, voices are stored as a single `voices-v1.0.bin` file containing all voice embeddings as numpy float32 arrays. The ONNX model input shape for the style vector is `(1, 1, 256)`.

**Voice naming convention**: `[language_code][gender][_name]` where language codes are: a=American English, b=British English, j=Japanese, z=Mandarin Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Brazilian Portuguese. Gender: f=female, m=male.

**Voice blending**: Custom voices can be created by weighted averaging of existing voice tensors (e.g., 70% voice_A + 30% voice_B). The default voice `af` is a 50/50 blend of af_bella and af_sarah.

**Total voices**: 47 voices across 9 languages (v1.0 release had 54, current count is 47 per VOICES.md).

### Model Weight File Format

The official model is distributed as `kokoro-v1_0.pth` (327 MB, FP32 PyTorch pickle). When loaded via `torch.load()`, it returns a **dictionary of dictionaries** with four top-level keys:

```
{
    "bert":         { ... state_dict for CustomAlbert ... },
    "text_encoder": { ... state_dict for TextEncoder ... },
    "predictor":    { ... state_dict for ProsodyPredictor ... },
    "decoder":      { ... state_dict for Decoder/iSTFTNet ... }
}
```

Each value is the `state_dict` for that component. The `bert_encoder` (the Linear(768, 512) projection) weights appear to be stored under `"bert"` or loaded with a key-stripping fallback (`k[7:]` to handle `module.` prefixes from DataParallel training).

**Note**: The model is distributed as `.pth` (PyTorch pickle), NOT as safetensors. There is no official safetensors release. The [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) has a safetensors conversion, but it is a community conversion for Apple MLX.

### ONNX Model Format

The [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) provides the model as a single ONNX graph:

**Inputs**:
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | `(1, <=512)` | int64 | Phoneme token IDs, padded with 0 at start and end |
| `style` | `(1, 1, 256)` | float32 | Voice style embedding |
| `speed` | `(1,)` | float32 | Speech speed multiplier |

**Outputs**:
| Name | Shape | Type | Description |
|------|-------|------|-------------|
| audio | `(1, num_samples)` | float32 | Raw waveform at 24 kHz |

**Size variants**: FP32 (326 MB), FP16 (163 MB), INT8 quantized (92.4 MB), mixed q8f16 (86 MB), 4-bit (305 MB), q4f16 (154 MB).

Although ONNX is a useful integration reference, **we do not depend on ONNX Runtime** (project rule: pure C# only). We re-implement the architecture natively. The ONNX export is useful only as: (a) a sanity-check during validation, (b) confirmation of which inputs/outputs the model exposes.

### Differences: Kokoro vs Standard StyleTTS 2

- **No diffusion**: Kokoro drops the diffusion-based style sampling from StyleTTS 2, using pre-computed style vectors instead.
- **No encoder release**: The speech encoder (used for extracting style from reference audio during training) is not released.
- **iSTFTNet vs HiFi-GAN**: Kokoro uses iSTFTNet instead of standard HiFi-GAN. iSTFTNet replaces the final upsampling stages with inverse STFT, reducing computation.
- **PLBERT instead of PnG-BERT**: Kokoro uses a custom ALBERT-based phoneme encoder rather than the PnG-BERT from the original StyleTTS 2.
- **Smaller model**: 82M parameters vs StyleTTS 2's larger configurations.

### Differences: Official PyTorch vs ONNX

- **PyTorch**: Model is 4 separate components loaded from a single `.pth` dict. Voice packs are individual `.pt` files per voice.
- **ONNX**: Single monolithic ONNX graph. Voice packs are consolidated into a single `.bin` file. The ONNX model takes `(input_ids, style, speed)` as a single forward call — no need to orchestrate the 4 components separately.
- **ONNX input shape**: style is `(1, 1, 256)` vs PyTorch `(1, 256)` — extra dimension for batch compatibility.

### Kokoro-v1.1-zh

`hexgrad/Kokoro-82M-v1.1-zh` is a fine-tuned variant with improved Mandarin Chinese support. Uses `kokoro-v1_1-zh.pth` weight file. Same architecture, different weights.

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

## Algorithm Steps

### Full Inference Pipeline

```
1. TEXT PREPROCESSING (misaki G2P — see G2P_PHONEMIZATION.md)
   Input: "Hello world!"
   a. Normalize text (numbers, currency, abbreviations)
   b. Tokenize with spaCy, POS tag
   c. Dictionary lookup for known words
   d. Handle heteronyms via averaged perceptron tagger
   e. espeak-ng fallback for OOV words
   Output: "həlˈO wˈɜɹld!" (IPA phoneme string)

2. TOKENIZATION
   a. Map each phoneme char -> integer via vocab dict
   b. Drop unknown characters
   c. Prepend and append padding token 0
   d. Assert len <= 512 (max_position_embeddings)
   Output: LongTensor (1, seq_len)

3. VOICE SELECTION
   a. Load voice .pt file -> tensor (511, 1, 256)
   b. Select by sequence length: ref_s = voicepack[seq_len]
   c. ref_s shape: (1, 256)

4. PLBERT ENCODING
   a. Create attention mask from input lengths
   b. Forward through 12-layer ALBERT (all 12 layers share the same parameters)
      (1, seq_len) -> (1, seq_len, 768)
   c. Linear projection: (1, seq_len, 768) -> (1, seq_len, 512)
   d. Transpose to (1, 512, seq_len)

5. DURATION PREDICTION
   a. DurationEncoder processes BERT output with style s=ref_s[:, 128:]
   b. LSTM processes encoded features
   c. Linear projects to (1, seq_len, 50), sigmoid, sum -> durations
   d. Divide by speed, round, clamp(min=1)
   e. Construct alignment matrix (1, seq_len, total_frames)

6. PROSODY PREDICTION
   a. Expand encoded features by alignment: (1, 512, total_frames)
   b. Predict F0 and energy (N) via AdainResBlk1d chains
   Output: F0_pred, N_pred — frame-level pitch and energy

7. TEXT ENCODING
   a. Embed tokens -> Conv1d layers -> BiLSTM
   b. Expand by alignment: asr = text_enc @ alignment
   Output: (1, 512, total_frames)

8. DECODING (iSTFTNet — see HIFIGAN_VOCODER.md)
   Output: audio tensor (1, num_samples) at 24 kHz

9. POST-PROCESSING
   a. Squeeze to 1D tensor
   b. Move to CPU
   c. Write to WAV file at 24,000 Hz sample rate
```

### Duration Prediction Detail

The duration predictor outputs a tensor of shape `(batch, seq_len, max_dur=50)`. Each of the 50 values is passed through sigmoid, giving probabilities. The sum of these 50 sigmoid values gives the predicted duration in frames for each phoneme (range 0-50). This is divided by the speed parameter and rounded to the nearest integer (minimum 1).

## Open Questions

- [ ] Exact parameter count breakdown per component (PLBERT vs TextEncoder vs Predictor vs Decoder). The 82M total is known but per-component breakdown is not documented.
- [ ] How voice packs are originally created from reference audio (the style encoder is not released).
- [ ] Whether the model supports streaming output or requires full sequence generation. The architecture suggests synchronous (whole-sequence) decode; chunked output would require running the decoder on partial alignments.
- [ ] Whether we should ship the v1.1-zh variant alongside v1.0 — the architecture is identical, the weights are different. Probably yes, load via the same model class with a config switch.

## Implementation Notes for SharpInference

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

5. **BiLSTM implementation**: PyTorch's BiLSTM is fused. For SharpInference we need:
   - Forward LSTM over input
   - Backward LSTM over reversed input
   - Concat along feature dim
   LSTM cell: standard `i, f, g, o = chunk(W*[x; h] + b, 4)`, `c = f*c_prev + i*tanh(g)`, `h = o*tanh(c)`. Plan to add to `SharpInference.Core/Modules` as `LstmCell` / `BiLstm`.

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
