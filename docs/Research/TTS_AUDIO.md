# TTS Audio — Research Notes

## Kokoro-82M

### Summary

Kokoro is an open-weight text-to-speech model with 82 million parameters, licensed under Apache 2.0. It is based on the **StyleTTS 2** architecture ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) with an **iSTFTNet** vocoder ([arXiv:2203.02395](https://arxiv.org/abs/2203.02395)). The model is decoder-only (no diffusion, no encoder release) and produces 24 kHz audio. The full inference pipeline is: text input -> G2P (misaki library + espeak-ng fallback) -> phoneme token IDs -> PLBERT contextual encoding -> prosody prediction (duration, F0, energy) -> text encoding -> length regulation -> iSTFTNet decoder -> raw audio waveform. Voice identity is controlled by a 256-dimensional style vector split into two 128-dimensional halves: one for the decoder and one for the prosody predictor. The model was trained on a few hundred hours of permissive/non-copyrighted audio data with IPA phoneme labels, costing roughly 1,000 A100-80GB GPU hours (~$1,000).

### Overall Architecture

Kokoro follows the StyleTTS 2 design with five primary neural network components:

1. **PLBERT (Phoneme-Level BERT)** — A `CustomAlbert` (modified ALBERT from HuggingFace Transformers) that produces contextual phoneme embeddings. Configured with hidden_size=768, 12 attention heads, 12 transformer layers, intermediate_size=2048, max_position_embeddings=512.

2. **BERT Encoder** — A single `nn.Linear(768, 512)` projection layer that maps PLBERT outputs from 768 dimensions down to the model's hidden_dim of 512.

3. **Text Encoder** — Embedding layer (178 tokens -> 512 dim) followed by multiple 1D CNN layers (kernel_size=5) with LayerNorm, LeakyReLU, and 0.2 dropout, then a bidirectional LSTM (512 -> 256 per direction = 512 output). Depth is 3 layers (matching `n_layer`).

4. **Prosody Predictor** — Predicts duration, F0 (pitch), and energy (N). Contains:
   - **DurationEncoder**: 3 layers of bidirectional LSTM with AdaLayerNorm, processing text embeddings concatenated with the style vector.
   - **Shared LSTM**: Input dimension `(hidden_dim + style_dim)` = 640, hidden `hidden_dim//2` = 256 per direction.
   - **Duration projection**: Linear(512, 50) where 50 = max_dur. Duration is predicted as sigmoid probabilities summed across the 50 bins.
   - **F0 pathway**: 3 `AdainResBlk1d` modules with upsampling.
   - **Energy (N) pathway**: 3 `AdainResBlk1d` modules.

5. **Decoder (iSTFTNet)** — Converts aligned text features + F0 + energy + style into a waveform. Uses Adaptive Instance Normalization (AdaIN) throughout for style conditioning, Snake activation functions, a harmonic+noise source module (SourceModuleHnNSF), multi-stage upsampling with transposed convolutions, and a final inverse STFT to produce the waveform.

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

### G2P: Misaki Library

Source: [github.com/hexgrad/misaki](https://github.com/hexgrad/misaki)

Kokoro uses **misaki** for grapheme-to-phoneme conversion. Misaki is a standalone G2P library that outputs IPA phoneme strings. It is NOT built into the model — it is a preprocessing step.

**English G2P pipeline**:
1. Text normalization (numbers, currency, abbreviations via num2words)
2. NLP processing via spaCy (tokenization, POS tagging)
3. Dictionary-based lookup for known words
4. Averaged perceptron tagger for heteronyms (words with context-dependent pronunciation)
5. Optional BERT-based contextual processing (`trf=True`)
6. **espeak-ng fallback** for out-of-vocabulary words

**Language support**: American English ('a'), British English ('b'), Spanish ('e'), French ('f'), Hindi ('h'), Italian ('i'), Japanese ('j', requires `misaki[ja]` with pyopenjtalk), Brazilian Portuguese ('p'), Mandarin Chinese ('z', requires `misaki[zh]`).

**espeak-ng dependency**: Required as a fallback for OOV English words and as the primary G2P engine for some non-English languages. This is a system-level dependency (C library). For a pure C# implementation, the espeak-ng dependency would need to be replaced with a custom G2P solution or a phoneme dictionary.

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

### Key Numbers/Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Total parameters | 82M | ~327 MB in FP32 |
| Sample rate | 24,000 Hz | Fixed, not configurable |
| Mel channels (n_mels) | 80 | Standard mel spectrogram bins |
| Hidden dimension | 512 | Main model hidden size |
| Style dimension | 128 | Per-half (256 total in voice pack) |
| PLBERT hidden size | 768 | ALBERT encoder output dim |
| PLBERT layers | 12 | Transformer layers in ALBERT |
| PLBERT attention heads | 12 | Multi-head attention |
| PLBERT intermediate size | 2048 | Feed-forward inner dim |
| Max position embeddings | 512 | Context window (token limit) |
| Vocabulary size (n_token) | 178 | Phoneme + punctuation tokens |
| Decoder layers (n_layer) | 3 | Used in TextEncoder, ProsodyPredictor |
| Text encoder kernel size | 5 | Conv1d kernel for TextEncoder |
| Max duration (max_dur) | 50 | Duration prediction bins |
| Dropout | 0.2 | Model-wide dropout rate |
| PLBERT dropout | 0.1 | Separate BERT dropout |
| iSTFT n_fft | 20 | Final inverse STFT window |
| iSTFT hop_size | 5 | Final inverse STFT hop |
| Upsample rates | [10, 6] | ConvTranspose1d strides |
| Upsample kernel sizes | [20, 12] | ConvTranspose1d kernels |
| Upsample initial channel | 512 | Starting channel count |
| Resblock kernel sizes | [3, 7, 11] | Multi-receptive-field fusion |
| Resblock dilations | [[1,3,5],[1,3,5],[1,3,5]] | Per-kernel dilation rates |
| Source harmonics | 8 | SineGen harmonic_num |
| Voice embedding shape | (511, 1, 256) | Per voice .pt file |
| Effective hop size | 300 | 10 * 6 * 5 = frames to samples |

### Data Layouts/Formats

#### Phoneme Token Encoding
```
Input:  "həlˈO wˈɜɹld"  (IPA phoneme string from misaki)
Mapped: [50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46]
Padded: [0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 0]
Shape:  (1, 14)  as LongTensor
```

Token 0 is the padding/BOS/EOS token. Characters not in the vocab are silently dropped.

#### Voice Pack (.pt file)
```
Shape: (511, 1, 256)  float32
Index: voicepack[len(tokens)] -> (1, 256)
Split: ref_s[:, :128] for decoder, ref_s[:, 128:] for predictor
File size: ~500 KB per voice
```

#### ONNX Voice Pack (.bin file)
```
Shape: (num_voices * 512, 1, 256)  float32
Access: voices[voice_index * 512 + len(tokens)] -> (1, 256)
```

#### Audio Output
```
Shape: (num_samples,) float32, range approximately [-1, 1]
Sample rate: 24,000 Hz
Format: raw PCM, typically written to WAV via soundfile
```

#### Model Weight File (.pth)
```
Top-level dict with 4 keys:
  "bert"         -> CustomAlbert state_dict
  "text_encoder" -> TextEncoder state_dict
  "predictor"    -> ProsodyPredictor state_dict
  "decoder"      -> Decoder (iSTFTNet) state_dict
Total size: 327 MB (FP32)
```

### Algorithm Steps

#### Full Inference Pipeline

```
1. TEXT PREPROCESSING (misaki G2P)
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
   b. Forward through 12-layer ALBERT: (1, seq_len) -> (1, seq_len, 768)
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

8. DECODING (iSTFTNet)
   a. Downsample F0 and N by 2x via strided Conv1d
   b. Concatenate [asr, F0, N], encode with AdainResBlk1d
   c. 4-stage decode with style-conditioned residual blocks
   d. Source module generates harmonic+noise from F0
   e. 2-stage upsampling (10x, 6x) with residual blocks
   f. Project to magnitude + phase
   g. Inverse STFT (n_fft=20, hop=5) -> waveform
   Output: audio tensor (1, num_samples) at 24 kHz

9. POST-PROCESSING
   a. Squeeze to 1D tensor
   b. Move to CPU
   c. Write to WAV file at 24,000 Hz sample rate
```

#### Duration Prediction Detail

The duration predictor outputs a tensor of shape `(batch, seq_len, max_dur=50)`. Each of the 50 values is passed through sigmoid, giving probabilities. The sum of these 50 sigmoid values gives the predicted duration in frames for each phoneme (range 0-50). This is divided by the speed parameter and rounded to the nearest integer (minimum 1).

### Open Questions

- [ ] Exact parameter count breakdown per component (PLBERT vs TextEncoder vs Predictor vs Decoder). The 82M total is known but per-component breakdown is not documented.
- [ ] How voice packs are originally created from reference audio (the style encoder is not released).
- [ ] Whether the model supports streaming output or requires full sequence generation.

### Implementation Notes for SharpInference

1. **G2P is the hardest part**: The misaki G2P library is Python-only and depends on spaCy, num2words, and espeak-ng. For C#, options include:
   - Port the phoneme dictionary lookup (misaki uses a dictionary file)
   - Use a pre-built pronunciation dictionary (e.g., CMU Pronouncing Dict converted to IPA)
   - Call espeak-ng via P/Invoke or process spawning (espeak-ng has a C API)
   - Consider the Rust port (misaki-rs) or Swift port (MisakiSwift) as porting references
   - For MVP, accept pre-phonemized input and defer G2P to a later milestone

2. **ONNX is the simplest path**: The ONNX model is a single forward call with 3 inputs. Using ONNX Runtime for C# (Microsoft.ML.OnnxRuntime NuGet package) avoids reimplementing all the neural network math.

3. **Native implementation path**: If implementing natively in C#:
   - ALBERT (PLBERT) is the most complex component — 12 transformer layers with shared weights (ALBERT's parameter-sharing trick)
   - The iSTFTNet decoder requires: Conv1d, ConvTranspose1d, InstanceNorm1d, Snake activation, and torch.istft (inverse STFT via FFT)
   - The inverse STFT at the end can use standard FFT libraries (e.g., MathNet.Numerics or custom SIMD FFT)

4. **Voice pack loading**: The .pt files are PyTorch pickle format. For C#, either:
   - Convert to a simpler format (raw float32 binary, JSON, safetensors) offline
   - Use the ONNX .bin voice format which is just raw numpy arrays
   - Parse PyTorch pickle (complex and fragile — not recommended)

5. **Memory considerations**: The full FP32 model is 327 MB. The INT8 ONNX variant is 92.4 MB. Voice packs are ~500 KB each.

6. **No training needed**: All inference-only. The discriminator and training components are not part of the released model.

7. **Key tensor operations needed**:
   - Matrix multiplication (for alignment expansion)
   - 1D convolution and transposed convolution
   - Bidirectional LSTM
   - ALBERT transformer (with ALBERT's weight-sharing: all 12 layers share the same weights)
   - Instance normalization
   - Inverse STFT (FFT-based)
   - Snake activation: `x + (1/alpha) * sin^2(alpha * x)`
   - SineGen: cumulative sum of phase increments, sine wave generation

### Reference Implementations

- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) — Official Python/PyTorch reference, pip installable
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — Official model card and weights (.pth)
- [hexgrad/misaki](https://github.com/hexgrad/misaki) — G2P library for phoneme conversion
- [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) — ONNX conversion with multiple quantization levels
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — Standalone ONNX Runtime wrapper
- [kokoro.js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) — Official JS implementation
- [Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) — Apple MLX port
- [mlalma/MisakiSwift](https://github.com/mlalma/MisakiSwift) — Swift port of misaki G2P
- [misaki-rs](https://lib.rs/crates/misaki-rs) — Rust port of misaki G2P
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — Base architecture paper implementation

---

## Vocoders: iSTFTNet & HiFiGAN

### Standard HiFiGAN Generator Architecture (Kong et al., 2020)

HiFiGAN is a GAN-based neural vocoder that converts mel spectrograms into raw audio waveforms. The generator uses transposed convolutions for progressive temporal upsampling, with Multi-Receptive Field Fusion (MRF) modules at each stage. The standard V1 configuration uses upsampling factors 8 x 8 x 2 x 2 = 256 (matching a 256-sample hop size) at 22.05 kHz. Only the generator is needed at inference time.

The generator has three main sections:

1. **Pre-convolution** (`conv_pre`): Conv1d projecting input mel spectrogram (80 channels) into the hidden dimension (512 for V1). Kernel size 7, padding 3.

2. **Upsampling blocks**: Transposed 1D convolutions, each increasing temporal resolution by a factor specified in `upsample_rates`. After each, the channel count is halved. Each upsampling layer is followed by an MRF module.

3. **Post-convolution** (`conv_post`): Conv1d projecting to a single output channel (waveform). Kernel size 7, padding 3. Followed by `tanh` to constrain output to [-1, 1].

#### Multi-Receptive Field Fusion (MRF) Module

```
MRF(x) = sum(ResBlock_k(x) for k in resblock_kernel_sizes) / num_kernels
```

For V1 with `resblock_kernel_sizes = [3, 7, 11]`, three residual blocks run in parallel per upsampling stage.

#### ResBlock1 (resblock_type=1, used by V1 and V2)

Contains 3 pairs of dilated convolution layers with residual connections:

```
for each dilation_group in dilation_sizes:
    xt = LeakyReLU(x, slope=0.1)
    xt = Conv1d(xt, kernel_size=kr, dilation=d[0])  # dilated conv
    xt = LeakyReLU(xt)
    xt = Conv1d(xt, kernel_size=kr, dilation=1)      # non-dilated conv
    x = x + xt                                        # residual connection
```

For V1 with dilation `[1, 3, 5]` and kernel size 3, a single ResBlock1 has 6 Conv1d layers total. All convolutions use weight normalization during training (removed at inference).

#### ResBlock2 (resblock_type=2, used by V3)

Contains 2 dilated convolution layers per dilation group (instead of 3 pairs). Same structure but fewer layers.

#### Channel Progression (V1)

| Stage | Operation | Input Ch | Output Ch | Temporal Scale |
|-------|-----------|----------|-----------|----------------|
| pre | Conv1d(k=7) | 80 | 512 | 1x |
| up_0 | ConvTranspose1d(k=16, s=8) | 512 | 256 | 8x |
| MRF_0 | 3 x ResBlock1 | 256 | 256 | 8x |
| up_1 | ConvTranspose1d(k=16, s=8) | 256 | 128 | 64x |
| MRF_1 | 3 x ResBlock1 | 128 | 128 | 64x |
| up_2 | ConvTranspose1d(k=4, s=2) | 128 | 64 | 128x |
| MRF_2 | 3 x ResBlock1 | 64 | 64 | 128x |
| up_3 | ConvTranspose1d(k=4, s=2) | 64 | 32 | 256x |
| MRF_3 | 3 x ResBlock1 | 32 | 32 | 256x |
| post | Conv1d(k=7) | 32 | 1 | 256x |

Total upsampling: 8 x 8 x 2 x 2 = 256 (matches hop_size=256).

### HiFiGAN Configuration Variants

| Parameter | V1 | V2 | V3 |
|-----------|----|----|-----|
| `upsample_initial_channel` (hu) | 512 | 128 | 256 |
| `upsample_rates` | [8, 8, 2, 2] | [8, 8, 2, 2] | [8, 8, 4] |
| `upsample_kernel_sizes` (ku) | [16, 16, 4, 4] | [16, 16, 4, 4] | [16, 16, 8] |
| `resblock_type` | 1 | 1 | 2 |
| `resblock_kernel_sizes` (kr) | [3, 7, 11] | [3, 7, 11] | [3, 5, 7] |
| `resblock_dilation_sizes` (Dr) | [[1,3,5], [1,3,5], [1,3,5]] | [[1,3,5], [1,3,5], [1,3,5]] | [[1,2], [2,6], [3,12]] |
| Sample rate | 22,050 Hz | 22,050 Hz | 22,050 Hz |
| Hop size | 256 | 256 | 256 |
| FFT size | 1,024 | 1,024 | 1,024 |
| Mel bins | 80 | 80 | 80 |

V1 is the highest quality (largest), V2 is the fastest (smallest hidden dim), V3 uses fewer upsampling stages with ResBlock2.

### iSTFTNet Modification (Kaneko et al., 2022)

iSTFTNet modifies HiFiGAN by replacing output-side transposed convolution layers with an inverse Short-Time Fourier Transform. Instead of generating waveform samples directly, the network predicts magnitude and phase spectrograms, which are then converted to audio via iSTFT.

Key insight: after sufficient upsampling reduces the frequency dimension, the remaining time-to-waveform conversion can be handled analytically by iSTFT rather than learned convolutions.

Three variants (applied to HiFiGAN V1 with upsample_rates [8,8,2,2]):
- **C8C8I**: Keep 2 upsampling stages (8x, 8x), replace last 2 with iSTFT. Best quality/speed tradeoff.
- **C8I**: Keep 1 upsampling stage (8x), replace rest with iSTFT. Fastest but lower quality.
- **CI**: No upsampling, all done by iSTFT. Poor quality.

The final layer outputs `(n_fft/2 + 1) * 2` channels: half for magnitude (passed through `exp()`) and half for phase (passed through `sin()`). The iSTFT then reconstructs the waveform.

### Kokoro/StyleTTS2 iSTFTNet Decoder (What We Implement)

Kokoro-82M is fine-tuned from [StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) and uses StyleTTS2's iSTFTNet decoder with additional modifications. This is NOT standard HiFiGAN.

#### Configuration (from StyleTTS2 config.yml)

```yaml
decoder:
  type: 'istftnet'
  resblock_kernel_sizes: [3, 7, 11]
  upsample_rates: [10, 6]
  upsample_initial_channel: 512
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
  upsample_kernel_sizes: [20, 12]
  gen_istft_n_fft: 20
  gen_istft_hop_size: 5

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300
    n_mels: 80
```

#### Key Differences from Standard HiFiGAN

1. **Only 2 upsampling stages** (not 4): rates [10, 6] instead of [8, 8, 2, 2]
2. **iSTFT output head**: Replaces the final conv_post + tanh with magnitude/phase prediction + iSTFT
3. **AdaIN conditioning**: Each ResBlock uses Adaptive Instance Normalization (AdaINResBlock1) that modulates features using a style vector
4. **Snake activation**: Uses `x + (1/alpha) * sin^2(alpha * x)` instead of LeakyReLU in residual blocks
5. **Harmonic-plus-noise source module** (SourceModuleHnNSF): Generates F0-conditioned harmonic source signals (8 harmonics) and noise, injected at each upsampling stage
6. **Style dimension**: 64 (internal decoder style, separate from the 256-dim model-level style embedding)
7. **24 kHz sample rate** (not 22.05 kHz)

#### Decoder Class Detail

Source: [`kokoro/istftnet.py`](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py)

1. **F0 conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples F0 by 2x.
2. **N conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples energy by 2x.
3. **Encode block**: `AdainResBlk1d(dim_in + 2*hidden_dim, 1024, style_dim)` — fuses ASR features, F0, and energy.
4. **Decode blocks**: 4 stages of `AdainResBlk1d` that progressively refine, with residual ASR features, F0, and N concatenated at each stage.
5. **Generator**: The actual iSTFTNet waveform synthesis.

#### Generator Class Detail

1. **Source module (SourceModuleHnNSF)**: Generates harmonic sine waves (8 harmonics) + noise from F0 at the target sample rate. Uses SineGen with `voiced_threshold=10`.
2. **Upsampling**: Two stages of `ConvTranspose1d` with rates `[10, 6]` and kernel sizes `[20, 12]`. Starting from 512 channels, halving at each stage: 512 -> 256 -> 128.
3. **Residual blocks**: At each upsampling level, 3 `AdaINResBlock1` blocks with kernel sizes `[3, 7, 11]` and dilations `[[1,3,5], [1,3,5], [1,3,5]]`. All conditioned on the style vector.
4. **Noise conditioning**: Parallel `Conv1d` + `AdaINResBlock1` path processes the STFT of the source signal at each level.
5. **Final projection**: `Conv1d` -> `(n_fft + 2)` channels, split into magnitude (exponentiated) and phase (sin/cos), then inverse STFT.

#### Channel Progression (Kokoro/StyleTTS2)

| Stage | Operation | Input Ch | Output Ch | Temporal Scale |
|-------|-----------|----------|-----------|----------------|
| pre | Conv1d(k=7) | 512 (style-conditioned input) | 512 | 1x |
| up_0 | ConvTranspose1d(k=20, s=10) | 512 | 256 | 10x |
| noise_0 | Conv1d(har, k=12, s=6) | 22 | 256 | (harmonic injection) |
| MRF_0 | 3 x AdaINResBlock1 | 256 | 256 | 10x |
| up_1 | ConvTranspose1d(k=12, s=6) | 256 | 128 | 60x |
| noise_1 | Conv1d(har, k=1, s=1) | 22 | 128 | (harmonic injection) |
| MRF_1 | 3 x AdaINResBlock1 | 128 | 128 | 60x |
| post | Conv1d -> split | 128 | 22 (11 mag + 11 phase) | 60x |
| iSTFT | iSTFT(n_fft=20, hop=5) | 11 complex | 1 | 300x (60 x 5) |

Total effective upsampling: 10 x 6 x 5 = 300 (matches hop_length=300 at 24 kHz).

#### iSTFT Output Head

The post-convolution outputs `gen_istft_n_fft + 2 = 22` channels:
- Channels 0-10: log-magnitude, passed through `exp()` to get magnitude
- Channels 11-21: raw phase, passed through `sin()` to get phase

Then `iSTFT(magnitude * exp(j * phase), n_fft=20, hop_size=5, window=hann)` reconstructs the waveform.

**Snake activation**: `x + (1/alpha) * sin^2(alpha * x)` where alpha is a learnable parameter. This provides a periodic, smooth non-linearity well-suited for audio.

**AdaIN1d**: `(1 + gamma) * InstanceNorm(x) + beta` where gamma and beta are projected from the style vector via `Linear(style_dim, num_features * 2)`.

### Vocos Architecture (Alternative)

Vocos (Siuzdak, 2023) is a fundamentally different approach:

- **No transposed convolutions**: Maintains constant temporal resolution throughout
- **ConvNeXt backbone**: Stack of 1D depthwise convolution + inverted bottleneck blocks with GELU activation and LayerNorm
- **All upsampling via iSTFT**: The only temporal expansion happens in the final iSTFT layer
- **ISTFT head**: Predicts magnitude via `exp(m)` and phase via unit circle projection `(cos(p), sin(p))`
- **Speed**: Approximately 13x faster than HiFiGAN, approximately 70x faster than BigVGAN on GPU
- **Quality**: Comparable or better than HiFiGAN (VISQOL 4.66 vs 4.57; PESQ 3.70 vs 3.09)
- **Parameters**: Trained on LibriTTS at 24 kHz with n_fft=1024, hop=256, 100 mel bins

Vocos is a compelling alternative for future consideration, but Kokoro currently uses iSTFTNet, so implementing iSTFTNet is required first.

### Comparison: Standard HiFiGAN vs Kokoro iSTFTNet vs Vocos

| Aspect | Standard HiFiGAN V1 | Kokoro/StyleTTS2 iSTFTNet | Vocos |
|--------|---------------------|--------------------------|-------|
| Sample rate | 22,050 Hz | 24,000 Hz | 24,000 Hz |
| Hop size | 256 | 300 | 256 |
| Upsampling stages | 4 (8,8,2,2) | 2 (10,6) + iSTFT(hop=5) | All iSTFT |
| Total upsampling | 256x | 300x (60x conv + 5x iSTFT) | Via iSTFT |
| Output method | tanh(conv_post) | iSTFT(magnitude, phase) | iSTFT(magnitude, phase) |
| Activation | LeakyReLU(0.1) | Snake activation | GELU |
| Normalization | Weight norm | AdaIN (style-conditioned) | LayerNorm |
| Conditioning | None (unconditional) | Style vector + F0 | None (mel-only) |
| Source model | None | HnNSF (harmonic+noise, F0) | None |
| Backbone | Transposed conv + ResBlocks | Transposed conv + ResBlocks | ConvNeXt blocks |
| Temporal resolution | Changes through network | Changes through network | Constant |
| Speed vs HiFiGAN | 1x | ~1.7x faster | ~13x faster |
| Parameters (generator) | ~14M (V1) | Part of 82M total | — |

### Key Numbers/Constants (Vocoder-Specific)

#### Standard HiFiGAN V1 (Reference)

| Constant | Value | Notes |
|----------|-------|-------|
| Input mel bins | 80 | Standard across all variants |
| Sample rate | 22,050 Hz | Standard HiFiGAN |
| Hop size | 256 | Matches total upsampling factor |
| FFT size | 1,024 | For mel spectrogram computation |
| Window size | 1,024 | Hann window |
| Total upsampling | 256x | 8 x 8 x 2 x 2 |
| Initial channels | 512 | V1 hidden dimension |
| LeakyReLU slope | 0.1 | Used throughout generator |
| Output range | [-1.0, 1.0] | tanh activation |

#### Kokoro/StyleTTS2 iSTFTNet

| Constant | Value | Notes |
|----------|-------|-------|
| Analysis FFT size | 2,048 | `n_fft` for mel computation |
| Analysis window size | 1,200 | `win_length` for mel computation |
| Synthesis iSTFT n_fft | 20 | `gen_istft_n_fft` |
| Synthesis iSTFT hop | 5 | `gen_istft_hop_size` |
| Style dimension (decoder) | 64 | AdaIN conditioning |
| Harmonic count | 8 | SourceModuleHnNSF |
| Voiced threshold | 10 | SourceModuleHnNSF |
| Post conv output channels | 22 | n_fft + 2 = 22 (11 mag + 11 phase) |

### Data Layouts/Formats (Vocoder)

#### Input: Mel Spectrogram
```
Shape: [batch, 80, T_mel]
Type: float32
Range: log-scale mel spectrogram values (typically -11.0 to 2.0)
```

#### Internal: After Upsampling
```
After pre-conv:   [batch, 512, T_mel]
After up_0:       [batch, 256, T_mel * 10]
After up_1:       [batch, 128, T_mel * 60]
After post-conv:  [batch, 22,  T_mel * 60]
  - channels 0-10: log-magnitude spectrogram
  - channels 11-21: phase spectrogram
```

#### Internal: Harmonic Source
```
F0 input:         [batch, 1, T_mel]  (fundamental frequency per mel frame)
F0 upsampled:     [batch, 1, T_mel * 300]  (upsampled to sample rate)
Harmonic source:  [batch, 1, T_mel * 300]  (sinusoidal signal)
Harmonic STFT:    [batch, 22, T_mel * 60]  (STFT of harmonic source, n_fft=20)
  - channels 0-10: harmonic magnitude
  - channels 11-21: harmonic phase
```

#### Output: Audio Waveform
```
Shape: [batch, 1, T_audio] where T_audio = T_mel * 300
Type: float32, range [-1.0, 1.0]
For 16-bit PCM: multiply by 32767, clamp to [-32768, 32767], cast to int16
```

### Algorithm Steps (Vocoder)

#### Standard HiFiGAN V1 Forward Pass

```
1. x = conv_pre(mel)                           # [B, 80, T] -> [B, 512, T]
2. For each upsampling stage i in [0, 1, 2, 3]:
   a. x = LeakyReLU(x, slope=0.1)
   b. x = ups[i](x)                            # transposed convolution
   c. xs = 0
   d. For each kernel j in [0, 1, 2]:           # resblock_kernel_sizes = [3, 7, 11]
      xs += resblocks[i*3 + j](x)              # ResBlock1 with dilations [1,3,5]
   e. x = xs / 3                                # average MRF outputs
3. x = LeakyReLU(x, slope=0.1)
4. x = conv_post(x)                             # [B, 32, T*256] -> [B, 1, T*256]
5. x = tanh(x)
6. return x
```

#### ResBlock1 Forward Pass

```
For each dilation d in [1, 3, 5]:
  xt = LeakyReLU(x, slope=0.1)
  xt = Conv1d(xt, kernel=kr, dilation=d, padding=d*(kr-1)//2)
  xt = LeakyReLU(xt, slope=0.1)
  xt = Conv1d(xt, kernel=kr, dilation=1, padding=(kr-1)//2)
  x = x + xt
return x
```

#### Kokoro/StyleTTS2 iSTFTNet Forward Pass

```
1. Upsample F0 to audio sample rate:
   f0_up = interpolate(f0, scale_factor=300)    # [B, 1, T] -> [B, 1, T*300]

2. Generate harmonic+noise source from F0:
   har_source, noi_source, uv = SourceModuleHnNSF(f0_up)

3. Compute STFT of harmonic source:
   har_spec, har_phase = STFT(har_source, n_fft=20, hop=5)
   har = concat(har_spec, har_phase, dim=1)     # [B, 22, T*60]

4. Process through upsampling stages:
   For each stage i in [0, 1]:
     a. x = LeakyReLU(x, slope=0.1)
     b. x_source = noise_convs[i](har)           # harmonic feature injection
        x_source = noise_res[i](x_source, style) # style-conditioned
     c. x = ups[i](x)                            # transposed conv upsample
     d. x = x + x_source                         # add harmonic conditioning
     e. xs = 0
     f. For each kernel j in [0, 1, 2]:
        xs += adain_resblocks[i*3+j](x, style)  # AdaIN + Snake activation
     g. x = xs / 3

5. Generate magnitude and phase:
   x = conv_post(x)                              # [B, 128, T*60] -> [B, 22, T*60]
   magnitude = exp(x[:, :11, :])                 # positive magnitude
   phase = sin(x[:, 11:, :])                     # phase wrapped to [-1, 1]

6. Reconstruct waveform via iSTFT:
   audio = iSTFT(magnitude, phase, n_fft=20, hop_size=5, window=hann)

7. return audio
```

#### AdaINResBlock1 Forward Pass (Snake + AdaIN)

```
For each dilation d in [1, 3, 5]:
  xt = AdaIN1d(x, style)                       # normalize, then scale/shift by style
  xt = Snake(xt, alpha)                         # x + (1/alpha) * sin^2(alpha * x)
  xt = Conv1d(xt, kernel=kr, dilation=d)
  xt = AdaIN1d(xt, style)
  xt = Snake(xt, alpha)
  xt = Conv1d(xt, kernel=kr, dilation=1)
  x = x + xt
return x
```

#### SourceModuleHnNSF (Harmonic Source Generation)

```
Given f0 (fundamental frequency in Hz) at sample rate:
1. For k in 0..harmonic_num (0..8):
   phase[k] = cumsum(f0 * (k+1) / sample_rate) * 2 * pi
   sine[k] = sin(phase[k])
   sine[k] *= voiced_mask(f0 > threshold)       # zero out unvoiced
2. harmonic_source = linear_combination(sines)   # learned weights
3. noise_source = gaussian_noise * learned_scale
4. return harmonic_source, noise_source, uv_flag
```

### Weight Normalization Notes

Standard HiFiGAN uses `weight_norm` on all Conv1d layers during training. At inference, weight normalization should be fused/removed for speed:
```
weight = weight_g * (weight_v / norm(weight_v))
```
This can be pre-computed and stored as a single weight tensor. Kokoro's iSTFTNet uses a similar pattern with `ConvWeighted` separating magnitude (`weight_g`) and direction (`weight_v`).

### Open Questions

- [ ] Exact parameter count of the iSTFTNet decoder portion alone (vs. full 82M model)
- [ ] Whether Kokoro's iSTFTNet weights can be loaded independently from the rest of the model
- [ ] Whether the F0 (pitch) input is required or can be set to zero for inference
- [ ] Exact weight tensor names in the Kokoro safetensors file for the decoder portion

### Implementation Notes for SharpInference (Vocoder)

1. **Implement iSTFTNet first, not standard HiFiGAN.** Kokoro uses iSTFTNet. Standard HiFiGAN is only useful as a reference/test target.

2. **Key operations to implement:**
   - 1D transposed convolution (ConvTranspose1d) with stride > 1
   - 1D dilated convolution (Conv1d with dilation parameter)
   - Adaptive Instance Normalization (AdaIN1d): normalize per-channel, then apply learned affine from style vector
   - Snake activation: `x + (1/alpha) * sin^2(alpha * x)` — requires element-wise sin and multiply
   - iSTFT: inverse Short-Time Fourier Transform with Hann window, n_fft=20, hop=5
   - SourceModuleHnNSF: cumulative sum of phase, sinusoidal generation, learned mixing

3. **iSTFT implementation:** With n_fft=20 and hop_size=5, this is a very small FFT (only 20 points). Can be implemented efficiently with a direct DFT formula rather than a full FFT library. The overlap-add synthesis window is a 20-point Hann window.

4. **Memory layout:** All intermediate tensors are `[batch, channels, time]` (channels-first). The time dimension grows through the network: T_mel -> T_mel*10 -> T_mel*60 -> T_audio (T_mel*300).

5. **Weight normalization fusion:** Pre-compute `weight = weight_g * weight_v / norm(weight_v)` during model loading to avoid runtime overhead.

6. **Transposed convolution padding:** For `ConvTranspose1d(kernel=k, stride=s)`, the output length is `(input_length - 1) * stride + kernel - 2 * padding`. The kernel sizes are chosen as 2x the stride (k=20 for s=10, k=12 for s=6), which with appropriate padding produces clean upsampling.

7. **16-bit PCM output:** After the vocoder produces float32 audio in [-1, 1]:
   ```
   pcm16 = clamp(round(audio * 32767), -32768, 32767)
   ```
   Write as little-endian int16 for WAV file output at 24,000 Hz sample rate.

8. **Cross-reference:** See [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for mel spectrogram computation details (note: that doc covers Whisper's parameters at 16kHz; Kokoro uses different STFT parameters at 24kHz with n_fft=2048, hop=300, win=1200, 80 mel bins).

### Reference Implementations (Vocoder)

- [jik876/hifi-gan](https://github.com/jik876/hifi-gan) — Official HiFiGAN V1/V2/V3
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — iSTFTNet decoder with AdaIN, what Kokoro uses
- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) — Kokoro TTS, fine-tuned from StyleTTS2-LJSpeech
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch) — Standalone iSTFTNet implementation
- [Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) — MLX port, good architecture reference
- [characterai/vocos](https://github.com/gemelo-ai/vocos) — Vocos reference implementation
- [torchaudio HiFiGANVocoder](https://docs.pytorch.org/audio/2.8/generated/torchaudio.prototype.models.HiFiGANVocoder.html) — Clean reference with V1/V2/V3 factory functions
- [SpeechBrain HiFiGAN](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.HifiGAN.html) — Well-documented HiFiGAN with both ResBlock types
