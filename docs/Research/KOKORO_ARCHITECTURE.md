# Kokoro Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Audio

## Summary

Kokoro is an open-weight text-to-speech model with 82 million parameters, licensed under Apache 2.0. It is based on the **StyleTTS 2** architecture ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) with an **iSTFTNet** vocoder ([arXiv:2203.02395](https://arxiv.org/abs/2203.02395)). The model is decoder-only (no diffusion, no encoder release) and produces 24 kHz audio. The full inference pipeline is: text input -> G2P (misaki library + espeak-ng fallback) -> phoneme token IDs -> PLBERT contextual encoding -> prosody prediction (duration, F0, energy) -> text encoding -> length regulation -> iSTFTNet decoder -> raw audio waveform. Voice identity is controlled by a 256-dimensional style vector split into two 128-dimensional halves: one for the decoder and one for the prosody predictor. The model was trained on a few hundred hours of permissive/non-copyrighted audio data with IPA phoneme labels, costing roughly 1,000 A100-80GB GPU hours (~$1,000).

## Detailed Findings

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

### Decoder Architecture (iSTFTNet) Detail

Source: [`kokoro/istftnet.py`](https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py)

The `Decoder` class:
1. **F0 conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples F0 by 2x.
2. **N conv**: `Conv1d(1, hidden_dim, kernel_size=3, stride=2)` — downsamples energy by 2x.
3. **Encode block**: `AdainResBlk1d(dim_in + 2*hidden_dim, 1024, style_dim)` — fuses ASR features, F0, and energy.
4. **Decode blocks**: 4 stages of `AdainResBlk1d` that progressively refine, with residual ASR features, F0, and N concatenated at each stage.
5. **Generator**: The actual iSTFTNet waveform synthesis.

The `Generator` class:
1. **Source module (SourceModuleHnNSF)**: Generates harmonic sine waves (8 harmonics) + noise from F0 at the target sample rate. Uses SineGen with `voiced_threshold=10`.
2. **Upsampling**: Two stages of `ConvTranspose1d` with rates `[10, 6]` and kernel sizes `[20, 12]`. Starting from 512 channels, halving at each stage: 512 -> 256 -> 128.
3. **Residual blocks**: At each upsampling level, 3 `AdaINResBlock1` blocks with kernel sizes `[3, 7, 11]` and dilations `[[1,3,5], [1,3,5], [1,3,5]]`. All conditioned on the style vector.
4. **Noise conditioning**: Parallel `Conv1d` + `AdaINResBlock1` path processes the STFT of the source signal at each level.
5. **Final projection**: `Conv1d` -> `(n_fft + 2)` channels, split into magnitude (exponentiated) and phase (sin/cos), then inverse STFT.

The inverse STFT parameters: `n_fft=20`, `hop_size=5`. The total upsampling factor is `10 * 6 * 5 = 300`, meaning each input frame produces 300 audio samples. At 24 kHz, this gives 80 frames/second (matching the 80-mel setup at standard hop lengths).

**Snake activation**: `x + (1/alpha) * sin^2(alpha * x)` where alpha is a learnable parameter. This provides a periodic, smooth non-linearity well-suited for audio.

**AdaIN1d**: `(1 + gamma) * InstanceNorm(x) + beta` where gamma and beta are projected from the style vector via `Linear(style_dim, num_features * 2)`.

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

## Key Numbers/Constants

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

## Data Layouts/Formats

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

### Duration Prediction Detail

The duration predictor outputs a tensor of shape `(batch, seq_len, max_dur=50)`. Each of the 50 values is passed through sigmoid, giving probabilities. The sum of these 50 sigmoid values gives the predicted duration in frames for each phoneme (range 0-50). This is divided by the speed parameter and rounded to the nearest integer (minimum 1).

### iSTFT Waveform Synthesis Detail

Unlike standard HiFi-GAN which directly predicts the waveform through transposed convolutions, iSTFTNet:
1. Upsamples features using only 2 transposed convolution stages (10x and 6x = 60x)
2. Projects to `n_fft + 2 = 22` channels
3. Splits into magnitude (first `n_fft//2 + 1 = 11` channels) and phase (remaining 11)
4. Applies `exp()` to magnitude and wraps phase through `sin()`/`cos()`
5. Uses `torch.istft(n_fft=20, hop_length=5)` to produce the final waveform
6. The iSTFT provides an additional 5x upsampling, giving total upsampling of 60 * 5 = 300x

This is faster and more lightweight than pure neural upsampling because the iSTFT is a deterministic transform.

## Reference Implementations

| Implementation | Language | URL | Notes |
|---|---|---|---|
| hexgrad/kokoro (official) | Python/PyTorch | [github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro) | Reference implementation, pip installable |
| hexgrad/Kokoro-82M | Model weights | [huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Official model card and weights (.pth) |
| hexgrad/misaki | Python | [github.com/hexgrad/misaki](https://github.com/hexgrad/misaki) | G2P library for phoneme conversion |
| onnx-community/Kokoro-82M-v1.0-ONNX | ONNX | [huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) | ONNX conversion with multiple quantization levels |
| thewh1teagle/kokoro-onnx | Python/ONNX | [github.com/thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Standalone ONNX Runtime wrapper |
| kokoro.js | JavaScript | [github.com/hexgrad/kokoro/tree/main/kokoro.js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) | Official JS implementation in the main repo |
| Blaizzy/mlx-audio | Python/MLX | [deepwiki.com/Blaizzy/mlx-audio](https://deepwiki.com/Blaizzy/mlx-audio/3.2-api-reference) | Apple MLX port |
| mlalma/MisakiSwift | Swift | [github.com/mlalma/MisakiSwift](https://github.com/mlalma/MisakiSwift) | Swift port of misaki G2P |
| misaki-rs | Rust | [lib.rs/crates/misaki-rs](https://lib.rs/crates/misaki-rs) | Rust port of misaki G2P |
| StyleTTS 2 (original) | Python/PyTorch | [github.com/yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | Base architecture paper implementation |

## Differences Between Implementations

### Official PyTorch vs ONNX

- **PyTorch**: Model is 4 separate components loaded from a single `.pth` dict. Voice packs are individual `.pt` files per voice.
- **ONNX**: Single monolithic ONNX graph. Voice packs are consolidated into a single `.bin` file. The ONNX model takes `(input_ids, style, speed)` as a single forward call — no need to orchestrate the 4 components separately.
- **ONNX input shape**: style is `(1, 1, 256)` vs PyTorch `(1, 256)` — extra dimension for batch compatibility.

### Kokoro vs Standard StyleTTS 2

- **No diffusion**: Kokoro drops the diffusion-based style sampling from StyleTTS 2, using pre-computed style vectors instead.
- **No encoder release**: The speech encoder (used for extracting style from reference audio during training) is not released.
- **iSTFTNet vs HiFi-GAN**: Kokoro uses iSTFTNet instead of standard HiFi-GAN. iSTFTNet replaces the final upsampling stages with inverse STFT, reducing computation.
- **PLBERT instead of PnG-BERT**: Kokoro uses a custom ALBERT-based phoneme encoder rather than the PnG-BERT from the original StyleTTS 2.
- **Smaller model**: 82M parameters vs StyleTTS 2's larger configurations.

### Kokoro vs Kokoro-v1.1-zh

- `hexgrad/Kokoro-82M-v1.1-zh` is a fine-tuned variant with improved Mandarin Chinese support.
- Uses `kokoro-v1_1-zh.pth` weight file.
- Same architecture, different weights.

## Open Questions

- [x] Exact phoneme set and encoding scheme used by Kokoro — **Resolved**: 178 IPA-based tokens, documented in config.json vocab mapping. 49 English phonemes (41 shared + 4 US-only + 4 UK-only) plus multilingual symbols.
- [x] Whether Kokoro requires espeak-ng or has a built-in G2P model — **Resolved**: Kokoro uses the external `misaki` library for G2P, which in turn uses espeak-ng as a fallback for OOV words. espeak-ng is a system dependency, not built into the model.
- [x] Voice embedding dimensionality and how custom voices are created — **Resolved**: 256-dimensional style vectors stored as (511, 1, 256) tensors in .pt files. Custom voices created by weighted averaging of existing voice tensors.
- [ ] Exact parameter count breakdown per component (PLBERT vs TextEncoder vs Predictor vs Decoder). The 82M total is known but per-component breakdown is not documented.
- [ ] How voice packs are originally created from reference audio (the style encoder is not released).
- [ ] Whether the model supports streaming output or requires full sequence generation.

## Implementation Notes

### For SharpInference C# Implementation

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
