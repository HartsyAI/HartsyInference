# Mel Spectrogram (Audio Preprocessing) — Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (any speech model)

## Summary

Mel spectrogram preprocessing converts raw audio into a log-mel spectrogram for use as input to STT models (Whisper, Parakeet, Canary, etc.), music models (Stable Audio Open, MusicGen for its conditioning encoders), and TTS feature extractors. The pipeline is: resample to model's sample rate → STFT (FFT of windowed frames) → power spectrogram → mel filterbank projection → log compression → normalization. Whisper's parameters are the most widely replicated reference; Kokoro/StyleTTS2/HiFiGAN use a different parameter set at 24 kHz. Validation tolerance is 1e-4 per element vs Python references (librosa, torchaudio, whisper.cpp).

All math is well-defined and bit-reproducible across implementations *if* the parameters match exactly — even small deviations (e.g. Slaney vs HTK mel scale, periodic vs symmetric Hann window) produce wrong outputs. Documenting exact parameter sets is the entire point of this doc.

Sources: [OpenAI Whisper audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py), [whisper.cpp mel](https://github.com/ggml-org/whisper.cpp), [HuggingFace WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py), [librosa.filters.mel](https://librosa.org/doc/main/generated/librosa.filters.mel.html), [torchaudio.transforms.MelSpectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html), [StyleTTS2 config](https://github.com/yl4579/StyleTTS2)

## Detailed Findings

### Pipeline Overview

```
audio (float32, mono, model_sample_rate Hz)
  │
  │  zero-pad/crop to integer multiple of hop_length, or fixed chunk
  ▼
[STFT] window=Hann(periodic, win_length), hop=hop_length, n_fft=n_fft
  ▼
complex spectrogram (n_fft/2+1, T)
  │
  │  drop last frame (Whisper convention)
  ▼
power spectrogram |STFT|^2 (n_fft/2+1, T)
  │
  │  matmul with mel filterbank (n_mels, n_fft/2+1)
  ▼
mel spectrogram (n_mels, T)
  │
  │  log10(max(mel, log_floor))
  ▼
log-mel (n_mels, T)
  │
  │  optional normalization (Whisper: dynamic-range + (+4)/4 shift)
  ▼
output (n_mels, T)
```

### Parameter Sets by Model Family

| Param | Whisper | Kokoro/StyleTTS2 | HiFiGAN V1 | Parakeet/Canary | Moonshine |
|---|---|---|---|---|---|
| Sample rate | 16,000 | 24,000 | 22,050 | 16,000 | 16,000 |
| n_fft | 400 | 2,048 | 1,024 | 512 | (operates on raw audio) |
| win_length | 400 | 1,200 | 1,024 | 400 | — |
| hop_length | 160 | 300 | 256 | 160 | — |
| Window | Hann periodic | Hann periodic | Hann periodic | Hann periodic | — |
| n_mels | 80 (or 128 for large-v3+) | 80 | 80 | 80 | — |
| Mel scale | Slaney | Slaney | Slaney | Slaney | — |
| Mel norm | Slaney area | Slaney area | None / sqrt-area | Slaney area | — |
| f_min | 0 | 0 | 0 | 0 | — |
| f_max | 8,000 | 12,000 (Nyquist) | 8,000 | 8,000 | — |
| Power | 2 (|STFT|^2) | 1 (|STFT|) | 1 | 2 | — |
| Log | log10 | log_e | log_e | log_e (clamp eps=1e-5) | — |
| Normalization | (x+4)/4, clamp max-8 | none | none | per-feature mean/std | — |
| Output frames per 30s | 3,000 (drop last) | 2,400 | 2,587 | 3,000 | n/a |

Moonshine is unique — it consumes raw waveform directly (no mel) via a strided conv front-end. See [MOONSHINE_ARCHITECTURE.md](MOONSHINE_ARCHITECTURE.md).

### STFT Computation

Standard formula. For each frame `k` of length `win_length`:

```
window = Hann(win_length, periodic=True)
x_padded = pad audio to (n_frames - 1) * hop_length + win_length
for k in range(n_frames):
  frame = x_padded[k*hop_length : k*hop_length + win_length] * window
  if win_length < n_fft:
    frame = zero-pad to n_fft on right
  X[:, k] = FFT(frame)[:n_fft/2 + 1]   # keep only positive frequencies
```

**Critical**: Hann window is `periodic`, not `symmetric`. In NumPy/SciPy this is `np.hanning` (symmetric, wrong!) vs `scipy.signal.windows.hann(N, sym=False)` (periodic, correct). PyTorch's `torch.hann_window` defaults to `periodic=True`, matching the convention.

Periodic Hann formula:
```
w[n] = 0.5 * (1 - cos(2*PI*n / N))     for n in 0..N-1     # NOTE: N (not N-1)
```

Symmetric Hann (NOT used):
```
w[n] = 0.5 * (1 - cos(2*PI*n / (N-1))) for n in 0..N-1     # divisor is N-1
```

### Whisper-Specific Quirks

From [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```python
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=hann_window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2   # 1) drop last frame  2) power not magnitude
mel_spec = mel_filters @ magnitudes
log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Steps:
1. **Drop last STFT frame** — `stft[..., :-1]`. With `center=True` (PyTorch default for `torch.stft`), `N_SAMPLES/HOP + 1 = 3001` frames are produced; dropping gives 3000.
2. **Power spectrum** — `abs()^2` (squared magnitudes), not raw magnitudes.
3. **Floor clamp** — `max(mel_spec, 1e-10)` to prevent log(0).
4. **log10** — base-10 logarithm (NOT natural log).
5. **Dynamic range** — clamp to within 8.0 log10 units (80 dB) below the per-spectrogram max.
6. **Normalize** — `(log_spec + 4.0) / 4.0` maps typical range to ~[0, 1].

**No per-channel mean subtraction. No global mean/var normalization.** Each 30-second clip is self-normalized.

### Mel Filterbank (Slaney Scale)

The Slaney mel scale is piecewise linear below 1 kHz and logarithmic above:

```
Hz to Mel:
  if f < 1000:  mel = 3 * f / 200
  if f >= 1000: mel = 15 + 27 * log(f / 1000) / log(6.4)

Mel to Hz:
  if mel < 15:  f = 200 * mel / 3
  if mel >= 15: f = 1000 * exp((mel - 15) * log(6.4) / 27)
```

The HTK mel scale (used in older speech systems) is `mel = 2595 * log10(1 + f / 700)` — produces different filterbank weights, **wrong for Whisper / Kokoro / Parakeet**.

**Filterbank construction (Slaney)**:
1. Compute `n_mels + 2` mel-spaced center frequencies between `f_min` and `f_max`.
2. Convert center frequencies back to Hz, then to FFT bin indices.
3. For each of `n_mels` filters, build a triangular filter from `bin[i]` to `bin[i+2]`, peak at `bin[i+1]`.
4. **Slaney area normalization**: divide each filter by `(mel_freq[i+2] - mel_freq[i]) / 2` so each filter has constant area in mel space.
5. Result: `[n_mels, n_fft/2 + 1]` matrix.

**Shape examples**:
- Whisper (n_mels=80, n_fft=400): `[80, 201]`
- Whisper-large-v3 (n_mels=128, n_fft=400): `[128, 201]`
- Kokoro (n_mels=80, n_fft=2048): `[80, 1025]`
- HiFiGAN (n_mels=80, n_fft=1024): `[80, 513]`

**Pre-computed vs runtime**: OpenAI loads from `whisper/assets/mel_filters.npz` (generated with librosa). HuggingFace generates at runtime with `mel_filter_bank(mel_scale="slaney", norm="slaney")`. We should generate at startup and cache — avoids asset dependency and matches HF behavior bit-for-bit.

### Padding & Chunking

**Whisper**: zero-pad to exactly 480,000 samples (30s) on the right. Audio longer than 30s is split into 30s chunks (sequential decoding handles partial trailing chunk via timestamp tokens).

**Parakeet / Canary**: pad to multiple of `hop_length` (typically); no fixed chunk length, but typical inference splits at 30-40s for memory reasons. See [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md).

**StyleTTS2 / Kokoro mel inversion**: mel spectrogram is computed only at training time for the vocoder; at inference the predictor generates mel internally without external preprocessing.

**Streaming**: see [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for chunk-and-overlap mel pipelines (mel can be computed incrementally as audio arrives, with the last `win_length - hop_length` samples held as "context tail" between chunks).

### Resampling

If input audio is not at the target sample rate (most files are 44.1 / 48 kHz), resampling is required. Standard algorithm:
1. Compute resample ratio `out_rate / in_rate` (e.g. 16000/44100).
2. Apply polyphase filter (windowed sinc, typically with Kaiser β=8.6, num_taps=64-256).

Reference: `scipy.signal.resample_poly`, `librosa.resample(res_type="kaiser_best")`. PyTorch: `torchaudio.functional.resample`.

For a pure-C# implementation, ship a simple polyphase resampler — sinc table generated at startup, applied as a 1D convolution. Quality is set by num_taps; 64 taps is fine for STT, 256+ for music.

### Output Tensor Shape

```
Whisper:    [1, n_mels, 3000]            (n_mels = 80 or 128)
Kokoro:     [1, 80, T_mel]                T_mel = ceil(audio_samples / 300)
HiFiGAN:    [1, 80, T_mel]                T_mel = ceil(audio_samples / 256)
Parakeet:   [1, 80, T_mel]                T_mel = ceil(audio_samples / 160)
```

All implementations use channels-first layout `[B, n_mels, T]`.

## Key Numbers / Constants

| Constant | Whisper | Kokoro |
|----------|---------|--------|
| Sample rate | 16,000 Hz | 24,000 Hz |
| Chunk length | 30 s (fixed) | variable |
| N_SAMPLES per chunk | 480,000 | variable |
| N_FFT | 400 | 2,048 |
| Hop length | 160 (10 ms) | 300 |
| Window length | 400 (25 ms) | 1,200 |
| Window function | Hann periodic | Hann periodic |
| N_FRAMES per chunk | 3000 (after drop) | variable |
| Mel bins | 80 / 128 | 80 |
| Mel freq range | 0 - 8000 Hz | 0 - 12000 Hz |
| Filterbank shape | [80 or 128, 201] | [80, 1025] |
| Log floor (eps) | 1e-10 | (depends on usage) |
| Log base | 10 | natural |
| Dynamic range clamp | 8.0 log10 units | none |
| Normalization | (x+4)/4 | none |

## Algorithm Steps

### Whisper Mel Computation (Reference)

```
INPUT: audio (float32, mono, 16000 Hz)
1. Resample to 16000 Hz if needed (polyphase, Kaiser β=8.6)
2. Zero-pad to 480000 samples on the right
3. Compute Hann window (periodic, length 400)
4. STFT: window=400, hop=160, n_fft=400, center=True
   -> complex tensor [201, 3001]
5. Drop last frame: [201, 3000]
6. Power spectrogram: |STFT|^2 -> real tensor [201, 3000]
7. Apply mel filterbank: [80 or 128, 201] @ [201, 3000] -> [80 or 128, 3000]
8. Log compress: log10(max(mel, 1e-10))
9. Dynamic range: max(log_spec, max(log_spec) - 8.0)
10. Normalize: (log_spec + 4.0) / 4.0
11. Add batch dim: [1, 80 or 128, 3000]
OUTPUT: log-mel spectrogram
```

### Slaney Mel Filterbank Construction

```
INPUT: n_mels, n_fft, sample_rate, f_min=0, f_max=sample_rate/2
1. mel_low = hz_to_mel(f_min)
2. mel_high = hz_to_mel(f_max)
3. mel_points = linspace(mel_low, mel_high, n_mels + 2)
4. hz_points = mel_to_hz(mel_points)                      # length n_mels + 2
5. bin_freqs = linspace(0, sample_rate / 2, n_fft//2 + 1)
6. filterbank = zeros(n_mels, n_fft//2 + 1)
7. for i in 0..n_mels:
     left, center, right = hz_points[i], hz_points[i+1], hz_points[i+2]
     for k in 0..n_fft//2:
       if bin_freqs[k] < left or bin_freqs[k] > right:
         continue
       if bin_freqs[k] < center:
         filterbank[i, k] = (bin_freqs[k] - left) / (center - left)
       else:
         filterbank[i, k] = (right - bin_freqs[k]) / (right - center)
     # Slaney area normalization
     enorm = 2.0 / (hz_points[i+2] - hz_points[i])
     filterbank[i, :] *= enorm
OUTPUT: [n_mels, n_fft//2 + 1] mel filterbank
```

### Polyphase Resampling

```
INPUT: audio (in_rate Hz), target_rate
1. ratio = target_rate / gcd(target_rate, in_rate)
   up = target_rate / gcd
   down = in_rate / gcd
2. taps = num_taps (default 64; 256 for music quality)
3. cutoff = 0.5 / max(up, down)
4. h = windowed_sinc(taps * up, cutoff, kaiser_beta=8.6)
   # h has length taps * up; reshape to [up, taps] for polyphase
5. h_poly = reshape h to [up, taps]
6. for output sample m:
     phase = m % up
     in_idx = (m // up) * down
     y[m] = sum(audio[in_idx + k] * h_poly[phase, k] for k in 0..taps)
OUTPUT: audio at target_rate Hz
```

## Reference Implementations

- [OpenAI Whisper audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py) — canonical mel pipeline, `log_mel_spectrogram()` function.
- [whisper.cpp whisper.cpp:log_mel_spectrogram](https://github.com/ggml-org/whisper.cpp/blob/master/whisper.cpp) — C reference, custom FFT, validation target.
- [HuggingFace WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) — runtime filterbank generation, identical math.
- [librosa.filters.mel](https://librosa.org/doc/main/generated/librosa.filters.mel.html) — original Slaney filterbank reference.
- [torchaudio MelSpectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html) — PyTorch reference, parameterized for arbitrary configs.
- [scipy.signal.windows.hann](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html) — periodic Hann reference.
- [NVIDIA NeMo audio_to_mel_spectrogram_preprocessor](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py) — Parakeet/Canary mel pipeline (per-feature normalization).

## Differences Between Implementations

| Aspect | OpenAI Whisper | whisper.cpp | HF Whisper | NeMo |
|--------|----------------|-------------|------------|------|
| Filterbank source | Pre-computed npz | Compute at load | Compute at runtime | Compute at runtime |
| Mel scale | Slaney (via librosa) | Slaney | Slaney | Slaney |
| FFT | torch.stft | Custom radix-2 | numpy/torch | torch.stft |
| Normalization | Identical to OpenAI | Identical | Identical | Per-feature mean/std |
| Output dtype | float32 | float32 / float16 (q) | float32 | float32 |

Whisper implementations agree bit-for-bit (within float32 rounding) when parameters and normalization are matched. NeMo (Parakeet) uses different normalization — feature-by-feature mean/var standardization from training stats.

## Open Questions

- [ ] FFT implementation choice for C# — options: (a) MathNet.Numerics (managed, ~5x slower than native), (b) implement radix-2 Cooley-Tukey in SIMD (Sse2/Avx2/Avx512), (c) cuFFT/Vulkan FFT on GPU. For n_fft=400 (Whisper), the FFT is tiny — a hand-rolled radix-2 in SIMD with zero-pad to 512 is the right call. For n_fft=2048+ (music), an FFT compute kernel on GPU starts to matter.
- [ ] Whether to ship a pre-computed filterbank table per model (faster startup) or always compute at startup (no asset shipping). Lean toward compute-at-startup — it's ~milliseconds.
- [ ] Polyphase resample quality — 64 taps for STT is fine. For music models that take 44.1 kHz input, do we resample once on file load to 24 kHz/16 kHz, or do all internal compute at native sample rate? Decision belongs in each music model's pipeline.

## Implementation Notes for SharpInference

1. **Hann window must be periodic** — `w[n] = 0.5 * (1 - cos(2 * PI * n / N))` where N is the divisor (NOT N-1). Easy to get wrong. Validate by hashing window values against PyTorch's `torch.hann_window(N, periodic=True)`.

2. **Slaney mel scale, Slaney area normalization** — never HTK. Validate filterbank against librosa output element-wise (1e-6 tolerance).

3. **Use power spectrum for Whisper** — `|STFT|^2`, not `|STFT|`. Other models (Kokoro, HiFiGAN) use magnitude (`|STFT|`); document per-model.

4. **Log base** — Whisper uses log10. Most other models use log_e. The dynamic range clamp (8.0 for Whisper = 80 dB) is base-dependent — convert appropriately if reusing across models.

5. **Drop last STFT frame** for Whisper. The `center=True` STFT produces `n_samples/hop + 1` frames; we want `n_samples/hop`.

6. **Pure C# implementation path**:
   - **FFT**: hand-roll radix-2 in `System.Runtime.Intrinsics` (Sse2/Avx2). For Whisper's n_fft=400, pad to 512 and do a 512-point radix-2. Pre-compute twiddle factors at module init.
   - **Resampler**: pure C# polyphase with windowed sinc, pre-compute sinc table.
   - **Filterbank**: pure C# allocation as `Tensor`; computed at startup, cached in model handle.
   - **Mel matmul**: route through `IBackend.MatMul`. Filterbank is a small `[80, 201]` tensor; preload to GPU at model load.

7. **Validation**: write a Python script that dumps `MelExtractor.forward(test_audio)` from HF Whisper for a known test clip; compare C# output element-wise with tolerance 1e-4.

8. **No managed array allocations on hot path** — mel preprocessing runs on every audio chunk. Pre-allocate scratch buffers (`stft_real[201, 3001]`, `power[201, 3000]`, `mel[80, 3000]`) at pipeline construction.

9. **Cross-references**: [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md) for the model that consumes this; [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for incremental mel; [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) for the StyleTTS2/Kokoro 24 kHz mel parameters.
