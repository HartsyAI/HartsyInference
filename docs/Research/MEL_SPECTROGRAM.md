# Mel Spectrogram — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Audio

## Summary

Whisper's audio preprocessing converts raw audio into a log-mel spectrogram with very specific parameters that must be matched exactly — even small deviations cause transcription errors. The pipeline is: resample to 16kHz -> zero-pad to 30s -> STFT (N_FFT=400, hop=160, Hann window) -> power spectrogram -> mel filterbank projection (80 or 128 bins, Slaney scale) -> log10 + dynamic range compression + normalization. Validation against whisper.cpp within 1e-4 tolerance is the acceptance criterion.

Sources: [OpenAI Whisper audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py)

## Detailed Findings

### Exact Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| N_FFT (window size) | 400 samples (25ms) |
| Hop length | 160 samples (10ms) |
| Window function | Hann (periodic) |
| Mel bins | 80 (tiny-large-v2) or 128 (large-v3, turbo) |
| Chunk length | 30 seconds |
| N_SAMPLES | 480,000 (16000 * 30) |
| N_FRAMES | 3000 (480000 / 160) |

### STFT Computation

From [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```python
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=hann_window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2   # power spectrogram, drop last frame
mel_spec = mel_filters @ magnitudes       # project to mel space
```

Key: last STFT frame is DROPPED, and `abs()^2` computes POWER spectrogram (squared magnitudes).

### Mel Filterbank

**Frequency range**: 0 Hz to 8000 Hz (Nyquist for 16kHz).

**Mel scale**: Slaney scale with Slaney area normalization (used by both OpenAI via librosa and HuggingFace).

Slaney mel scale formulas:
```
Hz to Mel:
  if f < 1000:  mel = 3 * f / 200
  if f >= 1000: mel = 15 + 27 * log(f / 1000) / log(6.4)

Mel to Hz:
  if mel < 15:  f = 200 * mel / 3
  if mel >= 15: f = 1000 * exp((mel - 15) * log(6.4) / 27)
```

**Filterbank shape**: [n_mels, N_FFT/2 + 1] = [80, 201] or [128, 201]

**Pre-computed**: OpenAI loads from `whisper/assets/mel_filters.npz` (generated with librosa). HuggingFace generates at runtime with `mel_filter_bank(mel_scale="slaney", norm="slaney")`.

### Log Compression (EXACT formula)

From [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```python
log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Step by step:
1. Floor clamp: `max(mel_spec, 1e-10)` — prevents log(0)
2. Log10: base-10 logarithm (NOT natural log)
3. Dynamic range: clamp to within 8.0 log10 units (80 dB) below max
4. Normalize: `(log_spec + 4.0) / 4.0` — maps typical range to ~[0, 1]

### Normalization

Global max-relative within each spectrogram:
- max_val = log_spec.max() across entire spectrogram
- Clamp to max_val - 8.0
- Shift and scale: (log_spec + 4.0) / 4.0

NO per-channel mean subtraction. NO per-channel normalization.

### Padding

Audio shorter than 30 seconds is zero-padded on the right to exactly 480,000 samples. Audio longer than 30s is processed in 30-second chunks.

### Output Tensor Shape

```
Input audio:        [480000]           (30s * 16000 Hz)
After STFT:         [201, 3001]        (N_FFT/2+1, N_SAMPLES/HOP+1)
Drop last frame:    [201, 3000]
After mel filters:  [80, 3000]         (or [128, 3000])
After log+norm:     [80, 3000]
Batched:            [1, 80, 3000]      (or [1, 128, 3000])
```

## Key Numbers / Constants

| Constant | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| N_FFT | 400 |
| Hop length | 160 |
| Mel bins (standard) | 80 |
| Mel bins (large-v3) | 128 |
| Mel freq range | 0 - 8000 Hz |
| Log floor | 1e-10 |
| Dynamic range | 8.0 (log10 units = 80 dB) |
| Normalization offset | +4.0 |
| Normalization scale | /4.0 |
| N_SAMPLES (30s) | 480,000 |
| N_FRAMES (30s) | 3,000 |
| Filterbank shape | [80, 201] or [128, 201] |

## Algorithm Steps

### Complete Mel Spectrogram Pipeline

```
1. Resample audio to 16kHz if needed
2. Zero-pad to 480,000 samples (30s) on the right
3. Apply Hann window (periodic, length 400)
4. Compute STFT: window=400, hop=160 -> complex [201, 3001]
5. Drop last frame -> [201, 3000]
6. Compute power spectrogram: |STFT|^2 -> real [201, 3000]
7. Apply mel filterbank: [n_mels, 201] @ [201, 3000] -> [n_mels, 3000]
8. Log compress: log10(max(mel, 1e-10))
9. Dynamic range: max(log_spec, max(log_spec) - 8.0)
10. Normalize: (log_spec + 4.0) / 4.0
11. Output: [1, n_mels, 3000]
```

### Mel Filterbank Construction (Slaney)

```
1. Compute n_mels+2 mel-spaced center frequencies between 0 and 8000 Hz
2. Convert center frequencies to FFT bin indices
3. For each of n_mels filters:
   a. Create triangular filter from center[i] to center[i+2], peak at center[i+1]
   b. Normalize by mel band width (Slaney area normalization)
4. Result: [n_mels, N_FFT/2+1] filterbank matrix
```

## Reference Implementations

| Implementation | Location | Notes |
|---------------|----------|-------|
| OpenAI Whisper | [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py) | Canonical. Uses pre-computed filterbank from mel_filters.npz |
| whisper.cpp | [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp) | C++ reference. Validation target (1e-4 tolerance) |
| HuggingFace | [feature_extraction_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) | Runtime filterbank generation |
| librosa | [filters.mel](https://librosa.org/doc/main/generated/librosa.filters.mel.html) | Original filterbank generation reference |

## Differences Between Implementations

| Aspect | OpenAI | whisper.cpp | HuggingFace |
|--------|--------|-------------|-------------|
| Filterbank | Pre-computed (mel_filters.npz) | Computed at load | Computed at runtime |
| Mel scale | Slaney (via librosa) | Slaney | Slaney |
| STFT | torch.stft | Custom FFT | numpy/torch |
| Normalization | Identical | Identical | Identical |

## Open Questions

- [x] ~~Pre-computed vs runtime filterbank~~ — OpenAI pre-computes, HuggingFace generates at runtime. Both produce identical results.
- [x] ~~Normalization type~~ — Global max-relative, NOT per-channel
- [x] ~~Padding behavior~~ — Zero-pad right to 480,000 samples
- [ ] Whether to ship pre-computed filterbank or generate at startup (generate recommended — avoids asset dependency)
- [ ] FFT implementation choice for C# (MathNet.Numerics vs custom)

## Implementation Notes

1. **Use Slaney mel scale** — both OpenAI and HuggingFace use this. HTK scale will produce wrong results.
2. **Power spectrum, not magnitude** — use |STFT|^2, not |STFT|
3. **Drop last STFT frame** — `stft[..., :-1]` is critical for correct shape
4. **Log10, not natural log** — the formula uses base-10 logarithm
5. **Generate filterbank at startup** — avoid shipping mel_filters.npz asset. Compute once and cache.
6. **Validation** — compare output against whisper.cpp mel computation within 1e-4 tolerance per element
7. **Support both 80 and 128 mel bins** — parameterize by model config
8. **Hann window** — use periodic Hann window (not symmetric). In C#: `w[n] = 0.5 * (1 - cos(2*PI*n/N))` where N=400
9. **FFT** — need N_FFT=400 point FFT. Can use MathNet.Numerics or implement radix-2 with zero-padding to 512
