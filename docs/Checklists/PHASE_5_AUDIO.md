# Phase 5 — Audio (Whisper + Kokoro)

> **Goal:** Whisper STT and Kokoro TTS working end-to-end.
> **Packages:** SharpInference.Audio

---

## 1. Research

- [x] WHISPER_ARCHITECTURE, MEL_SPECTROGRAM, HIFIGAN_VOCODER
- [ ] KOKORO_ARCHITECTURE — **still Draft**

## 2. Planning

- [ ] Whisper model sizes (tiny→large-v3), audio preprocessing flow
- [ ] Streaming STT chunking (chunk size, overlap, context carry)
- [ ] Kokoro phoneme conversion (espeak-ng vs built-in G2P), TTS streaming output

## 3. Audio Preprocessing

- [ ] `AudioPreprocessor.cs` — resample 16kHz, normalize, windowing
- [ ] `StftProcessor.cs` — Cooley-Tukey FFT + STFT + Hann window
- [ ] `MelSpectrogramProcessor.cs` — mel filterbank, log compression, normalization
- [ ] PTX: `fft_radix2.ptx`, `mel_filterbank.ptx`

## 4. Whisper STT

- [ ] `WhisperEncoder.cs` — Conv1D feature extractor + transformer encoder
- [ ] `WhisperDecoder.cs` — autoregressive decoder + cross-attention + KV cache
- [ ] `WhisperPipeline.cs` — audio → mel → encode → decode → transcript
- [ ] `WhisperStreamingPipeline.cs` — chunk-by-chunk real-time
- [ ] `WhisperOptions.cs` — language, task, timestamps, model size
- [ ] Timestamp token decoding, language auto-detection

## 5. Kokoro TTS

- [ ] `KokoroPhonemeEncoder.cs`, `KokoroPipeline.cs`
- [ ] `HiFiGanVocoder.cs` — mel → waveform (upsampling + dilated conv)
- [ ] `VocosVocoder.cs` (alternative), `TtsOptions.cs`, streaming output

## 6. Voice Conversion (stubs)

- [ ] `RvcPipeline.cs`, `F0Extractor.cs`

## 7. Testing

- [ ] STFT vs NumPy/SciPy, mel vs whisper.cpp (1e-4)
- [ ] Whisper encoder/decoder vs reference, full pipeline WER < 1%
- [ ] Streaming match, timestamps ±50ms, Kokoro mel tolerance
- [ ] HiFiGAN round-trip, 1hr memory leak test
- [ ] All tests pass on CI

## 8. Review & Merge

- [ ] Code review (audio buffer boundaries, streaming thread safety)
- [ ] Benchmark Whisper RTF per model size, Kokoro latency
- [ ] Merge to main branch
