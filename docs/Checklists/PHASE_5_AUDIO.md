# Phase 5 — Audio (Whisper + Kokoro)

> **Goal:** Whisper STT and Kokoro TTS working end-to-end.
> **Packages:** SharpInference.Audio

---

## 1. Research

- [x] Complete [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) research — done and verified
- [x] Complete [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) research — done and verified
- [ ] Complete [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) research — **still Draft**
- [x] Complete [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) research — done and verified

## 2. Planning

- [ ] Map Whisper model sizes (tiny → large-v3) — layer counts, dimensions, parameters
- [ ] Plan audio preprocessing pipeline data flow
- [ ] Plan streaming STT chunking strategy (chunk size, overlap, context carry)
- [ ] Plan Kokoro phoneme conversion approach (espeak-ng vs built-in G2P)
- [ ] Plan TTS streaming output (chunk size, latency tradeoff)
- [ ] Write agent instructions for Phase 5

## 3. Implementation — Audio Preprocessing

- [ ] `AudioPreprocessor.cs` — resample to 16kHz, normalize, windowing
- [ ] `StftProcessor.cs` — Cooley-Tukey FFT, STFT with Hann window
- [ ] `MelSpectrogramProcessor.cs` — mel filterbank construction, log compression, normalization
- [ ] CUDA PTX: `fft_radix2.ptx` — FFT on GPU
- [ ] CUDA PTX: `mel_filterbank.ptx` — mel filter application on GPU

## 4. Implementation — Whisper STT

- [ ] `WhisperEncoder.cs` — Conv1D feature extractor + transformer encoder blocks
- [ ] `WhisperDecoder.cs` — autoregressive decoder with cross-attention + KV cache
- [ ] `WhisperPipeline.cs` — full pipeline: audio → mel → encode → decode → transcript
- [ ] `WhisperStreamingPipeline.cs` — chunk-by-chunk real-time transcription
- [ ] `WhisperOptions.cs` — language, task (transcribe/translate), timestamps, model size
- [ ] Timestamp token decoding (word-level timing)
- [ ] Language auto-detection from first encoder pass

## 5. Implementation — Kokoro TTS

- [ ] `KokoroPhonemeEncoder.cs` — text → phoneme conversion + phoneme embedding
- [ ] `KokoroPipeline.cs` — phonemes → acoustic model → mel spectrogram
- [ ] `HiFiGanVocoder.cs` — mel → waveform synthesis (upsampling + dilated conv)
- [ ] `VocosVocoder.cs` — alternative vocoder (if Kokoro uses it)
- [ ] `TtsOptions.cs` — voice selection, speed, output format
- [ ] Streaming TTS output — synthesize and emit audio chunks progressively

## 6. Implementation — Voice Conversion (stub)

- [ ] `RvcPipeline.cs` — stub for RVC v2 (Phase 2 model)
- [ ] `F0Extractor.cs` — stub for pitch extraction

## 7. Testing & Validation

- [ ] `StftTests.cs` — compare FFT output to NumPy/SciPy reference
- [ ] `MelSpectrogramTests.cs` — compare mel output to whisper.cpp within 1e-4
- [ ] `WhisperEncoderTests.cs` — same mel input → same encoder output as reference
- [ ] `WhisperDecoderTests.cs` — same encoder output → same token sequence
- [ ] `WhisperIntegrationTests.cs` — transcribe 10 test audio files, WER < 1% vs whisper.cpp
- [ ] Whisper streaming test — chunked transcription matches full-file transcription
- [ ] Whisper timestamp test — word timestamps within ±50ms of reference
- [ ] Kokoro TTS — same text → mel spectrogram within tolerance of reference
- [ ] Kokoro TTS — generated audio is intelligible (manual listening test)
- [ ] HiFiGAN — mel → waveform → mel round-trip consistency
- [ ] Memory test — transcribe 1 hour of audio, verify no memory leak
- [ ] All tests pass on CI

## 8. Review & Merge

- [ ] Code review — audio buffer management (no clicks/pops from buffer boundaries)
- [ ] Code review — streaming pipeline thread safety
- [ ] Benchmark: Whisper RTF (real-time factor) for each model size
- [ ] Benchmark: Kokoro TTS latency (time to first audio chunk)
- [ ] Merge to main branch
