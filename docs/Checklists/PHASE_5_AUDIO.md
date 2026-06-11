# Phase 5 — Audio (STT + TTS + Music)

> **Goal:** End-to-end inference for SOTA STT, TTS, and music-generation models in pure C#.
> **Packages:** SharpInference.Audio (STT + TTS), SharpInference.Music (music generation)
> **Research scope:** all 27 audio research docs landed 2026-05-17. See [RESEARCH_REQUIREMENTS.md](../Design/RESEARCH_REQUIREMENTS.md) "Audio" sections.

---

## 0. Scope at a Glance

| Group | Models | Research docs |
|---|---|---|
| Shared infra | mel preprocessing, audio codecs, flow-matching, G2P, streaming, HiFiGAN/Vocos | 6 docs ✅ |
| STT | Whisper (all sizes + turbo + Distil), Parakeet (CTC/RNN-T/TDT), Canary, Moonshine, SenseVoice, FireRedASR | 5 docs ✅ |
| TTS | Kokoro, F5-TTS, XTTS-v2, Bark, CosyVoice 1/2, IndexTTS 1.5/2, SparkTTS, ChatTTS, Higgs Audio v2, Sesame CSM, GPT-SoVITS, OpenVoice v2, MeloTTS, StyleTTS 2, VibeVoice (1.5B / 7B / Streaming-0.5B) | 15 docs ✅ |
| Music | ACE-Step (v1/v1.5/XL), Stable Audio Open (1.0/Small/2), MusicGen + AudioGen, YuE, DiffRhythm, AudioLDM 2 | 6 docs ✅ |

All 31 audio research docs are complete (see `docs/Research/`). No further research is required before implementation can begin.

---

## 1. Research

### Shared infrastructure
- [x] [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) — STFT, Hann window, Slaney mel filterbank, log compression, polyphase resampling
- [x] [AUDIO_CODECS.md](../Research/AUDIO_CODECS.md) — EnCodec / DAC / Mimi / SNAC / WavTokenizer / SpeechTokenizer / XCodec / BigCodec
- [x] [FLOW_MATCHING_AUDIO.md](../Research/FLOW_MATCHING_AUDIO.md) — Sway sampling (F5), ACE-Step omega/APG/CFG-Zero*, pingpong (Stable Audio Small)
- [x] [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md) — CMUDict + ARPABET→IPA + misaki + per-language strategy + neural OOV
- [x] [STREAMING_AUDIO_INFERENCE.md](../Research/STREAMING_AUDIO_INFERENCE.md) — chunked-and-overlap, KV-cache append-and-grow, ring buffers, latency budgets
- [x] [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) — HiFiGAN V1/V2/V3, iSTFTNet (Kokoro/StyleTTS2), Vocos (F5/ChatTTS)

### STT model architectures
- [x] [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) — tiny/base/small/medium/large-v2/v3/turbo + Distil-Whisper variants
- [x] [PARAKEET_ARCHITECTURE.md](../Research/PARAKEET_ARCHITECTURE.md) — FastConformer + CTC/RNN-T/TDT heads
- [x] [CANARY_ARCHITECTURE.md](../Research/CANARY_ARCHITECTURE.md) — FastConformer + Transformer decoder + AST translation
- [x] [MOONSHINE_ARCHITECTURE.md](../Research/MOONSHINE_ARCHITECTURE.md) — raw-waveform front-end + RoPE encoder/decoder
- [x] [SENSEVOICE_FIREREDASR_ARCHITECTURE.md](../Research/SENSEVOICE_FIREREDASR_ARCHITECTURE.md) — non-AR CTC (SenseVoice) + Conformer+AED/LLM (FireRedASR)

### TTS model architectures
- [x] [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) — StyleTTS2-based 82M
- [x] [F5_TTS_ARCHITECTURE.md](../Research/F5_TTS_ARCHITECTURE.md) — DiT + flow matching (Sway sampling) + Vocos
- [x] [XTTS_ARCHITECTURE.md](../Research/XTTS_ARCHITECTURE.md) — GPT + HiFiGAN, multilingual zero-shot clone
- [x] [BARK_ARCHITECTURE.md](../Research/BARK_ARCHITECTURE.md) — 3-stage GPT + EnCodec 24kHz 8-codebook
- [x] [COSYVOICE_ARCHITECTURE.md](../Research/COSYVOICE_ARCHITECTURE.md) — Qwen2.5-0.5B + FSQ codec + CFM + HiFTNet
- [x] [INDEX_TTS_ARCHITECTURE.md](../Research/INDEX_TTS_ARCHITECTURE.md) — GPT T2S + S2Mel DiT + Vocos codec + BigVGAN
- [x] [SPARK_TTS_ARCHITECTURE.md](../Research/SPARK_TTS_ARCHITECTURE.md) — Qwen2.5-0.5B + BiCodec (semantic VQ + global FSQ)
- [x] [CHATTTS_ARCHITECTURE.md](../Research/CHATTTS_ARCHITECTURE.md) — LLaMA-style GPT + dilated ConvNeXt decoder + Vocos
- [x] [HIGGS_AUDIO_ARCHITECTURE.md](../Research/HIGGS_AUDIO_ARCHITECTURE.md) — Llama-3.2-3B + DualFFN + HuBERT+DAC dual-branch codec
- [x] [SESAME_CSM_ARCHITECTURE.md](../Research/SESAME_CSM_ARCHITECTURE.md) — Llama-3.2-1B + small decoder + Mimi codec (12.5 Hz)
- [x] [GPT_SOVITS_ARCHITECTURE.md](../Research/GPT_SOVITS_ARCHITECTURE.md) — GPT T2S + SoVITS SynthesizerTrn + HuBERT
- [x] [OPENVOICE_ARCHITECTURE.md](../Research/OPENVOICE_ARCHITECTURE.md) — MeloTTS stage 1 + Tone Color Converter flow
- [x] [MELOTTS_ARCHITECTURE.md](../Research/MELOTTS_ARCHITECTURE.md) — VITS-based + BERT auxiliary + per-language phoneme front-end
- [x] [STYLETTS2_ARCHITECTURE.md](../Research/STYLETTS2_ARCHITECTURE.md) — Kokoro parent + diffusion style sampler + speech encoder
- [x] [VIBEVOICE_ARCHITECTURE.md](../Research/VIBEVOICE_ARCHITECTURE.md) — Qwen2.5 LM (1.5B/7B/0.5B) + next-token DDPM head + 7.5 Hz causal acoustic/semantic VAEs + multi-speaker prompt format (long-form 90 min, MIT license)

### Music generation architectures
- [x] [ACE_STEP_ARCHITECTURE.md](../Research/ACE_STEP_ARCHITECTURE.md) — v1 (3.5B DiT) + v1.5 (Qwen3-based FSQ AR) + XL 5B + Music DCAE + ADaMoSHiFiGAN
- [x] [STABLE_AUDIO_ARCHITECTURE.md](../Research/STABLE_AUDIO_ARCHITECTURE.md) — Open 1.0 / Small / 2 — Oobleck VAE + timing-conditioned DiT
- [x] [MUSICGEN_ARCHITECTURE.md](../Research/MUSICGEN_ARCHITECTURE.md) — decoder-LM + EnCodec + delay pattern + mono/melody/stereo variants
- [x] [YUE_ARCHITECTURE.md](../Research/YUE_ARCHITECTURE.md) — Llama-2 7B (S1) + 1B (S2) + xcodec, dual-track full-song generation
- [x] [DIFFRHYTHM_ARCHITECTURE.md](../Research/DIFFRHYTHM_ARCHITECTURE.md) — Stable-Audio-derived VAE + 1.1B DiT + MuQ-MuLan conditioning
- [x] [AUDIOLDM2_ARCHITECTURE.md](../Research/AUDIOLDM2_ARCHITECTURE.md) — CLAP + T5 + GPT-2 prefix + UNet latent diffusion + SpeechT5HiFiGAN

---

## 2. Planning

- [ ] Decide STT model rollout order — recommend **Whisper first** (existing research, broad coverage), then **Parakeet-TDT** (real-time English), then **Moonshine** (edge / streaming), then **Canary** (multilingual + translation), then **SenseVoice + FireRedASR** (Chinese strong).
- [ ] Decide TTS model rollout order — recommend **Kokoro first** (simplest, no codec, deterministic voices), then **F5-TTS** (no G2P needed, flow matching reuses image scheduler infra), then **XTTS-v2** (multilingual character-level), then **CosyVoice 2** (streaming + Qwen reuse), then the rest. Slot **VibeVoice** after F5-TTS — it reuses Qwen2.5 from dotLLM, the same DPM++ v-prediction cosine scheduler F5 already lives next to, and AdaLN-Zero block patterns from Flux/SD3; novelty is the causal 1D-ConvNeXt VAEs and the per-token DDPM sub-loop. Build all three checkpoints (1.5B / 7B / Streaming-0.5B) behind one `VibeVoicePipeline` switched on `config.model_type` — they share the acoustic VAE, diffusion head, and DPM-Solver path.
- [ ] Decide music model rollout order — recommend **ACE-Step first** (flagship per user request, flow-matching reuses image pipeline patterns), then **Stable Audio Open** (similar tech stack, smaller scope), then **MusicGen** (clean AR + EnCodec reference), then **DiffRhythm / YuE** (full-song generation), then **AudioLDM 2** (text-to-audio for SFX).
- [ ] G2P language-coverage cut: **English only in v1** via CMUDict + ARPABET→IPA + heteronym table + small neural OOV fallback. Chinese (pinyin via jieba.NET port) and Japanese (NMeCab + UniDic repackaging) phased.
- [ ] Audio codec build order: **EnCodec 24kHz first** (Bark, MusicGen, AudioGen, AudioCraft), then **Mimi** (Sesame CSM streaming demo), then **DAC** (IndexTTS, Higgs Audio), then **xcodec** (YuE), then **Vocos** (covers F5/Kokoro variants/ChatTTS as vocoder too).
- [ ] Decide whether to package music separately under `SharpInference.Music` or keep all audio under `SharpInference.Audio`. Music brings full-song generation, long-context, and large LLM backbones that have a different memory profile.

## 3. Audio Preprocessing & Shared Primitives

- [x] `AudioPreprocessor.cs` — resample (polyphase, windowed sinc, Kaiser β=8.6, configurable taps), normalize, windowing  *(shipped as [Io/Resampler.cs](../../src/SharpInference.Audio/Io/Resampler.cs) + [Preprocessing/HannWindow.cs](../../src/SharpInference.Audio/Preprocessing/HannWindow.cs))*
- [x] `StftProcessor.cs` — radix-2 Cooley-Tukey FFT, STFT, periodic Hann window. **SIMD path landed for the largest stage** ([Preprocessing/Fft.cs](../../src/SharpInference.Audio/Preprocessing/Fft.cs) — Vector&lt;float&gt;-vectorized when step==1; scalar fallback for early stages)
- [x] `MelSpectrogramProcessor.cs` — Slaney mel filterbank, log compression, normalization variants ([Preprocessing/MelSpectrogramExtractor.cs](../../src/SharpInference.Audio/Preprocessing/MelSpectrogramExtractor.cs); Whisper / Kokoro / HiFiGAN presets)
- [x] `IStftLayer.cs` — inverse STFT for vocoders ([Models/Vocoders/IStft.cs](../../src/SharpInference.Audio/Models/Vocoders/IStft.cs))
- [x] `AudioRingBuffer.cs` — circular PCM buffer ([Streaming/AudioRingBuffer.cs](../../src/SharpInference.Audio/Streaming/AudioRingBuffer.cs))
- [x] `StreamingMelExtractor.cs` — incremental STFT with context tail ([Streaming/StreamingMelExtractor.cs](../../src/SharpInference.Audio/Streaming/StreamingMelExtractor.cs))
- [x] `StreamingKvCache.cs` — per-layer K/V append-and-grow ([Streaming/StreamingKvCache.cs](../../src/SharpInference.Audio/Streaming/StreamingKvCache.cs))
- [ ] PTX kernels: `fft_radix2.ptx`, `mel_filterbank.ptx`, `istft_overlap_add.ptx`, `snake_activation.ptx`, `conv_transpose1d.ptx` — **CUDA source written**, awaiting nvcc build pass ([native/cuda/audio/](../../native/cuda/audio/)). Vulkan shaders source-only ([native/vulkan/shaders/snake.comp.glsl](../../native/vulkan/shaders/snake.comp.glsl) + conv1d/conv_transpose1d), awaiting glslc build pass.
- [x] `IAsyncEnumerable<AudioChunk>` output streaming surface ([Streaming/AudioStreamer.cs](../../src/SharpInference.Audio/Streaming/AudioStreamer.cs) + [Streaming/AudioChunk.cs](../../src/SharpInference.Audio/Streaming/AudioChunk.cs))
- [x] `LstmCell` / `BiLstm` modules ([Layers/LstmCell.cs](../../src/SharpInference.Audio/Layers/LstmCell.cs), [Layers/BiLstm.cs](../../src/SharpInference.Audio/Layers/BiLstm.cs)). Bonus: [Layers/UnidirectionalLstm.cs](../../src/SharpInference.Audio/Layers/UnidirectionalLstm.cs) for stacked multi-layer LSTM (EnCodec bottleneck).
- [x] **Shared DSP statics** ([`Dsp/`](../../src/SharpInference.Audio/Dsp/)) — **reuse these, don't re-roll per model** (see AGENTS.md "Reuse shared primitives"): [`NsfVocoderDsp`](../../src/SharpInference.Audio/Dsp/NsfVocoderDsp.cs) (NSF harmonic source + forward-STFT mag/phase + iSTFT exp/sin head + reflection-pad + add-cropped + scale — shared by Kokoro iSTFTNet + CosyVoice HiFTNet) and [`DeterministicRng`](../../src/SharpInference.Audio/Dsp/DeterministicRng.cs) (seeded xorshift+Box-Muller Gaussian/uniform — shared by NSF source, CFM/diffusion samplers, the speech-token sampler). Layout transpose `[1,C,T]↔[1,T,C]` is the existing `backend.Transpose2D` op (used 30+ places); the central linear is `WhisperOps.ProjectLinear`.

### Backend ops landed in 3.x (post-original-checklist additions)

- [x] `Conv1d` / `ConvTranspose1d` on `IBackend` — CPU complete, CUDA + Vulkan stubbed with native source files ready for build
- [x] `Sigmoid` / `Tanh` / `Elu` / `Snake` activations on `IBackend` — CPU complete; Vulkan `Sigmoid` working today via existing elementwise op 6; rest awaiting glslc/nvcc recompile

## 4. Neural Audio Codecs (`SharpInference.Audio.Codecs`)

- [x] `EnCodec24kHz` encoder + decoder ([Models/Codecs/EnCodec/](../../src/SharpInference.Audio/Models/Codecs/EnCodec/) — covers Bark, MusicGen, AudioGen)
- [x] `Mimi` decoder + encoder + transformer-of-codecs ([Models/Codecs/Mimi/](../../src/SharpInference.Audio/Models/Codecs/Mimi/) — covers Sesame CSM, Moshi)
- [x] `DAC` encoder + decoder for 44.1 / 24 / 16 kHz variants ([Models/Codecs/Dac/](../../src/SharpInference.Audio/Models/Codecs/Dac/) — covers IndexTTS, Higgs Audio, Spark-TTS HiFi-GAN)
- [x] `XCodec` decoder + encoder ([Models/Codecs/XCodec/](../../src/SharpInference.Audio/Models/Codecs/XCodec/) — covers YuE; lifts DAC verbatim with codec-specific config)
- [x] `Vocos` (mel-input) ([Models/Vocoders/Vocos.cs](../../src/SharpInference.Audio/Models/Vocoders/Vocos.cs) — pre-existing for F5-TTS)
- [x] `SNAC` decoder + encoder with hierarchical RVQ ([Models/Codecs/Snac/](../../src/SharpInference.Audio/Models/Codecs/Snac/) — covers Orpheus TTS; 24/32/44.1 kHz presets)
- [x] `WavTokenizer` encoder + decoder with iSTFT head ([Models/Codecs/WavTokenizer/](../../src/SharpInference.Audio/Models/Codecs/WavTokenizer/) — single 4096-entry codebook)
- [x] `BiCodec` semantic + global encoders ([Models/Codecs/BiCodec/](../../src/SharpInference.Audio/Models/Codecs/BiCodec/) — covers Spark-TTS)
- [x] FSQ codec primitives ([Models/Codecs/Fsq.cs](../../src/SharpInference.Audio/Models/Codecs/Fsq.cs) — parity-aware tanh bound + base-L packing)
- [x] Weight-norm fusion — runtime via [Models/Codecs/WeightNormFusion.cs](../../src/SharpInference.Audio/Models/Codecs/WeightNormFusion.cs); offline CLI at [samples/FuseWeightNorm/](../../samples/FuseWeightNorm/)
- [x] **Streaming codec wrapper** ([Models/Codecs/StreamingCodec.cs](../../src/SharpInference.Audio/Models/Codecs/StreamingCodec.cs)) — generic `StreamingCodecEncoder<T>` / `StreamingCodecDecoder<T>` over any of the 9 codecs, for live-mic encode and live-playback decode use cases.
- [x] Codec smoke tests — config + construction tests for every codec ([tests/SharpInference.Audio.Tests/CodecSmokeTests.cs](../../tests/SharpInference.Audio.Tests/CodecSmokeTests.cs))
- [ ] Validation: round-trip encode→decode STOI > 0.95 vs Python reference — **pending checkpoint downloads + per-codec integration tests**. Smoke-test scaffolding in place; the gated test attribute pattern from F5TtsSmokeTests is the convention to follow.

### Reusable codec infrastructure (bonus, not in original §4 checklist but enables it)

- [x] `ResidualVectorQuantizer` — generic euclidean RVQ for EnCodec ([Models/Codecs/ResidualVectorQuantizer.cs](../../src/SharpInference.Audio/Models/Codecs/ResidualVectorQuantizer.cs))
- [x] `DacResidualVectorQuantizer` — cosine RVQ with per-codebook in/out projections (DAC, XCodec, Mimi)
- [x] `SnacResidualVectorQuantizer` — hierarchical RVQ with per-codebook stride
- [x] `WeightNormFusionT` — transpose-conv variant of weight-norm fusion (separate axis handling)
- [x] `MimiTransformer` — small causal transformer-of-codecs (RoPE + MHA + GeLU FFN; reused on encoder + decoder sides of Mimi)

## 5. Speech-to-Text (STT)

### Whisper family — BUILT (greedy decode); long-form + word-timestamps pending
- [x] [`WhisperEncoder.cs`](../../src/SharpInference.Audio/Models/Whisper/WhisperEncoder.cs) — Conv1D × 2 + sinusoidal pos + N × ResidualAttentionBlock
- [x] [`WhisperDecoder.cs`](../../src/SharpInference.Audio/Models/Whisper/WhisperDecoder.cs) — embed + learned pos + N × cross-attention block + self-attn KV cache + `PrecomputeCrossKv`
- [x] [`WhisperPipeline.cs`](../../src/SharpInference.Audio/Pipelines/WhisperPipeline.cs) — mel → encode → cross-KV precompute → greedy decode loop with suppress-tokens. **Temperature fallback deferred** (greedy only in v1).
- [x] [`WhisperOptions.cs`](../../src/SharpInference.Audio/Pipelines/WhisperOptions.cs) — language, task (translate), timestamps, max-tokens; model size via `WhisperConfig`. **Beam search deferred** (greedy only).
- [x] Distil-Whisper variant support — `WhisperConfig.{DistilLargeV2, DistilLargeV3, DistilMediumEn, DistilSmallEn}` set `DecoderLayers = 2`, reuse the same encoder/decoder classes
- [x] Whisper-large-v3-turbo support — `WhisperConfig.LargeV3Turbo` (128 mel bins from `LargeV3`, 4-layer decoder)
- [ ] Long-form sequential decoding (timestamp-driven) + chunked variant — documented as a later pass in `WhisperPipeline`
- [ ] Word-level timestamps via cross-attention DTW (alignment heads table per model size)
- **Validated:** `WhisperEndToEndTests` transcribes canonical JFK audio (real end-to-end; skips cleanly when no cached model / network).

### NVIDIA NeMo family (`SharpInference.Audio.Nemo` or similar)
- [ ] `FastConformerEncoder.cs` — 8x conv subsampling + Conformer blocks with limited-context attention (shared by Parakeet, Canary, FireRedASR-AED)
- [ ] `CtcDecoder.cs` — greedy + beam search with blank collapse (Parakeet-CTC)
- [ ] `RnntDecoder.cs` — prediction net (LSTM) + joint net + hypothesis-extension beam (Parakeet-RNNT)
- [ ] `TdtDecoder.cs` — joint net with token + duration heads, greedy decode (Parakeet-TDT)
- [ ] `ParakeetPipeline.cs` — variant-dispatched
- [ ] `CanaryPipeline.cs` — FastConformer + Transformer decoder + AST prompt tokens
- [ ] Cache-aware streaming for Parakeet and Canary chunked modes

### Moonshine — BUILT
- [x] [`MoonshinePipeline.cs`](../../src/SharpInference.Audio/Pipelines/MoonshinePipeline.cs) — Conv1D front-end + RoPE encoder/decoder ([`MoonshineEncoder`](../../src/SharpInference.Audio/Models/Moonshine/MoonshineEncoder.cs)/[`MoonshineDecoder`](../../src/SharpInference.Audio/Models/Moonshine/MoonshineDecoder.cs)) + SentencePiece byte-fallback BPE ([`MoonshineTokenizer`](../../src/SharpInference.Audio/Models/Moonshine/MoonshineTokenizer.cs))
- [x] Hallucination guard — **approximated** via a token-count cap proportional to encoder-seconds (`MoonshinePipeline`); the dynamic rate-based ~6.5 tok/s ceiling is not yet enforced
- **Validated:** `MoonshineEndToEndTests` transcribes canonical JFK audio (real end-to-end; skip-gated on cache/network).

### SenseVoice + FireRedASR
- [ ] `SenseVoiceEncoder.cs` — 50-layer SANM encoder + LFR frontend + CTC head
- [ ] `SenseVoicePipeline.cs` — special-token parser (`<emotion><lang><event><text>`)
- [ ] `FireRedAsrPipeline.cs` (AED variant) — Conformer + Whisper-style decoder
- [ ] `FireRedAsrLlmPipeline.cs` (LLM variant) — Conformer + adapter + Qwen2 decoder (dotLLM dependency)

### .nemo file format support
- [ ] `NemoFileLoader.cs` — extract tar to {ckpt, config.yaml}; route to model loader
- [ ] Offline converter: `.nemo` → safetensors + config.json for our standard pipeline

## 6. Text-to-Speech (TTS)

### Kokoro (first TTS to ship) — BUILT (real iSTFTNet audio); G2P pending
- [x] `PlBertEncoder` — [`KokoroPlBert.cs`](../../src/SharpInference.Audio/Models/Kokoro/KokoroPlBert.cs): ALBERT with weight sharing — one shared `AlbertLayer` instance looped 12× in the forward pass
- [x] [`KokoroTextEncoder.cs`](../../src/SharpInference.Audio/Models/Kokoro/KokoroTextEncoder.cs) — Embed + Conv1D × 3 (ChannelLN + LeakyReLU) + BiLSTM
- [x] [`KokoroProsodyPredictor.cs`](../../src/SharpInference.Audio/Models/Kokoro/KokoroProsodyPredictor.cs) — DurationEncoder (3× BiLSTM + AdaLayerNorm) + duration LSTM + projection + F0/N AdaINResBlock chains
- [x] [`KokoroIStftNetDecoder.cs`](../../src/SharpInference.Audio/Models/Kokoro/KokoroIStftNetDecoder.cs) — **real iSTFTNet generator forward** (replaces the prior sine placeholder). Full chain: F0/N convs + asr_res → encode/decode AdainResBlk1d → HnNSF harmonic source (9 harmonics, deterministic phase accumulation + fixed-seed Gaussian noise, `tanh(Linear[9→1])`) → forward STFT → two ConvTranspose1d upsamples (10×, 6×) with per-level noise injection + 3-kernel MRF AdaIN/**Snake** resblocks → `conv_post` → magnitude(`exp`)/phase(`sin`) iSTFT head. Mirrors StyleTTS2 `kokoro/istftnet.py` Decoder+Generator. Required a non-power-of-two direct-DFT fallback in [`Fft.cs`](../../src/SharpInference.Audio/Preprocessing/Fft.cs) for the n_fft=20 synthesis transform. See [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md).
- [x] [`KokoroPipeline.cs`](../../src/SharpInference.Audio/Pipelines/KokoroPipeline.cs) — end-to-end `Synthesize` (voice-pack load → tokenize → PLBERT → TextEncoder → duration predict → length-regulate → F0/N predict → decoder). Now produces real iSTFTNet speech.
- [ ] G2P backed by [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md) (English-first) — **not built**; the pipeline accepts pre-phonemized IPA strings only (callers phonemize externally via misaki / eSpeak-NG). `KokoroPhonemeTokenizer` is IPA→token, not text→IPA. **Next Kokoro deliverable.**
- **Validated:** `KokoroFoundationTests` + `KokoroPipelineSmokeTests` (6 tests) load all 548 tensors and run the full real generator to finite non-degenerate 24 kHz audio; skip cleanly when the cache is missing. Numeric reference-diff vs Python (§9) still pending a checkpoint-paired run.

### F5-TTS — BUILT
- [x] `F5TtsDiT` — [`F5Dit.cs`](../../src/SharpInference.Audio/Models/F5Tts/F5Dit.cs) + [`F5DitBlock.cs`](../../src/SharpInference.Audio/Models/F5Tts/F5DitBlock.cs): 22 DiT blocks, AdaLN-Zero modulation, RoPE on Q/K, ConvNeXt stem; full `Forward` runs timestep + text + input embed → blocks → AdaLN head + proj
- [x] [`F5TtsPipeline.cs`](../../src/SharpInference.Audio/Pipelines/F5TtsPipeline.cs) — flow-matching Euler with CFG + in-context infilling; re-clamps the reference mel portion each step (ref-overwrite)
- [x] Sway sampling scheduler — [`F5SwaySamplingScheduler.cs`](../../src/SharpInference.Audio/Models/F5Tts/F5SwaySamplingScheduler.cs) (sway-warped Euler timesteps)
- [x] `vocos-mel-24khz` vocoder — [`Vocos.cs`](../../src/SharpInference.Audio/Models/Vocoders/Vocos.cs): Conv embed → 8 ConvNeXt blocks → mag/phase head → iSTFT
- **Validated:** `F5TtsSmokeTests` + `F5ForwardLiveTest` (skip/early-exit when checkpoint absent).

### XTTS-v2
- [ ] `XttsGpt.cs` — GPT-2-style 30L × 1024
- [ ] `XttsTokenizer.cs` — shared BPE + `[<lang>]` prefix tokens + per-language romanizers (pinyin, cutlet)
- [ ] `XttsConditioningEncoder.cs` — Perceiver resampler → 32×1024 gpt_cond_latent
- [ ] `XttsSpeakerEncoder.cs` — H/ASP ECAPA-TDNN → 512-d
- [ ] `XttsHifiGan.cs` — takes 1024-d GPT latents (not mel), FiLM-conditioned per ResBlock
- [ ] Streaming variant (chunk_size=20 mel tokens, overlap_wav_chunks=1024)

### Bark — SCAFFOLD COMPLETE (3-stage cascade); checkpoint-gated
> Files under [`Models/Bark/`](../../src/SharpInference.Audio/Models/Bark/) + [`BarkPipeline.cs`](../../src/SharpInference.Audio/Pipelines/BarkPipeline.cs). **Shared GPT-2 backbone built** ([`Models/LanguageModels/Gpt/`](../../src/SharpInference.Audio/Models/LanguageModels/Gpt/) — `GptBackbone`/`GptBlock`/`GptConfig`) that all three Bark stages reuse and **XTTS/ChatTTS will reuse**. EnCodec 24 kHz decoder already built (§4).
- [x] **Shared `GptBackbone`** — GPT-2 pre-norm (learned abs pos, MHA, 4× GELU MLP, bias=False, LayerNorm), causal + non-causal. **Validated** via synthetic-weight forward (3 tests, both mask modes finite). Full-sequence (no KV cache); AR re-feeds the prefix — perf-tunable later.
- [x] [`BarkCausalStage.cs`](../../src/SharpInference.Audio/Models/Bark/BarkCausalStage.cs) — semantic + coarse stages (token embed + `GptBackbone` + lm_head, AR via shared `NucleusSampler`).
- [x] [`BarkFineModel.cs`](../../src/SharpInference.Audio/Models/Bark/BarkFineModel.cs) — non-causal refiner: 8 codebook embeds summed + 7 heads, iterative argmax fill of codebooks 2..7.
- [x] [`BarkConfig.cs`](../../src/SharpInference.Audio/Models/Bark/BarkConfig.cs) — Full/Small presets + the Bark token-offset constants. **Tested**.
- [x] [`BarkPipeline.cs`](../../src/SharpInference.Audio/Pipelines/BarkPipeline.cs) — semantic → coarse (de-interleave 2 codebooks) → fine → EnCodec 24 kHz decode → audio.
- [ ] `BarkTokenizer.cs` (mBERT WordPiece + `+10048` offset) — token-IDs-in for now; caller tokenizes.
- [ ] `BarkSpeakerPrompt.cs` (3-stream speaker-prompt history) + checkpoint validation (`suno/bark`).

### CosyVoice 2 — SCAFFOLD COMPLETE (non-streaming); checkpoint-gated validation pending
> All components build and the exactly-specified pieces are unit-tested; the rest are structurally-correct scaffolds per the FunAudioLLM repo, **awaiting the 4.4 GB `FunAudioLLM/CosyVoice2-0.5B` checkpoint** for first-run validation (none on this host). Files under [`src/SharpInference.Audio/Models/CosyVoice/`](../../src/SharpInference.Audio/Models/CosyVoice/). **Architecture correction:** the LM is *not* the research doc's "single extended-vocab softmax" — the real `llm.pt` keeps the Qwen text embedding and adds separate `speech_embedding` / `llm_decoder` / `llm_embedding` heads; built to match.
- [x] [`CosyVoiceConfig.cs`](../../src/SharpInference.Audio/Models/CosyVoice/CosyVoiceConfig.cs) — CV2-0.5B composition config (LM + flow + HiFTNet + sampling sub-configs). **Tested** (preset values).
- [x] [`CosyVoiceQwenLm.cs`](../../src/SharpInference.Audio/Models/CosyVoice/CosyVoiceQwenLm.cs) — Qwen2.5-0.5B backbone (reuses the local `Qwen2Model` + `StreamingKvCache`) + separate `speech_embedding`/`llm_decoder`/`llm_embedding`; zero-shot text→speech-token AR loop.
- [x] [`SpeechSampler.cs`](../../src/SharpInference.Audio/Models/CosyVoice/SpeechSampler.cs) — rep-penalty → temp → top-k → top-p → RAS, deterministic. **Tested** (top-k=argmax, determinism, candidate masking, RAS breaks degenerate loops).
- [x] [`S3Tokenizer.cs`](../../src/SharpInference.Audio/Models/CosyVoice/S3Tokenizer.cs) — FSQ speech tokenizer (D=8/L=3 → 6561). **`PackFsqTokens` exact + tested** (center/min/max/mixed codes); the 6-block RoPE encoder is a conv-subsample scaffold (`speech_tokenizer_v2.onnx` available as fallback).
- [x] [`CamPlusSpeakerEncoder.cs`](../../src/SharpInference.Audio/Models/CosyVoice/CamPlusSpeakerEncoder.cs) — 192-d L2-normalized embedding via TDNN → stats-pooling → FC. Contract-correct scaffold (full FCM + dense D-TDNN + CAM masking checkpoint-gated; `campplus.onnx` fallback).
- [x] CosyVoice CFM — [`ICfmEstimator`](../../src/SharpInference.Audio/Models/CosyVoice/ICfmEstimator.cs) + [`ConditionalCfm.cs`](../../src/SharpInference.Audio/Models/CosyVoice/ConditionalCfm.cs) (OT-CFM Euler + CFG solver, **exact + tested**: constant-velocity integration, CFG combine, determinism) + [`CausalConditionalDecoder.cs`](../../src/SharpInference.Audio/Models/CosyVoice/CausalConditionalDecoder.cs) (timestep-injected resnet+attention UNet1D estimator) + [`CosyVoiceFlow.cs`](../../src/SharpInference.Audio/Models/CosyVoice/CosyVoiceFlow.cs) (token-embed → encoder scaffold → `encoder_proj`/`spk_affine` → CFM). The chunk-causal `UpsampleConformerEncoder` + exact estimator down/up topology are the checkpoint-gated pieces.
- [x] [`HiFTNetVocoder.cs`](../../src/SharpInference.Audio/Models/CosyVoice/HiFTNetVocoder.cs) — mel → 24 kHz: internal F0 predictor (ConvRNN) + NSF harmonic source + 3 ConvTranspose upsamples with source injection + plain-Snake MRF resblocks + magnitude/phase iSTFT head. Shares the validated Kokoro iSTFTNet NSF/iSTFT algorithm.
- [x] [`CosyVoicePipeline.cs`](../../src/SharpInference.Audio/Pipelines/CosyVoicePipeline.cs) — non-streaming orchestration (zero-shot reference → S3 + CAM++ → LM → flow → vocoder; or precomputed-embedding mode). **13 checkpoint-free tests pass.**
- [ ] **Streaming pipeline** — the 5:15 text:speech interleave + chunk-aware CFM flush + per-chunk vocoder + `IAsyncEnumerable<AudioChunk>` (150 ms first-packet). Non-streaming path built first; streaming is the follow-up.
- [ ] **Checkpoint converter + first-run validation** — bucket `llm.pt` / `flow.pt` / `hift.pt` (+ `campplus.onnx` / `speech_tokenizer_v2.onnx`) into the per-component LoadWeights dicts; reconcile exact state-dict keys (LM head/embedding, flow estimator topology, CAM++ D-TDNN, HiFTNet source-inject params) against the real weights, then env-gated `CosyVoiceGenerationTests`.
- [ ] **CosyVoice 1** (300M custom TransformerLM + VQ-4096 + UNet1D CFM) — out of scope for this pass; CV2 shipped first.

### IndexTTS 1.5 + 2
- [ ] `IndexT2sGpt.cs` — 24L × 1280
- [ ] `IndexS2MelDit.cs` — 13L × 512 with WaveNet final layer (non-causal — no streaming)
- [ ] `IndexSemanticCodec.cs` — Vocos-style ConvNeXt encoder
- [ ] `IndexConformerPerceiver.cs` for speaker + emotion conditioning
- [ ] BigVGAN v2 22kHz vocoder (download from `nvidia/bigvgan_v2_22khz_80band_256x`)
- [ ] Optional Qwen-3 0.6B-emo for text-emotion conditioning

### SparkTTS — SCAFFOLD COMPLETE (attribute-controlled generation); checkpoint-gated
> Files under [`Models/SparkTts/`](../../src/SharpInference.Audio/Models/SparkTts/) + [`SparkTtsPipeline.cs`](../../src/SharpInference.Audio/Pipelines/SparkTtsPipeline.cs). **Maximal reuse:** the LM is a plain Qwen2.5-0.5B (single extended-vocab softmax) reusing `Qwen2Model` verbatim; sampling reuses the new shared `NucleusSampler`; BiCodec global dequant reuses `Fsq`; the wave-gen MRF reuses the new shared `SnakeResBlock`.
- [x] [`SparkTtsConfig.cs`](../../src/SharpInference.Audio/Models/SparkTts/SparkTtsConfig.cs) — `V0_5B` (Qwen2.5-0.5B with `VocabSize=166000`) + token-ID bases (semantic/global/EOS/structure) + BiCodec decode config. **Tested** (vocab/codec sizes, FSQ-levels↔global-vocab consistency). Token offsets are config fields pending `added_tokens.json` reconciliation.
- [x] [`SparkTtsLm.cs`](../../src/SharpInference.Audio/Models/SparkTts/SparkTtsLm.cs) — reuses `Qwen2Model` + `StreamingKvCache` + shared `NucleusSampler`; AR loop parses emitted absolute IDs → global/semantic codec indices by range.
- [x] [`BiCodecDecoder.cs`](../../src/SharpInference.Audio/Models/SparkTts/BiCodecDecoder.cs) — semantic VQ lookup + global FSQ dequant (`Fsq.Dequantize`) + DAC-style HiFi-GAN wave generator (ConvTranspose [8,5,4,2] + shared `SnakeResBlock` MRF + conv_post→tanh, FiLM-lite speaker conditioning). 16 kHz. (Full Vocos-ConvNeXt AdaLN prenet + exact BiCodec keys are the checkpoint-gated piece.)
- [x] [`SparkTtsPipeline.cs`](../../src/SharpInference.Audio/Pipelines/SparkTtsPipeline.cs) — token-IDs-in → LM → BiCodec decode → 16 kHz audio. Zero-shot cloning (w2v-BERT reference tokenization on the *encode* side) is deferred — attribute-controlled generation needs no reference.
- [x] **Shared primitives hoisted this pass** (reuse principle): [`NucleusSampler`](../../src/SharpInference.Audio/Sampling/NucleusSampler.cs) (top-k/top-p/temp draw — `SpeechSampler` refactored to delegate, bit-identical) + [`SnakeResBlock`](../../src/SharpInference.Audio/Models/Vocoders/SnakeResBlock.cs) (plain-Snake MRF — HiFTNet refactored to use it).
- [ ] **BiCodec encoder (cloning) + w2v-BERT feature extractor** — encode-side (reference→tokens) for zero-shot voice cloning; deferred (attribute generation works without it).
- [ ] **Checkpoint validation** — `SparkAudio/Spark-TTS-0.5B` (3.95 GB): reconcile `added_tokens.json` IDs + BiCodec state-dict keys, then env-gated generation test.

### ChatTTS
- [ ] `ChatTtsGpt.cs` — single LLaMA-style 20L × 768, 4-codebook GFSQ output
- [ ] `ChatTtsDvaeDecoder.cs` — 12-layer dilated ConvNeXt
- [ ] `ChatTtsSpeakerLatent.cs` — sample from `spk_stat.pt` Gaussian
- [ ] Vocos vocoder (shared with F5-TTS)
- [ ] 61 paralinguistic special tokens; sampling defaults differ for RefineText vs InferCode

### Higgs Audio v2
- [ ] Llama-3.2-3B with DualFFN (per-token-type routing) — extended dotLLM
- [ ] Dual-branch tokenizer (HuBERT-base + DAC acoustic, 8 codebooks consumed of 12 declared)
- [ ] RAS (Repetition-Aware Sampling)
- [ ] Multi-speaker prompt format with `[SPEAKER0]` / `[SPEAKER1]` tags

### Sesame CSM — SCAFFOLD COMPLETE (dual-transformer); checkpoint-gated
> Files under [`Models/Csm/`](../../src/SharpInference.Audio/Models/Csm/) + [`CsmPipeline.cs`](../../src/SharpInference.Audio/Pipelines/CsmPipeline.cs). **Reuse:** both transformers are headless Llama-3.2 bodies → reuse `Qwen2Model` (generalized this pass with a `Qwen2Config.AttentionBias` flag, default true, so Llama loads bias-free while CosyVoice/SparkTTS/VibeVoice are unchanged); audio decode reuses the built Mimi codec; sampling reuses `NucleusSampler`.
- [x] [`CsmConfig.cs`](../../src/SharpInference.Audio/Models/Csm/CsmConfig.cs) — `V1B`: backbone (Llama-1B: 16L/2048/GQA 32:8, bias-off) + decoder (Llama-100M: 4L/1024/8:2) + 8 codebooks + Mimi. **Tested** (dual-transformer shape + the AttentionBias regression guard).
- [x] **`Qwen2Model.LoadWeightsHeadless`** — loads the transformer body (layers + final norm) without `embed_tokens`/`lm_head` (CSM's are `Identity`; embeds + heads live on the outer model). Reusable for any headless-LM design.
- [x] [`CsmModel.cs`](../../src/SharpInference.Audio/Models/Csm/CsmModel.cs) — dual `Qwen2Model` + text/audio embed tables + codebook-0 head + backbone→decoder projection + 7 decoder heads; `GenerateFrame` (backbone → codebook 0, decoder AR → codebooks 1..7).
- [x] [`CsmPipeline.cs`](../../src/SharpInference.Audio/Pipelines/CsmPipeline.cs) — frame loop (re-embedded context) → Mimi 24 kHz decode. Conversational `Segment` history is token-IDs-in (caller assembles).
- [ ] **Aggressive streaming + persistent KV cache** (~50 ms/frame) — current path re-feeds context per frame (correct, not yet streaming-optimized).
- [ ] **Checkpoint validation** — `sesame/csm-1b`: reconcile HF key names + decoder-side audio embeddings + Llama-3.2 RoPE rescaling (scale_factor=32), then env-gated generation test.

### GPT-SoVITS
- [ ] T2S GPT (predict semantic tokens from text + reference)
- [ ] SoVITS SynthesizerTrn (VITS posterior + flow + decoder + duration predictor + Stochastic Duration Predictor)
- [ ] `cn-hubert` HuBERT-base for semantic token extraction
- [ ] BERT auxiliary (Chinese-RoBERTa for zh, custom for other langs)
- [ ] ref_enc (mel ResNet + attention pool) for speaker conditioning
- [ ] Per-language phoneme tokenizers: pinyin (zh), ARPABET (en), pyopenjtalk romaji (ja)

### OpenVoice v2
- [ ] Tone Color Converter (residual flow over mel)
- [ ] Tone Color Extractor (Conv1D + attention pool → speaker embedding)
- [ ] Stage 1 dispatches to MeloTTS (below)

### MeloTTS
- [ ] `MeloTtsTextEncoder.cs` — Transformer over phonemes + BERT aux concat
- [ ] `MeloTtsFlow.cs` — coupling layers with WaveNet residual stack
- [ ] `MeloTtsStochasticDurationPredictor.cs` — normalizing flow over duration
- [ ] `MeloTtsHifiGanDecoder.cs` — 44.1 kHz output (hop=512)
- [ ] Per-language BERT loaders (XLM-R / Chinese-RoBERTa / Japanese-RoBERTa)
- [ ] Per-language speaker embeddings (`spk2id` table)

### StyleTTS 2 — BUILT (reuses validated Kokoro stack); checkpoint-gated for the new style modules
> The parent architecture of Kokoro. PLBERT + TextEncoder + ProsodyPredictor + iSTFTNet decoder are reused **verbatim** from the (validated, end-to-end-tested) Kokoro modules — only the runtime style path is new. Files under [`Models/StyleTts2/`](../../src/SharpInference.Audio/Models/StyleTts2/) + [`StyleTts2Pipeline.cs`](../../src/SharpInference.Audio/Pipelines/StyleTts2Pipeline.cs).
- [x] **Kokoro module reuse** — the 4 shared classes were made `public` and [`KokoroPipeline`](../../src/SharpInference.Audio/Pipelines/KokoroPipeline.cs) gained a `SynthesizeFromStyle(phonemes, refStyle256, speed)` entry point (the voice-pack path now routes through the same `SynthesizeCore`). StyleTTS 2 composes a `KokoroPipeline` from its own weights and drives it with an externally-computed style. **Kokoro's 6 tests still pass after the refactor.**
- [x] [`StyleTts2Config.cs`](../../src/SharpInference.Audio/Models/StyleTts2/StyleTts2Config.cs) — `LibriTts` (multispeaker) / `LjSpeech` presets. **Tested** (StyleDim=256, MultiSpeaker flag).
- [x] `StyleTransformer1d` diffusion style sampler — [`StyleDiffusionSampler.cs`](../../src/SharpInference.Audio/Models/StyleTts2/StyleDiffusionSampler.cs) (Karras schedule + ADPM2 ancestral second-order sampler, **exact + tested**: schedule endpoints/monotonicity, determinism, shape) + [`StyleDenoiser.cs`](../../src/SharpInference.Audio/Models/StyleTts2/StyleDenoiser.cs) (EDM/KDiffusion preconditioning + CFG combine — exact; the StyleTransformer1d network forward is a checkpoint-gated scaffold).
- [x] [`StyleEncoder.cs`](../../src/SharpInference.Audio/Models/StyleTts2/StyleEncoder.cs) — 2D-Conv ResNet (StarGAN-v2 ResBlks) + adaptive-avg-pool → 128-d, run twice (acoustic + prosodic → 256-d). Uses the existing `Conv2D` + inline pooling. (spectral-norm sigma folding checkpoint-gated.)
- [x] [`StyleTts2Pipeline.cs`](../../src/SharpInference.Audio/Pipelines/StyleTts2Pipeline.cs) — three inference modes: `SynthesizeClone` (zero-shot from reference mel), `SynthesizeRandom` (diffusion, no reference — the LJSpeech mode), `SynthesizeClonePerturbed` (diffusion seeded by the reference style). **8 checkpoint-free tests pass** (sampler + config).
- [ ] **Checkpoint validation** — download `yl4579/StyleTTS2-LibriTTS` (.pth → strip training modules → ~590 MB), reconcile the `style_encoder` / `predictor_encoder` / `diffusion` (StyleTransformer1d) state-dict keys + spectral-norm folding, wire a loader, and add an env-gated generation test. The Kokoro-shared modules load from the same checkpoint family.
- [ ] **Long-form continuation** mode (the 3rd StyleTTS2 mode beyond clone/random) — deferred.

### VibeVoice (1.5B / 7B / Streaming-0.5B) — multi-speaker BUILT; streaming pipeline pending
> The local LM is a self-contained Qwen2.5 reimplementation under [`Models/LanguageModels/Qwen2/`](../../src/SharpInference.Audio/Models/LanguageModels/Qwen2/) (no dotLLM dependency taken). Multi-speaker path is fully wired; the split-LM streaming variant is deferred.
- [x] [`VibeVoiceConfig.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceConfig.cs) — composition config (acoustic + semantic tokenizer + Qwen2.5 decoder + diffusion head); `V15B` / `V7B` / `Streaming05B` presets. (Loaded via downstream JSON parsing; no `[JsonSerializable]` attribute.)
- [x] `VibeVoiceStreamingConfig` — **folded into `VibeVoiceConfig`**: `Streaming05B` preset + `IsStreaming` (null semantic tokenizer, `TtsBackboneNumHiddenLayers = 20`)
- [x] `Block1D` — [`VibeVoiceConvNeXtBlock.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceConvNeXtBlock.cs) (ConvNeXt-V1 1D: ConvRMSNorm + depthwise causal Conv1d + LayerScale + GELU FFN)
- [x] [`SConv1d.cs`](../../src/SharpInference.Audio/Models/VibeVoice/SConv1d.cs) / [`SConvTranspose1d.cs`](../../src/SharpInference.Audio/Models/VibeVoice/SConvTranspose1d.cs) — causal padding wrappers with streaming-cache hooks
- [x] [`VibeVoiceTokenizerStreamingCache.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceTokenizerStreamingCache.cs) — per-`(layer_id, sample_index)` left-padded history buffer
- [x] `TokenizerEncoder` — [`VibeVoiceTokenizerEncoder.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceTokenizerEncoder.cs): 6-stage stem+downsample, reversed ratios `[2,2,4,5,5,8]`, channels `32→…→2048`, Linear head 2048→vae_dim
- [x] `TokenizerDecoder` — [`VibeVoiceTokenizerDecoder.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceTokenizerDecoder.cs): mirror of encoder via `SConvTranspose1d`
- [x] [`VibeVoiceAcousticTokenizerModel.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceAcousticTokenizerModel.cs) — encoder + decoder + `fix_std=0.5` + Gaussian sampling; `encode()`/`decode()`
- [x] [`VibeVoiceSemanticTokenizerModel.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceSemanticTokenizerModel.cs) — encoder only, `vae_dim=128`, deterministic
- [x] [`SpeechConnector.cs`](../../src/SharpInference.Audio/Models/VibeVoice/SpeechConnector.cs) — Linear + RMSNorm + Linear
- [x] [`VibeVoiceDiffusionHead.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceDiffusionHead.cs) — 4 × HeadLayer (RMSNorm + AdaLN + SwiGLU) + FinalLayer; `noisy_images_proj` / `cond_proj` / `t_embedder`
- [x] `VibeVoiceCosineBetaSchedule` — Nichol-Dhariwal cosine `alpha_bar`, in [`VibeVoiceCosineDpmSolver.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceCosineDpmSolver.cs)
- [x] DPM-Solver multistep — v-prediction + cosine + order=2 + 20 steps (`VibeVoiceCosineDpmSolver`)
- [x] `VibeVoiceTextTokenizer` — [`VibeVoiceTokenizer.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceTokenizer.cs): Qwen2 tokenizer exposing `SpeechStart/End/Diffusion` + EOT ids
- [x] `VibeVoiceTokenConstraintProcessor` — **folded into `VibeVoicePipeline.SampleConstrained`** (logit mask over `{speech_start, end, diffusion, eos}`)
- [x] [`VibeVoiceProcessor.cs`](../../src/SharpInference.Audio/Models/VibeVoice/VibeVoiceProcessor.cs) — multi-speaker script parser (JSON / plain / `Speaker N:`), −25 dBFS normalizer, voice-prompt builder, `speech_input_mask`
- [x] [`VibeVoicePipeline.cs`](../../src/SharpInference.Audio/Pipelines/VibeVoicePipeline.cs) (multi-speaker) — prefill (acoustic-encode voice → splice into LM embed at mask) + AR loop (Qwen2.5 → constrained logits → DDPM sub-loop with CFG → acoustic decode → semantic re-encode → embed feedback)
- [ ] `VibeVoiceStreamingPipeline.cs` — split-LM forward + windowed text/speech interleave + `BinaryClassifier` EOS + acoustic-only feedback. **Not built** (multi-speaker pipeline notes it as deferred).
- [ ] `BinaryClassifier.cs` — `Linear → ReLU → Linear→1` streaming EOS head. **Not built** (part of streaming).
- [ ] **Per-layer-stop on the local `Qwen2Model`** — needed only for the streaming split-LM; not required by the multi-speaker path that ships today.
- [ ] **Audio streamer surface** — per-`speech_diffusion`-token `IAsyncEnumerable<AudioChunk>` emission (part of the streaming variant).
- [ ] Long-context smoke test: 65k-token KV cache stable on 1.5B (90 min output), 32k on 7B (45 min)
- [ ] License surface: VibeVoice is **MIT** — flag in the model registry vs XTTS (CPML), AudioLDM2 (CC-BY-NC), SparkTTS (CC-BY-NC)
- **Validated:** `VibeVoiceStage1SmokeTests` loads all 1204 tensors + acoustic-VAE round-trip + semantic-VAE 128-d latent + diffusion-head single-step; `VibeVoiceEndToEndTests` runs prefill + short-script synthesis. Both skip-gated on cache (E2E also behind `SHARPINFERENCE_RUN_VIBEVOICE_E2E`).

## 7. Music Generation (`SharpInference.Music`)

### ACE-Step (flagship)

> **v1 BUILT END-TO-END (2026-06-10)** — lives in **SharpInference.Diffusion** per the §"diffusion music models" routing
> note (`Models/Music/` + `Models/Denoisers/AceStep*` + `Pipelines/AceStepPipeline.cs`), NOT a separate Music package.
> Structurally CPU-verified (6 model tests incl. a tiny e2e stereo generation); **numerics validation-pending** against
> the Python reference. Source-confirmed correction to the research doc: the self-attention is Sana **LiteLA ReLU-linear
> attention** (`CustomLiteLAProcessor2_0`), not softmax SDPA. Validation-gated items: several checkpoint key spellings
> carry fallback candidates pending a real key dump (`proj_in.*`, `t_block`, `genre_embedder`, GLUMBConv conv names),
> cross-attn K RoPE positions, Conformer macaron/conv-norm variants (presence-driven at load).

- [x] UMT5-base — reused the shared `T5TextEncoder` (new `T5TextEncoderConfig.Umt5Base` preset, per-layer position bias); no ACE-specific encoder class needed
- [x] [`AceStepDit.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/AceStepDit.cs) v1 — 24L × 20 heads × 128 (inner 2560), LiteLA self-attn + RoPE θ=1e6, softmax cross-attn over [speaker ‖ text ‖ lyrics], GLUMBConv FFN (ratio 2.5), patch [16,1] height collapse; owns speaker/genre/lyric projections + the lyric Conformer
- [ ] `AceStepFsqLm.cs` v1.5 — Qwen3-based decoder predicting FSQ audio tokens (separate later effort; FSQ codec already exists in Audio)
- [x] [`MusicDcaeDecoder.cs`](../../src/SharpInference.Diffusion/Models/Music/MusicDcaeDecoder.cs) — Sana AutoencoderDC decoder over 2-D stereo mel ([`ResBlock2d`](../../src/SharpInference.Diffusion/Models/Music/ResBlock2d.cs) + [`EfficientVitBlock`](../../src/SharpInference.Diffusion/Models/Music/EfficientVitBlock.cs) multiscale ReLU-linear attention + GLUMBConv, repeat-interleave/pixel-shuffle shortcuts). **Encoder not built yet** — needed for edit/repaint/reference-audio modes only
- [x] [`AdaMosHiFiGanV1.cs`](../../src/SharpInference.Diffusion/Models/Music/AdaMosHiFiGanV1.cs) — ConvNeXt backbone (depths [3,3,9,3]) + HiFi-GAN head (7 ups, ×512 total, MRF kernels [3,7,11,13]); weight-norm fused at conversion
- [x] [`AceStepLyricEncoder.cs`](../../src/SharpInference.Diffusion/Models/Music/AceStepLyricEncoder.cs) — 8-layer wenet Conformer (rel-pos MHSA with pos_bias_u/v, GLU conv module with fused BatchNorm, presence-driven macaron) — first Conformer in the codebase
- [x] Voice BPE tokenizer — [`AceStepLyricTokenizer`](../../src/SharpInference.Tokenizers/AceStepLyricTokenizer.cs) (XTTS-style: `[lang]` prefix, `[SPACE]`, vocab+merges from the one-time tokenizer.json export documented in the converter)
- [x] Lyric structure tags + `tokenize_lyrics` (261/2 line protocol) + per-line script-heuristic language detection (statistical 17-lang detector + CJK G2P = caller responsibility, documented)
- [x] Flow-match shift=3.0 + APG/CFG-Zero★/CFG — [`AceStepGuidance`](../../src/SharpInference.Diffusion/Utilities/AceStepGuidance.cs) (momentum buffer, norm threshold, parallel/orthogonal decomposition; orthogonality + formula tests)
- [x] Three samplers: Euler / Heun (in-pipeline 2nd-order) / [`FlowMatchPingPongScheduler`](../../src/SharpInference.Diffusion/Schedulers/FlowMatchPingPongScheduler.cs)
- [x] [`AceStepPipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/AceStepPipeline.cs) — denoise → pipeline latent scale (0.1786/−1.9091, NOT the diffusers 0.41407) → DCAE decode → de-standardize to log-mel [−11,3] → per-channel vocoder → 44.1 kHz stereo; [`AceStepCheckpointConverter`](../../src/SharpInference.ModelHandler/CheckpointConverters/AceStepCheckpointConverter.cs) (SSL-head drop, weight-norm fusion w/ numeric test) + `TestPaths.AceStep`
- [ ] Flow-edit / repaint algorithm (masked dual-conditioning velocity loop — needs the DCAE **encoder** first)
- [ ] Env-gated `AceStepGenerationTests` against the real `ACE-Step/ACE-Step-v1-3.5B` checkpoint + key-dump confirmation of the fallback spellings + python diff harness

### Stable Audio Open
- [ ] `OobleckVae.cs` — 5-stage Conv1D + snake activation + weight-norm (no GroupNorm), 2048× downsample, 64-ch latent
- [ ] `StableAudioDit.cs` — 1536 / 24L / 24 heads (Q) / 12 KV heads (cross-attn), AdaLN-6, RoPE, SwiGLU
- [ ] `FourierFeatures1D` for timing embedding
- [ ] Timing conditioning (seconds_start + seconds_total → cross-attn tokens + global AdaLN)
- [ ] T5 reuse from SharpInference.Diffusion
- [ ] `dpmpp-3m-sde` scheduler (Open 1.0 v-prediction)
- [ ] Pingpong scheduler (Open Small distilled)

### MusicGen + AudioGen
- [ ] `MusicGenCausalLm.cs` — decoder-only, sum-of-K embeddings input, sinusoidal pos, GELU 4× FFN, K parallel heads
- [ ] T5-base text encoder
- [ ] EnCodec 32kHz 4-codebook (music) / EnCodec 16kHz 4-codebook (audio)
- [ ] Delay-pattern state machine (default `[0,1,2,3]`; stereo `[0,0,1,1,2,2,3,3]`)
- [ ] CFG two-stream batched (uncond + 3.0*(cond-uncond))
- [ ] Melody conditioning (chromagram extractor + argmax-and-zero + cross-attn prepend)
- [ ] Stereo variants (8 codebooks, paired delay)

### YuE — SCAFFOLD COMPLETE (Stage-1, first music model); checkpoint-gated
> **First music model.** Files under [`Models/Music/`](../../src/SharpInference.Audio/Models/Music/) + [`YuePipeline.cs`](../../src/SharpInference.Audio/Pipelines/YuePipeline.cs). Built almost entirely from reuse — the codec-LM music models live in the Audio package since they reuse its codecs + LM infra (the diffusion music models — ACE-Step/Stable Audio/DiffRhythm/AudioLDM2 — will go in the Diffusion package).
- [x] [`YueStage1Lm.cs`](../../src/SharpInference.Audio/Models/Music/YueStage1Lm.cs) — **reuses `Qwen2Model`** (Llama-2-7B: bias-off + MHA via `NumKeyValueHeads == NumAttentionHeads`) + shared `NucleusSampler`; emits the interleaved **dual-track** `[vocal_cb0, accomp_cb0, …]` stream and parses it into the two per-track codebook-0 streams. Mandatory `repetition_penalty=1.1` applied.
- [x] [`YueConfig.cs`](../../src/SharpInference.Audio/Models/Music/YueConfig.cs) — Stage-1 (Llama-2-7B) + Stage-2 (~1.5B) presets + extended-vocab audio-token bases. **Tested**.
- [x] [`YuePipeline.cs`](../../src/SharpInference.Audio/Pipelines/YuePipeline.cs) — Stage-1 → cb0 → **built `XCodec`** 16 kHz decode.
- [ ] **Stage-2 residual upsampler** (cb0 → 8 codebooks) — reuses `Qwen2Model` again; currently codebooks 1..7 are zero-filled (decodes cb0 semantic content only). The accompaniment track + mix also ride on Stage-2.
- [ ] **Vocos 16→44.1 kHz upsampler** + CoT/ICL prompt variants + long-context RoPE scaling.
- [ ] **Checkpoint validation** — `m-a-p/YuE-s1-7B-*` + `xcodec_mini_infer`: reconcile the tokenizer audio-token bases + HF key names, then env-gated test.

### DiffRhythm
- [ ] Stable-Audio-derived VAE (21.5 Hz / 64-ch / 2048× downsample)
- [ ] 1.1B / 16L / d=2048 / 32-head LLaMA-style DiT with cross-attention
- [ ] G2P phonemes via [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md)
- [ ] MuQ-MuLan style conditioning + timestep AdaLN-Zero
- [ ] Flow-matching Euler + optional Sway (32 NFE, CFG ≈ 4.0)
- [ ] Sentence-level LRC alignment

### AudioLDM 2
- [ ] CLAP text encoder (RoBERTa + projection head) — new
- [ ] T5-Large reuse
- [ ] GPT-2 small in continuous-feature mode (no `lm_head`, deterministic, max_new_tokens=8)
- [ ] AudioLDM2ProjectionModel
- [ ] UNet with dual cross-attention (GPT-2 + T5) — extends existing SD UNet
- [ ] Mel VAE (3 down-blocks, scaling_factor=0.4110932946205139)
- [ ] SpeechT5HifiGan (5-stage upsample, 16 kHz, 64 mel bins)
- [ ] DDIM scheduler (existing) or DPM++ 2M

## 8. Server / Streaming Endpoints (deferred to Phase 7)

- [ ] STT endpoints: `/v1/audio/transcriptions`, `/v1/audio/translations`
- [ ] TTS endpoints: `/v1/audio/speech` (synchronous) + SSE streaming variant
- [ ] Music endpoints: `/v1/audio/music` (long-running job queue — songs take minutes)
- [ ] OpenAI-compatible request shapes (model, voice, response_format)

## 9. Testing

- [ ] STFT vs NumPy/SciPy (1e-6); mel vs whisper.cpp (1e-4); per-feature norm vs NeMo (1e-5)
- [ ] Codec round-trip STOI > 0.95 per codec on test clip
- [ ] Whisper full pipeline WER < 1% of reference on LibriSpeech test-other
- [ ] Parakeet-TDT WER match (within 0.1% absolute of upstream report)
- [ ] Kokoro mel + waveform within 1e-3 of reference on canonical phoneme input
- [ ] F5-TTS Sway sampling scheduler produces identical sigma values to upstream `cfm.py` (1e-6)
- [ ] VibeVoice acoustic VAE round-trip on 5s clip — STOI > 0.95 vs Python reference
- [ ] VibeVoice diffusion head single-step forward (random latent + timestep + condition) — match Python within 1e-3 (bf16)
- [ ] VibeVoice DPM-Solver 20-step trajectory matches `vibevoice/schedule/dpm_solver.py` on identical seed within 1e-3
- [ ] VibeVoice end-to-end multi-speaker (4-way) demo script reproduces the upstream demo audio (subjective listening + spectrogram comparison)
- [ ] VibeVoice-Streaming-0.5B first-packet latency < 250 ms on RTX 3060 12GB; 90-min VRAM-stable on 1.5B
- [ ] ACE-Step single-step DiT forward matches Python reference (avg_err < 1e-3)
- [ ] Each codec encoder/decoder round-trip on a 5s test clip — STOI > 0.95
- [ ] Long-form (30 min audio) memory leak test for each STT pipeline
- [ ] 1-hour streaming test for streaming TTS (CosyVoice 2, XTTS, Sesame CSM)
- [ ] All tests pass on GPU CI

## 10. Performance Targets (initial, revise after first run)

| Pipeline | Hardware | Target |
|---|---|---|
| Whisper-large-v3-turbo | RTX 3060 12GB | RTF ≥ 5x (process 5 min audio in 1 min) |
| Parakeet-TDT-0.6B-v2 | RTX 3060 12GB | RTF ≥ 100x |
| Moonshine-base | CPU (modern x64) | RTF ≥ 2x |
| Kokoro 82M | RTX 3060 12GB | RTF ≥ 20x (generate 20s audio in 1s) |
| F5-TTS | RTX 3060 12GB | RTF ≥ 2x with 32 NFE |
| VibeVoice-1.5B | RTX 3060 12GB | RTF ≥ 3x (single speaker, 20 DDPM steps per 133 ms frame) |
| VibeVoice-7B (Large) | RTX 3060 12GB | RTF ≥ 1x with bf16 quantization (bf16 fits ~14 GB — may need int8 LM on 12 GB cards) |
| VibeVoice-Streaming-0.5B | RTX 3060 12GB | First-packet < 250 ms, sustained RTF ≥ 1x |
| CosyVoice 2 streaming | RTX 3060 12GB | First-packet < 250 ms |
| Sesame CSM | RTX 3060 12GB | Frame latency < 100 ms (real-time conversation) |
| ACE-Step 3.5B | RTX 3060 12GB | 4-min song in ≤ 60s (target — likely ~120s realistically) |
| Stable Audio Open Small | RTX 3060 12GB | 12s clip in ≤ 5s |

## 11. Review & Merge

- [ ] Code review per package (audio buffer boundaries, streaming thread safety, KV cache correctness)
- [ ] Benchmark report comparing SharpInference vs Python reference for each model
- [ ] License audit — flag non-commercial models (XTTS = CPML, AudioLDM2 = CC-BY-NC-SA, SparkTTS = CC-BY-NC-SA) in model registry; surface to users. VibeVoice is **MIT** (commercially permissive — surface as such).
- [ ] Update `MODEL_STATUS.md` with audio model coverage
- [ ] Merge to main branch

## 12. Status Summary (2026-05-29)

### Done

- **STT — Whisper + Moonshine shipped.** Whisper (encoder/decoder/pipeline + Distil + large-v3-turbo presets, greedy decode + suppress-tokens) and Moonshine (Conv1D + RoPE + SentencePiece BPE) both transcribe canonical JFK audio end-to-end. NeMo / SenseVoice / FireRedASR families remain unstarted.
- **TTS — F5-TTS + Kokoro shipped; VibeVoice multi-speaker + CosyVoice 2 scaffolded.** F5-TTS (DiT + sway-sampling + Vocos) is complete. Kokoro is now end-to-end real speech: the full **iSTFTNet generator forward** (HnNSF harmonic source + 2 upsample stages + MRF AdaIN/Snake + magnitude/phase iSTFT) replaced the sine placeholder; the only remaining Kokoro gap is a C# **G2P** (IPA-in only today). VibeVoice's multi-speaker pipeline (tokenizers, diffusion head, cosine DPM-Solver, processor, AR + DDPM loop) is wired; the **streaming-0.5B split-LM pipeline + `BinaryClassifier` EOS are not built**. **CosyVoice 2** is a full non-streaming scaffold across all 7 components (config / Qwen LM + RAS sampler / S3 FSQ tokenizer / CAM++ / OT-CFM flow / HiFTNet vocoder / pipeline) — the exactly-specified pieces (sampler, CFM Euler+CFG solver, FSQ packing) are unit-tested (13 tests); the rest are structurally-correct scaffolds **awaiting the CV2-0.5B checkpoint** for first-run validation. **StyleTTS 2** (Kokoro's parent) reuses the validated Kokoro stack verbatim via a new `KokoroPipeline.SynthesizeFromStyle` path and adds the style modules — the diffusion style sampler (Karras + ADPM2 + EDM/CFG) and FSQ-free 2D-Conv `StyleEncoder` — unlocking zero-shot voice cloning + random-style modes; sampler/config unit-tested (8 tests), the StyleTransformer1d network awaiting `StyleTTS2-LibriTTS`. **Spark-TTS** (Qwen2.5-0.5B LM → BiCodec) is a non-streaming scaffold built almost entirely from reuse — the LM reuses `Qwen2Model` verbatim (single extended-vocab softmax), and two shared primitives were hoisted in the process: `NucleusSampler` (top-k/top-p draw, now shared by CosyVoice + Spark) and `SnakeResBlock` (plain-Snake MRF, now shared by HiFTNet + BiCodec); config + sampler unit-tested (10 tests), BiCodec decode awaiting `Spark-TTS-0.5B`.
- **All 9 codecs** in §4 are structurally complete with config + Encode + Decode + EnumerateWeights surfaces. Build passes clean across 12 projects × 2 frameworks.
- **All 10 shared-primitive items** in §3 are landed (streaming surface, ring buffer, KV cache, LSTMs, mel extractor, SIMD FFT).
- **Backend op gaps from §3 PTX list** addressed at the C# IBackend level: `Conv1d`, `ConvTranspose1d`, `Sigmoid`, `Tanh`, `Snake`, `Elu` are wired on CpuBackend with real implementations. CudaBackend + VulkanBackend currently throw `NotSupportedException` for the not-yet-compiled-PTX/SPIR-V variants, with native source files staged.
- **Streaming wrappers** for any codec available via `StreamingCodecEncoder<T>` / `StreamingCodecDecoder<T>` (generic; works on EnCodec, DAC, SNAC, Mimi, XCodec, WavTokenizer).
- **Offline weight-norm fusion** CLI at `samples/FuseWeightNorm/`.
- **Smoke tests** for all 9 codecs (config shapes + construction sanity).
- **Native source files** for new CUDA + Vulkan ops written and committed under `native/cuda/audio/` and `native/vulkan/shaders/` (snake / conv1d / conv_transpose1d / extended elementwise with tanh+elu+sigmoid).

### Outstanding

- **Kokoro G2P frontend** — English text → IPA (per `G2P_PHONEMIZATION.md`); pipeline is IPA-in only today. Now the highest-value remaining Kokoro item (the iSTFTNet generator forward shipped).
- **VibeVoice streaming variant** — `VibeVoiceStreamingPipeline` + `BinaryClassifier` EOS + split-LM per-layer-stop + per-token audio-chunk streaming.
- **CosyVoice 2 checkpoint validation** — download `FunAudioLLM/CosyVoice2-0.5B` (4.4 GB), reconcile the per-component state-dict keys + the checkpoint-gated topologies (flow `UpsampleConformerEncoder` + estimator down/up wiring, CAM++ D-TDNN, HiFTNet source-inject params), then build the converter + env-gated generation test. Also the streaming pipeline (5:15 interleave, 150 ms first-packet).
- **Whisper long-form + word-level timestamps** — sequential/chunked decoding and cross-attention DTW alignment; temperature fallback + beam search.
- **PTX / SPIR-V compilation pass** — needs nvcc + glslc invocations in the native build pipeline; source-only today.
- **Per-codec STOI validation** against Python references — requires downloading official checkpoints. Smoke tests provide the scaffolding pattern; the gated-attribute approach used by F5TtsSmokeTests is the convention.
- **Reference-diff validation for the shipped models** (Whisper WER, Kokoro/F5/VibeVoice numeric matches in §9) — all download-gated; current tests assert structural correctness + skip cleanly.
- **Per-codec streaming caches** with proper context propagation (current `StreamingCodecEncoder<T>` is causal-naive — re-encodes each chunk fresh). Acceptable for chunk_size ≥ 200 ms; sub-200 ms live use needs per-codec cache wiring (Mimi-specific is the highest-priority).
- **TTS / music pipelines** that consume these codecs (Bark / MusicGen / Sesame CSM / Orpheus / IndexTTS / Higgs Audio / Spark-TTS / YuE) — separate work items, codec foundation is in place.
