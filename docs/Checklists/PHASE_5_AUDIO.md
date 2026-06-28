# Phase 5 — Audio (STT + TTS + Music)

> **Goal:** End-to-end inference for SOTA STT, TTS, and music-generation models in pure C#.
> **Packages:** HartsyInference.Audio (STT + TTS), HartsyInference.Music (music generation)
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
- [ ] Decide whether to package music separately under `HartsyInference.Music` or keep all audio under `HartsyInference.Audio`. Music brings full-song generation, long-context, and large LLM backbones that have a different memory profile.

## 3. Audio Preprocessing & Shared Primitives

- [x] `AudioPreprocessor.cs` — resample (polyphase, windowed sinc, Kaiser β=8.6, configurable taps), normalize, windowing  *(shipped as [Io/Resampler.cs](../../src/HartsyInference.Audio/Io/Resampler.cs) + [Preprocessing/HannWindow.cs](../../src/HartsyInference.Audio/Preprocessing/HannWindow.cs))*
- [x] `StftProcessor.cs` — radix-2 Cooley-Tukey FFT, STFT, periodic Hann window. **SIMD path landed for the largest stage** ([Preprocessing/Fft.cs](../../src/HartsyInference.Audio/Preprocessing/Fft.cs) — Vector&lt;float&gt;-vectorized when step==1; scalar fallback for early stages)
- [x] `MelSpectrogramProcessor.cs` — Slaney mel filterbank, log compression, normalization variants ([Preprocessing/MelSpectrogramExtractor.cs](../../src/HartsyInference.Audio/Preprocessing/MelSpectrogramExtractor.cs); Whisper / Kokoro / HiFiGAN presets)
- [x] `IStftLayer.cs` — inverse STFT for vocoders ([Models/Vocoders/IStft.cs](../../src/HartsyInference.Audio/Models/Vocoders/IStft.cs))
- [x] `AudioRingBuffer.cs` — circular PCM buffer ([Streaming/AudioRingBuffer.cs](../../src/HartsyInference.Audio/Streaming/AudioRingBuffer.cs))
- [x] `StreamingMelExtractor.cs` — incremental STFT with context tail ([Streaming/StreamingMelExtractor.cs](../../src/HartsyInference.Audio/Streaming/StreamingMelExtractor.cs))
- [x] `StreamingKvCache.cs` — per-layer K/V append-and-grow ([Streaming/StreamingKvCache.cs](../../src/HartsyInference.Audio/Streaming/StreamingKvCache.cs))
- [ ] PTX kernels: `fft_radix2.ptx`, `mel_filterbank.ptx`, `istft_overlap_add.ptx`, `snake_activation.ptx`, `conv_transpose1d.ptx` — **CUDA source written**, awaiting nvcc build pass ([native/cuda/audio/](../../native/cuda/audio/)). Vulkan shaders source-only ([native/vulkan/shaders/snake.comp.glsl](../../native/vulkan/shaders/snake.comp.glsl) + conv1d/conv_transpose1d), awaiting glslc build pass.
- [x] `IAsyncEnumerable<AudioChunk>` output streaming surface ([Streaming/AudioStreamer.cs](../../src/HartsyInference.Audio/Streaming/AudioStreamer.cs) + [Streaming/AudioChunk.cs](../../src/HartsyInference.Audio/Streaming/AudioChunk.cs))
- [x] `LstmCell` / `BiLstm` modules ([Layers/LstmCell.cs](../../src/HartsyInference.Audio/Layers/LstmCell.cs), [Layers/BiLstm.cs](../../src/HartsyInference.Audio/Layers/BiLstm.cs)). Bonus: [Layers/UnidirectionalLstm.cs](../../src/HartsyInference.Audio/Layers/UnidirectionalLstm.cs) for stacked multi-layer LSTM (EnCodec bottleneck).
- [x] **Shared DSP statics** ([`Dsp/`](../../src/HartsyInference.Audio/Dsp/)) — **reuse these, don't re-roll per model** (see AGENTS.md "Reuse shared primitives"): [`NsfVocoderDsp`](../../src/HartsyInference.Audio/Dsp/NsfVocoderDsp.cs) (NSF harmonic source + forward-STFT mag/phase + iSTFT exp/sin head + reflection-pad + add-cropped + scale — shared by Kokoro iSTFTNet + CosyVoice HiFTNet) and [`DeterministicRng`](../../src/HartsyInference.Audio/Dsp/DeterministicRng.cs) (seeded xorshift+Box-Muller Gaussian/uniform — shared by NSF source, CFM/diffusion samplers, the speech-token sampler). Layout transpose `[1,C,T]↔[1,T,C]` is the existing `backend.Transpose2D` op (used 30+ places); the central linear is `WhisperOps.ProjectLinear`.

### Backend ops landed in 3.x (post-original-checklist additions)

- [x] `Conv1d` / `ConvTranspose1d` on `IBackend` — CPU complete, CUDA + Vulkan stubbed with native source files ready for build
- [x] `Sigmoid` / `Tanh` / `Elu` / `Snake` activations on `IBackend` — CPU complete; Vulkan `Sigmoid` working today via existing elementwise op 6; rest awaiting glslc/nvcc recompile

## 4. Neural Audio Codecs (`HartsyInference.Audio.Codecs`)

- [x] `EnCodec24kHz` encoder + decoder ([Models/Codecs/EnCodec/](../../src/HartsyInference.Audio/Models/Codecs/EnCodec/) — covers Bark, MusicGen, AudioGen)
- [x] `Mimi` decoder + encoder + transformer-of-codecs ([Models/Codecs/Mimi/](../../src/HartsyInference.Audio/Models/Codecs/Mimi/) — covers Sesame CSM, Moshi)
- [x] `DAC` encoder + decoder for 44.1 / 24 / 16 kHz variants ([Models/Codecs/Dac/](../../src/HartsyInference.Audio/Models/Codecs/Dac/) — covers IndexTTS, Higgs Audio, Spark-TTS HiFi-GAN)
- [x] `XCodec` decoder + encoder ([Models/Codecs/XCodec/](../../src/HartsyInference.Audio/Models/Codecs/XCodec/) — covers YuE; lifts DAC verbatim with codec-specific config)
- [x] `Vocos` (mel-input) ([Models/Vocoders/Vocos.cs](../../src/HartsyInference.Audio/Models/Vocoders/Vocos.cs) — pre-existing for F5-TTS)
- [x] `SNAC` decoder + encoder with hierarchical RVQ ([Models/Codecs/Snac/](../../src/HartsyInference.Audio/Models/Codecs/Snac/) — covers Orpheus TTS; 24/32/44.1 kHz presets)
- [x] `WavTokenizer` encoder + decoder with iSTFT head ([Models/Codecs/WavTokenizer/](../../src/HartsyInference.Audio/Models/Codecs/WavTokenizer/) — single 4096-entry codebook)
- [x] `BiCodec` semantic + global encoders ([Models/Codecs/BiCodec/](../../src/HartsyInference.Audio/Models/Codecs/BiCodec/) — covers Spark-TTS)
- [x] `NeuCodec` decode path ([Models/Codecs/NeuCodec/](../../src/HartsyInference.Audio/Models/Codecs/NeuCodec/) — covers NeuTTS Air; single FSQ codebook `4^8=65536`, Vocos-transformer backbone + iSTFT head, 16 kHz in / 24 kHz out). Encoder (ref-audio → codes) deferred.
- [x] FSQ codec primitives ([Models/Codecs/Fsq.cs](../../src/HartsyInference.Audio/Models/Codecs/Fsq.cs) — parity-aware tanh bound + base-L packing)
- [x] Weight-norm fusion — runtime via [Models/Codecs/WeightNormFusion.cs](../../src/HartsyInference.Audio/Models/Codecs/WeightNormFusion.cs); offline CLI at [samples/FuseWeightNorm/](../../samples/FuseWeightNorm/)
- [x] **Streaming codec wrapper** ([Models/Codecs/StreamingCodec.cs](../../src/HartsyInference.Audio/Models/Codecs/StreamingCodec.cs)) — generic `StreamingCodecEncoder<T>` / `StreamingCodecDecoder<T>` over any of the 9 codecs, for live-mic encode and live-playback decode use cases.
- [x] Codec smoke tests — config + construction tests for every codec ([tests/HartsyInference.Audio.Tests/CodecSmokeTests.cs](../../tests/HartsyInference.Audio.Tests/CodecSmokeTests.cs))
- [ ] Validation: round-trip encode→decode STOI > 0.95 vs Python reference — **pending checkpoint downloads + per-codec integration tests**. Smoke-test scaffolding in place; the gated test attribute pattern from F5TtsSmokeTests is the convention to follow.

### Reusable codec infrastructure (bonus, not in original §4 checklist but enables it)

- [x] `ResidualVectorQuantizer` — generic euclidean RVQ for EnCodec ([Models/Codecs/ResidualVectorQuantizer.cs](../../src/HartsyInference.Audio/Models/Codecs/ResidualVectorQuantizer.cs))
- [x] `DacResidualVectorQuantizer` — cosine RVQ with per-codebook in/out projections (DAC, XCodec, Mimi)
- [x] `SnacResidualVectorQuantizer` — hierarchical RVQ with per-codebook stride
- [x] `WeightNormFusionT` — transpose-conv variant of weight-norm fusion (separate axis handling)
- [x] `MimiTransformer` — small causal transformer-of-codecs (RoPE + MHA + GeLU FFN; reused on encoder + decoder sides of Mimi)

## 5. Speech-to-Text (STT)

### Whisper family — BUILT (greedy decode); long-form + word-timestamps pending
- [x] [`WhisperEncoder.cs`](../../src/HartsyInference.Audio/Models/Whisper/WhisperEncoder.cs) — Conv1D × 2 + sinusoidal pos + N × ResidualAttentionBlock
- [x] [`WhisperDecoder.cs`](../../src/HartsyInference.Audio/Models/Whisper/WhisperDecoder.cs) — embed + learned pos + N × cross-attention block + self-attn KV cache + `PrecomputeCrossKv`
- [x] [`WhisperPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/WhisperPipeline.cs) — mel → encode → cross-KV precompute → greedy decode loop with suppress-tokens. **Temperature fallback deferred** (greedy only in v1).
- [x] [`WhisperOptions.cs`](../../src/HartsyInference.Audio/Pipelines/WhisperOptions.cs) — language, task (translate), timestamps, max-tokens; model size via `WhisperConfig`. **Beam search deferred** (greedy only).
- [x] Distil-Whisper variant support — `WhisperConfig.{DistilLargeV2, DistilLargeV3, DistilMediumEn, DistilSmallEn}` set `DecoderLayers = 2`, reuse the same encoder/decoder classes
- [x] Whisper-large-v3-turbo support — `WhisperConfig.LargeV3Turbo` (128 mel bins from `LargeV3`, 4-layer decoder)
- [ ] Long-form sequential decoding (timestamp-driven) + chunked variant — documented as a later pass in `WhisperPipeline`
- [ ] Word-level timestamps via cross-attention DTW (alignment heads table per model size)
- **Validated:** `WhisperEndToEndTests` transcribes canonical JFK audio (real end-to-end; skips cleanly when no cached model / network).

### NVIDIA NeMo family (`HartsyInference.Audio.Nemo` or similar)
- [ ] `FastConformerEncoder.cs` — 8x conv subsampling + Conformer blocks with limited-context attention (shared by Parakeet, Canary, FireRedASR-AED)
- [ ] `CtcDecoder.cs` — greedy + beam search with blank collapse (Parakeet-CTC)
- [ ] `RnntDecoder.cs` — prediction net (LSTM) + joint net + hypothesis-extension beam (Parakeet-RNNT)
- [ ] `TdtDecoder.cs` — joint net with token + duration heads, greedy decode (Parakeet-TDT)
- [ ] `ParakeetPipeline.cs` — variant-dispatched
- [ ] `CanaryPipeline.cs` — FastConformer + Transformer decoder + AST prompt tokens
- [ ] Cache-aware streaming for Parakeet and Canary chunked modes

### Moonshine — BUILT
- [x] [`MoonshinePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/MoonshinePipeline.cs) — Conv1D front-end + RoPE encoder/decoder ([`MoonshineEncoder`](../../src/HartsyInference.Audio/Models/Moonshine/MoonshineEncoder.cs)/[`MoonshineDecoder`](../../src/HartsyInference.Audio/Models/Moonshine/MoonshineDecoder.cs)) + SentencePiece byte-fallback BPE ([`MoonshineTokenizer`](../../src/HartsyInference.Audio/Models/Moonshine/MoonshineTokenizer.cs))
- [x] Hallucination guard — **approximated** via a token-count cap proportional to encoder-seconds (`MoonshinePipeline`); the dynamic rate-based ~6.5 tok/s ceiling is not yet enforced
- **Validated:** `MoonshineEndToEndTests` transcribes canonical JFK audio (real end-to-end; skip-gated on cache/network).

### Kyutai STT (delayed-streams) — BUILT (structural); checkpoint-gated validation pending
> `kyutai/stt-1b-en_fr` + `kyutai/stt-2.6b-en`. Helium = `Qwen2Model` (attn-bias off) driven headless via
> `ForwardEmbeds`; audio in via the built Mimi (32-codebook DSM variant). Research: [`KYUTAI_DSM_ARCHITECTURE.md`](../Research/KYUTAI_DSM_ARCHITECTURE.md). Files under [`Models/Kyutai/`](../../src/HartsyInference.Audio/Models/Kyutai/) + [`KyutaiSttPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/KyutaiSttPipeline.cs).
- [x] [`KyutaiSttConfig.cs`](../../src/HartsyInference.Audio/Models/Kyutai/KyutaiSttConfig.cs) — `Stt1B` (16L/16 heads/head_dim 128/vocab 8001) + `Stt2_6B` (48L/32 heads/head_dim 64/vocab 4001) Helium presets + DSM params (32 codebooks, 2049 codebook-vocab, audio offsets, silence-prefix/delay). **Tested.**
- [x] [`MimiConfig.Mimi24kHzDsm`](../../src/HartsyInference.Audio/Models/Codecs/Mimi/MimiConfig.cs) — 32-codebook (1 semantic + 31 acoustic) DSM variant.
- [x] [`KyutaiSttModel.cs`](../../src/HartsyInference.Audio/Models/Kyutai/KyutaiSttModel.cs) — shared `embed_tokens` table (text + 32×2049 audio rows) + headless Helium; per-frame input = text-row + Σ audio-code rows; tied head projects over text-vocab rows. Audio-offset math **tested**.
- [x] [`KyutaiSttPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/KyutaiSttPipeline.cs) — silence-pad → Mimi encode → per-frame Helium step → greedy text token → PAD-stripped token ids out.
- [ ] **Reconcile on checkpoint load:** gated MLP ships fused `fc1`/`fc2` (split to gate|up for our Qwen2 layer), shared-embedding double-nested key, the WORD-boundary token id, and the 32-vs-8 codebook count + Mimi 12.5 Hz frame rate (shared Mimi reconcile with CSM).
- [x] **SentencePiece text decoder BUILT (2026-06-21)** — [`KyutaiSttTokenizer.cs`](../../src/HartsyInference.Tokenizers/KyutaiSttTokenizer.cs): `Decode(ids)` → text (ML.Tokenizers `SentencePieceTokenizer`, ▁→space) + `Encode`. Consumer supplies the model's `text.model`. Compile-verified (SentencePiece can't be hand-crafted tiny for a unit test); runtime decode validated when the real spm is present.
- [ ] Word-level timestamps from the WORD/PAD stream + delay subtraction; streaming `IAsyncEnumerable` surface.

### SenseVoice + FireRedASR
- [ ] `SenseVoiceEncoder.cs` — 50-layer SANM encoder + LFR frontend + CTC head
- [ ] `SenseVoicePipeline.cs` — special-token parser (`<emotion><lang><event><text>`)
- [ ] `FireRedAsrPipeline.cs` (AED variant) — Conformer + Whisper-style decoder
- [ ] `FireRedAsrLlmPipeline.cs` (LLM variant) — Conformer + adapter + Qwen2 decoder (dotLLM dependency)

### .nemo file format support
- [ ] `NemoFileLoader.cs` — extract tar to {ckpt, config.yaml}; route to model loader
- [ ] Offline converter: `.nemo` → safetensors + config.json for our standard pipeline

## 6. Text-to-Speech (TTS)

### Kokoro (first TTS to ship) — BUILT (real iSTFTNet audio); English G2P BUILT
- [x] `PlBertEncoder` — [`KokoroPlBert.cs`](../../src/HartsyInference.Audio/Models/Kokoro/KokoroPlBert.cs): ALBERT with weight sharing — one shared `AlbertLayer` instance looped 12× in the forward pass
- [x] [`KokoroTextEncoder.cs`](../../src/HartsyInference.Audio/Models/Kokoro/KokoroTextEncoder.cs) — Embed + Conv1D × 3 (ChannelLN + LeakyReLU) + BiLSTM
- [x] [`KokoroProsodyPredictor.cs`](../../src/HartsyInference.Audio/Models/Kokoro/KokoroProsodyPredictor.cs) — DurationEncoder (3× BiLSTM + AdaLayerNorm) + duration LSTM + projection + F0/N AdaINResBlock chains
- [x] [`KokoroIStftNetDecoder.cs`](../../src/HartsyInference.Audio/Models/Kokoro/KokoroIStftNetDecoder.cs) — **real iSTFTNet generator forward** (replaces the prior sine placeholder). Full chain: F0/N convs + asr_res → encode/decode AdainResBlk1d → HnNSF harmonic source (9 harmonics, deterministic phase accumulation + fixed-seed Gaussian noise, `tanh(Linear[9→1])`) → forward STFT → two ConvTranspose1d upsamples (10×, 6×) with per-level noise injection + 3-kernel MRF AdaIN/**Snake** resblocks → `conv_post` → magnitude(`exp`)/phase(`sin`) iSTFT head. Mirrors StyleTTS2 `kokoro/istftnet.py` Decoder+Generator. Required a non-power-of-two direct-DFT fallback in [`Fft.cs`](../../src/HartsyInference.Audio/Preprocessing/Fft.cs) for the n_fft=20 synthesis transform. See [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md).
- [x] [`KokoroPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/KokoroPipeline.cs) — end-to-end `Synthesize` (voice-pack load → tokenize → PLBERT → TextEncoder → duration predict → length-regulate → F0/N predict → decoder). Now produces real iSTFTNet speech.
- [x] **English G2P BUILT (2026-06-21)** — [`EnglishG2P.cs`](../../src/HartsyInference.Audio/Frontends/EnglishG2P.cs) + [`ArpabetToIpa.cs`](../../src/HartsyInference.Audio/Frontends/ArpabetToIpa.cs): CMUdict→IPA (stress marks, AH0→ə/ER0→ɚ) + integer normalization + letter-to-sound OOV fallback. `ToIpa(text)` → the IPA string `KokoroPhonemeTokenizer` consumes. Approximation of misaki (intelligible, not bit-exact; stress before vowel not syllable); consumer supplies the public-domain `cmudict.dict`. 6 tests pass (in-memory dict). Chinese/Japanese still phased per [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md).
- **Validated:** `KokoroFoundationTests` + `KokoroPipelineSmokeTests` (6 tests) load all 548 tensors and run the full real generator to finite non-degenerate 24 kHz audio; skip cleanly when the cache is missing. Numeric reference-diff vs Python (§9) still pending a checkpoint-paired run.

### F5-TTS — ✅ VERIFIED END-TO-END (real-weight, bit-exact DiT + sample loop)
> Numerically validated vs upstream F5-TTS: DiT velocity corr 1.0, full `CFM.sample` generated mel corr 1.0, Vocos corr 0.9999. 4 bugs fixed (the structural code was already complete) — see [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) + memory `f5tts-build`.
- [x] `F5TtsDiT` — [`F5Dit.cs`](../../src/HartsyInference.Audio/Models/F5Tts/F5Dit.cs) + [`F5DitBlock.cs`](../../src/HartsyInference.Audio/Models/F5Tts/F5DitBlock.cs): 22 DiT blocks, AdaLN-Zero, RoPE, ConvNeXt stem. **Velocity bit-exact** after fixing: filler-tail masking (ConvNeXt), ×1000 timestep sinusoid scale, erf-GELU stem vs tanh-GELU FFN.
- [x] [`F5TtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/F5TtsPipeline.cs) — flow-matching Euler with **cond-anchored** CFG (`pred + cfg·(pred−null)`); the reference region evolves freely and is replaced only at the end (not per-step). `GenerateMel` exposes the pre-vocoder mel + accepts injected noise for deterministic tests.
- [x] Sway sampling scheduler — [`F5SwaySamplingScheduler.cs`](../../src/HartsyInference.Audio/Models/F5Tts/F5SwaySamplingScheduler.cs) (sway-warped Euler timesteps)
- [x] `vocos-mel-24khz` vocoder — [`Vocos.cs`](../../src/HartsyInference.Audio/Models/Vocoders/Vocos.cs): Conv embed → 8 ConvNeXt blocks → mag/phase head → iSTFT
- **Validated:** `F5TtsSmokeTests` + `F5ForwardLiveTest` (skip/early-exit when checkpoint absent).

### Dia-1.6B (Nari Labs) — 🔬 TRANSFORMER VERIFIED bit-exact (encoder + decoder corr 1.0); DAC wiring remains
> Real-weight parity vs the upstream `DiaModel`: encoder (12L) + decoder (18L, cross-attn / 9-channel summed embeddings / fused `logits_dense`) both corr 1.0. New `DiaWeights.cs` adapts the nari-native DenseGeneral layout ([in,h,k]→[out,in], `wi_fused`/`wo`/`logits_dense` rename+transpose) and provides split-half RoPE. 5 bugs (split-half RoPE, attn scale 1.0, DenseGeneral transpose, prefix `encoder.`/`decoder.`, missing `cache.AdvanceLength()`). Remaining: download + wire DAC 44.1kHz 9cb (shared `DacDecoder` ✅) + the delay-pattern AR loop. See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) + memory `dia-fishspeech-build`.

### Dia-1.6B — (prior scaffold notes)
> `nari-labs/Dia-1.6B` — a T5/Whisper-style **encoder-decoder** TTS (the **first cross-attention transformer
> in the Audio package**) generating a 9-codebook DAC grid (delay pattern) → 44.1 kHz. Research:
> [`DIA_TTS_ARCHITECTURE.md`](../Research/DIA_TTS_ARCHITECTURE.md). Files under [`Models/Dia/`](../../src/HartsyInference.Audio/Models/Dia/) + [`DiaPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/DiaPipeline.cs).
- [x] [`DiaConfig.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaConfig.cs) — `Dia1_6B` (enc 12L/1024/MHA, dec 18L/2048/GQA 16:4 self + 16:16 cross, head_dim 128, 9 channels, audio vocab 1028, delay `[0,8..15]`, CFG 3.0). **Tested.**
- [x] [`DiaAttention.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaAttention.cs) + [`DiaHeads.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaHeads.cs) — parameterized attention (encoder self non-causal MHA, decoder self causal GQA + RoPE + KV-cache, decoder cross-attn over precomputed encoder K/V) reusing `RotaryEmbedding` + `ScaledDotProductAttention` + `ProjectLinear`.
- [x] [`DiaMlp.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaMlp.cs) (fused gate_up SwiGLU) + [`DiaEncoder.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaEncoder.cs) + [`DiaDecoder.cs`](../../src/HartsyInference.Audio/Models/Dia/DiaDecoder.cs) (9 summed channel embeds + fused `logits_dense` → 9×1028 head).
- [x] [`MusicGenDelay.Apply`](../../src/HartsyInference.Audio/Models/Music/MusicGenDelay.cs) BOS/PAD overload (distinct lead-in vs tail fill). **Tested** (round-trip).
- [x] [`DiaPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/DiaPipeline.cs) — cond+uncond encode → per-branch cross-KV → CFG delayed-AR decode (two decoders sharing weights) → channel-0 EOS + flush → revert delay → DAC decode → 44.1 kHz.
- [x] **Synthetic-weight forward verified** ([`DiaTests`](../../tests/HartsyInference.Audio.Tests/DiaTests.cs)) — tiny config runs encoder + cross-attn decoder step to finite output (exercises every net-new attention flavor).
- [ ] **Reconcile on checkpoint load:** original-repo `DenseGeneral` fused-tensor reshape vs HF key layout, RoPE on cross-attn (currently off), and audio-prompt voice cloning (prefix DAC codes). eSpeak byte tokenization is caller-side.

### XTTS-v2
- [ ] `XttsGpt.cs` — GPT-2-style 30L × 1024
- [ ] `XttsTokenizer.cs` — shared BPE + `[<lang>]` prefix tokens + per-language romanizers (pinyin, cutlet)
- [ ] `XttsConditioningEncoder.cs` — Perceiver resampler → 32×1024 gpt_cond_latent
- [ ] `XttsSpeakerEncoder.cs` — H/ASP ECAPA-TDNN → 512-d
- [ ] `XttsHifiGan.cs` — takes 1024-d GPT latents (not mel), FiLM-conditioned per ResBlock
- [ ] Streaming variant (chunk_size=20 mel tokens, overlap_wav_chunks=1024)

### Bark — SCAFFOLD COMPLETE (3-stage cascade); checkpoint-gated
> Files under [`Models/Bark/`](../../src/HartsyInference.Audio/Models/Bark/) + [`BarkPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/BarkPipeline.cs). **Shared GPT-2 backbone built** ([`Models/LanguageModels/Gpt/`](../../src/HartsyInference.Audio/Models/LanguageModels/Gpt/) — `GptBackbone`/`GptBlock`/`GptConfig`) that all three Bark stages reuse and **XTTS/ChatTTS will reuse**. EnCodec 24 kHz decoder already built (§4).
- [x] **Shared `GptBackbone`** — GPT-2 pre-norm (learned abs pos, MHA, 4× GELU MLP, bias=False, LayerNorm), causal + non-causal. **Validated** via synthetic-weight forward (3 tests, both mask modes finite). Full-sequence (no KV cache); AR re-feeds the prefix — perf-tunable later.
- [x] [`BarkCausalStage.cs`](../../src/HartsyInference.Audio/Models/Bark/BarkCausalStage.cs) — semantic + coarse stages (token embed + `GptBackbone` + lm_head, AR via shared `NucleusSampler`).
- [x] [`BarkFineModel.cs`](../../src/HartsyInference.Audio/Models/Bark/BarkFineModel.cs) — non-causal refiner: 8 codebook embeds summed + 7 heads, iterative argmax fill of codebooks 2..7.
- [x] [`BarkConfig.cs`](../../src/HartsyInference.Audio/Models/Bark/BarkConfig.cs) — Full/Small presets + the Bark token-offset constants. **Tested**.
- [x] [`BarkPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/BarkPipeline.cs) — semantic → coarse (de-interleave 2 codebooks) → fine → EnCodec 24 kHz decode → audio.
- [x] **BERT WordPiece tokenizer BUILT (2026-06-21)** — [`BertWordPieceTokenizer.cs`](../../src/HartsyInference.Tokenizers/BertWordPieceTokenizer.cs) (mBERT-cased, raw ids, no special tokens) + [`AudioTextFrontend.BarkText`](../../src/HartsyInference.Audio/Frontends/AudioTextFrontend.cs) applies the `+10048` offset. Consumer supplies `vocab.txt`. 2 tests pass.
- [ ] `BarkSpeakerPrompt.cs` (3-stream speaker-prompt history) + checkpoint validation (`suno/bark`).

### CosyVoice 2 — SCAFFOLD COMPLETE (non-streaming); checkpoint-gated validation pending
> All components build and the exactly-specified pieces are unit-tested; the rest are structurally-correct scaffolds per the FunAudioLLM repo, **awaiting the 4.4 GB `FunAudioLLM/CosyVoice2-0.5B` checkpoint** for first-run validation (none on this host). Files under [`src/HartsyInference.Audio/Models/CosyVoice/`](../../src/HartsyInference.Audio/Models/CosyVoice/). **Architecture correction:** the LM is *not* the research doc's "single extended-vocab softmax" — the real `llm.pt` keeps the Qwen text embedding and adds separate `speech_embedding` / `llm_decoder` / `llm_embedding` heads; built to match.
- [x] [`CosyVoiceConfig.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/CosyVoiceConfig.cs) — CV2-0.5B composition config (LM + flow + HiFTNet + sampling sub-configs). **Tested** (preset values).
- [x] [`CosyVoiceQwenLm.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/CosyVoiceQwenLm.cs) — Qwen2.5-0.5B backbone (reuses the local `Qwen2Model` + `StreamingKvCache`) + separate `speech_embedding`/`llm_decoder`/`llm_embedding`; zero-shot text→speech-token AR loop.
- [x] [`SpeechSampler.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/SpeechSampler.cs) — rep-penalty → temp → top-k → top-p → RAS, deterministic. **Tested** (top-k=argmax, determinism, candidate masking, RAS breaks degenerate loops).
- [x] [`S3Tokenizer.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/S3Tokenizer.cs) — FSQ speech tokenizer (D=8/L=3 → 6561). **`PackFsqTokens` exact + tested** (center/min/max/mixed codes); the encoder is now the **real 6-block RoPE transformer** (conv stem 4× subsample → `encoders.{i}` pre-norm MHSA+GELU-FFN with interleaved RoPE → `quantizer.project_down`), synthetic-weights forward tested (`speech_tokenizer_v2.onnx` remains an exact-weights fallback).
- [x] [`CamPlusSpeakerEncoder.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/CamPlusSpeakerEncoder.cs) — 192-d L2-normalized embedding via TDNN → stats-pooling → FC. Contract-correct scaffold (full FCM + dense D-TDNN + CAM masking checkpoint-gated; `campplus.onnx` fallback).
- [x] CosyVoice CFM — [`ICfmEstimator`](../../src/HartsyInference.Audio/Models/CosyVoice/ICfmEstimator.cs) + [`ConditionalCfm.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/ConditionalCfm.cs) (OT-CFM Euler + CFG solver, **exact + tested**: constant-velocity integration, CFG combine, determinism) + [`CausalConditionalDecoder.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/CausalConditionalDecoder.cs) (timestep-injected resnet+attention UNet1D estimator) + [`CosyVoiceFlow.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/CosyVoiceFlow.cs) (token-embed → **real `UpsampleConformerEncoder`** → `encoder_proj`/`spk_affine` → CFM) + [`UpsampleConformerEncoder.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/UpsampleConformerEncoder.cs) (`embed.out.0` linear → N macaron-FFN conformer blocks → `up_layer` ConvTranspose1d 2× upsample 25→50 Hz → N post-up conformer blocks; macaron FFN + MHSA + pointwise/GLU/depthwise conv module). Synthetic-weights forward tested. The exact estimator down/up topology is the remaining checkpoint-gated piece.
- [x] [`HiFTNetVocoder.cs`](../../src/HartsyInference.Audio/Models/CosyVoice/HiFTNetVocoder.cs) — mel → 24 kHz: internal F0 predictor (ConvRNN) + NSF harmonic source + 3 ConvTranspose upsamples with source injection + plain-Snake MRF resblocks + magnitude/phase iSTFT head. Shares the validated Kokoro iSTFTNet NSF/iSTFT algorithm.
- [x] [`CosyVoicePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/CosyVoicePipeline.cs) — non-streaming orchestration (zero-shot reference → S3 + CAM++ → LM → flow → vocoder; or precomputed-embedding mode). **13 checkpoint-free tests pass.**
- [ ] **Streaming pipeline** — the 5:15 text:speech interleave + chunk-aware CFM flush + per-chunk vocoder + `IAsyncEnumerable<AudioChunk>` (150 ms first-packet). Non-streaming path built first; streaming is the follow-up.
- [ ] **Checkpoint converter + first-run validation** — bucket `llm.pt` / `flow.pt` / `hift.pt` (+ `campplus.onnx` / `speech_tokenizer_v2.onnx`) into the per-component LoadWeights dicts; reconcile exact state-dict keys (LM head/embedding, flow estimator topology, CAM++ D-TDNN, HiFTNet source-inject params) against the real weights, then env-gated `CosyVoiceGenerationTests`.
- [ ] **CosyVoice 1** (300M custom TransformerLM + VQ-4096 + UNet1D CFM) — out of scope for this pass; CV2 shipped first.

### Kyutai TTS (delayed-streams) — SCAFFOLD COMPLETE (depformer); checkpoint-gated
> `kyutai/tts-1.6b-en_fr`. Moshi RQ-Transformer: temporal Helium (`Qwen2Model` headless, RoPE θ=10000) +
> the **depformer** (RoPE-free per-step-weighted depth transformer over 32 codebooks) + Mimi decode. Research:
> [`KYUTAI_DSM_ARCHITECTURE.md`](../Research/KYUTAI_DSM_ARCHITECTURE.md). Files under [`Models/Kyutai/`](../../src/HartsyInference.Audio/Models/Kyutai/) + [`KyutaiTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/KyutaiTtsPipeline.cs).
- [x] [`KyutaiTtsConfig.cs`](../../src/HartsyInference.Audio/Models/Kyutai/KyutaiTtsConfig.cs) — temporal Helium (2048/16L, θ=10000, SwiGLU 8448) + depth sub-config + delays (text/cb0=0, acoustic=2) + stream delay (16 steps) + speaker dim. **Tested.**
- [x] [`MoshiDepthConfig.cs`](../../src/HartsyInference.Audio/Models/Kyutai/MoshiDepthConfig.cs) — depformer (dim 1024 / 4L / 16 heads / FFN 3072 / dep_q 32 / low-rank 128) + the per-step weight-set schedule (11 sets: cb 0–7 unique, 8–15→8, 16–23→9, 24–31→10). **Tested.**
- [x] [`MoshiDelay.cs`](../../src/HartsyInference.Audio/Models/Kyutai/MoshiDelay.cs) — per-codebook delay Apply/Revert. **Exact + tested.**
- [x] [`MoshiDepthTransformer.cs`](../../src/HartsyInference.Audio/Models/Kyutai/MoshiDepthTransformer.cs) — AR over 32 codebooks: per-step input projection over low-rank embeddings → per-set RMSNorm + no-RoPE causal attention (reuses `backend.ScaledDotProductAttention`) + SwiGLU → per-step head → `NucleusSampler`.
- [x] [`KyutaiTtsModel.cs`](../../src/HartsyInference.Audio/Models/Kyutai/KyutaiTtsModel.cs) + [`KyutaiTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/KyutaiTtsPipeline.cs) — temporal step over summed text+audio embeds → depformer frame → Mimi decode → 24 kHz PCM.
- [ ] **Reconcile on checkpoint load:** depformer weight-key layout (per-step sets / low-rank / multi-linear), the delayed-coordinate handling (wire `MoshiDelay` into the gen loop), the PAD/EPAD/WORD text state machine + 2-step lookahead stream, and **speaker cross-attention** (the decoder-only backbone has no cross-attn sublayer yet — runs unconditioned now), plus CFG/control LUT conditioners.

### Qwen3-TTS — backbone (stage a) BUILT + verified; talker/MTP/vocoder/ECAPA staged
> Alibaba Qwen3-TTS (12 Hz). The most complex AudioLab model (5 net-new components). Research: [`QWEN3_TTS_ARCHITECTURE.md`](../Research/QWEN3_TTS_ARCHITECTURE.md). Backbone under [`Models/LanguageModels/Qwen3/`](../../src/HartsyInference.Audio/Models/LanguageModels/Qwen3/).
- [x] [`Qwen3Model.cs`](../../src/HartsyInference.Audio/Models/LanguageModels/Qwen3/Qwen3Model.cs) — reusable headless Qwen3 decoder: **per-head q_norm/k_norm** (RMSNorm over head_dim, pre-RoPE) + **decoupled head_dim** + GQA + SwiGLU, reusing `DiaHeads`/`RotaryEmbedding`/`WhisperOps`. **Synthetic-forward verified.** `Talker1_7B`/`CodePredictor` presets. (Also unlocks future Qwen3 image/text models.)
- [ ] **Staged:** talker dual embedding (text + codec) + `codec_head` + `text_projection`; **3D interleaved mRoPE** (currently 1D); 5-layer **MTP CodePredictor** (15 codebooks); the custom **Snake/ConvNeXt causal-conv codec decoder** (split-RVQ); Mimi encoder reuse; **ECAPA-TDNN** speaker encoder; clone/custom_voice/voice_design modes.

### IndexTTS 1.5 + 2
- [ ] `IndexT2sGpt.cs` — 24L × 1280
- [ ] `IndexS2MelDit.cs` — 13L × 512 with WaveNet final layer (non-causal — no streaming)
- [ ] `IndexSemanticCodec.cs` — Vocos-style ConvNeXt encoder
- [ ] `IndexConformerPerceiver.cs` for speaker + emotion conditioning
- [ ] BigVGAN v2 22kHz vocoder (download from `nvidia/bigvgan_v2_22khz_80band_256x`)
- [ ] Optional Qwen-3 0.6B-emo for text-emotion conditioning

### SparkTTS — ✅ VERIFIED END-TO-END (real-weight, controllable mode, bit-exact)
> Files under [`Models/SparkTts/`](../../src/HartsyInference.Audio/Models/SparkTts/) + [`SparkTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/SparkTtsPipeline.cs). **Maximal reuse:** the LM is a plain Qwen2.5-0.5B reusing `Qwen2Model` verbatim; the wave generator reuses `DacDecoder`; the prompt tokenizer reuses `BpeTokenizer` + `ByteLevelCodec`. Validation detail in [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) + memory `sparktts-build`.
- [x] [`SparkTtsConfig.cs`](../../src/HartsyInference.Audio/Models/SparkTts/SparkTtsConfig.cs) — `V0_5B` (Qwen2.5-0.5B, `VocabSize=166000`) + token-ID bases **reconciled to the real `added_tokens.json`** (global=151665, semantic=155761, structure markers) + BiCodec decode config.
- [x] [`SparkTtsLm.cs`](../../src/HartsyInference.Audio/Models/SparkTts/SparkTtsLm.cs) — reuses `Qwen2Model` + `StreamingKvCache` + `NucleusSampler` (+ a `greedy` path). **LM logits corr 1.0 (top-1 100%); greedy tokens match the reference exactly (32/32 global + 179/179 semantic).**
- [x] [`BiCodecDecoder.cs`](../../src/HartsyInference.Audio/Models/SparkTts/BiCodecDecoder.cs) + [`SparkVocosBackbone.cs`] — **rewritten to the real `detokenize`**: factorized 8-D VQ → `out_project`; global FSQ → channel-major-flatten d-vector; AdaLN Vocos PreNet (12 ConvNeXt + 2 SamplingBlock×3); reused `DacDecoder` wave generator. PostNet is training-only. **z_q / d_vector / prenet / wav all corr 1.0.**
- [x] [`SparkTtsTokenizer.cs`](../../src/HartsyInference.Tokenizers/SparkTtsTokenizer.cs) — composes `BpeTokenizer` + `ByteLevelCodec` + the `added_tokens.json` map (`<|…|>` split). **Reproduces the validated prompt ids exactly.**
- [x] [`SparkTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/SparkTtsPipeline.cs) — `LoadFromDirectory`/`LoadAsync` + `BuildControllablePrompt` + `SynthesizeControllable(text, gender, pitch, speed)`. **Full in-engine e2e: text → 57280 samples (3.58s), identical to reference.** Gated test `SparkTtsTests.EndToEnd_RealWeights_ControllablePrompt` (env `SPARK_DIR`).
- [x] **Shared-component opt-ins added** (defaults unchanged): `DacConfig.TransposeWeightNormDim0` (descript dim-0 weight-norm) + `DacConfig.DecoderKernelSizes` (explicit `[16,11,8,4]`).
- [ ] **BiCodec encoder (cloning) + wav2vec2-XLSR + ECAPA/Perceiver** — encode-side (reference→tokens) for zero-shot voice cloning; not built (controllable mode needs only the decoder, which is done).

### Chatterbox (Resemble AI) — CORE BUILT (T3 + voice encoder, synthetic-forward verified); S3Gen reuse-wiring pending
> `ResembleAI/chatterbox` (MIT). T3 Llama LM → S3 tokens (25 Hz FSQ 6561) → S3Gen (CosyVoice2 flow + HiFTNet)
> → 24 kHz. Research: [`CHATTERBOX_ARCHITECTURE.md`](../Research/CHATTERBOX_ARCHITECTURE.md). Files under [`Models/Chatterbox/`](../../src/HartsyInference.Audio/Models/Chatterbox/). **Maximal reuse:** S3Gen = the existing CosyVoice `ConditionalCfm` + `HiFTNetVocoder` + `CosyVoiceFlow` + `S3Tokenizer` + `CamPlusSpeakerEncoder`.
- [x] [`ChatterboxConfig.cs`](../../src/HartsyInference.Audio/Models/Chatterbox/ChatterboxConfig.cs) — T3 `Llama_520M` (1024/30L/16 MHA/head_dim 64/SwiGLU 4096/θ=500000) + vocabs + token ids + gen defaults. **Tested.**
- [x] [`ChatterboxT3.cs`](../../src/HartsyInference.Audio/Models/Chatterbox/ChatterboxT3.cs) — headless `Qwen2Model` + text/speech embeds + learned positions + `speech_head` + cond encoder (speaker `Linear(256→1024)` + exaggeration `Linear(1→1024)`); `[cond++text++speech]` prefill + AR speech-token gen (rep-penalty + min-p + top-p). **Synthetic-weights forward verified** (generates valid in-range tokens).
- [x] [`ChatterboxVoiceEncoder.cs`](../../src/HartsyInference.Audio/Models/Chatterbox/ChatterboxVoiceEncoder.cs) — GE2E 3-layer LSTM (40→256) + proj + L2-norm, reusing `UnidirectionalLstm`. **Synthetic-weights forward verified** (finite + L2-normalized).
- [x] **Fixed shared `UnidirectionalLstm` bug** — multi-layer with `InputDim < HiddenDim` reshaped the per-step buffer to too-few elements (only worked when input dim == hidden); now allocates per-layer at `dimIn`. Kokoro/EnCodec LSTM tests still green.
- [x] `min_p` added to `NucleusSampler` (backward-compatible) + `t3` learned-position handling.
- [ ] **S3Gen pipeline wiring** — `ChatterboxPipeline` assembling T3 tokens → `CosyVoiceFlow` → `HiFTNetVocoder` (the reused stack, needs a Chatterbox-tuned config: cosine CFM schedule + cfg 0.7 + 10 steps + HiFT [8,5,3]); prompt-speech perceiver resampler; llama3 RoPE scaling; Perth watermark. Deferred.

### Fish-Speech 1.5 — 🔬 DualAR LM VERIFIED (slow corr 1.0, fast corr 0.9999); firefly codec remains
> Real-weight parity vs the upstream `DualARTransformer`: slow backbone (24L Llama) logits bit-exact, fast depth-LM (4L) corr 0.9999. The checkpoint is a fused-key Llama; `FishSpeechDualAr.Remap` adapts it to the Qwen2 split layout. 4 fixes: key adapter (wqkv/w1w2w3/norm renames), interleaved RoPE (new `Qwen2Config.Rope`), `scale_codebook_embeddings=False`, and the fast model takes the PRE-norm slow hidden (`norm_fastlayer_input=False`). Remaining: the firefly-gan-vq codec (FSQ + HiFiGAN-SiLU — a separate reconcile). See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) + memory `dia-fishspeech-build`.

### Fish-Speech 1.5 / OpenAudio S1 — (prior scaffold notes)
> DualAR text2semantic (slow Llama + fast depth transformer) + firefly-gan-vq codec. Research: [`FISH_SPEECH_ARCHITECTURE.md`](../Research/FISH_SPEECH_ARCHITECTURE.md). Files under [`Models/FishSpeech/`](../../src/HartsyInference.Audio/Models/FishSpeech/) + [`FishSpeechPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/FishSpeechPipeline.cs).
- [x] [`FishSpeechDualAr.cs`](../../src/HartsyInference.Audio/Models/FishSpeech/FishSpeechDualAr.cs) — slow + fast both **reuse `Qwen2Model`** (headless); dual summed-codebook embedding (offset table, ×1/√(N+1)); slow→semantic, fast depth-AR→8 codebooks. Synthetic-forward verified (valid semantic + codebooks).
- [x] [`FireflyDecoder.cs`](../../src/HartsyInference.Audio/Models/FishSpeech/FireflyDecoder.cs) — codebook dequant → **reused `VitsHiFiGan`** (firefly upsample) → 44.1 kHz. Synthetic-forward verified.
- [x] [`FishSpeechPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/FishSpeechPipeline.cs) — text prefill → AR frames → firefly decode.
- [ ] **Staged:** firefly grouped-residual **FSQ** (levels (8,5,5,5)) + ConvNeXt resample + SiLU (current uses learned embeds + LeakyReLU), Fish fused `wqkv`/`w1w2w3` key adapter, BPE tokenizer + special-token ids, and the openaudio-s1 DAC-codec variant (Snake + RVQ).

### Demucs (HTDemucs) — audio FX (stem separation): novel components BUILT + verified; full assembly staged
> Hybrid Transformer Demucs v4 — dual-branch U-Net + cross-domain transformer → 4 stereo stems. Research:
> [`HTDEMUCS_ARCHITECTURE.md`](../Research/HTDEMUCS_ARCHITECTURE.md). Files under [`Models/Demucs/`](../../src/HartsyInference.Audio/Models/Demucs/).
- [x] [`DemucsCrossTransformer.cs`](../../src/HartsyInference.Audio/Models/Demucs/DemucsCrossTransformer.cs) — the defining cross-domain transformer (self/cross alternating, 2D/1D sin pos-emb, gated LayerScale), reusing SDPA + `DiaHeads`. **Synthetic-forward verified.**
- [x] [`DemucsConvBlock.cs`](../../src/HartsyInference.Audio/Models/Demucs/DemucsConvBlock.cs) — HEnc/HDec 1D+2D GLU conv block (Conv2D/ConvTranspose2d/Conv1d + GELU + plain GLU). **Synthetic-forward verified** (enc + dec). [`HtDemucsConfig.cs`](../../src/HartsyInference.Audio/Models/Demucs/HtDemucsConfig.cs) tested.
- [ ] **Staged:** the full `HtDemucs` assembly (STFT→cac→dual-branch→freq-collapse+time-inject merge→transformer→decode→mask→iSTFT→sum — the bit-exact-risky merge needs a reference run), DConv residual, `freq_emb`, and the `.th` bag loader (reuse the GameCraft `.pt` pickle loader).

### Resemble Enhance — audio FX (enhance/denoise): LCFM core BUILT + verified; denoiser/UnivNet staged
> Speech enhancer: a 2D-STFT denoiser + an LCFM (latent CFM + IRMAE) + UnivNet vocoder. Research: [`RESEMBLE_ENHANCE_ARCHITECTURE.md`](../Research/RESEMBLE_ENHANCE_ARCHITECTURE.md). Files under [`Models/ResembleEnhance/`](../../src/HartsyInference.Audio/Models/ResembleEnhance/) + [`ResembleEnhancePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/ResembleEnhancePipeline.cs).
- [x] [`ResembleWnEstimator.cs`](../../src/HartsyInference.Audio/Models/ResembleEnhance/ResembleWnEstimator.cs) — the WN CFM velocity net (30 dilated gated layers, InstanceNorm local cond + sinusoidal time global), implementing `ICfmEstimator`. Synthetic-forward verified.
- [x] [`ResembleIrmaeDecoder.cs`](../../src/HartsyInference.Audio/Models/ResembleEnhance/ResembleIrmaeDecoder.cs) — IRMAE latent→mel decoder (res-stacks + head). Synthetic-forward verified.
- [x] [`ResembleEnhancePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/ResembleEnhancePipeline.cs) — **reuses the CosyVoice `ConditionalCfm`** (CFG off) to solve the latent CFM, then IRMAE-decodes → enhanced mel. Synthetic-forward verified (mel → mel).
- [ ] **Staged:** the 2D-STFT denoiser UNet (denoise-only path / mel pre-conditioner), the UnivNet LVCNet vocoder, slaney-dB mel front-end, midpoint/rk4 + exponential time-mapping (currently euler), τ-prior, and the DeepSpeed `.pt` loader.

### ChatTTS
- [ ] `ChatTtsGpt.cs` — single LLaMA-style 20L × 768, 4-codebook GFSQ output
- [ ] `ChatTtsDvaeDecoder.cs` — 12-layer dilated ConvNeXt
- [ ] `ChatTtsSpeakerLatent.cs` — sample from `spk_stat.pt` Gaussian
- [ ] Vocos vocoder (shared with F5-TTS)
- [ ] 61 paralinguistic special tokens; sampling defaults differ for RefineText vs InferCode

### Orpheus TTS — BUILT (single-softmax Llama + SNAC); checkpoint-gated validation pending
> `canopylabs/orpheus-3b-0.1-ft` — a Llama-3.2-3B causal LM that emits SNAC 24 kHz audio tokens through one
> extended-vocab softmax. **Maximal reuse:** backbone is `Qwen2Model` with `AttentionBias` off (the CSM
> Llama path); sampling is the shared `NucleusSampler` (repetition penalty pre-shapes the logit buffer, its
> documented usage); decode is the built SNAC. Files under [`Models/Orpheus/`](../../src/HartsyInference.Audio/Models/Orpheus/) + [`OrpheusPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/OrpheusPipeline.cs).
- [x] [`OrpheusConfig.cs`](../../src/HartsyInference.Audio/Models/Orpheus/OrpheusConfig.cs) — `Orpheus3B` preset (Llama-3.2-3B dims + extended vocab 156,940 + the framing/audio token-id constants) + SNAC codec. **Tested** (preset shape).
- [x] [`OrpheusCodeFrames.cs`](../../src/HartsyInference.Audio/Models/Orpheus/OrpheusCodeFrames.cs) — flat-stream parse (crop before last `CodeStart`, drop EOS, trim to 7) + 7→3 hierarchical redistribution (1/2/4 with per-position `base+p*4096` offset). **Exact + tested** (the Orpheus analog of `MusicGenDelay`).
- [x] **SNAC 24 kHz preset corrected** to the real `hubertsiuzdak/snac_24khz` (3 codebooks, strides `[4,2,1]`) — it was mis-set to 4; Orpheus's 7=1+2+4 packing confirms 3.
- [x] [`OrpheusPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/OrpheusPipeline.cs) — text-ids-in → human-frame wrap → AR loop (rep-penalty + nucleus draw, stop at EndOfSpeech) → extract → redistribute → SNAC decode → 24 kHz PCM.
- [ ] `OrpheusTokenizer` — Llama BPE of `"{voice}: {text}"`; token-ids-in for now (caller tokenizes), same convention as SparkTTS/CosyVoice.
- [ ] **Checkpoint validation** — reconcile the Llama-3.2 "llama3" RoPE NTK-by-parts rescale (factor 32, the CSM-shared deferral) + exact extended vocab rows, bucket weights into the backbone/SNAC LoadWeights, then env-gated generation test.

### NeuTTS Air — BUILT (decode path); checkpoint-gated validation pending
> `neuphonic/neutts-air` — a Qwen2.5-0.5B LM (vocab extended to 217,652 with 65,536 `<|speech_N|>` tokens)
> emitting a single NeuCodec FSQ stream, decoded to 24 kHz. Voice cloning conditions on reference NeuCodec
> codes in the prompt. **Reuse:** stock `Qwen2Model` + `NucleusSampler` (top-k=50) + the new `NeuCodecDecoder`
> (which itself reuses `Fsq`, `IStft`, Moonshine `RotaryEmbedding`, and `IBackend` conv/norm/attn). Files under
> [`Models/NeuTts/`](../../src/HartsyInference.Audio/Models/NeuTts/) + [`Models/Codecs/NeuCodec/`](../../src/HartsyInference.Audio/Models/Codecs/NeuCodec/) + [`NeuTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/NeuTtsPipeline.cs).
- [x] [`NeuTtsConfig.cs`](../../src/HartsyInference.Audio/Models/NeuTts/NeuTtsConfig.cs) — `Air` preset (Qwen2.5-0.5B + vocab 217652) + speech-token base 151671 + framing token ids + sampling (top-k 50, temp 1.0, min-new 50). **Tested.**
- [x] [`NeuCodecConfig.cs`](../../src/HartsyInference.Audio/Models/Codecs/NeuCodec/NeuCodecConfig.cs) + [`NeuCodecDecoder.cs`](../../src/HartsyInference.Audio/Models/Codecs/NeuCodec/NeuCodecDecoder.cs) — FSQ `4^8` de-quant → project-out → fc_post_a → Conv embed + 2 ResNet pre-net + 12 RoPE transformer blocks + final LN + 2 ResNet post-net → iSTFT head. **FSQ vocab + config tested.**
- [x] [`NeuTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/NeuTtsPipeline.cs) — prompt prefix + speech-gen-start + ref codes → AR (top-k, min-new EOS suppression, stop at SpeechGenEnd) → codes → NeuCodec decode → 24 kHz PCM.
- [ ] **Reconcile on checkpoint load:** NeuCodec key spelling, the RoPE convention (torchtune vs interleaved), the iSTFT "same"-padding edge handling, and the FSQ project_out presence; NeuCodec **encoder** (ref-audio → codes) for live cloning is deferred (caller supplies pre-encoded ref codes).
- [ ] eSpeak phonemizer + Qwen tokenizer for the prompt — token-ids-in for now (caller phonemizes/tokenizes), same convention as the other TTS models.

### Zonos-v0.1 (transformer) — BUILT (structural, synthetic-forward verified); checkpoint-gated
> `Zyphra/Zonos-v0.1-transformer` (~2B). Llama-style GQA decoder (**LayerNorm**, interleaved RoPE) → 9-codebook
> DAC 44.1 kHz (k+1 delay) conditioned on a phoneme + speaker + controls prefix. Research:
> [`ZONOS_ARCHITECTURE.md`](../Research/ZONOS_ARCHITECTURE.md). Files under [`Models/Zonos/`](../../src/HartsyInference.Audio/Models/Zonos/) + [`ZonosPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/ZonosPipeline.cs).
- [x] [`ZonosConfig.cs`](../../src/HartsyInference.Audio/Models/Zonos/ZonosConfig.cs) — backbone (2048/26L/GQA 16:4/head_dim 128/SwiGLU 8192/θ=10000) + codebooks (9, in-vocab 1026, out-vocab 1025, EOS 1024, masked 1025) + delay k+1 + CFG 2.0. **Tested.**
- [x] [`ZonosBackbone.cs`](../../src/HartsyInference.Audio/Models/Zonos/ZonosBackbone.cs) — 26 LayerNorm blocks **reusing `DiaAttention` + `DiaMlp`** (split fused `in_proj`, remap `fc1/fc2` at load). **Synthetic-weights forward verified** (finite).
- [x] [`ZonosCodebooks.cs`](../../src/HartsyInference.Audio/Models/Zonos/ZonosCodebooks.cs) — 9 summed embeddings + 9 stacked heads. **Tested.** [`ZonosFourierConditioner.cs`](../../src/HartsyInference.Audio/Models/Zonos/ZonosFourierConditioner.cs) — Gaussian random-feature cos/sin. **Tested.**
- [x] [`ZonosPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/ZonosPipeline.cs) — cond/uncond backbones (shared weights) → delayed-AR over 9 codebooks → CFG → cb0 EOS + 9-step flush → revert → DAC decode.
- [ ] **Reconcile/deferred:** espeak-ng phonemization + the full conditioning-prefix assembly (speaker/integer/passthrough conditioners), ResNet293 speaker encoder, the NovelAI "unified" sampler, and the interleaved-RoPE convention check.

### Higgs Audio v2
- [ ] Llama-3.2-3B with DualFFN (per-token-type routing) — extended dotLLM
- [ ] Dual-branch tokenizer (HuBERT-base + DAC acoustic, 8 codebooks consumed of 12 declared)
- [ ] RAS (Repetition-Aware Sampling)
- [ ] Multi-speaker prompt format with `[SPEAKER0]` / `[SPEAKER1]` tags

### Sesame CSM — SCAFFOLD COMPLETE (dual-transformer); checkpoint-gated
> Files under [`Models/Csm/`](../../src/HartsyInference.Audio/Models/Csm/) + [`CsmPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/CsmPipeline.cs). **Reuse:** both transformers are headless Llama-3.2 bodies → reuse `Qwen2Model` (generalized this pass with a `Qwen2Config.AttentionBias` flag, default true, so Llama loads bias-free while CosyVoice/SparkTTS/VibeVoice are unchanged); audio decode reuses the built Mimi codec; sampling reuses `NucleusSampler`.
- [x] [`CsmConfig.cs`](../../src/HartsyInference.Audio/Models/Csm/CsmConfig.cs) — `V1B`: backbone (Llama-1B: 16L/2048/GQA 32:8, bias-off) + decoder (Llama-100M: 4L/1024/8:2) + 8 codebooks + Mimi. **Tested** (dual-transformer shape + the AttentionBias regression guard).
- [x] **`Qwen2Model.LoadWeightsHeadless`** — loads the transformer body (layers + final norm) without `embed_tokens`/`lm_head` (CSM's are `Identity`; embeds + heads live on the outer model). Reusable for any headless-LM design.
- [x] [`CsmModel.cs`](../../src/HartsyInference.Audio/Models/Csm/CsmModel.cs) — dual `Qwen2Model` + text/audio embed tables + codebook-0 head + backbone→decoder projection + 7 decoder heads; `GenerateFrame` (backbone → codebook 0, decoder AR → codebooks 1..7).
- [x] [`CsmPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/CsmPipeline.cs) — frame loop (re-embedded context) → Mimi 24 kHz decode. Conversational `Segment` history is token-IDs-in (caller assembles).
- [ ] **Aggressive streaming + persistent KV cache** (~50 ms/frame) — current path re-feeds context per frame (correct, not yet streaming-optimized).
- [ ] **Checkpoint validation** — `sesame/csm-1b`: reconcile HF key names + decoder-side audio embeddings + Llama-3.2 RoPE rescaling (scale_factor=32), then env-gated generation test.

### RVC v2 — BUILT (HuBERT + NSF synthesizer, synthetic-forward verified); F0 extraction BUILT (YIN)
> `SynthesizerTrnMs768NSFsid`. Content (the built `Hubert`) + F0 → content+pitch encoder → reused `VitsFlow`
> → NSF-HiFi-GAN. Research: [`RVC_ARCHITECTURE.md`](../Research/RVC_ARCHITECTURE.md). Files under [`Models/Rvc/`](../../src/HartsyInference.Audio/Models/Rvc/) + [`RvcPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/RvcPipeline.cs).
- [x] **NSF source injection added to `VitsHiFiGan`** (`noise_convs` + optional `harSource`; backward-compatible) — reuses `NsfVocoderDsp.GenerateHarmonicSource`. The plain-HiFiGAN consumers (Piper/MeloTTS/OpenVoice/SoVITS) stay green.
- [x] [`RvcTextEncoder.cs`](../../src/HartsyInference.Audio/Models/Rvc/RvcTextEncoder.cs) — emb_phone(768→192)+emb_pitch(256→192) front-end → **reused `VitsTextEncoder` layers**.
- [x] [`RvcPitch.cs`](../../src/HartsyInference.Audio/Models/Rvc/RvcPitch.cs) — mel coarse-pitch quantization + octave shift. **Exact + tested.**
- [x] [`RvcPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/RvcPipeline.cs) — content + F0 → enc_p → reused `VitsFlow` → NSF `VitsHiFiGan` → audio. **Synthetic-forward verified.**
- [x] **F0 extraction BUILT (2026-06-21)** — [`F0Estimator.cs`](../../src/HartsyInference.Audio/Dsp/F0Estimator.cs): pure-C# YIN (`EstimateYin`, 16 kHz / hop 320 → 50 Hz, aligned with HuBERT content frames), the RMVPE stand-in — caller no longer must supply F0. 5 tests pass (recovers 120/220/440 Hz sine within 5 Hz; silence unvoiced).
- [ ] **Staged:** RMVPE neural pitch (higher accuracy), FAISS index-retrieval blend, v1 (256-d ContentVec).

### GPT-SoVITS v2 — BUILT (3 stages, all synthetic-forward verified); enc_p MRTE + ref_enc staged
> Two-stage (T2S GPT → SoVITS) + cn-HuBERT content encoder. Research: [`GPT_SOVITS_ARCHITECTURE.md`](../Research/GPT_SOVITS_ARCHITECTURE.md). Files under [`Models/Hubert/`](../../src/HartsyInference.Audio/Models/Hubert/) + [`Models/GptSoVits/`](../../src/HartsyInference.Audio/Models/GptSoVits/).
- [x] **`cn-HuBERT`** ([`Hubert.cs`](../../src/HartsyInference.Audio/Models/Hubert/Hubert.cs)) — 7-conv extractor (GroupNorm-0, GELU) + grouped pos-conv + 12 post-LN transformer layers → `last_hidden_state`. **Shared with RVC.** Synthetic-forward verified.
- [x] **T2S GPT** ([`Text2Semantic.cs`](../../src/HartsyInference.Audio/Models/GptSoVits/Text2Semantic.cs)) — post-LN 24L/512/16h transformer, sinusoidal `alpha` positions, biased fused-QKV, text-bidir/audio-causal mask, top-k + rep-penalty(1.35) AR. Synthetic-forward verified (valid semantic tokens).
- [x] **SoVITS** ([`SoVitsSynthesizer.cs`](../../src/HartsyInference.Audio/Models/GptSoVits/SoVitsSynthesizer.cs)) — semantic VQ dequant (`[1024,768]`, ×2) → ssl_proj → prior → **reused g-conditioned `VitsFlow` + `VitsHiFiGan`** (32 kHz). Synthetic-forward verified (finite audio from semantic tokens + `ge`).
- [ ] **Staged:** full `enc_p` (dual attention encoders + **MRTE** cross-attn — currently a structural conv prior), `ref_enc` MelStyleEncoder (produces `ge`, caller-supplied now), Chinese-RoBERTa BERT + phoneme tokenizers, exact `quantizer.vq.*`/ckpt-`"weight"` key reconciliation.

### OpenVoice v2 — BUILT (Tone Color Converter, synthetic-forward verified)
> The converter reuses the VITS posterior + flow + HiFi-GAN with **speaker (`g`) conditioning** (added this
> pass to `VitsWaveNet`/`VitsFlow`/`VitsHiFiGan` — also completing the multispeaker path for Piper/MeloTTS).
> Stage-1 base TTS = MeloTTS (built above). Files under [`Models/Vits/`](../../src/HartsyInference.Audio/Models/Vits/) + [`OpenVoicePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/OpenVoicePipeline.cs).
- [x] **`g` conditioning added** to `VitsWaveNet` (cond_layer slice into the gate), `VitsFlow` (forward + reverse, `g` passthrough), `VitsHiFiGan` (cond after conv_pre) — all backward-compatible (`g=null` keeps single-speaker green).
- [x] [`VitsPosteriorEncoder.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsPosteriorEncoder.cs) (`enc_q`: pre → WN(g) → proj → sample) — the VC inference entry, reuses `VitsWaveNet`.
- [x] [`OpenVoicePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/OpenVoicePipeline.cs) — `Convert`: posterior(g_src) → flow.Forward(g_src) → flow.Reverse(g_tgt) → decoder(g_tgt). **Synthetic-forward verified** (finite audio through the full g-conditioned path).
- [ ] **Staged:** the tone-color extractor (reference → speaker embedding) — caller supplies `g_src`/`g_tgt` for now.

### VITS SynthesizerTrn (shared core) + Piper — BUILT (full graph, synthetic-forward verified); SDP staged
> The reusable VITS TTS core behind Piper, MeloTTS, GPT-SoVITS (SoVITS half), and OpenVoice. Research:
> [`VITS_ARCHITECTURE.md`](../Research/VITS_ARCHITECTURE.md). Files under [`Models/Vits/`](../../src/HartsyInference.Audio/Models/Vits/) + [`PiperPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/PiperPipeline.cs).
- [x] [`VitsTextEncoder.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsTextEncoder.cs) — emb + 6 layers of **relative-position MHA** (computed directly, no pad-reshape) + Conv FFN + proj → (m_p, logs_p).
- [x] [`VitsWaveNet.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsWaveNet.cs) (dilated gated WN) + [`VitsFlow.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsFlow.cs) (ResidualCouplingLayer mean-only + Flip, reverse pass).
- [x] [`VitsDurationPredictor.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsDurationPredictor.cs) (deterministic) + [`VitsLengthRegulator.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsLengthRegulator.cs) (intersperse + monotonic expand). **Exact + tested.**
- [x] [`VitsHiFiGan.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsHiFiGan.cs) — conv_pre + ConvTranspose upsamplers + MRF ResBlock1/2 + conv_post + tanh (reuses `IBackend.Conv1d`/`ConvTranspose1d` with dilation).
- [x] [`VitsWeights.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsWeights.cs) — weight-norm fuse (g/v → weight) at load. [`VitsSynthesizer.cs`](../../src/HartsyInference.Audio/Models/Vits/VitsSynthesizer.cs) assembles the full infer graph.
- [x] [`PiperPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/PiperPipeline.cs) — medium/high presets + Piper phoneme interspersing (BOS 1 / blank 0 / EOS 2).
- [x] **Full-graph synthetic-weights forward verified** ([`VitsTests`](../../tests/HartsyInference.Audio.Tests/VitsTests.cs)) — tiny config runs text-encoder → DP → length-reg → flow → HiFi-GAN to finite audio.
- [ ] **Staged:** the **stochastic duration predictor** (DDSConv + ConvFlow rational-quadratic spline — Piper's default `use_sdp=True`), multispeaker `g` conditioning, and direct ONNX load. espeak phonemization is caller-side.

### MeloTTS — BUILT (reuses the VITS core; synthetic-forward verified)
> MyShell MeloTTS = VITS + an extended text encoder (tone/language/BERT) + multispeaker. Files under
> [`Models/MeloTts/`](../../src/HartsyInference.Audio/Models/MeloTts/) + [`MeloTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/MeloTtsPipeline.cs).
- [x] [`MeloTtsTextEncoder.cs`](../../src/HartsyInference.Audio/Models/MeloTts/MeloTtsTextEncoder.cs) — summed phoneme + tone + language + `bert_proj`(1024→h) + `ja_bert_proj`(768→h) embedding → **reuses `VitsTextEncoder` layers** via its new embedding-in entry point (`ForwardFromEmbedding` + `LoadWeightsLayersOnly`). **Synthetic-forward verified** (finite prior).
- [x] [`MeloTtsConfig.cs`](../../src/HartsyInference.Audio/Models/MeloTts/MeloTtsConfig.cs) (wraps `VitsConfig` core + tone/lang counts + BERT dims + `sdp_ratio`) + [`MeloTtsPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/MeloTtsPipeline.cs) — encoder → **reused** `VitsDurationPredictor` + `VitsFlow` + `VitsHiFiGan`.
- [ ] **Staged:** SDP/DP blend (`sdp_ratio`), TransformerCouplingBlock flow option, multispeaker `g` conditioning, per-language BERT + phonemizer (caller-side).

### StyleTTS 2 — BUILT (reuses validated Kokoro stack); checkpoint-gated for the new style modules
> The parent architecture of Kokoro. PLBERT + TextEncoder + ProsodyPredictor + iSTFTNet decoder are reused **verbatim** from the (validated, end-to-end-tested) Kokoro modules — only the runtime style path is new. Files under [`Models/StyleTts2/`](../../src/HartsyInference.Audio/Models/StyleTts2/) + [`StyleTts2Pipeline.cs`](../../src/HartsyInference.Audio/Pipelines/StyleTts2Pipeline.cs).
- [x] **Kokoro module reuse** — the 4 shared classes were made `public` and [`KokoroPipeline`](../../src/HartsyInference.Audio/Pipelines/KokoroPipeline.cs) gained a `SynthesizeFromStyle(phonemes, refStyle256, speed)` entry point (the voice-pack path now routes through the same `SynthesizeCore`). StyleTTS 2 composes a `KokoroPipeline` from its own weights and drives it with an externally-computed style. **Kokoro's 6 tests still pass after the refactor.**
- [x] [`StyleTts2Config.cs`](../../src/HartsyInference.Audio/Models/StyleTts2/StyleTts2Config.cs) — `LibriTts` (multispeaker) / `LjSpeech` presets. **Tested** (StyleDim=256, MultiSpeaker flag).
- [x] `StyleTransformer1d` diffusion style sampler — [`StyleDiffusionSampler.cs`](../../src/HartsyInference.Audio/Models/StyleTts2/StyleDiffusionSampler.cs) (Karras schedule + ADPM2 ancestral second-order sampler, **exact + tested**: schedule endpoints/monotonicity, determinism, shape) + [`StyleDenoiser.cs`](../../src/HartsyInference.Audio/Models/StyleTts2/StyleDenoiser.cs) (EDM/KDiffusion preconditioning + CFG combine — exact; the StyleTransformer1d network forward is a checkpoint-gated scaffold).
- [x] [`StyleEncoder.cs`](../../src/HartsyInference.Audio/Models/StyleTts2/StyleEncoder.cs) — 2D-Conv ResNet (StarGAN-v2 ResBlks) + adaptive-avg-pool → 128-d, run twice (acoustic + prosodic → 256-d). Uses the existing `Conv2D` + inline pooling. (spectral-norm sigma folding checkpoint-gated.)
- [x] [`StyleTts2Pipeline.cs`](../../src/HartsyInference.Audio/Pipelines/StyleTts2Pipeline.cs) — three inference modes: `SynthesizeClone` (zero-shot from reference mel), `SynthesizeRandom` (diffusion, no reference — the LJSpeech mode), `SynthesizeClonePerturbed` (diffusion seeded by the reference style). **8 checkpoint-free tests pass** (sampler + config).
- [ ] **Checkpoint validation** — download `yl4579/StyleTTS2-LibriTTS` (.pth → strip training modules → ~590 MB), reconcile the `style_encoder` / `predictor_encoder` / `diffusion` (StyleTransformer1d) state-dict keys + spectral-norm folding, wire a loader, and add an env-gated generation test. The Kokoro-shared modules load from the same checkpoint family.
- [ ] **Long-form continuation** mode (the 3rd StyleTTS2 mode beyond clone/random) — deferred.

### VibeVoice (1.5B / 7B / Streaming-0.5B) — multi-speaker BUILT; streaming pipeline pending
> The local LM is a self-contained Qwen2.5 reimplementation under [`Models/LanguageModels/Qwen2/`](../../src/HartsyInference.Audio/Models/LanguageModels/Qwen2/) (no dotLLM dependency taken). Multi-speaker path is fully wired; the split-LM streaming variant is deferred.
- [x] [`VibeVoiceConfig.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceConfig.cs) — composition config (acoustic + semantic tokenizer + Qwen2.5 decoder + diffusion head); `V15B` / `V7B` / `Streaming05B` presets. (Loaded via downstream JSON parsing; no `[JsonSerializable]` attribute.)
- [x] `VibeVoiceStreamingConfig` — **folded into `VibeVoiceConfig`**: `Streaming05B` preset + `IsStreaming` (null semantic tokenizer, `TtsBackboneNumHiddenLayers = 20`)
- [x] `Block1D` — [`VibeVoiceConvNeXtBlock.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceConvNeXtBlock.cs) (ConvNeXt-V1 1D: ConvRMSNorm + depthwise causal Conv1d + LayerScale + GELU FFN)
- [x] [`SConv1d.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/SConv1d.cs) / [`SConvTranspose1d.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/SConvTranspose1d.cs) — causal padding wrappers with streaming-cache hooks
- [x] [`VibeVoiceTokenizerStreamingCache.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceTokenizerStreamingCache.cs) — per-`(layer_id, sample_index)` left-padded history buffer
- [x] `TokenizerEncoder` — [`VibeVoiceTokenizerEncoder.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceTokenizerEncoder.cs): 6-stage stem+downsample, reversed ratios `[2,2,4,5,5,8]`, channels `32→…→2048`, Linear head 2048→vae_dim
- [x] `TokenizerDecoder` — [`VibeVoiceTokenizerDecoder.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceTokenizerDecoder.cs): mirror of encoder via `SConvTranspose1d`
- [x] [`VibeVoiceAcousticTokenizerModel.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceAcousticTokenizerModel.cs) — encoder + decoder + `fix_std=0.5` + Gaussian sampling; `encode()`/`decode()`
- [x] [`VibeVoiceSemanticTokenizerModel.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceSemanticTokenizerModel.cs) — encoder only, `vae_dim=128`, deterministic
- [x] [`SpeechConnector.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/SpeechConnector.cs) — Linear + RMSNorm + Linear
- [x] [`VibeVoiceDiffusionHead.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceDiffusionHead.cs) — 4 × HeadLayer (RMSNorm + AdaLN + SwiGLU) + FinalLayer; `noisy_images_proj` / `cond_proj` / `t_embedder`
- [x] `VibeVoiceCosineBetaSchedule` — Nichol-Dhariwal cosine `alpha_bar`, in [`VibeVoiceCosineDpmSolver.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceCosineDpmSolver.cs)
- [x] DPM-Solver multistep — v-prediction + cosine + order=2 + 20 steps (`VibeVoiceCosineDpmSolver`)
- [x] `VibeVoiceTextTokenizer` — [`VibeVoiceTokenizer.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceTokenizer.cs): Qwen2 tokenizer exposing `SpeechStart/End/Diffusion` + EOT ids
- [x] `VibeVoiceTokenConstraintProcessor` — **folded into `VibeVoicePipeline.SampleConstrained`** (logit mask over `{speech_start, end, diffusion, eos}`)
- [x] [`VibeVoiceProcessor.cs`](../../src/HartsyInference.Audio/Models/VibeVoice/VibeVoiceProcessor.cs) — multi-speaker script parser (JSON / plain / `Speaker N:`), −25 dBFS normalizer, voice-prompt builder, `speech_input_mask`
- [x] [`VibeVoicePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/VibeVoicePipeline.cs) (multi-speaker) — prefill (acoustic-encode voice → splice into LM embed at mask) + AR loop (Qwen2.5 → constrained logits → DDPM sub-loop with CFG → acoustic decode → semantic re-encode → embed feedback)
- [ ] `VibeVoiceStreamingPipeline.cs` — split-LM forward + windowed text/speech interleave + `BinaryClassifier` EOS + acoustic-only feedback. **Not built** (multi-speaker pipeline notes it as deferred).
- [ ] `BinaryClassifier.cs` — `Linear → ReLU → Linear→1` streaming EOS head. **Not built** (part of streaming).
- [ ] **Per-layer-stop on the local `Qwen2Model`** — needed only for the streaming split-LM; not required by the multi-speaker path that ships today.
- [ ] **Audio streamer surface** — per-`speech_diffusion`-token `IAsyncEnumerable<AudioChunk>` emission (part of the streaming variant).
- [ ] Long-context smoke test: 65k-token KV cache stable on 1.5B (90 min output), 32k on 7B (45 min)
- [ ] License surface: VibeVoice is **MIT** — flag in the model registry vs XTTS (CPML), AudioLDM2 (CC-BY-NC), SparkTTS (CC-BY-NC)
- **Validated:** `VibeVoiceStage1SmokeTests` loads all 1204 tensors + acoustic-VAE round-trip + semantic-VAE 128-d latent + diffusion-head single-step; `VibeVoiceEndToEndTests` runs prefill + short-script synthesis. Both skip-gated on cache (E2E also behind `HARTSYINFERENCE_RUN_VIBEVOICE_E2E`).

## 7. Music Generation (`HartsyInference.Music`)

### ACE-Step (flagship)

> **v1 BUILT END-TO-END (2026-06-10) + RECONCILED AGAINST THE REAL 3.5B CHECKPOINT (2026-06-11)** — lives in
> **HartsyInference.Diffusion** (`Models/Music/` + `Models/Denoisers/AceStep*` + `Pipelines/AceStepPipeline.cs`).
> The checkpoint is downloaded to `Models/Music/ACE-Step-v1-3.5B/` (with the exported lyric `vocab.json`/`merges.txt`)
> and a **real-weight load smoke PASSES**: all 862 DiT tensors + DCAE (192) + vocoder (536 post-fusion) resolve, and
> the DCAE/vocoder run real-weight forwards with finite outputs. Source/dump corrections applied vs the research doc:
> self-attention is Sana **LiteLA ReLU-linear** (not softmax); patch embed is `conv(16,1)→2048 → GroupNorm(32) → 1×1`
> (not 3×3+GN(64)); `timestep_embedder.*` is top-level; all attn q/k/v have **biases**; RoPE is the Qwen2 half-concat
> table combined with interleaved-pair rotation (trained-in quirk, reproduced literally); cross-attn K/V come from
> `to_k/to_v` over the context with K roped at context positions (`add_*`/`to_add_out` are constructed-but-unused —
> not loaded); lyric encoder is **6 layers** with NO conv module / NO macaron (plain rel-pos transformer); structure
> tags have **dedicated added tokens** 6681–6692 (vocab = 6681 + 12 = 6693, resolving the open question);
> `[zh-cn]` not `[zh]`. **Remaining for "validated": numeric parity diff vs the Python reference** (the gated
> generate test proves plumbing, not parity).

- [x] UMT5-base — reused the shared `T5TextEncoder` (new `T5TextEncoderConfig.Umt5Base` preset, per-layer position bias); no ACE-specific encoder class needed
- [x] [`AceStepDit.cs`](../../src/HartsyInference.Diffusion/Models/Denoisers/AceStepDit.cs) v1 — 24L × 20 heads × 128 (inner 2560), LiteLA self-attn + RoPE θ=1e6, softmax cross-attn over [speaker ‖ text ‖ lyrics], GLUMBConv FFN (ratio 2.5), patch [16,1] height collapse; owns speaker/genre/lyric projections + the lyric Conformer
- [ ] `AceStepFsqLm.cs` v1.5 — Qwen3-based decoder predicting FSQ audio tokens (separate later effort; FSQ codec already exists in Audio)
- [x] [`MusicDcaeDecoder.cs`](../../src/HartsyInference.Diffusion/Models/Music/MusicDcaeDecoder.cs) — Sana AutoencoderDC decoder over 2-D stereo mel ([`ResBlock2d`](../../src/HartsyInference.Diffusion/Models/Music/ResBlock2d.cs) + [`EfficientVitBlock`](../../src/HartsyInference.Diffusion/Models/Music/EfficientVitBlock.cs) multiscale ReLU-linear attention + GLUMBConv, repeat-interleave/pixel-shuffle shortcuts). **Encoder not built yet** — needed for edit/repaint/reference-audio modes only
- [x] [`AdaMosHiFiGanV1.cs`](../../src/HartsyInference.Diffusion/Models/Music/AdaMosHiFiGanV1.cs) — ConvNeXt backbone (depths [3,3,9,3]) + HiFi-GAN head (7 ups, ×512 total, MRF kernels [3,7,11,13]); weight-norm fused at conversion
- [x] [`AceStepLyricEncoder.cs`](../../src/HartsyInference.Diffusion/Models/Music/AceStepLyricEncoder.cs) — **6-layer** wenet-lineage rel-pos transformer (pos_bias_u/v + learned position projection; the conv module/macaron paths exist presence-driven but the real checkpoint ships without them)
- [x] Voice BPE tokenizer — [`AceStepLyricTokenizer`](../../src/HartsyInference.Tokenizers/AceStepLyricTokenizer.cs), verified against the real shipped tokenizer: `[lang]` prefix (zh→`[zh-cn]`), spaces→`[SPACE]`=2, HF Whitespace pre-tokenization, `[UNK]`=1 fallback; vocab/merges exported to `Models/Music/ACE-Step-v1-3.5B/`
- [x] Lyric structure tags + `tokenize_lyrics` (`[START]`=261 / 2 line protocol; tags hit dedicated added tokens 6681–6692) + per-line script-heuristic language detection (statistical 17-lang detector + number expansion + CJK G2P = caller responsibility, documented)
- [x] Flow-match shift=3.0 + APG/CFG-Zero★/CFG — [`AceStepGuidance`](../../src/HartsyInference.Diffusion/Utilities/AceStepGuidance.cs) (momentum buffer, norm threshold, parallel/orthogonal decomposition; orthogonality + formula tests)
- [x] Three samplers: Euler / Heun (in-pipeline 2nd-order) / [`FlowMatchPingPongScheduler`](../../src/HartsyInference.Diffusion/Schedulers/FlowMatchPingPongScheduler.cs)
- [x] [`AceStepPipeline.cs`](../../src/HartsyInference.Diffusion/Pipelines/AceStepPipeline.cs) — denoise → pipeline latent scale (0.1786/−1.9091, NOT the diffusers 0.41407) → DCAE decode → de-standardize to log-mel [−11,3] → per-channel vocoder → 44.1 kHz stereo; [`AceStepCheckpointConverter`](../../src/HartsyInference.ModelHandler/CheckpointConverters/AceStepCheckpointConverter.cs) (SSL-head drop, weight-norm fusion w/ numeric test) + `TestPaths.AceStep`
- [ ] Flow-edit / repaint algorithm (masked dual-conditioning velocity loop — needs the DCAE **encoder** first)
- [x] Env-gated [`AceStepGenerationTests`](../../tests/HartsyInference.Diffusion.Tests/AceStepGenerationTests.cs) — `LoadSmoke` (cheap, mmap; key/shape validation vs the real checkpoint — **verified passing via a standalone console run 2026-06-11**) + `Generate` (full 27-step CPU inference → stereo WAV in `Output/`; double-gated on `HARTSYINFERENCE_RUN_ACE_E2E=1` because the F32 DiT cast peaks ~13 GB RSS — run from a bare terminal, not an IDE test runner). The converter gained `castToF32` (CPU kernels are F32-only; BF16 stays mmap-borrowed otherwise)
- [x] **DiT-core Python parity harness (2026-06-15)** — [`dump_ace_step_dit.py`](../../tests/python-reference/dump_ace_step_dit.py) + [`diff_ace_step_layers.py`](../../tests/python-reference/diff_ace_step_layers.py) + [`AceStepDitDiffTests.cs`](../../tests/HartsyInference.Diffusion.Tests/AceStepDitDiffTests.cs) + [`AceStepDebugDump`](../../src/HartsyInference.Diffusion/Models/Denoisers/AceStepDebugDump.cs). Self-contained CPU/synthetic (C# generates the tiny checkpoint via `AceStepSyntheticWeights` + `SafeTensorsWriter`, Python independently re-implements `AceStepDit.Forward`). **All taps (patch_embed/tblock/layers/velocity) match to ~1e-8** — validates LiteLA linear-attn, the interleaved-pair/half-concat RoPE quirk, GLUMBConv, cross-attn, AdaLN-6, patch-embed, final layer. Remaining: extend the harness over BuildContext/lyric-Conformer + DCAE decoder + vocoder, then a real-weight single-step diff on the cloud GPU.
- [ ] GGUF Q4/Q8 DiT path for low-RAM boxes (community quants exist; would drop the 13 GB F32 requirement)

### Stable Audio Open
- [ ] `OobleckVae.cs` — 5-stage Conv1D + snake activation + weight-norm (no GroupNorm), 2048× downsample, 64-ch latent
- [ ] `StableAudioDit.cs` — 1536 / 24L / 24 heads (Q) / 12 KV heads (cross-attn), AdaLN-6, RoPE, SwiGLU
- [ ] `FourierFeatures1D` for timing embedding
- [ ] Timing conditioning (seconds_start + seconds_total → cross-attn tokens + global AdaLN)
- [ ] T5 reuse from HartsyInference.Diffusion
- [ ] `dpmpp-3m-sde` scheduler (Open 1.0 v-prediction)
- [ ] Pingpong scheduler (Open Small distilled)

### MusicGen + AudioGen — BUILT (text-only mono); checkpoint-gated validation pending
> One generic stack serves both — they share the AudioCraft recipe and differ only in codec config. Files
> under [`Models/Music/`](../../src/HartsyInference.Audio/Models/Music/) + [`MusicGenPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/MusicGenPipeline.cs).
- [x] `MusicGenDecoder.cs` — decoder-only, sum-of-K embeddings input, sinusoidal pos, GELU 4× FFN, K parallel heads, cross-attn to precomputed T5 states ([Models/Music/MusicGenDecoder.cs](../../src/HartsyInference.Audio/Models/Music/MusicGenDecoder.cs))
- [x] T5-base text encoder — pipeline takes precomputed `[1,T,768]` cross-attn states (caller supplies T5). Engine now ships the pieces to build it: [`T5TextEncoderConfig.T5Base`](../../src/HartsyInference.Diffusion/Models/TextEncoders/T5TextEncoderConfig.cs) (v1.0 google/t5-base: non-gated, ReLU, 32k vocab — embedded t5_spiece) + [`MusicGenCheckpointConverter.LoadTextEncoder`](../../src/HartsyInference.ModelHandler/CheckpointConverters/MusicGenCheckpointConverter.cs) (extracts `text_encoder.*` from the combined checkpoint). **Added 2026-06-21** — lifts the consumer EngineGap; AudioGen reuses verbatim.
- [x] EnCodec 32kHz 4-codebook (music, [`EnCodecConfig.EnCodec32kHz`](../../src/HartsyInference.Audio/Models/Codecs/EnCodec/EnCodecConfig.cs)) / EnCodec 16kHz 4-codebook (audio, `EnCodecConfig.EnCodec16kHz`) — both 2048-entry, 50 Hz, non-causal; `n_filters` for the 16 kHz codec is the one value to reconcile on first checkpoint load
- [x] Delay-pattern state machine (default `[0,1,2,3]`; stereo `[0,0,1,1,2,2,3,3]`) — [`MusicGenDelay.cs`](../../src/HartsyInference.Audio/Models/Music/MusicGenDelay.cs); Apply/Revert/IsActive **exact + tested** (`MusicGenTests`)
- [x] CFG two-stream (cond + g*(cond-uncond)) — `MusicGenPipeline` (cond + null-cross uncond branch, special token masked out of every active codebook's draw)
- [x] `MusicGenConfig.AudioGen` preset (medium dims 1536/48/24 + 16 kHz codec) — AudioGen is served verbatim by `MusicGenPipeline`; no separate pipeline (would duplicate). Smoke: `CodecSmokeTests` (32k/16k presets) + `MusicGenTests`.
- [ ] Melody conditioning (chromagram extractor + argmax-and-zero + cross-attn prepend) — deferred
- [ ] Stereo variants (8 codebooks, paired delay) — config supports the paired delay; stereo decode path deferred
- [ ] **Checkpoint validation** — bucket `facebook/musicgen-medium` + `facebook/audiogen-medium` weights into the decoder/codec LoadWeights dicts, then env-gated generation test

### HeartMuLa (HeartMuLa-oss-3B) — BUILT (LM reuses verified CSM); HeartCodec + MuQ staged
> Music/song generation = the CSM two-transformer pattern (Llama-3B global + Llama-300M depth, 8 codebooks vocab
> 8197, 48 kHz) + a flow-matching HeartCodec + MuQ-MuLan style cond. Research: [`HEARTMULA_ARCHITECTURE.md`](../Research/HEARTMULA_ARCHITECTURE.md).
- [x] [`HeartMulaConfig.cs`](../../src/HartsyInference.Audio/Models/HeartMula/HeartMulaConfig.cs) + [`HeartMulaPipeline.cs`](../../src/HartsyInference.Audio/Pipelines/HeartMulaPipeline.cs) — **LM reuses the verified `CsmModel`** (dual transformer + codebook heads); lyrics → AR 8-codebook frames. **Config + construct tested.**
- [ ] **Staged:** HeartCodec flow-matching RVQ decoder (reuse `ConditionalCfm`), MuQ-MuLan style embedder, sharded-safetensors load.

### PocketTTS (Kyutai pocket-tts) — RESEARCHED + config skeleton; config-gated (continuous-latent)
> ~100M CPU TTS, **continuous-latent AR** (NOT a discrete codec-LM) + per-frame flow-matching over a continuous-Mimi
> VAE. Research: [`POCKET_TTS_ARCHITECTURE.md`](../Research/POCKET_TTS_ARCHITECTURE.md).
- [x] [`PocketTtsConfig.cs`](../../src/HartsyInference.Audio/Models/PocketTts/PocketTtsConfig.cs) — documented skeleton (24 kHz/12.5 Hz, 26 voices, LSD steps; `DModel`/`LatentDim` placeholders). **Config tested.**
- [ ] **Deferred (config-gated):** the continuous-latent AR loop + flow-matching/LSD head (reuse `ConditionalCfm`) + continuous-Mimi path (reuse built `Mimi`, bypass RVQ) — implementation waits on the checkpoint's exact dims (NOT public).

### YuE — SCAFFOLD COMPLETE (Stage-1, first music model); checkpoint-gated
> **First music model.** Files under [`Models/Music/`](../../src/HartsyInference.Audio/Models/Music/) + [`YuePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/YuePipeline.cs). Built almost entirely from reuse — the codec-LM music models live in the Audio package since they reuse its codecs + LM infra (the diffusion music models — ACE-Step/Stable Audio/DiffRhythm/AudioLDM2 — will go in the Diffusion package).
- [x] [`YueStage1Lm.cs`](../../src/HartsyInference.Audio/Models/Music/YueStage1Lm.cs) — **reuses `Qwen2Model`** (Llama-2-7B: bias-off + MHA via `NumKeyValueHeads == NumAttentionHeads`) + shared `NucleusSampler`; emits the interleaved **dual-track** `[vocal_cb0, accomp_cb0, …]` stream and parses it into the two per-track codebook-0 streams. Mandatory `repetition_penalty=1.1` applied.
- [x] [`YueConfig.cs`](../../src/HartsyInference.Audio/Models/Music/YueConfig.cs) — Stage-1 (Llama-2-7B) + Stage-2 (~1.5B) presets + extended-vocab audio-token bases. **Tested**.
- [x] [`YuePipeline.cs`](../../src/HartsyInference.Audio/Pipelines/YuePipeline.cs) — Stage-1 → cb0 → **built `XCodec`** 16 kHz decode.
- [x] **mm tokenizer BUILT (2026-06-21)** — [`YueTokenizer.cs`](../../src/HartsyInference.Tokenizers/YueTokenizer.cs): SentencePiece + YuE special tokens resolved from the loaded `tokenizer.model`; `EncodeStage1Prompt(genre, lyrics)`. Lyrics can now be encoded. Special-token strings + Stage-1 token-sequence parity still need bit-exact validation against `infer.py` (the audio-token bases in `YueConfig` remain placeholders pending that).
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
- [x] ACE-Step single-step DiT forward matches Python reference (avg_err < 1e-3) — **achieved ~1e-8 on synthetic weights** (`AceStepDitDiffTests` + `dump_ace_step_dit.py`); real-weight diff on GPU still pending
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
- [ ] Benchmark report comparing HartsyInference vs Python reference for each model
- [ ] License audit — flag non-commercial models (XTTS = CPML, AudioLDM2 = CC-BY-NC-SA, SparkTTS = CC-BY-NC-SA) in model registry; surface to users. VibeVoice is **MIT** (commercially permissive — surface as such).
- [x] Audio model coverage tracked in [`MODEL_STATUS_AUDIO.md`](MODEL_STATUS_AUDIO.md)
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

## 13. Engine Staged-Component Build-Out (2026-06-22)

Completed the full neural build-out of every previously **staged / checkpoint-gated** component across the 19 AudioLab models, so the SwarmUI-AudioLab extension can wire each model end-to-end (numeric parity remains the usual checkpoint-download follow-up). All structural, synthetic-forward verified.

### Built + verified this pass (parallel agent build, then central integration)

- **CosyVoice 2** — real S3 tokenizer RoPE encoder + `UpsampleConformerEncoder` (replaced both conv-subsample scaffolds).
- **GPT-SoVITS** — real `enc_p` (MRTE cross-attention + dual encoders) + `ref_enc` MelStyleEncoder (replaced the structural conv prior).
- **VITS core** — Stochastic Duration Predictor (DDSConv + ConvFlow rational-quadratic spline, inverse pass) wired into `VitsSynthesizer` (`UseSdp`).
- **Fish-Speech / OpenAudio-S1** — real firefly grouped-residual FSQ + SiLU generator + the modded-DAC codec variant + byte-BPE `FishSpeechTokenizer` (asset-loaded).
- **StyleTTS 2** — real `StyleTransformer1d` diffusion network (AdaLN + self/cross-attention) replacing the denoiser scaffold.
- **Demucs (HTDemucs)** — full assembly (STFT→cac→dual-branch encode→freq-collapse/time-inject merge→cross-transformer→decode→iSTFT+time-sum→4 stems) + DConv + pipeline.
- **Resemble Enhance** — 2D-STFT denoiser UNet + UnivNet LVCNet vocoder + midpoint-solver enhance path.
- **Qwen3-TTS** — talker (dual-embed + codec head) + MTP code-predictor + custom Snake/ConvNeXt split-RVQ vocoder (×1920) + ECAPA-TDNN + 3D mRoPE + 3-mode pipeline.
- **HeartMuLa** — HeartCodec flow-matching RVQ decoder (48 kHz) + MuQ-MuLan style embedder, wired into the CSM pipeline.
- **PocketTTS** — continuous-latent FlowLMModel + per-frame flow head + RVQ-bypass `MimiContinuousLatent` (all dims config-parameterized for the gated checkpoint).
- **Speaker/pitch encoders** — Zonos ResNet293 + full 7-conditioner prefix, RVC RMVPE pitch extractor, OpenVoice tone-color encoder + linear-spectrogram extractor, NeuCodec encoder.
- **Chatterbox** — end-to-end pipeline wiring T3 → S3Gen (reuses CosyVoice flow + HiFTNet) → 24 kHz.

### Integration result
- Library: **0 warnings / 0 errors** (net8.0 + net10.0).
- Tests: **357 passed / 5 skipped / 0 failed** (both TFMs).
- 4 integration bugs found + fixed by central verification: ECAPA + Zonos attentive-stat-pooling allocated rank-2 (overran `BatchedMatMul`, which needs rank-3); FishDac DAC residual unit's pointwise conv was padded as if k=7; plus a Qwen3-TTS tiny-config mRoPE-section mismatch (made `MropeSection` configurable).

### Remaining (genuinely checkpoint/asset-gated — not buildable without the weights)
- Numeric parity vs Python references for every component (download-gated, the standing convention).
- Exact state-dict key reconciliation + tokenizer/control-token assets (vocab files, `added_tokens.json` ids) per model.
- Kyutai STT engine republish + the extension-side handler tweak.
