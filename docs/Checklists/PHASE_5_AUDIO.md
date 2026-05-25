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

### Whisper family
- [ ] `WhisperEncoder.cs` — Conv1D × 2 + sinusoidal pos + N × ResidualAttentionBlock
- [ ] `WhisperDecoder.cs` — embed + learned pos + N × cross-attention block + KV cache
- [ ] `WhisperPipeline.cs` — mel → encode → cross-KV precompute → decode loop with suppress-tokens + temperature fallback
- [ ] `WhisperOptions.cs` — language, task, timestamps, model size, beam vs greedy
- [ ] Distil-Whisper variant support (2-layer decoder; same model class, different config)
- [ ] Whisper-large-v3-turbo support (128 mel bins, 4-layer decoder)
- [ ] Long-form sequential decoding (timestamp-driven) + chunked variant
- [ ] Word-level timestamps via cross-attention DTW (alignment heads table per model size)

### NVIDIA NeMo family (`SharpInference.Audio.Nemo` or similar)
- [ ] `FastConformerEncoder.cs` — 8x conv subsampling + Conformer blocks with limited-context attention (shared by Parakeet, Canary, FireRedASR-AED)
- [ ] `CtcDecoder.cs` — greedy + beam search with blank collapse (Parakeet-CTC)
- [ ] `RnntDecoder.cs` — prediction net (LSTM) + joint net + hypothesis-extension beam (Parakeet-RNNT)
- [ ] `TdtDecoder.cs` — joint net with token + duration heads, greedy decode (Parakeet-TDT)
- [ ] `ParakeetPipeline.cs` — variant-dispatched
- [ ] `CanaryPipeline.cs` — FastConformer + Transformer decoder + AST prompt tokens
- [ ] Cache-aware streaming for Parakeet and Canary chunked modes

### Moonshine
- [ ] `MoonshinePipeline.cs` — Conv1D front-end + RoPE encoder/decoder + Llama BPE
- [ ] Hallucination guard (~6.5 tokens/sec ceiling)

### SenseVoice + FireRedASR
- [ ] `SenseVoiceEncoder.cs` — 50-layer SANM encoder + LFR frontend + CTC head
- [ ] `SenseVoicePipeline.cs` — special-token parser (`<emotion><lang><event><text>`)
- [ ] `FireRedAsrPipeline.cs` (AED variant) — Conformer + Whisper-style decoder
- [ ] `FireRedAsrLlmPipeline.cs` (LLM variant) — Conformer + adapter + Qwen2 decoder (dotLLM dependency)

### .nemo file format support
- [ ] `NemoFileLoader.cs` — extract tar to {ckpt, config.yaml}; route to model loader
- [ ] Offline converter: `.nemo` → safetensors + config.json for our standard pipeline

## 6. Text-to-Speech (TTS)

### Kokoro (first TTS to ship)
- [ ] `ClipTextEncoder.cs` adapted as `PlBertEncoder` (ALBERT with weight sharing — load once, loop 12x)
- [ ] `KokoroTextEncoder.cs` (Embed + Conv1D × 3 + BiLSTM)
- [ ] `KokoroProsodyPredictor.cs` (DurationEncoder + LSTM + duration projection + F0/N AdaINResBlocks)
- [ ] `KokoroIStftNetDecoder.cs` — see [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md)
- [ ] `KokoroPipeline.cs` — end-to-end with voice pack loading
- [ ] G2P backed by [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md) recommendation (English-first)

### F5-TTS
- [ ] `F5TtsDiT.cs` — reuse Flux/SD3 DiT blocks + ConvNeXt V2 stem (GRN) + AdaLN-Zero
- [ ] `F5TtsPipeline.cs` — in-context infilling with ref-overwrite
- [ ] Sway sampling scheduler (`FLOW_MATCHING_AUDIO.md`)
- [ ] `vocos-mel-24khz` vocoder

### XTTS-v2
- [ ] `XttsGpt.cs` — GPT-2-style 30L × 1024
- [ ] `XttsTokenizer.cs` — shared BPE + `[<lang>]` prefix tokens + per-language romanizers (pinyin, cutlet)
- [ ] `XttsConditioningEncoder.cs` — Perceiver resampler → 32×1024 gpt_cond_latent
- [ ] `XttsSpeakerEncoder.cs` — H/ASP ECAPA-TDNN → 512-d
- [ ] `XttsHifiGan.cs` — takes 1024-d GPT latents (not mel), FiLM-conditioned per ResBlock
- [ ] Streaming variant (chunk_size=20 mel tokens, overlap_wav_chunks=1024)

### Bark
- [ ] `BarkSemanticTransformer.cs` / `BarkCoarseTransformer.cs` / `BarkFineTransformer.cs`
- [ ] `BarkTokenizer.cs` — mBERT-cased with +10048 offset
- [ ] `BarkSpeakerPrompt.cs` — load 3-stream pickles (offline-convert to safetensors)
- [ ] `BarkPipeline.cs` — three-stage orchestration, coarse alternation, fine window stitching
- [ ] EnCodec 24kHz 8-codebook decoder

### CosyVoice 1 + 2
- [ ] `CosyVoiceQwenLm.cs` — wraps Qwen2.5-0.5B (dotLLM reuse) with 6561 extended vocab
- [ ] `S3Tokenizer.cs` — FSQ semantic codec (CV2 variant)
- [ ] `CamPlusSpeakerEncoder.cs` — D-TDNN with context-aware masking, 192-d
- [ ] `CosyVoiceCfm.cs` — chunk-aware ConditionalDecoder with causal masking modes
- [ ] `HifTNetVocoder.cs` (CV2) — HiFTNet variant
- [ ] `CosyVoiceStreamingPipeline.cs` — 15-token chunks, 150 ms first-packet latency

### IndexTTS 1.5 + 2
- [ ] `IndexT2sGpt.cs` — 24L × 1280
- [ ] `IndexS2MelDit.cs` — 13L × 512 with WaveNet final layer (non-causal — no streaming)
- [ ] `IndexSemanticCodec.cs` — Vocos-style ConvNeXt encoder
- [ ] `IndexConformerPerceiver.cs` for speaker + emotion conditioning
- [ ] BigVGAN v2 22kHz vocoder (download from `nvidia/bigvgan_v2_22khz_80band_256x`)
- [ ] Optional Qwen-3 0.6B-emo for text-emotion conditioning

### SparkTTS
- [ ] Qwen2.5-0.5B with 166000 vocab (151936 base + 14064 BiCodec tokens)
- [ ] BiCodec encoder + decoder (50 Hz semantic VQ 8192 + 32 global FSQ tokens 4096)
- [ ] DAC HiFi-GAN wave generator (16 kHz)
- [ ] w2v-BERT 2.0 feature extractor (cached per audio hash)

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

### Sesame CSM
- [ ] Llama-3.2-1B backbone for codebook 1
- [ ] Small (~100M) decoder for codebooks 2-N (per-frame, AR within frame)
- [ ] Mimi codec decoder (12.5 Hz / 8 codebooks)
- [ ] Conversational context format with audio interleaving
- [ ] Aggressive streaming pipeline (~50 ms per-frame latency target)

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

### StyleTTS 2
- [ ] Reuse Kokoro modules verbatim (PLBERT + TextEncoder + ProsodyPredictor + iSTFTNet)
- [ ] `StyleTransformer1d` diffusion style sampler (Karras / ADPM2, 5-10 steps)
- [ ] `StyleEncoder` (2D-Conv ResNet + AdaptiveAvgPool) — extracts style from reference audio
- [ ] Three inference modes: zero-shot clone / random style / long-form continuation

### VibeVoice (1.5B / 7B / Streaming-0.5B) — long-form multi-speaker, MIT
- [ ] `VibeVoiceConfig.cs` — composition config (acoustic_tokenizer_config + semantic_tokenizer_config + decoder_config (Qwen2.5) + diffusion_head_config). Source-gen JSON via `[JsonSerializable]`.
- [ ] `VibeVoiceStreamingConfig.cs` — streaming-0.5B variant (no semantic_tokenizer_config; adds `tts_backbone_num_hidden_layers = 20`)
- [ ] `Block1D.cs` — ConvNeXt-V1-style 1D block (ConvRMSNorm + depthwise causal Conv1d + LayerScale γ + FFN(SiLU-free GELU, 4× expansion) + LayerScale ffn_γ). Mixer = `depthwise_conv`, layer_scale_init_value=1e-6.
- [ ] `SConv1d.cs` / `SConvTranspose1d.cs` — causal padding wrappers with streaming cache hooks. Pad mode = constant zero. `trim_right_ratio=1.0` for transpose.
- [ ] `VibeVoiceTokenizerStreamingCache.cs` — per-`(layer_id, sample_index)` history buffer; `get` pads left to max length so the batch tensor stays rectangular. Deterministic layer IDs assigned at construction time (no `id(self)` equivalent).
- [ ] `TokenizerEncoder.cs` — 6-stage stem + downsample chain over reversed ratios `[2,2,4,5,5,8]`, channel doubling `32→64→128→256→512→1024→2048`, per-stage Block1D depths `[3,3,3,3,3,3,8]`, optional last-norm (disabled for VibeVoice), Linear head 2048→64. Shape: `(B, 1, T_pcm) → (B, 64, T_pcm/3200)`.
- [ ] `TokenizerDecoder.cs` — mirror of encoder using `SConvTranspose1d`. Decoder depths = reverse of encoder depths.
- [ ] `VibeVoiceAcousticTokenizerModel.cs` — encoder + decoder + `fix_std=0.5` + Gaussian sampling. `encode()` returns `VibeVoiceTokenizerEncoderOutput { Mean, Std }`. `decode()` runs the streaming cache.
- [ ] `VibeVoiceSemanticTokenizerModel.cs` — encoder only, `vae_dim=128`, deterministic (`fix_std=0`, `dist_type=none`)
- [ ] `SpeechConnector.cs` — Linear(in, lm_hidden) + LlamaRMSNorm(lm_hidden, eps=1e-6) + Linear(lm_hidden, lm_hidden)
- [ ] `VibeVoiceDiffusionHead.cs` — 4 × HeadLayer(RMSNorm + AdaLN(SiLU+Linear, zero-init) + SwiGLU FFN with `head_ffn_ratio=3.0`) + FinalLayer (RMSNorm(no-affine) + AdaLN-2 + zero-init Linear→64). `noisy_images_proj` (64→hidden), `cond_proj` (hidden→hidden), `t_embedder` (sinusoidal 256 + MLP→hidden).
- [ ] `VibeVoiceCosineBetaSchedule.cs` — Nichol-Dhariwal cosine `alpha_bar(t) = cos²((t/T+s)/(1+s) · π/2)`, `s=0.008`, beta clip `[0, 0.999]`. **First verify our existing `DpmppMultiStepScheduler` matches** before writing a new one.
- [ ] DPM-Solver multistep wiring — v-prediction + cosine + order=2 + 20 inference steps. Confirm match against `vibevoice/schedule/dpm_solver.py`.
- [ ] `VibeVoiceTextTokenizer.cs` — thin subclass of dotLLM Qwen2 tokenizer exposing `SpeechStartId` / `SpeechEndId` / `SpeechDiffusionId` mapped from `<|vision_start|>` / `<|vision_end|>` / `<|vision_pad|>`. No vocab extension.
- [ ] `VibeVoiceTokenConstraintProcessor.cs` — logit mask limiting next-token sampling to `{speech_start, speech_end, speech_diffusion, eos, bos}` only.
- [ ] `VibeVoiceProcessor.cs` — multi-speaker script parser (regex `^Speaker\s+(\d+)\s*:\s*(.*)$`, JSON + plain-text + Speaker-N: formats), audio dB-FS normalizer (target=-25 dBFS), voice-prompt builder (`N_i = ceil(audio_samples / 3200)` `<|vision_pad|>` tokens per speaker), `speech_input_mask` construction.
- [ ] `VibeVoicePipeline.cs` (multi-speaker) — prefill (run voice audio through acoustic encoder, splice into LM embed at mask positions) + AR loop (Qwen2.5 forward → constrained logits → branch on token kind → DDPM sub-loop with CFG on `speech_diffusion` → acoustic decode → semantic re-encode → embed feedback). Includes negative-stream KV-cache bookkeeping for per-token CFG.
- [ ] `VibeVoiceStreamingPipeline.cs` — split-LM forward (lower Qwen2.5 layers text-only with `norm = Identity`, upper layers TTS with `tts_input_types` embedding marking text(1)/speech(0)) + 5-text/6-speech windowed interleave + `BinaryClassifier` EOS head + acoustic-only feedback. Batch=1 enforced.
- [ ] `BinaryClassifier.cs` — `Linear(hidden, hidden) → ReLU → Linear(hidden, 1)` for streaming EOS.
- [ ] **dotLLM dependency**: add per-layer-stop or two-instance API on `Qwen2Model` so we can run lower-N then upper-(total-N) layers independently (needed for streaming variant). Coordinate with dotLLM maintainer.
- [ ] **Audio streamer surface** — reuse the `IAsyncEnumerable<AudioChunk>` pattern from §3 of this doc, emit one chunk per `speech_diffusion` token (3200 samples = 133 ms at 24 kHz). Streaming-0.5B targets < 250 ms first-packet latency.
- [ ] Long-context smoke test: 65k-token KV cache stable on 1.5B (90 min output), 32k on 7B (45 min)
- [ ] License surface: VibeVoice is **MIT** (commercially usable) — flag this in the model registry as a positive contrast to XTTS (CPML), AudioLDM2 (CC-BY-NC), SparkTTS (CC-BY-NC)

## 7. Music Generation (`SharpInference.Music`)

### ACE-Step (flagship)
- [ ] `AceStepUmt5TextEncoder.cs` — reuse AuraFlow's UMT5 from SharpInference.Diffusion
- [ ] `AceStepDit.cs` v1 — 24L × 20 heads × head_dim 128 × inner 2560
- [ ] `AceStepFsqLm.cs` v1.5 — Qwen3-based decoder predicting FSQ audio tokens
- [ ] `MusicDcaeVae.cs` — Sana AutoencoderDC over 2D stereo mel
- [ ] `AdaMosHiFiGanV1.cs` vocoder
- [ ] Voice BPE tokenizer (XTTS-style)
- [ ] Lyric structure tags + `tokenize_lyrics` function + 19-lang ID map
- [ ] Flow-match scheduler with `shift=3.0` + omega/APG/CFG-Zero*
- [ ] Three schedulers: Euler / Heun / PingPong
- [ ] Flow-edit / repaint algorithm

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

### YuE
- [ ] Llama-2 7B (S1) with extended vocab for x-codec audio tokens — overlap with dotLLM
- [ ] Llama-2 1B (S2) upsampler
- [ ] X-Codec mini decoder (8 codebooks × 1024 entries @ 50 Hz, 16 kHz mono)
- [ ] Dual-track interleaving (v_0, a_0, v_1, a_1, ...)
- [ ] Mandatory `repetition_penalty=1.1`
- [ ] CoT and ICL variants
- [ ] Long-context RoPE scaling for 5+ minute songs

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

## 12. Status Summary (2026-05-23)

### Done

- **All 9 codecs** in §4 are structurally complete with config + Encode + Decode + EnumerateWeights surfaces. Build passes clean across 12 projects × 2 frameworks.
- **All 10 shared-primitive items** in §3 are landed (streaming surface, ring buffer, KV cache, LSTMs, mel extractor, SIMD FFT).
- **Backend op gaps from §3 PTX list** addressed at the C# IBackend level: `Conv1d`, `ConvTranspose1d`, `Sigmoid`, `Tanh`, `Snake`, `Elu` are wired on CpuBackend with real implementations. CudaBackend + VulkanBackend currently throw `NotSupportedException` for the not-yet-compiled-PTX/SPIR-V variants, with native source files staged.
- **Streaming wrappers** for any codec available via `StreamingCodecEncoder<T>` / `StreamingCodecDecoder<T>` (generic; works on EnCodec, DAC, SNAC, Mimi, XCodec, WavTokenizer).
- **Offline weight-norm fusion** CLI at `samples/FuseWeightNorm/`.
- **Smoke tests** for all 9 codecs (config shapes + construction sanity).
- **Native source files** for new CUDA + Vulkan ops written and committed under `native/cuda/audio/` and `native/vulkan/shaders/` (snake / conv1d / conv_transpose1d / extended elementwise with tanh+elu+sigmoid).

### Outstanding

- **PTX / SPIR-V compilation pass** — needs nvcc + glslc invocations in the native build pipeline; source-only today.
- **Per-codec STOI validation** against Python references — requires downloading official checkpoints. Smoke tests provide the scaffolding pattern; the gated-attribute approach used by F5TtsSmokeTests is the convention.
- **Per-codec streaming caches** with proper context propagation (current `StreamingCodecEncoder<T>` is causal-naive — re-encodes each chunk fresh). Acceptable for chunk_size ≥ 200 ms; sub-200 ms live use needs per-codec cache wiring (Mimi-specific is the highest-priority).
- **TTS / music pipelines** that consume these codecs (Bark / MusicGen / Sesame CSM / Orpheus / IndexTTS / Higgs Audio / Spark-TTS / YuE) — separate work items, codec foundation is in place.
