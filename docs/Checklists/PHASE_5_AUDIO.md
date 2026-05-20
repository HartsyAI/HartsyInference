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
| TTS | Kokoro, F5-TTS, XTTS-v2, Bark, CosyVoice 1/2, IndexTTS 1.5/2, SparkTTS, ChatTTS, Higgs Audio v2, Sesame CSM, GPT-SoVITS, OpenVoice v2, MeloTTS, StyleTTS 2 | 14 docs ✅ |
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
- [ ] Decide TTS model rollout order — recommend **Kokoro first** (simplest, no codec, deterministic voices), then **F5-TTS** (no G2P needed, flow matching reuses image scheduler infra), then **XTTS-v2** (multilingual character-level), then **CosyVoice 2** (streaming + Qwen reuse), then the rest.
- [ ] Decide music model rollout order — recommend **ACE-Step first** (flagship per user request, flow-matching reuses image pipeline patterns), then **Stable Audio Open** (similar tech stack, smaller scope), then **MusicGen** (clean AR + EnCodec reference), then **DiffRhythm / YuE** (full-song generation), then **AudioLDM 2** (text-to-audio for SFX).
- [ ] G2P language-coverage cut: **English only in v1** via CMUDict + ARPABET→IPA + heteronym table + small neural OOV fallback. Chinese (pinyin via jieba.NET port) and Japanese (NMeCab + UniDic repackaging) phased.
- [ ] Audio codec build order: **EnCodec 24kHz first** (Bark, MusicGen, AudioGen, AudioCraft), then **Mimi** (Sesame CSM streaming demo), then **DAC** (IndexTTS, Higgs Audio), then **xcodec** (YuE), then **Vocos** (covers F5/Kokoro variants/ChatTTS as vocoder too).
- [ ] Decide whether to package music separately under `SharpInference.Music` or keep all audio under `SharpInference.Audio`. Music brings full-song generation, long-context, and large LLM backbones that have a different memory profile.

## 3. Audio Preprocessing & Shared Primitives

- [ ] `AudioPreprocessor.cs` — resample (polyphase, windowed sinc, Kaiser β=8.6, configurable taps), normalize, windowing
- [ ] `StftProcessor.cs` — radix-2 Cooley-Tukey FFT (Sse2/Avx2/Avx512), STFT, periodic Hann window
- [ ] `MelSpectrogramProcessor.cs` — Slaney mel filterbank (generated at startup), log compression, normalization variants (Whisper / NeMo / StyleTTS)
- [ ] `IStftLayer.cs` — inverse STFT for vocoders (iSTFTNet, Vocos)
- [ ] `AudioRingBuffer.cs` — circular PCM buffer for streaming input
- [ ] `StreamingMelExtractor.cs` — incremental STFT with context tail between chunks
- [ ] `StreamingKvCache.cs` — per-layer K/V append-and-grow with position tracking
- [ ] PTX kernels: `fft_radix2.ptx`, `mel_filterbank.ptx`, `istft_overlap_add.ptx`, `snake_activation.ptx`, `conv_transpose1d.ptx`
- [ ] `IAsyncEnumerable<AudioChunk>` output streaming surface (matches dotLLM streaming convention)
- [ ] `LstmCell` / `BiLstm` modules (needed by Kokoro, GPT-SoVITS, several VITS-based TTS)

## 4. Neural Audio Codecs (`SharpInference.Audio.Codecs`)

- [ ] `EnCodec24kHz` encoder + decoder (24 kHz mono, 4-8 codebooks @ 75 Hz) — covers Bark, MusicGen, AudioGen
- [ ] `Mimi` decoder (12.5 Hz, 8 codebooks) — covers Sesame CSM, Moshi
- [ ] `DAC` encoder + decoder (44.1 kHz / 24 kHz / 16 kHz variants) — covers IndexTTS, Higgs Audio (acoustic branch), Spark-TTS HiFi-GAN wavegen
- [ ] `XCodec` / `XCodec2` decoder — covers YuE (xcodec_mini_infer)
- [ ] `Vocos` (mel-input and EnCodec-input) — covers F5-TTS, ChatTTS, vocoder for Kokoro alternative
- [ ] `SNAC` decoder — Orpheus TTS and forks
- [ ] `WavTokenizer` decoder — single-codebook variants
- [ ] `BiCodec` (Spark-TTS): semantic VQ + global FSQ dual-stream
- [ ] FSQ codec primitives: `(L-1)/2 * tanh(z) → round → base-L pack`
- [ ] Weight-norm fusion at load time (offline tool)
- [ ] Validation: round-trip encode→decode on test clip within ≤0.05 STOI vs Python reference

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
| CosyVoice 2 streaming | RTX 3060 12GB | First-packet < 250 ms |
| Sesame CSM | RTX 3060 12GB | Frame latency < 100 ms (real-time conversation) |
| ACE-Step 3.5B | RTX 3060 12GB | 4-min song in ≤ 60s (target — likely ~120s realistically) |
| Stable Audio Open Small | RTX 3060 12GB | 12s clip in ≤ 5s |

## 11. Review & Merge

- [ ] Code review per package (audio buffer boundaries, streaming thread safety, KV cache correctness)
- [ ] Benchmark report comparing SharpInference vs Python reference for each model
- [ ] License audit — flag non-commercial models (XTTS = CPML, AudioLDM2 = CC-BY-NC-SA, SparkTTS = CC-BY-NC-SA) in model registry; surface to users
- [ ] Update `MODEL_STATUS.md` with audio model coverage
- [ ] Merge to main branch
