# Audio Models — status

Concise status for every audio model: TTS, STT, and the codec / voice-conversion / music / separation
family. Build detail lives in [PHASE_5_AUDIO.md](PHASE_5_AUDIO.md); the music-specific completion plan is
[MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md). Parity evidence (maxAbs, bugs found)
lives in [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## TTS

| Model | Status | Notes |
|---|---|---|
| **GPT-SoVITS v2** | ✅ | HuBERT 1.07e-5, s1 GPT + s2 SoVITS verified, EN end-to-end → 32 kHz on real `lj1995` weights. |
| **Chatterbox** (ResembleAI) | ✅ | Full S3Gen rewrite (== CosyVoice2); enc 2.6e-6 / dec 4.4e-5 / vocoder 1.6e-5; end-to-end on CUDA. |
| **CosyVoice 2** | ✅ | Validated via the shared Chatterbox S3Gen. |
| **Qwen3-TTS** | ✅ | Bit-exact (RoPE split-half + byte-level tokenizer fixes). |
| **Piper** (VITS) | ✅ | corr 0.9998 vs onnxruntime; 7 VITS bugs fixed (affect all VITS). Espeak phonemization is the only gap. |
| **Kokoro** (StyleTTS2) | ✅ | ~1e-4 on the CUDA path (added `audio_leaky_relu` / `audio_adain1d` kernels). Loader 404s until repacked. |
| **F5-TTS** (v1 Base) | ✅ | Flow-matching DiT verified bit-exact: velocity corr 1.0, full CFM sample loop (generated mel) corr 1.0, Vocos corr 0.9999. 4 bugs fixed (ConvNeXt filler-mask, ×1000 timestep scale, erf/tanh GELU split, cond-anchored CFG + end-only ref-clamp). |
| **Kyutai TTS** (tts-1.6b-en_fr) | 🔬 | All numerical cores verified (backbone 1.3e-4, depformer 32/32, conditioner ~1e-8). Greedy e2e diverges by argmax cascade (not a bug); Mimi decode reconcile in progress. |
| **ResembleEnhance** | 🔬 | Modules synthetic-verified + converter built; real-weight mel→mel parity pending. |
| **F5-TTS** | 🔧 | Built + wired in SwarmUI; parity dump pending. |
| **HeartMuLa** | 🔧 | Built + wired; parity pending. |
| **MeloTTS** (English-v3) | ✅ | Real-weight e2e in pure C#: g2p ids exact, BERT bit-exact, audio corr 0.9993 (len exact) vs the noise-0 reference. `MeloTts` facade (LoadFromFiles/LoadAsync/SynthesizeText) + gated parity test. |
| **Spark-TTS-0.5B** | ✅ | Real-weight e2e bit-exact, fully in-engine (controllable mode): LM logits corr 1.0 (top-1 100%), greedy tokens 32/32 global + 179/179 semantic match Python, BiCodec wav corr 1.0 (factorized VQ, FSQ d-vector, AdaLN PreNet all corr 1.0). `SparkTtsPipeline.LoadFromDirectory`/`LoadAsync` + `SynthesizeControllable(text, gender, pitch, speed)`; `SparkTtsTokenizer` reuses the shared BPE + ByteLevelCodec. Zero-shot cloning would need the BiCodec encoder side (wav2vec2 + ECAPA), not built. |
| **FishSpeech 1.5** | 🔬 | DualAR LM verified: slow (24-layer) corr 1.0, fast depth-LM (4-layer) corr 0.9999. fused-key adapter + interleaved RoPE + no embed-scale + pre-norm fast input. Only the firefly-gan-vq codec remains. |
| **Dia-1.6B** | 🔬 | Full transformer verified bit-exact (corr 1.0): encoder (12L) + decoder (18L, cross-attn/9-ch/fused head). DenseGeneral adapter + split-half RoPE + attn scale 1.0 + KV-cache AdvanceLength fix. Only DAC wiring (shared/✅) + delay-AR remain. |
| **VibeVoice / NeuTTS / Orpheus / Bark / StyleTTS2** | 🔧 | Built (varying completeness); no real-weight parity yet. Orpheus/NeuTTS are phoneme-id-blocked (caller supplies ids). |
| **Zonos** | ⛔ | Blocked: espeak phonemes + ResNet293 speaker encoder + NovelAI sampler. Deferred. |

## STT

| Model | Status | Notes |
|---|---|---|
| **Whisper** (tiny → large-v3) | ✅ | JFK clip transcribes correct content words (`WhisperEndToEndTests`). |
| **Whisper streaming** (RealtimeSTT) | ✅ | LocalAgreement-2 + JFK streaming. |
| **Moonshine** | ✅ | Tests pass. |
| **Kyutai STT** (stt-1b / 2.6b) | 🔧 | Shares the moshi backbone; parity pending (no depformer). |

## Codec / voice conversion / music / separation

| Model | Status | Notes |
|---|---|---|
| **OpenVoice** (tone-color VC) | ✅ | Conv2d + GRU + speaker encoder validated. |
| **CAM++ / CamPlus** (speaker) | ✅ | From `funasr/campplus_cn_common.bin`. |
| **S3Tokenizer** | ✅ | From the `s3tokenizer` package. |
| **Vocos / vocoders** | ✅ | Test passes. |
| **GPT-SoVITS HuBERT / CosyVoice sub-encoders** | ✅ | Validated above. |
| **ACE-Step** (music DiT 3.5B) | 🔬 | DiT parity ~1e-8; E2E gen env-gated (13 GB F32 cast, bare terminal only). |
| **Mimi** (codec) | 🔬 | SeaNet composed-weight load fixed (DSM checkpoint); DSM 32-cb decode reconcile in progress. Shared with CSM. |
| **MusicGen / AudioGen** | 🔧 | Decoder LM built; codec-blocked (needs 32 kHz / 16 kHz EnCodec). Parity pending. |
| **YuE** (music) | 🔧 | Stage-1 built; Stage-2 residual upsampler + dual-track mixing pending. |
| **RVC** (voice conversion) | 🔧 | RMVPE front-end built; parity pending. |
| **Demucs** (separation) | 🔧 | Built; parity pending. |
| **CSM** (Sesame) | 🔧 | Uses Mimi; parity pending. |
| **Stable Audio Open / DiffRhythm / AudioLDM 2 / ACE-Step v1.5 + XL** | ❌/🔧 | Music roadmap; see [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md) for the per-model build state and ROI order. |
| **PocketTTS** (continuous-latent) | ⛔ | Gated `kyutai/pocket-tts`; config dims are placeholders. Reuses the moshi backbone. |

## Notes

- Music models have their own definition of "production-ready" and a sequenced completion plan in
  [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md); the universal missing piece there is
  the audio parity harness, now proven on ACE-Step's DiT.
- Build audio with `-m:1`; the Audio test suite crashes under xunit parallel, run it sequentially
  (`-- xUnit.ParallelizeTestCollections=false`) and reuse a model cache via `HARTSYINFERENCE_MODEL_CACHE`.
