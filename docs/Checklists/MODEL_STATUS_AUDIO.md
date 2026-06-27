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
| **Kyutai TTS** (tts-1.6b-en_fr) | 🔬 | All numerical cores verified (backbone 1.3e-4, depformer 32/32, conditioner ~1e-8). Greedy e2e diverges by argmax cascade (not a bug); Mimi decode reconcile in progress. |
| **ResembleEnhance** | 🔬 | Modules synthetic-verified + converter built; real-weight mel→mel parity pending. |
| **F5-TTS** | 🔧 | Built + wired in SwarmUI; parity dump pending. |
| **HeartMuLa** | 🔧 | Built + wired; parity pending. |
| **VibeVoice / SparkTTS / NeuTTS / Orpheus / MeloTTS / Bark / Dia / FishSpeech / StyleTTS2** | 🔧 | Built (varying completeness); no real-weight parity yet. Orpheus/NeuTTS are phoneme-id-blocked (caller supplies ids). |
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
