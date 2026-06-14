# Research Requirements

Every area below needs a `docs/Research/` document **before** implementation begins. See `docs/Agents/RESEARCH.md` for research output format.

## Model Formats & Loading

| Document | Needed Before |
|---|---|
| [SAFETENSORS_FORMAT.md](../Research/SAFETENSORS_FORMAT.md) | ModelHandler |
| [GGUF_FORMAT.md](../Research/GGUF_FORMAT.md) | ModelHandler |
| [QUANTIZATION_DIFFUSION.md](../Research/QUANTIZATION_DIFFUSION.md) | ModelHandler, Diffusion |

## GPU / Compute

| Document | Needed Before |
|---|---|
| [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) | Cuda |
| [PTX_KERNELS.md](../Research/PTX_KERNELS.md) | Cuda / Ptx |
| [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) | Cuda |
| [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md) | Vulkan ✅ |
| [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md) | Vulkan / Spirv ✅ |
| [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md) | Vulkan ✅ |
| [SIMD_INTRINSICS_DOTNET.md](../Research/SIMD_INTRINSICS_DOTNET.md) | Cpu |

## CPU Kernel Algorithms

| Document | Needed Before |
|---|---|
| [IM2COL_CPU.md](../Research/IM2COL_CPU.md) | Cpu |
| [GROUPNORM_MATH.md](../Research/GROUPNORM_MATH.md) | Cpu |
| [FLASH_ATTENTION.md](../Research/FLASH_ATTENTION.md) | Cpu, Cuda |

## Diffusion Architectures

| Document | Needed Before |
|---|---|
| [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) | Diffusion (UNet) |
| [SDXL_ARCHITECTURE.md](../Research/SDXL_ARCHITECTURE.md) | Diffusion (SDXL) |
| [FLUX_ARCHITECTURE.md](../Research/FLUX_ARCHITECTURE.md) | Diffusion (Flux) |
| [SD3_ARCHITECTURE.md](../Research/SD3_ARCHITECTURE.md) | Diffusion (SD3) |
| [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) | Diffusion (VAE) |
| [LANCE_ARCHITECTURE.md](../Research/LANCE_ARCHITECTURE.md) ✅ | Diffusion (Lance image pipeline, Phase 4) + Video (Lance video pipeline, Phase 9) |

## Diffusion Techniques

| Document | Needed Before |
|---|---|
| [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) | Diffusion (Schedulers) |
| [CFG_AND_GUIDANCE.md](../Research/CFG_AND_GUIDANCE.md) | Diffusion (Pipelines) |
| [LORA_FORMAT.md](../Research/LORA_FORMAT.md) | Diffusion (Adapters) |
| [CONTROLNET.md](../Research/CONTROLNET.md) | Diffusion (Adapters) |

## Text Encoders & Tokenizers

| Document | Needed Before |
|---|---|
| [CLIP_ARCHITECTURE.md](../Research/CLIP_ARCHITECTURE.md) | Vision, Diffusion |
| [CLIP_TOKENIZER.md](../Research/CLIP_TOKENIZER.md) | Tokenizers |
| [T5_ARCHITECTURE.md](../Research/T5_ARCHITECTURE.md) | Diffusion, Tokenizers |
| [T5_TOKENIZER.md](../Research/T5_TOKENIZER.md) | Tokenizers |

## Audio — Shared Infrastructure

| Document | Needed Before |
|---|---|
| [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) ✅ | All audio models |
| [AUDIO_CODECS.md](../Research/AUDIO_CODECS.md) ✅ | Bark, MusicGen, IndexTTS, CosyVoice, Higgs, Sesame CSM, Moshi, Orpheus, etc. |
| [FLOW_MATCHING_AUDIO.md](../Research/FLOW_MATCHING_AUDIO.md) ✅ | F5-TTS, ACE-Step, DiffRhythm, Stable Audio Open |
| [G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md) ✅ | Kokoro, MeloTTS, GPT-SoVITS, StyleTTS 2 (any phoneme-input TTS) |
| [STREAMING_AUDIO_INFERENCE.md](../Research/STREAMING_AUDIO_INFERENCE.md) ✅ | Parakeet, Moonshine, CosyVoice 2, Sesame CSM, Server (Phase 7) |
| [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) ✅ | Kokoro, AudioLDM2, MeloTTS, GPT-SoVITS, F5-TTS (Vocos), CosyVoice |

## Audio — Speech-to-Text (STT)

| Document | Needed Before |
|---|---|
| [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) ✅ | Whisper / Distil-Whisper pipeline |
| [PARAKEET_ARCHITECTURE.md](../Research/PARAKEET_ARCHITECTURE.md) ✅ | NVIDIA Parakeet (CTC / RNN-T / TDT) — SOTA fast English |
| [CANARY_ARCHITECTURE.md](../Research/CANARY_ARCHITECTURE.md) ✅ | NVIDIA Canary — SOTA multilingual + translation |
| [MOONSHINE_ARCHITECTURE.md](../Research/MOONSHINE_ARCHITECTURE.md) ✅ | Moonshine — edge / streaming STT (raw waveform input) |
| [SENSEVOICE_FIREREDASR_ARCHITECTURE.md](../Research/SENSEVOICE_FIREREDASR_ARCHITECTURE.md) ✅ | SenseVoice (Alibaba) + FireRedASR (Xiaohongshu) — Chinese-strong STT |

## Audio — Text-to-Speech (TTS)

| Document | Needed Before |
|---|---|
| [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) ✅ | Kokoro 82M TTS |
| [F5_TTS_ARCHITECTURE.md](../Research/F5_TTS_ARCHITECTURE.md) ✅ | F5-TTS — flow-matching zero-shot voice clone |
| [XTTS_ARCHITECTURE.md](../Research/XTTS_ARCHITECTURE.md) ✅ | Coqui XTTS-v2 — multilingual voice clone |
| [BARK_ARCHITECTURE.md](../Research/BARK_ARCHITECTURE.md) ✅ | Bark (Suno) — generative audio (speech + music + SFX) |
| [COSYVOICE_ARCHITECTURE.md](../Research/COSYVOICE_ARCHITECTURE.md) ✅ | CosyVoice 1/2 — streaming voice clone |
| [INDEX_TTS_ARCHITECTURE.md](../Research/INDEX_TTS_ARCHITECTURE.md) ✅ | IndexTTS 1.5 / 2 — Chinese + English voice clone |
| [SPARK_TTS_ARCHITECTURE.md](../Research/SPARK_TTS_ARCHITECTURE.md) ✅ | Spark-TTS — Qwen2.5-based zero-shot clone |
| [CHATTTS_ARCHITECTURE.md](../Research/CHATTTS_ARCHITECTURE.md) ✅ | ChatTTS — conversational TTS with paralinguistics |
| [HIGGS_AUDIO_ARCHITECTURE.md](../Research/HIGGS_AUDIO_ARCHITECTURE.md) ✅ | Higgs Audio v2 — Llama-3-based dialogue + voice clone |
| [SESAME_CSM_ARCHITECTURE.md](../Research/SESAME_CSM_ARCHITECTURE.md) ✅ | Sesame CSM — streaming conversational (Mimi codec) |
| [GPT_SOVITS_ARCHITECTURE.md](../Research/GPT_SOVITS_ARCHITECTURE.md) ✅ | GPT-SoVITS — few-shot voice clone (zh/ja/en) |
| [OPENVOICE_ARCHITECTURE.md](../Research/OPENVOICE_ARCHITECTURE.md) ✅ | OpenVoice v2 — two-stage tone-color conversion |
| [MELOTTS_ARCHITECTURE.md](../Research/MELOTTS_ARCHITECTURE.md) ✅ | MeloTTS — multilingual VITS-based (stage 1 of OpenVoice) |
| [STYLETTS2_ARCHITECTURE.md](../Research/STYLETTS2_ARCHITECTURE.md) ✅ | StyleTTS 2 — Kokoro's parent architecture, zero-shot extension |

## Audio — Music Generation

| Document | Needed Before |
|---|---|
| [ACE_STEP_ARCHITECTURE.md](../Research/ACE_STEP_ARCHITECTURE.md) ✅ | ACE-Step v1 / v1.5 / XL — flagship flow-matching music model |
| [STABLE_AUDIO_ARCHITECTURE.md](../Research/STABLE_AUDIO_ARCHITECTURE.md) ✅ | Stable Audio Open 1.0 / Small / 2 — 44.1 kHz stereo DiT |
| [MUSICGEN_ARCHITECTURE.md](../Research/MUSICGEN_ARCHITECTURE.md) ✅ | MusicGen + AudioGen + AudioCraft family |
| [YUE_ARCHITECTURE.md](../Research/YUE_ARCHITECTURE.md) ✅ | YuE — Llama-2-based long-form vocal music |
| [DIFFRHYTHM_ARCHITECTURE.md](../Research/DIFFRHYTHM_ARCHITECTURE.md) ✅ | DiffRhythm — fast latent-diffusion full-song generation |
| [AUDIOLDM2_ARCHITECTURE.md](../Research/AUDIOLDM2_ARCHITECTURE.md) ✅ | AudioLDM 2 — text-to-audio (music / SFX / speech) |

## Vision / Server / Reference

| Document | Needed Before |
|---|---|
| [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) | Vision |
| [OPENAI_IMAGE_API.md](../Research/OPENAI_IMAGE_API.md) | Server |
| [DOTLLM_ARCHITECTURE.md](../Research/DOTLLM_ARCHITECTURE.md) | All packages |

## Interactive / World Models (Phase 10)

> Foundational design doc + one architecture doc per Tier-1 model. All Apache-2.0 / MIT except Hunyuan-GameCraft (Tencent Hunyuan Community License — restricted, license-acceptance gated).

| Document | Needed Before |
|---|---|
| [INTERACTIVE_INFERENCE.md](../Research/INTERACTIVE_INFERENCE.md) ✅ | Phase 9 shared infra (`IActionEncoder`, `DenoiseKvCache`, `IDiscreteVideoTokenizer`, `DistilledFlowMatchEuler`, `VideoVaeStreamDecoder`, license plumbing) + Phase 10 (Interactive + all world-model pipelines) |
| [MATRIX_GAME_3_ARCHITECTURE.md](../Research/MATRIX_GAME_3_ARCHITECTURE.md) ✅ | HartsyInference.Interactive (Matrix-Game 3.0 pipeline, Phase 10) — flagship 5B world model, Wan2.2-TI2V-5B finetune, ActionModule + camera-aware memory, FlowUniPC + DMD distilled |
| [MATRIX_GAME_2_ARCHITECTURE.md](../Research/MATRIX_GAME_2_ARCHITECTURE.md) ✅ | HartsyInference.Interactive (Matrix-Game 2.0 pipeline, Phase 10) — 1.8B entry-level, SkyReels-V2 (Wan2.1) lineage, per-variant action vocabs, 3-4 step distilled |
| [OASIS_ARCHITECTURE.md](../Research/OASIS_ARCHITECTURE.md) ✅ | HartsyInference.Interactive (Oasis pipeline, Phase 10) — tiny 500M DiT-S/2 spatio-temporal axial attention, continuous Gaussian VAE, DDIM v-pred + Diffusion Forcing; CI smoke-test target |
| [HUNYUAN_GAMECRAFT_ARCHITECTURE.md](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md) ✅ | HartsyInference.Interactive (GameCraft pipeline, Phase 10) — **license-restricted**, gated on user acceptance |
| [COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md](../Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md) ✅ | HartsyInference.Video (Cosmos V2W pipeline, Phase 9 — AR video continuation) — discrete FSQ tokenizer + AR transformer infra reused by future Phase 10 AR world models |
