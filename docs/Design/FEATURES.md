# Features

A capability-level view of what HartsyInference does across modalities. For per-model **status**
(verified end-to-end vs built-but-pending), see the modality status docs indexed in
[`../Checklists/MODEL_STATUS.md`](../Checklists/MODEL_STATUS.md); the cross-modality real-weight parity
authority is [`../Checklists/PARITY_VERIFICATION.md`](../Checklists/PARITY_VERIFICATION.md). This
document lists capabilities, not per-model checkmarks.

> **How you use these features:** the recommended path is
> [SwarmUI + the HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend).
> You can also embed the per-modality NuGet libraries directly, or drive the bundled sample CLIs under
> [`../../samples/`](../../samples/).

## Core engine

- **Pure C#/.NET.** No Python, no native inference library, no external process. Targets net8.0 and
  net10.0.
- **Three backends behind one `IBackend`:** CUDA (PTX via the CUDA Driver API + cuBLAS), Vulkan
  (SPIR-V compute), and CPU (AVX2 / AVX-512 / NEON SIMD with a scalar fallback).
- **Eager execution.** Ops run immediately; no computation graph to compile.
- **Zero-allocation hot paths.** Tensor storage in `NativeMemory.AlignedAlloc`, memory-mapped weights,
  `Span<T>` throughout.
- **Direct checkpoint loading:** `.safetensors`, `.gguf`, and PyTorch `.pt`/`.ckpt` pickle checkpoints,
  including sharded and diffusers-layout directories, with on-the-fly architecture detection
  (`PipelineFactory.DetectArchitecture`).
- **Quantization:** GGUF (Q4/Q5/Q6/Q8 and K-quants) with fused dequant/GEMV kernels, plus block-scaled
  low-precision formats (MXFP4/8, NVFP4) for supported architectures.
- **HuggingFace download** of checkpoints on first use for pipelines that support it.
- **LoRA** loading and application.

## LLM text generation (`HartsyInference.LLM`)

- **Native, config-driven generic decoder transformer** that backs many model families from one code
  path: Qwen2 / Qwen3, Llama-3.x, Mistral, and more, plus quantized GGUF inference for any supported
  architecture.
- **Device-resident KV cache** for O(n) incremental decode, a sampler chain (temperature, top-p, top-k,
  greedy), and chat-template handling read from the model file where present.
- The same generic transformer also powers text encoders used by diffusion and audio pipelines.
- Optimized CUDA decode path: fused mul_mat_vec Q4_K/Q6_K/Q8_0 kernels, quantized `lm_head`, and
  split-K flash-decode attention (see [the benchmarks](../../benchmarks/README.md)).

## Image generation (`HartsyInference.Diffusion`)

- **UNet architectures:** Stable Diffusion 1.5, SDXL (+ Refiner), inpainting.
- **DiT / MMDiT / NextDiT architectures:** Flux.1 / Flux.2, Chroma / Chroma Radiance, SD3, Qwen-Image,
  HunyuanImage, HiDream, AuraFlow, Lumina 2, ERNIE-Image, Kandinsky 5, OmniGen 2, Ideogram 4, and more.
- **Pipelines:** text-to-image, image-to-image, inpainting, and tiled VAE encode/decode for large
  images.
- **Text encoders:** CLIP (L/G/H/bigG), T5 / UMT5 / Pile-T5, Gemma-2, Qwen2.5-VL and Qwen3 taps, shared
  across architectures.
- **Schedulers:** the common diffusion and flow-matching samplers (Euler, DPM-family, flow-match shift,
  UniPC and friends).
- **Conditioning / prompting:** prompt weighting, BREAK, prompt scheduling, regional prompting, textual
  inversion, and clip-skip. ControlNet and IP-Adapter loaders are present (see roadmap for coverage).
- **One-line auto-load** for SDXL via `PipelineFactory.LoadAuto`; other families are constructed
  explicitly from components.

## Audio: speech, TTS, music, codecs (`HartsyInference.Audio`)

- **Speech-to-text:** Whisper (tiny to large-v3) and Moonshine, with language selection and timestamps.
- **Text-to-speech:** Kokoro, StyleTTS2, Bark, Spark-TTS (BiCodec), CosyVoice, VibeVoice, Piper/VITS,
  MeloTTS, F5-TTS voice cloning, and more.
- **Music generation:** ACE-Step (flow-matching DiT), MusicGen, YuE.
- **Neural audio codecs:** Vocos, EnCodec, DAC, SNAC, Mimi, WavTokenizer, BiCodec, XCodec, Oobleck.
- **DSP in pure C#:** STFT/mel/FFT, vocoders (HiFi-GAN family), and streaming.
- **Grapheme-to-phoneme** via `HartsyInference.Phonemizer`, a pure-C# espeak-ng port (no native espeak
  dependency).

## Vision (`HartsyInference.Vision`)

- **Embeddings:** CLIP (ViT-L/14, H/14, bigG/14), SigLIP / SigLIP2, DINOv2 / DINOv3 dense features.
- **Object detection:** YOLO8 / YOLO11 (n to xl).
- **Segmentation:** SAM, SAM 2, SAM 2.1; CLIPSeg for prompt-based masks.
- **Face:** RetinaFace-style detection and landmarks.
- **Image codec helpers:** PNG decode/encode and post-processing utilities.

## Video generation (`HartsyInference.Video`)

- **Text-to-video and image-to-video:** LTX-Video, Wan 2.x (T2V + I2V), Lance (T2V), Kandinsky 5 Video.
- Shared 3D-video foundation: CausalConv3d, Wan-family VAE encode/decode with streaming, frame
  encoders, N-axis RoPE.
- ffmpeg-based muxing when driven through the SwarmUI extension.

## 3D generation (`HartsyInference.ThreeD`)

- **Image/text to mesh:** TripoSR (feed-forward triplane/NeRF), Hunyuan3D-2 (Flux-lineage flow-match
  DiT + ShapeVAE).
- **Mesh/splat/triplane foundation:** marching cubes, with glTF (GLB), OBJ, and PLY export.

## Interactive / world models (`HartsyInference.Interactive`)

- **Real-time, action-conditioned generation:** keyboard / mouse / camera-pose input streams in,
  frames stream out. Models: Hunyuan-GameCraft, Matrix-Game 2.0 / 3.0, Oasis.
- **Session abstraction:** `IInteractiveSession` with background compute, action encoders, camera/FOV
  memory, and per-frame action submission.

## Consumption surfaces

- **SwarmUI backend extension** (recommended): registers HartsyInference as a SwarmUI backend, a
  pure-C# alternative to the ComfyUI backend.
- **NuGet libraries:** per-modality packages (see [NuGet Package Design](NUGET_PACKAGE_DESIGN.md)); a
  `HartsyInference` meta-package pulls in the core, all three backends, and every modality package
  including `HartsyInference.LLM` and `HartsyInference.Phonemizer` (only the abandoned `Server` and the
  sample `Cli` are excluded).
- **Sample CLIs:** developer/validation tools under [`../../samples/`](../../samples/) and
  [`../../src/HartsyInference.Cli`](../../src/HartsyInference.Cli).

## Explicitly not included

- No first-party UI / web app (use the SwarmUI extension).
- No OpenAI-compatible REST server (dropped; the `HartsyInference.Server` project is abandoned
  scaffolding).
- No training or fine-tuning (inference engine only).

See the [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) for planned additions
(ControlNet/IP-Adapter breadth, more vision detectors, HunyuanVideo/CogVideoX, Gaussian-splat 3D
output, performance work).
