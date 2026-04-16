# Model Support Roadmap

> Back to [Core Design](CORE_DESIGN.md)

---

## Phase 1 — Initial Support (Months 1–4)

Highest-priority, most widely-used models. Getting these right validates the architecture. All models target both CUDA (NVIDIA) and Vulkan (AMD/Intel) backends via the `IBackend` abstraction.

### Image Generation

| Model | Format | GPU Backends | Why First |
|---|---|---|---|
| Stable Diffusion 1.5 | safetensors, GGUF | CUDA + Vulkan | Simplest UNet architecture. Best for validation. Huge ecosystem. First model on both backends. |
| SDXL 1.0 | safetensors | CUDA + Vulkan | Most popular SD model family. Dual CLIP encoders, larger UNet. |
| SDXL Turbo / Lightning | safetensors | CUDA + Vulkan | Few-step distilled models. Fast validation of scheduler correctness. |
| Flux.1-dev | safetensors, GGUF | CUDA + Vulkan | State-of-the-art DiT/flow-matching. Most requested in 2025–2026. |
| Flux.1-schnell | safetensors, GGUF | CUDA + Vulkan | 4-step distilled Flux. Extremely fast. |

**Image inference kernel requirements (beyond dotLLM's LLM kernels):** Conv2D (3×3, 1×1), GroupNorm, GroupNorm+SiLU fused, spatial SDPA, upsample 2D, timestep embedding, SiLU, GELU, dequant (Q8_0, Q4_K). Each implemented as CPU SIMD, CUDA PTX, and Vulkan SPIR-V.

### Audio

| Model | Format | Why First |
|---|---|---|
| Whisper (tiny → large-v3) | safetensors, GGUF | Universal STT standard. |
| Kokoro-82M | safetensors | Fastest high-quality TTS. Apache 2.0. |

### Vision

| Model | Format | Why First |
|---|---|---|
| CLIP ViT-L/14 | safetensors | Required by SD and SDXL internally. Ship as standalone too. |
| CLIP ViT-H/14 | safetensors | Required by SDXL. |

---

## Phase 2 — Extended Support (Months 5–8)

### Image Generation

| Model | Notes |
|---|---|
| SD 3 / SD 3.5 | MMDiT architecture, T5 text encoder, 3 CLIP variants |
| Stable Video Diffusion (SVD) | Image-to-video, temporal UNet |
| ControlNet (SD1.5 + SDXL) | Depth, Canny, OpenPose, Scribble, Tile variants |
| IP-Adapter | Image prompt conditioning |
| LCM / Hyper-SD | Very few step (1–4) distilled models |
| SDXL-Inpaint | Specialized inpainting model |
| AuraFlow | Open MMDiT competitor to Flux |

### Audio

| Model | Notes |
|---|---|
| Parler-TTS | Instruction-following TTS with voice description |
| WhisperX | Word-level timestamps, speaker diarization |
| F5-TTS | Voice cloning from short reference audio |
| RVC v2 | Voice conversion, widely used |

### Vision

| Model | Notes |
|---|---|
| SigLIP | Better than CLIP for zero-shot classification |
| YOLOv8 / YOLOv11 | Object detection and segmentation |
| Florence-2 | Vision-language, captioning, grounding |
| SAM 2 | Segment Anything, image and video |
| DINO v2 | Dense feature extraction |

---

## Phase 3 — Full Coverage (Months 9+)

### Image Generation
- All future Flux variants (Flux.1-fill, Flux.1-canny, etc.)
- HiDream
- LUMINA-Next
- Custom architecture support via ONNX passthrough

### Audio
- ACE-Step (music generation)
- MusicGen (Meta)
- Stable Audio
- VALL-E 2 / VoiceBox style models
- Fish TTS
- Orpheus TTS

### Video Generation
- LTX-Video
- Wan (2.1+)
- HunyuanVideo
- CogVideoX

### Multimodal / LLM+Vision
- LLaVA-style vision-language (via dotLLM integration)
- Qwen2.5-VL
- Pixtral

### Other
- Any model expressible as a computation graph with SharpInference's op set
- ONNX model passthrough for unsupported architectures
