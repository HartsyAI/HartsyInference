# Model Support Roadmap

> Back to [Core Design](CORE_DESIGN.md)

## Phase 1 — Initial Support (Months 1–4)

Highest-priority models; all target CUDA + Vulkan via `IBackend`.

### Image Generation

| Model | Format | Why First |
|---|---|---|
| SD 1.5 | safetensors, GGUF | Simplest UNet; best for validation; huge ecosystem |
| SDXL 1.0 | safetensors | Most popular SD family; dual CLIP, larger UNet |
| SDXL Turbo/Lightning | safetensors | Few-step distilled; fast scheduler validation |
| Flux.1-dev | safetensors, GGUF | SOTA DiT/flow-matching; most requested |
| Flux.1-schnell | safetensors, GGUF | 4-step distilled; extremely fast |

**Kernel requirements (beyond dotLLM):** Conv2D (3×3, 1×1), GroupNorm, GroupNorm+SiLU fused, spatial SDPA, upsample 2D, timestep embedding, SiLU, GELU, dequant (Q8_0, Q4_K). CPU SIMD + CUDA PTX + Vulkan SPIR-V.

### Audio

| Model | Format | Why First |
|---|---|---|
| Whisper (tiny → large-v3) | safetensors, GGUF | Universal STT standard |
| Kokoro-82M | safetensors | Fast, high-quality TTS; Apache 2.0 |

### Vision

| Model | Format | Why First |
|---|---|---|
| CLIP ViT-L/14 | safetensors | Required by SD/SDXL; ship standalone too |
| CLIP ViT-H/14 | safetensors | Required by SDXL |

## Phase 2 — Extended Support (Months 5–8)

### Image Generation

| Model | Notes |
|---|---|
| SD 3 / SD 3.5 | MMDiT, T5, 3 CLIP variants |
| SVD | Image-to-video, temporal UNet |
| ControlNet (SD1.5 + SDXL) | Depth, Canny, OpenPose, Scribble, Tile |
| IP-Adapter | Image prompt conditioning |
| LCM / Hyper-SD | 1–4 step distilled |
| SDXL-Inpaint | Specialized inpainting |
| AuraFlow | Open MMDiT competitor |

### Audio

| Model | Notes |
|---|---|
| Parler-TTS | Instruction-following TTS |
| WhisperX | Word-level timestamps, diarization |
| F5-TTS | Voice cloning |
| RVC v2 | Voice conversion |

### Vision

| Model | Notes |
|---|---|
| SigLIP | Better zero-shot than CLIP |
| YOLOv8 / YOLOv11 | Detection + segmentation |
| Florence-2 | Vision-language, captioning, grounding |
| SAM 2 | Segment Anything (image + video) |
| DINO v2 | Dense feature extraction |

## Phase 3 — Full Coverage (Months 9+)

### Image Generation
- Flux variants (fill, canny, etc.), HiDream, LUMINA-Next
- ONNX passthrough for unsupported architectures

### Audio
- ACE-Step, MusicGen, Stable Audio, VALL-E 2, Fish TTS, Orpheus TTS

### Video Generation
- LTX-Video, Wan (2.1+), HunyuanVideo, CogVideoX

### Multimodal / LLM+Vision
- LLaVA-style (via dotLLM), Qwen2.5-VL, Pixtral

### Other
- Any model expressible with SharpInference's op set
- ONNX passthrough for unsupported architectures
