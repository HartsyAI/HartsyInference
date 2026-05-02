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
| Flux.1 Tools (Fill, Redux, Canny, Depth) | Inpainting, outpainting, image variations, edge/depth-guided editing |
| Flux.1 Kontext | Edit/context model; image editing via text prompt describing changes |
| Flux.2 (Dev 32B, Klein 4B/9B) | Next-gen MMDiT; 16×16 VAE, Mistral/Qwen text enc; SOTA quality |
| Flux.2 Klein (4B, 9B) | Smaller Flux.2 variants; faster inference, Qwen text encoder |
| Hunyuan Image 2.1 | 17B MMDiT by Tencent; 32×32 VAE downscale, native 2048×2048; includes distilled + refiner variants |
| Chroma | 8.9B Flux derivative by Lodestone Rock; standard CFG (not distilled to 1) |
| Qwen-Image / Qwen-Image 2.0 | 7B–20B MMDiT; text-to-image gen + unified editing (inpaint, outpaint, relighting, style transfer) |
| Qwen-Image Edit | Mask-based inpainting, semantic/appearance editing, text rendering in images |
| Z-Image Turbo / Base | 6B Turbo (8-step distilled, Apache 2.0); Base unreleased (same 6B/3840-dim, un-distilled). Lumina2/NextDiT architecture (single-stream, 30 layers + 2 noise + 2 context refiners). FP8Mix distribution. Qwen3-4B text encoder + Flux VAE. See [Z_IMAGE_ARCHITECTURE.md](../Research/Z_IMAGE_ARCHITECTURE.md) |

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
- ERNIE-Image (8B single-stream DiT) + ERNIE-Image-Turbo (distilled, 8-step)
- F-Lite (10B/7B DiT by Freepik/Fal, copyright-safe training, CreativeML Open RAIL-M)
- Anima (2B Cosmos-Predict2 based, anime-focused, by CircleStone Labs / Comfy Org)
- Kandinsky 5 (6B DiT by Kandinsky Lab)
- Lumina 2.0 (2.6B NextDiT by Alpha-VLLM; Gemma 2 2B text encoder, Flux VAE)
- Chroma Radiance (8.9B pixel-space MMDiT, WIP) + Zeta Chroma (6B pixel S3-DiT)
- OmniGen 2 (7B MLLM by VectorSpaceLab; unified gen/edit/understanding)
- Flux.1-pro, HiDream i1 (17B MMDiT)
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

---

## Architecture Reuse Strategy

Most image generation models on the roadmap fall into a few architectural families. Build generic, configurable components for each family so new models require only a config + checkpoint converter, not a full reimplementation.

### Architectural Families

| Family | Architecture | Models | Key Traits |
|---|---|---|---|
| **Flux-lineage DiT** | Single-stream (+ optional double-stream) | Flux.1, Flux.2, Chroma, ERNIE-Image, F-Lite, Kandinsky 5 | AdaLN-Zero, RoPE, flow-matching, single/double stream blocks |
| **Lumina2 / NextDiT** | Single-stream w/ caption + noise refiner blocks | Z-Image (Turbo, Base), Lumina 2.0 | RMSNorm everywhere, multi-axis RoPE, AdaLN(4-out), SwiGLU, Qwen/Gemma LLM as text encoder |
| **MMDiT** | Symmetric dual-stream joint attention | SD3/3.5, Qwen-Image, Hunyuan Image 2.1, HiDream i1, AuraFlow | AdaLN-Zero, dual-stream with shared attention, typically 3 text encoders |
| **UNet** | Conv encoder-decoder with cross-attn | SD 1.5, SDXL, SVD, ControlNet, Inpaint variants | GroupNorm+SiLU, ResBlocks, cross-attention at select depths |
| **Unique** | Model-specific | Anima (Cosmos-Predict2), Lumina 2.0 (NextDiT), OmniGen 2 (MLLM) | Less reuse opportunity; implement per-model |

### Existing Shared Components (`DiTBlocks/`)

Already built and reusable across both DiT and MMDiT families:

| Component | Used By | Purpose |
|---|---|---|
| `AdaLNModulation` | Flux, SD3, AuraFlow, and all DiT/MMDiT variants | Timestep → shift/scale/gate; parameterized by output count |
| `QkNorm` | Flux, SD3.5+, AuraFlow | RMSNorm on Q/K heads |
| `SwiGluFfn` | Flux, SD3, AuraFlow | SwiGLU + GELU feedforward; handles both weight formats |
| `PatchEmbed` | SD3, AuraFlow, Hunyuan, Qwen-Image, and any DiT with patch input | Conv2D-based 2D patch embedding |
| `Unpatchify` | SD3, Flux, Hunyuan, Qwen-Image | Reconstruct spatial tensors from patch sequences |
| `FluxRope` | Flux family, Chroma, Flux.2 | Axial RoPE for 2D positions; reusable for any RoPE-based DiT |
| `DiTUtils` | All DiT/MMDiT transformers | Shared static helpers: LayerNormNoAffine, SinusoidalTimestepEmbedding, linear projections, reshape/concat ops |
| `AuraFlowJointBlock` | AuraFlow | Dual-stream joint block with image+text modulation |
| `AuraFlowSingleBlock` | AuraFlow | Single-stream image-only block |

### Shared Utilities (`DiTUtils`)

Consolidated from duplicated code across `FluxTransformer` and `Sd3Transformer`:

1. **`LayerNormNoAffine()`** — unparameterized LayerNorm
2. **`SinusoidalTimestepEmbedding()`** — sinusoidal timestep → embedding vector
3. **`ReshapeToMultiHead()` / `ReshapeFromMultiHead()`** — Q/K/V head reshaping
4. **`ConcatAlongSeqDim()` / `SplitAlongSeqDim()`** — joint attention sequence ops
5. **`LinearProject1D` / `LinearProjectBatched`** — CPU-side linear projection helpers
6. **`ConcatAlongLastDim()` / `ConcatPooled()` / `PadLastDim()`** — dimension manipulation

### Reuse Expectations Per Model

**Near-zero new code** (config + checkpoint converter only):
- **Flux.1 Tools** (Fill, Redux, Canny, Depth) — same Flux backbone with conditioning adapters
- **Flux.1 Kontext** — Flux backbone with edit-mode conditioning

**Minimal new code** (new config + small architectural tweaks):
- **Flux.2** — evolved Flux with new VAE (16×16) and text encoder (Mistral/Qwen). Core transformer likely reusable with config changes for block counts/dims.
- **Z-Image** — Lumina2/NextDiT architecture (NOT Flux-lineage as initially assumed). New top-level transformer class required, but sub-components (`SwiGluFfn`, `QkNorm`, `AdaLNModulation`) reusable. Uses Qwen3-4B as text encoder + Flux VAE verbatim. See [Z_IMAGE_ARCHITECTURE.md](../Research/Z_IMAGE_ARCHITECTURE.md).
- **F-Lite** — DiT-based, same family.

**Moderate new code** (new block class, reuse shared components):
- **SD3 / SD3.5** — `JointBlock` already implemented; needs `Sd3Transformer` forward pass + 3 text encoder setup
- **Qwen-Image** — MMDiT variant; reuse `AdaLNModulation`, `QkNorm`, `SwiGluFfn`; new block class for Qwen-specific attention + Qwen VL text encoder
- **Hunyuan Image 2.1** — MMDiT with 32×32 VAE; reuse shared components, new block class + VAE
- **HiDream i1** — MMDiT; reuse shared components, new config
- **AuraFlow** — Hybrid 4-MMDiT-then-32-single-DiT using **Pile-T5-XL** (NOT XXL — context dim 2048) + SDXL VAE; reuse shared components, FP32 LayerNorm everywhere, no biases on attention/FFN, SwiGLU FFN with `mlpDim = find_multiple(int(2*4*dim/3), 256)` = 8192 for v0.3 (NOT 4×hidden), 8 register tokens prepended to text after `context_embedder`. Learned 2D pos_embed via `AuraFlowPatchEmbed` (Linear, NOT Conv2d) with center-crop selection on √1024=32×32 grid.
- **Kandinsky 5** — DiT; reuse DiT shared components, new config
- **Chroma** — moved here. Despite the v1 expectation of a "Flux fork", Chroma actually replaces Flux's per-block AdaLN linears with a **shared distilled-guidance approximator MLP** (5-layer SiLU + RMSNorm residual) that produces a precomputed per-block modulation table. Needs a custom transformer class (`ChromaTransformer`), pruned-AdaLN block variants (`ChromaDoubleStreamBlock` / `ChromaSingleStreamBlock`), the `ChromaApproximator` MLP, and a T5-only pipeline with attention-mask plumbing. Single-stream block does external `[txt, img]` concat once, not per-block. Reusable from Flux: `FluxAttention`, `FluxPosEmbed`, FFN/RMSNorm/SDPA, latent pack/unpack, and the FlowMatchEuler scheduler with dynamic shift. **No `swap_scale_shift` needed** for the final layer (norm shift/scale comes from the runtime modulation table, not from the checkpoint).
- **ERNIE-Image** — moved here. Single-stream DiT (NOT Flux-lineage MMDiT). 36 identical `ErnieImageSharedAdaLNBlock` layers all sharing a single AdaLN modulation linear. RMSNorm everywhere (not LayerNorm), QK-RMSNorm over head_dim. **3D RoPE on `(text_pos, y, x)` axes (32, 48, 48)** with image tokens offset by `text_lens` per batch — non-standard pattern. GELU-gated FFN (NOT SwiGLU). Patch embed is a 1×1 Conv2d because patchification is delegated to a Flux2-style 128-channel VAE. Custom Baidu text encoder (`AutoModel`) — likely an ERNIE-class encoder (TBD without inspecting `text_encoder/config.json`). Optional `pe` Prompt Enhancer LLM (skip in v1). Diffusers reference exists at `transformer_ernie_image.py` + `pipeline_ernie_image.py`.

**Significant new code** (unique architectures):
- **Anima** — Cosmos-Predict2 is a fundamentally different architecture
- **Lumina 2.0** — NextDiT with Gemma 2 LLM text encoder; partially shares DiT patterns but different enough to need its own block
- **OmniGen 2** — MLLM-based, entirely different paradigm

### Implementation Guideline

When adding a new DiT/MMDiT model:

1. **Check the family** — if it's Flux-lineage or MMDiT, start from the existing transformer class
2. **Diff the architecture** — identify what's different (block count, hidden dim, positional encoding, text encoders, VAE)
3. **If only config differs** → add a new config record + checkpoint converter, reuse the transformer class
4. **If block internals differ** → create a new block class in `DiTBlocks/`, compose from shared components (`AdaLNModulation`, `QkNorm`, `SwiGluFfn`)
5. **Always reuse** `PatchEmbed`, `Unpatchify`, timestep embedding, and `IBackend` dispatch — never duplicate these
6. **Write a checkpoint converter** — map the model's weight names to our internal naming convention
