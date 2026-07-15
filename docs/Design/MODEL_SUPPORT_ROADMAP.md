# Model Support Roadmap

> Back to [Core Design](CORE_DESIGN.md)
>
> This is the forward-looking *plan*. For the current built-vs-verified state of each model, see the
> per-modality status docs indexed in [`../Checklists/MODEL_STATUS.md`](../Checklists/MODEL_STATUS.md).

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
| CLIP ViT-L/14 (OpenAI) | safetensors | Required by SD/SDXL; ship standalone too |
| CLIP ViT-H/14 (OpenCLIP) | safetensors | Used by IP-Adapter SDXL; standalone scoring |
| CLIP ViT-bigG/14 (OpenCLIP) | safetensors | Required by SDXL second encoder |
| YOLO11n / YOLO11s | safetensors (PT→ours) | Smallest/fastest detection; baseline coverage |

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
| SigLIP / **SigLIP 2** | Better zero-shot than CLIP; SigLIP 2 (2025) adds multilingual + improved alignment, drop-in replacement |
| **EVA-CLIP** | LAION-trained CLIP variant, much stronger than OpenAI baseline; useful for retrieval / search |
| YOLOv8 / YOLO11 / **YOLOv10** / **YOLOv12** | Detection + segmentation across the modern Ultralytics + community variants |
| **RT-DETR / RT-DETRv2** | Baidu transformer-based detector; anchor-free, NMS-free; faster than YOLO at equal mAP |
| **Grounding DINO 1.5 / Grounding DINO Pro** | Open-vocabulary detection ("detect a red mug") — text-prompted; pairs with SAM for open-vocab segmentation |
| **YOLO-World v2** / **YOLOE** | Open-vocabulary YOLO variants; faster than Grounding DINO for fixed-class subsets |
| **OWLv2** | Google open-vocabulary detector; text + image query support |
| Florence-2 / **Florence-2.5** | Vision-language: captioning, grounding, OCR, dense prediction (unified output format) |
| SAM 2 / **SAM 2.1** | Segment Anything (image + video); SAM 2.1 adds long-video memory bank |
| **HQ-SAM / MobileSAM / FastSAM / EfficientSAM** | SAM variants: higher quality, mobile-grade, ~50× faster, real-time targeting |
| **EVF-SAM** | Text-prompted SAM ("segment the dog") — open-vocabulary segmentation |
| DINO v2 / **DINOv3** | Dense self-supervised features; DINOv3 (2025) is the current SOTA for unsupervised visual representations |
| **Hiera** | Meta hierarchical ViT; backbone for SAM 2; useful standalone for dense prediction |
| **AM-RADIO** | NVIDIA agglomerative model — distills CLIP + DINO + SAM into one backbone; one forward pass for retrieval + features + segmentation |

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
- **Lance video** (`Lance_3B_Video`, ByteDance, Apache-2.0) — unified multimodal, Wan2.2 3D causal VAE. See [LANCE_ARCHITECTURE.md](../Research/LANCE_ARCHITECTURE.md).
- **Cosmos-Predict1 Video2World** (NVIDIA, Open Model License) — AR transformer + discrete video tokenizer (Cosmos DV). Video continuation (not action-conditioned), but the AR + DV tokenizer infra is reused by Phase 10 world models. See [COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md](../Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md).

### Interactive / World Models (Phase 10)

Action-conditioned, real-time, frame-by-frame video generators — distinct from offline video diffusion. Take typed input events (keyboard scancodes, mouse deltas, gamepad sticks, camera pose) and emit a streamed frame per step. New `HartsyInference.Interactive` package. See [INTERACTIVE_INFERENCE.md](../Research/INTERACTIVE_INFERENCE.md) for the cross-cutting foundation.

| Model | Org | License | Notes |
|---|---|---|---|
| **Matrix-Game 3.0** | Skywork | Apache-2.0 | 5B (+ 28B MoE variant), 720p @ 40 FPS, memory-augmented DiT finetuned from Wan2.2-TI2V-5B. Flagship; shares VAE with Lance video. See [MATRIX_GAME_3_ARCHITECTURE.md](../Research/MATRIX_GAME_3_ARCHITECTURE.md). |
| **Matrix-Game 2.0** | Skywork | MIT | 1.8B, 540p @ 25 FPS, real-time on 12 GB GPUs. SkyReels-V2/Wan lineage. Entry-level world model. See [MATRIX_GAME_2_ARCHITECTURE.md](../Research/MATRIX_GAME_2_ARCHITECTURE.md). |
| **Oasis-500m** | Decart + Etched | MIT | Tiny (~500M), autoregressive frame-by-frame Minecraft world model. Likely uses a discrete video tokenizer (VQ family). Pedagogical / CI smoke test. See [OASIS_ARCHITECTURE.md](../Research/OASIS_ARCHITECTURE.md). |
| **Hunyuan-GameCraft 1.0** 🔧 | Tencent | weights: Tencent Hunyuan Community | 12.5B, 704×1216 @ 33 frames, keyboard + camera-pose actions, hybrid history conditioning. **Built end-to-end (structural, numerics validation-pending, 2026-06-15).** No license gate — engine is MIT, ships no weights/Tencent code; weight-use is the user's responsibility (same as every model). See [PHASE_10_INTERACTIVE.md §7](../Checklists/PHASE_10_INTERACTIVE.md) + [HUNYUAN_GAMECRAFT_ARCHITECTURE.md](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md). |

**Considered + deferred** (not in v1 of Phase 10):
- **DIAMOND** (Alonso et al., MIT) — small (~381M) research-grade Atari / CS:GO world models. Useful as a reference for action-conditioning correctness but too narrow for shipping.
- **WHAM / Microsoft Muse** (Microsoft Research License — non-commercial) — architecturally interesting (VQ-GAN tokens + decoder-only AR + controller actions) but fails our permissive-license bar.
- **DreamerV3** — RL world-model training framework; no deployable inference checkpoints. Wrong shape for our engine.
- **V-JEPA 2** (Meta) — open and large but representation-only (predicts in embedding space, not pixels). Wrong modality for "world model that outputs frames."
- **Genie 3, GameNGen, VideoPoet, Sora, ByteDance Yan** — closed weights as of 2026-05; revisit if released.

### 3D Asset Generation (Phase 11)

Image/text → 3D mesh (and later Gaussian splats) — the one previously-empty modality. New `HartsyInference.ThreeD`
package with a representation-agnostic foundation (marching cubes, glTF/OBJ/PLY export, triplane/grid sampling,
DINOv2 conditioning in Vision) reused across models. See [PHASE_11_THREED.md](../Checklists/PHASE_11_THREED.md).

| Model | Org | License | Notes |
|---|---|---|---|
| **Hunyuan3D-2** (shape) ✅ | Tencent | weights: Tencent Hunyuan | Image→mesh: flow-match VecSet DiT + ShapeVAE occupancy → marching cubes. **Verified e2e** (DiT corr 0.99999738, VAE corr 0.99999518) + gen-perf optimized (71.3 → 9.2 s, 1.6× off Python). See [HUNYUAN3D_2_ARCHITECTURE.md](../Research/HUNYUAN3D_2_ARCHITECTURE.md). |
| **TripoSR** ✅ | Stability/Tripo | MIT | Image→mesh, feed-forward LRM → triplane → NeRF MLP → marching cubes. **Verified e2e** vs `tsr` (all stages corr ~1.0) + gen-perf optimized (26.2 → 2.1 s). See [TRIPOSR_ARCHITECTURE.md](../Research/TRIPOSR_ARCHITECTURE.md). |
| **TRELLIS** ❌ | Microsoft | MIT | Image→Gaussian-splat + mesh. Deferred — needs sparse 3D conv/attention + flexicubes + splat rendering. |

Reusable `.pt` (PyTorch pickle) checkpoint loader landed alongside GameCraft (`ModelHandler/PyTorch`), enabling
`.pt`-only models (Cosmos, GameCraft) without a Python conversion step.

### Vision (Phase 3 extensions)

| Category | Models |
|---|---|
| **Depth estimation** | Depth Anything v2, Depth Pro (Apple), Marigold (diffusion-based), MoGe, ZoeDepth, MiDaS (legacy) |
| **Pose estimation** | RTMPose, YOLOv8/v11-Pose, ViTPose++, OpenPose (legacy compat) |
| **Face detection / recognition** | RetinaFace, YOLOv8-Face, **InsightFace / ArcFace** (recognition + alignment), MediaPipe FaceMesh (468-pt landmarks) |
| **OCR** | GOT-OCR 2.0, PaddleOCR v4, Florence-2 OCR-mode, olmOCR (document-level) |
| **Object tracking / Re-ID** | ByteTrack, BoT-SORT, FairMOT — multi-object tracking with YOLO detector backbone |
| **Image captioning** | BLIP-2, BLIP-3, CogVLM2, GIT |
| **Dense prediction backbones** | Vision Mamba / VMamba (state-space), ConvNeXt v2 |
| **Image super-resolution** | Real-ESRGAN, ESRGAN, SwinIR — also serve as diffusion upscalers |

### Multimodal / LLM+Vision
- LLaVA-style (via the native `HartsyInference.LLM` package + Vision towers), Qwen2.5-VL, **Qwen3-VL** (2026), Pixtral
- **InternVL 2.5 / 3** — open weights, competitive with GPT-4V on benchmarks
- **PaliGemma 2** — Google, 2B/9B/28B sizes, vision-language transfer
- **Molmo** (Allen AI) — competitive open VLM
- **NVIDIA Eagle 2** — multi-vision-encoder fusion (CLIP + SigLIP + DINOv2 + SAM)

### Other
- Any model expressible with HartsyInference's op set
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
- **Qwen-Image** — ✅ scaffold complete (2026-05-07). Dual-stream MMDiT with `QwenImageBlock` (per-stream 6-output AdaLN + joint `[txt,img]` attention), `QwenImageRope` (3-axis [16,56,56] per-stream pre-concat), `QwenImageTransformer` (60 blocks), `QwenImagePipeline` (Qwen2.5-VL via `LlamaStyleEncoder` → flow-match → 16-channel VAE), `QwenImageCheckpointConverter`. Awaiting checkpoint download (or Q4_K GGUF for 12 GB cards) for first-run validation.
- **Hunyuan Image 2.1** — MMDiT with 32×32 VAE; reuse shared components, new block class + VAE
- **HiDream i1** — MMDiT; reuse shared components, new config
- **AuraFlow** — Hybrid 4-MMDiT-then-32-single-DiT using **Pile-T5-XL** (NOT XXL — context dim 2048) + SDXL VAE; reuse shared components, FP32 LayerNorm everywhere, no biases on attention/FFN, SwiGLU FFN with `mlpDim = find_multiple(int(2*4*dim/3), 256)` = 8192 for v0.3 (NOT 4×hidden), 8 register tokens prepended to text after `context_embedder`. Learned 2D pos_embed via `AuraFlowPatchEmbed` (Linear, NOT Conv2d) with center-crop selection on √1024=32×32 grid.
- **Kandinsky 5** — ✅ scaffold complete (2026-05-07). DiT with `Kandinsky5Block` + `Kandinsky5Rope` + `Kandinsky5Transformer` + `Kandinsky5Pipeline` (pre-computed dual Qwen2.5-VL + CLIP-L embeddings) + `Kandinsky5CheckpointConverter` (single-file + diffusers-folder loaders) + `Kandinsky5GenerationTests`. Awaiting `kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` for first-run.
- **Chroma** — moved here. Despite the v1 expectation of a "Flux fork", Chroma actually replaces Flux's per-block AdaLN linears with a **shared distilled-guidance approximator MLP** (5-layer SiLU + RMSNorm residual) that produces a precomputed per-block modulation table. Needs a custom transformer class (`ChromaTransformer`), pruned-AdaLN block variants (`ChromaDoubleStreamBlock` / `ChromaSingleStreamBlock`), the `ChromaApproximator` MLP, and a T5-only pipeline with attention-mask plumbing. Single-stream block does external `[txt, img]` concat once, not per-block. Reusable from Flux: `FluxAttention`, `FluxPosEmbed`, FFN/RMSNorm/SDPA, latent pack/unpack, and the FlowMatchEuler scheduler with dynamic shift. **No `swap_scale_shift` needed** for the final layer (norm shift/scale comes from the runtime modulation table, not from the checkpoint).
- **ERNIE-Image** — ✅ scaffold complete (2026-05-02 push, audited 2026-05-07). Single-stream DiT with 36 shared-AdaLN blocks, RMSNorm everywhere, non-interleaved 3D RoPE (32,48,48) with text_lens-based image-position offset, GELU-gated FFN, 1×1 Conv2d patch embed, Flux2-style 128-channel VAE. Text encoder is **Ministral 3B** (per `text_encoder/config.json`, `model_type: ministral3`) — already wired via `LlamaStyleEncoder` + `ErnieImageLlamaTextEncoder` exposing `hidden_states[-2]`. Awaiting checkpoint download for first-run validation.

**Significant new code** (unique architectures — all scaffold-complete 2026-05-07):
- **Anima** — ✅ scaffold complete. Cosmos-Predict2-2B-Text2Image with the temporal axis pinned to `T = 1`. `AnimaBlock` + `AnimaRope` (3-axis with T=1) + `AnimaTransformer` (drops video / world-model / autoregressive paths) + `AnimaPipeline` + `AnimaCheckpointConverter` + `AnimaGenerationTests`.
- **Lumina-Image-2.0** — ✅ scaffold complete. NextDiT family (sibling of Z-Image — 32-axis [32,32,32] vs [32,48,48], θ=10000 vs 256). `Lumina2Block` + `Lumina2ContextRefinerBlock` + `Lumina2Transformer` (separate Q/K/V projections, GQA 24:8, no learned cap_pad/x_pad) + `Lumina2Pipeline` (consumes pre-computed Gemma-2 caption embeddings; full Gemma-2 forward in C# is future work) + `Lumina2CheckpointConverter` + `Lumina2GenerationTests`. `LlamaStyleEncoderConfig.Gemma2_2B` preset added.
- **OmniGen 2** — ✅ scaffold complete (transformer forward is structural shell with first-run-debug marker). `OmniGen2Block` + `OmniGen2Rope` + `OmniGen2Transformer` + `OmniGen2Pipeline` + `OmniGen2CheckpointConverter` + `OmniGen2GenerationTests`. Editing / multi-image-input paths intentionally out of scope (t2i only).
- **HiDream i1** — ✅ scaffold complete. MMDiT with quad text encoder (CLIP-L + CLIP-G + T5-XXL + Llama-3.1). `HiDreamBlock` + `HiDreamRope` + `HiDreamTransformer` + `HiDreamPipeline` (full denoise loop with CFG) + `HiDreamCheckpointConverter` + `HiDreamGenerationTests`. MoE FFN currently runs as single-expert fallback — full expert routing pending.
- **Hunyuan Image 2.1** — ✅ scaffold complete. 17B MMDiT with 32× downsample VAE. `HunyuanImageBlock` + `HunyuanImageSingleBlock` + `HunyuanImageRope` + `HunyuanImageTransformer` + `HunyuanImageTokenRefiner` + `HunyuanImageByT5Projection` + `HunyuanImagePipeline` + `HunyuanImageCheckpointConverter` + `HunyuanImageGenerationTests`. Pipeline body has remaining first-run wiring TODOs the test catches and skips on.

### Implementation Guideline

When adding a new DiT/MMDiT model:

1. **Check the family** — if it's Flux-lineage or MMDiT, start from the existing transformer class
2. **Diff the architecture** — identify what's different (block count, hidden dim, positional encoding, text encoders, VAE)
3. **If only config differs** → add a new config record + checkpoint converter, reuse the transformer class
4. **If block internals differ** → create a new block class in `DiTBlocks/`, compose from shared components (`AdaLNModulation`, `QkNorm`, `SwiGluFfn`)
5. **Always reuse** `PatchEmbed`, `Unpatchify`, timestep embedding, and `IBackend` dispatch — never duplicate these
6. **Write a checkpoint converter** — map the model's weight names to our internal naming convention
