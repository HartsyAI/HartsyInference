# Model Implementation Status

Snapshot of where each non-LLM image-generation model stands in HartsyInference.
Last updated: 2026-06-16 session: reconciled the table against the actual code. Hunyuan-GameCraft 1.0 (built 2026-06-15), Lance image + video (built 2026-06-14), and Ideogram 4 (built 2026-06-14, newly added row) moved to 🔧; Microsoft Lens promoted 🚧→🔧 (full pipeline built 2026-06-14); the GameCraft license-gate note was removed (the engine is MIT and weights are user-supplied, so there is no load-time gate).
Previous: 2026-05-27 session: Microsoft Lens / Lens-Turbo / Lens-Base (3.8B MMDiT + GPT-OSS MoE encoder + Flux.2 VAE) added as research-complete / not-started.
Previous: 2026-05-07 session: Qwen-Image + ERNIE-Image audit + Hunyuan Image 2.1 + Lumina 2.0 + HiDream i1 + Kandinsky 5 + Anima + OmniGen 2 scaffolds.

For architecture details, deviations, and per-model task lists, see
[PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md) and
[PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md).

> **Other modalities** (this file tracks image only). Status for the other modalities lives in their phase
> checklists: **video** → [PHASE_9_VIDEO.md](PHASE_9_VIDEO.md); **interactive / world models** (Matrix-Game 2/3,
> Oasis 🔧; Hunyuan-GameCraft 🔧 built 2026-06-15) → [PHASE_10_INTERACTIVE.md](PHASE_10_INTERACTIVE.md);
> **3D** (Hunyuan3D-2 + TripoSR 🔧 built 2026-06-14) → [PHASE_11_THREED.md](PHASE_11_THREED.md). 🔧 = built
> structurally, numerics validation-pending — the same state most video/world models are in.

## Legend
- ✅ end-to-end clean visual output verified
- 🔧 implementation green, awaiting checkpoint download + first-run iteration
- 🚧 scaffolding green, full implementation pending
- ❌ not started

## Status

| Model | Status | Code | Test | Notes |
|---|---|---|---|---|
| **SD 1.5** | ✅ | done | passes | Clean astronaut-on-horse output. |
| **SDXL** | ✅ | done | passes | Clean output (1024×1024 in 5.5 s/step on RTX 3060 F16). |
| **SD3.5 Medium** | ✅ | done | passes | Clean photorealistic astronaut. Five pipeline bugs fixed; see PHASE_3_DEVIATIONS #31-35. |
| **Flux Dev / Schnell / Krea** | ✅ | done | passes | Photoreal output across all three. |
| **Z-Image Turbo / Base** | ✅ | done | passes | Clean photoreal output. Eight plumbing bugs fixed; see PHASE_3_DEVIATIONS #25-30. |
| **Flux.2 Klein 4B** | ✅ | done | passes | Clean astronaut. |
| **Flux.2 Dev (32B)** | 🔧 | done | blocked | Won't fit on 12 GB without GGUF Q4 + per-block streaming. |
| **AuraFlow v0.3** | 🔧 | done | scaffolded | Full impl green (~1650 lines + diff harness). Awaits checkpoint download. |
| **Qwen-Image** | 🔧 | done | scaffolded | Full impl green (~1419 lines): `QwenImageBlock` (dual-stream, 6-output AdaLN per stream, joint `[txt,img]` attention concat, GELU FFN, QK-norm) + `QwenImageRope` (3-axis [16,56,56] per-stream pre-concat) + `QwenImageTransformer` (img_in / txt_norm / txt_in / sinusoidal+MLP timestep / 60-block stack / `[shift,scale]` AdaLN-Continuous final / unpatchify) + `QwenImageDebugDump` + `QwenImagePipeline` (Qwen2.5-VL text encode → 2×2 patch pack → flow-match Euler → optional CFG → FreeWeights → 16-channel VAE) + `QwenImageCheckpointConverter` (diffusers single-file with fused-QKV split + FP8 scale folding) + `QwenImageGenerationTests` with VRAM probe. Awaits ≥22 GB free VRAM (FP8 stock) or Q4_K GGUF. |
| **Chroma** | 🔧 | done | scaffolded | Full impl green (~2350 lines): `ChromaApproximator` (5-layer SiLU+RMSNorm MLP) + `ChromaCombinedTimestepEmbeddings` + pruned-AdaLN double + single block variants + `ChromaTransformer` + `ChromaCheckpointConverter` (BFL→diffusers) + T5-only `ChromaPipeline` (with "first padding token unmasked" mask + true-CFG dual-pass) + `ChromaDebugDump` + `ChromaGenerationTests`. All `Forward` methods implemented (zero `NotImplementedException`). Awaits `chroma_v1.safetensors` download. |
| **ERNIE-Image** | 🔧 | done | scaffolded | Full impl green (~2090 lines): non-interleaved 3-axis `ErnieImageRope` (axes 32/48/48, theta=256, image-position offset by `text_lens`) + `ErnieImagePatchEmbed` (1×1 Conv2d) + `ErnieImageBlock` (single-stream w/ shared modulation) + `ErnieImageTransformer` + `ErnieImageCheckpointConverter` (accepts both diffusers and Comfy-Org folder layouts) + `ErnieImagePipeline` (Flux2-style BN-unnormalize + standard CFG) + `IErnieTextEncoder` interface + `ErnieImageLlamaTextEncoder` (Ministral 3B per the Baidu repo's `text_encoder/config.json`) + `ErnieImageDebugDump` + `ErnieImageGenerationTests` with VRAM probe (skips below 14 GB free). Awaits ≥14 GB free VRAM for FP16 stock or Q4_K GGUF (`unsloth/ERNIE-Image-GGUF`). |
| **Hunyuan Image 2.1** | 🔧 | done | scaffolded | Full code stack: `HunyuanImageBlock` + `HunyuanImageSingleBlock` + `HunyuanImageRope` + `HunyuanImageTransformer` + `HunyuanImageTokenRefiner` + `HunyuanImageByT5Projection` + `HunyuanImagePipeline` (full body 2026-05-07: T5-XXL primary text encode, patchify → transformer → unpatchify → flow-match Euler → distilled-guidance → 32-channel VAE decode) + `HunyuanImageCheckpointConverter` (diffusers-naming bucket split, FP8 scale folding) + `HunyuanImageDebugDump` + `HunyuanImageGenerationTests` with VRAM probe. 17B FP16 needs ≥36 GB VRAM — fits A100/H100 or via future Q4_K GGUF. |
| **Lumina-Image-2.0** | 🔧 | done | scaffolded | Full impl: `Lumina2Block` + `Lumina2ContextRefinerBlock` + `Lumina2Transformer` (NextDiT family, sibling of Z-Image) + `Lumina2DebugDump` + `Lumina2Pipeline` (pre-computed Gemma-2 caption embeddings, dynamic-shift flow-match) + `Lumina2CheckpointConverter` (single-file or `transformer.*` prefix) + `Lumina2GenerationTests`. Awaits checkpoint download — 2B FP16 fits 12 GB cards. |
| **HiDream i1 (Full / Dev)** | 🔧 | done | scaffolded | Full code stack: `HiDreamBlock` + `HiDreamRope` + `HiDreamTransformer` + `HiDreamPipeline` (quad-encoder: CLIP-L + CLIP-G + T5-XXL + Llama-3.1, full denoise loop + CFG) + `HiDreamCheckpointConverter` (diffusers single-file → six-bucket split) + `HiDreamDebugDump` + `HiDreamGenerationTests` with VRAM probe (skips < 30 GB). MoE FFN is currently single-expert fallback — full MoE routing pending. Awaits checkpoint download + Llama-3.1 tokenizer (placeholder tokens used). |
| **Kandinsky 5.0 Lite** | 🔧 | done | passes-skip | Full impl: `Kandinsky5Block` + `Kandinsky5Rope` + `Kandinsky5Transformer` + `Kandinsky5DebugDump` + `Kandinsky5Pipeline` (pre-computed dual Qwen2.5-VL + CLIP-L embeddings) + `Kandinsky5CheckpointConverter` (single-file + diffusers folder loaders) + `Kandinsky5GenerationTests` with VRAM probe before/after preload. Awaits checkpoint download (`kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers`). |
| **Anima (Cosmos-Predict2)** | 🔧 | done | scaffolded | Full impl: `AnimaBlock` (single-stream w/ V-residual, Gated GELU FFN, AdaRMSNorm-Zero modulation) + `AnimaRope` (3-axis (T,H,W) with T pinned to 1 for image-only) + `AnimaTransformer` (drops every video / world-model code path; image-only invariant <c>p_t = 1</c>) + `AnimaDebugDump` + `AnimaPipeline` (pre-computed text embeddings + standard CFG) + `AnimaCheckpointConverter` + `AnimaGenerationTests` with VRAM probe. Awaits checkpoint download (Cosmos-Predict2-2B-Text2Image or the Anima fork). **img2img/inpaint + LoRA added (2026-05-31)** for SwarmUI-extension parity: `QwenImageVaeEncoder` (mirror of `QwenImageVaeDecoder`; encoder `downsamples` schedule is a documented mirror pending checkpoint key-reconciliation, same as the decoder was probed) + `AnimaPipeline` `ImageToImageRequest` branch (ctor overload taking the encoder, `Img2ImgSetup` + flow-match `AddNoise` at `sigma[startStep]`, strength=0 byte-identical short-circuit). LoRA: Anima/Cosmos `transformer.blocks.*` diffusers-PEFT LoRAs now route through the existing architecture-agnostic diffusers passthrough mapper (only `LoraFormatDetector` was widened — no new mapper). **DiT ControlNet/IP-Adapter deferred** (the existing adapters are UNet-only; DiT-adapter support is unbuilt for *every* DiT in the project — a foundational framework feature, not Anima-specific). |
| **OmniGen 2** | 🔧 | done | scaffolded | Full impl: `OmniGen2Block` + `OmniGen2Rope` (+ new `Joint` mode for joint-stream rotation) + `OmniGen2Transformer` (full Forward chain wired 2026-05-07: patchify → patch_embed → caption_embed → time-caption embed → 2 noise_refiner with image RoPE → 2 context_refiner with text RoPE → concat([txt,img]) → 32 main blocks with joint RoPE → strip text prefix → AdaLN-Continuous final → proj_out → unpatchify) + `OmniGen2DebugDump` + `OmniGen2Pipeline` + `OmniGen2CheckpointConverter` + `OmniGen2GenerationTests`. Editing / multi-image-input paths intentionally out of scope (t2i only). Awaits `OmniGen2/OmniGen2` checkpoint. |
| **F-Lite (Freepik / Fal.ai)** | 🔧 | done | scaffolded | Full impl (~1700 lines): `FLiteAttention` + `FLiteBlock` + `FLiteRope` + `FLiteTransformer` + `FLitePipeline` (T5-XXL layer-17 encode, inline dynamic-shift flow-match, dual-pass CFG) + `FLiteCheckpointConverter` + 8 unit tests. Awaits `Freepik/F-Lite` download (~29.4 GB). |
| **Boogu-Image 0.1 (Base / Turbo / Edit)** | 🔧 | DiT built (2026-06-21) | structural tests pass | **DiT + scheduler + converter + pipelines built end-to-end 2026-06-21** (structural; numerics validation-pending). 10B OmniGen2/Lumina-2 lineage: `BooguImageTransformer` (hidden 3360, 8 dual-stream + 32 single-stream + 3×2 refiner blocks, GQA 28:7, head_dim 120) **reusing** `OmniGen2Block` (single/refiner — byte-identical sandwich-norm block; added a precomputed-RoPE-table `Forward` overload) + `OmniGen2Rope` (added `BuildTableFromPositions` for the edit `pe_shift` offsets) + `DiTUtils` (added `PatchifyNCHW`/`UnpatchifyToNCHW`) + new `BooguImageDoubleBlock` (joint cross-attention + image self-attention dual stream) + `BooguImageConfig.V01`. New `BooguFlowMatchScheduler` (v1 logistic static time-shift, **ascending** t: noise→data). `BooguImageCheckpointConverter` (transformer + FLUX VAE + Qwen3-VL language tower + vision tower buckets). `BooguImagePipeline` (`GenerateFromEmbeddings` T2I single CFG; `EditFromEmbeddings` double guidance with VAE-encoded reference latents). Reuses FLUX `VaeEncoder`/`VaeDecoder` (`VaeConfig.Flux`) and the `LlamaStyleEncoder` Qwen3-VL-8B language tower. 4 structural smoke tests pass (full T2I + edit DiT forward finite; v1 scheduler math). **Remaining:** the Qwen3-VL-8B **vision tower** (depth 27 / hidden 1152 / patch 16 / spatial-merge 2 / temporal-patch 2 / out 4096, with **deepstack** taps [8,16,24] + interleaved **M-RoPE** + image-token merge into the LM embeds) for full multimodal edit conditioning, plus Python-parity dumps. Research doc: [`docs/Research/BOOGU_IMAGE.md`](../Research/BOOGU_IMAGE.md). |
| **Ideogram 4** | 🔧 | done (2026-06-14) | scaffolded | **Built end-to-end 2026-06-14.** 9.3B single-stream DiT: `Ideogram4Transformer` + `Ideogram4Pipeline` (all `Forward` paths implemented, zero `NotImplementedException`) with generation / diff-harness / scheduler / prompt test suites. Qwen multi-layer feature tap is shared with Microsoft Lens. Numerics validation-pending (awaits checkpoint download + first-run iteration). Research notes in [`docs/Research`](../Research). |
| **Microsoft Lens / Lens-Turbo / Lens-Base** | 🔧 | pipeline built (2026-06-14) | unit tests pass | **Built end-to-end 2026-06-14** (`LensPipeline` + `LensPipelineFactory` + `LensPipelineBundle` + `LensTransformer`, all `Forward` paths implemented, zero `NotImplementedException`; encoder / quant / factory tests pass; a full generation test plus checkpoint-diff pass are still pending). Microsoft Research's 3.8B dual-stream MMDiT (MIT, released 2026-05-25). 48 layers / hidden 1536 / 24 heads × 64 = 1536 / SwiGLU FFN 4096 / 3-axis complex-polar RoPE (8,28,28) with `scale_rope=True`. Text encoder is **GPT-OSS** MoE (24L, hidden 2880, 32 local experts × 4 active per token, GQA 64:8, alternating sliding-128 + full attention, MXFP4-native) with **multi-layer feature concat at layers [5,11,17,23]** — Microsoft's "massive text encoder training on GPT image outputs" trick. VAE reuses Flux.2 semantic VAE verbatim. Scheduler is `FlowMatchEulerDiscreteScheduler` with **empirical mu** computed from `seq_len`+`num_steps`. CFG is dual-pass batch-of-2 with **norm-rescaling** (combined prediction rescaled to match cond branch's per-token L2 norm). Three variants share the exact same architecture: Lens (20 steps, CFG 5.0, RL-tuned), Lens-Turbo (4 steps, CFG 1.0, distilled), Lens-Base (50 steps, CFG 5.0, SFT-only). Supports 18 resolution buckets (1024 & 1440 base × 9 aspect ratios from 1:2 to 2:1, all divisible by 16). On a 4090 it generates 1440×1440 with sharp text/iconography in a few seconds per image. Net-new backend work: GPT-OSS MoE FFN with real top-k routing (HiDream's MoE infra is currently single-expert fallback — this would be the first proper top-k port), MXFP4 dequant-at-load, alternating sliding/full attention mask in `LlamaStyleEncoderConfig`. Research doc: [`docs/Research/MICROSOFT_LENS_ARCHITECTURE.md`](../Research/MICROSOFT_LENS_ARCHITECTURE.md). See PHASE_4_MODEL_BREADTH § Microsoft Lens for build plan. |
| **Lance (ByteDance) image** | 🔧 | built (2026-06-14) | scaffolded | **Built end-to-end 2026-06-14** (`LanceTransformer` + `LanceImagePipeline` + `LancePipelineCommon` + `LanceCheckpointConverter`, with latent-patch and converter tests). Numerics validation-pending. Unified multimodal 3B-active model from ByteDance Research (Apache 2.0). Image variant `Lance_3B` (24.7 GB safetensors) covers T2I + image edit + image understanding via a single Qwen2.5-VL backbone with **MoT** (per-layer dual-stream QKV+FFN+norm for understanding vs. generation tokens) and **MaPE** (modality-aware temporal offset in 3D M-RoPE). Frozen Qwen2.5-VL ViT (semantic) + frozen Wan2.2 3D causal VAE (z=48, 16× spatial down). Rectified-flow + 3-way CFG (text+vision). Research doc: [`docs/Research/LANCE_ARCHITECTURE.md`](../Research/LANCE_ARCHITECTURE.md). See PHASE_4_MODEL_BREADTH § Lance for build plan; PHASE_9_VIDEO § Lance for the shared video variant. Net-new backend work: packed/varlen attention, 3D CausalConv, MoT routing primitive, diffusion KV-cache. |
| **Lance (ByteDance) video** | 🔧 | built (2026-06-14) | scaffolded | **Built end-to-end 2026-06-14** (`LanceVideoPipeline` on the shared Lance backbone + Wan2.2 3D causal VAE, with generation and pipeline tests). Numerics validation-pending. Video variant `Lance_3B_Video` (28.4 GB safetensors); same backbone as image but with the temporal-decode path active (Wan2.2 VAE 4× temporal down, up to 121 frames @ 480p). Shares MoT/MaPE/CausalConv infra with the image path; this brought up the reusable 3D-video foundation (CausalConv3d, streaming VAE, frame encoders). |
| **Cosmos-Predict1 V2W (5B / 13B)** | ❌ | not started | not started | NVIDIA's AR video-continuation model (Phase 9). Text + initial image OR 9-frame video → up to 12,800 tokens of continuation. AR transformer (4B / 12B base + per-layer cross-attn adapter) + FSQ discrete tokenizer (`[8,8,8,5,5,5]` levels → 64,000 codes, compression `[8,16,16]`). T5-11B text encoder. NVIDIA Open Model License (commercial OK). `.pt` pickle weights (no safetensors). **The FSQ tokenizer + AR transformer infra is the reusable substrate for any future AR-token world model in Phase 10.** Research doc: [`docs/Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md`](../Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md). |
| **Matrix-Game 3.0** (Skywork) — Phase 10 | 🔧 | core built (2026-06-10) | structural tests pass | Flagship interactive world model. 5B (+ 28B MoE coming soon), 720p @ 40 FPS on 9-GPU cluster, Apache-2.0. Finetune of Wan2.2-TI2V-5B with `ActionModule` (mouse=self-attn, keyboard=cross-attn) + camera-aware long-horizon memory (5 past-frame slots selected by FOV overlap) + DMD-distilled 3-step inference. UMT5-XXL text encoder + Wan2.2 3D causal VAE + MG-LightVAE (50%/75% pruned decoder). Shares VAE with Lance video. Research doc: [`docs/Research/MATRIX_GAME_3_ARCHITECTURE.md`](../Research/MATRIX_GAME_3_ARCHITECTURE.md); build plan: [`docs/Checklists/PHASE_10_INTERACTIVE.md § 5`](PHASE_10_INTERACTIVE.md). |
| **Matrix-Game 2.0** (Skywork) — Phase 10 | 🔧 | built (2026-06-10) | structural tests pass | Entry-level interactive world model. 1.8B, 540p @ 25 FPS, MIT. Built on SkyReels-V2-I2V-1.3B-540P (30-layer Wan2.1 DiT). Three per-domain variants (Universal: 4-key+mouse, GTA: 2-key+mouse, TempleRun: 7-key). Sliding-window KV cache (`local_attn_size=6`). Wan2.1 3D causal VAE (16 ch, 8×8/4× compression). CLIP-ViT-H/14 image encoder for I2V seed. Research doc: [`docs/Research/MATRIX_GAME_2_ARCHITECTURE.md`](../Research/MATRIX_GAME_2_ARCHITECTURE.md); build plan: [`docs/Checklists/PHASE_10_INTERACTIVE.md § 4`](PHASE_10_INTERACTIVE.md). |
| **Oasis-500m** (Decart + Etched) — Phase 10 | 🔧 | built (2026-06-10) | structural tests pass | Tiny 500M autoregressive frame-by-frame Minecraft world model, MIT. DiT-S/2 (16L × 1024 × 16h) with alternating spatial axial + causal temporal axial attention (Latte-style). **Continuous Gaussian ViT-VAE (NOT VQ)** — patch 20, 360×640 → 18×32×16 latent. 25-dim Minecraft VPT action vector added to per-frame timestep embedding. 10-step DDIM v-pred + Diffusion Forcing. Pedagogical / CI smoke test for action-conditioning correctness. Research doc: [`docs/Research/OASIS_ARCHITECTURE.md`](../Research/OASIS_ARCHITECTURE.md); build plan: [`docs/Checklists/PHASE_10_INTERACTIVE.md § 6`](PHASE_10_INTERACTIVE.md). |
| **Hunyuan-GameCraft 1.0** (Tencent), Phase 10 | 🔧 | built (2026-06-15) | structural tests pass | **Built end-to-end 2026-06-15** (structural; numerics validation-pending). 12.5B HunyuanVideo MM-DiT (19 double + 38 single blocks) + CameraNet (Plücker 6-channel ray maps) + 33-channel composite history input `[noisy(16) + ref_history(16) + mask(1)]`. PCM + CFG distilled 8-step variant. Llava-Llama-3-8B + CLIP-ViT-L text encoders. HunyuanVideo 3D causal VAE. Brought up a reusable `.pt` pickle loader + N-axis rope (image and video share blocks). Weights are PyTorch `.pt` pickle (90 GB total) and need a one-off conversion. **No license gate:** the engine is MIT and weights are user-supplied, so HartsyInference does not bundle, auto-download, or gate the Tencent weights (users accept Tencent's terms when they obtain them). Research doc: [`docs/Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md`](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md); build plan: [`docs/Checklists/PHASE_10_INTERACTIVE.md § 7`](PHASE_10_INTERACTIVE.md). |

## What to do next

To get a 🔧 model to ✅:

### AuraFlow v0.3
1. Download `aura_flow_0.3.safetensors` (16.5 GB) from
   <https://huggingface.co/fal/AuraFlow-v0.3/blob/main/aura_flow_0.3.safetensors> →
   `Models/Stable-Diffusion/AuraFlow/`.
2. Download `text_encoder/` Pile-T5-XL shards from
   <https://huggingface.co/fal/AuraFlow-v0.3/tree/main/text_encoder> →
   `Models/text_encoders/pile-t5-xl/`.
3. Download SDXL VAE from <https://huggingface.co/stabilityai/sdxl-vae> →
   `Models/VAE/sdxl_vae.safetensors`.
4. Run: `dotnet test --filter AuraFlow_V03_Gpu_512_NoCfg`. Expect bugs on first
   run — use the layer diff harness:
   ```
   tests/python-reference/.venv/bin/python tests/python-reference/dump_auraflow_full_forward.py
   AURAFLOW_DEBUG_DIR=Output/auraflow_csharp_dump dotnet test --filter AuraFlowDiffTests_Cpu
   tests/python-reference/.venv/bin/python tests/python-reference/diff_auraflow_layers.py
   ```
   Find the first layer with `avg_err > 1e-3`, fix the corresponding C# code, repeat.

### Chroma
1. Download `chroma_v1.safetensors` from <https://huggingface.co/lodestones/Chroma> →
   `Models/Stable-Diffusion/Chroma/`.
2. Reuse Flux's T5-XXL setup + Flux VAE (already downloaded).
3. `dotnet test --filter ChromaGenerationTests` → iterate against
   `dump_chroma_full_forward.py` + `diff_chroma_layers.py`.

### ERNIE-Image
1. Download `baidu/ERNIE-Image` diffusers folder layout (~31.6 GB total) →
   `Models/Stable-Diffusion/ErnieImage/v1/`. Or grab a Q4_K GGUF dump from
   `unsloth/ERNIE-Image-GGUF` (~5 GB transformer; fits 12 GB GPU).
2. The Ministral 3B text encoder is already wired in via `LlamaStyleEncoder`.
   No additional encoder code is needed.
3. Reuse the existing Flux2-style 128-channel VAE (`VaeConfig.Flux2`).
4. `dotnet test --filter ErnieImageGenerationTests` → iterate via
   `dump_ernie_image_full_forward.py` + `diff_ernie_image_layers.py`.

### Qwen-Image
1. Download `Qwen/Qwen-Image` diffusers single-file (FP8, ~20 GB transformer) +
   Qwen2.5-VL-7B text encoder (~15 GB BF16) + the 16-channel Qwen-Image VAE →
   point `TestPaths.QwenImage.{V1, TextEncoder, Vae}` at the files. Or wait for a
   Q4_K GGUF dump — the GGUF backend is ready.
2. Set the Qwen3 BPE tokenizer assets (already vendored via `Qwen3Tokenizer`).
3. `dotnet test --filter QwenImage_V1_Gpu_512_NoCfg_Smoke` (skips on <22 GB free VRAM).
4. Author `dump_qwenimage_full_forward.py` and walk the `QwenImageDebugDump` hooks
   to fix any first-run plumbing bugs (expect 1-3 iterations per past patterns).

## Pending unblockers

- **Per-block weight streaming** for Flux.2 Dev (32B) and stock-FP8 Qwen-Image would
  let large models run on 12 GB cards by uploading-running-freeing one block at a time.
  Requires a substantial refactor of the GPU weight cache. Cost: PCIe-bound runtime.
- **Checkpoint download + first-run iteration** is the remaining unblocker for every 🔧
  row (Anima, Lumina 2.0, OmniGen 2, and the rest are all scaffolded and code-green; they
  just need their weights and a layer-diff pass to reach ✅).
