# Model Implementation Status

Snapshot of where each non-LLM image-generation model stands in SharpInference.
Last updated: 2026-05-07 session — Qwen-Image full scaffold + ERNIE-Image audit.

For architecture details, deviations, and per-model task lists, see
[PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md) and
[PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md).

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
| **Hunyuan Image 2.1** | 🚧 | stub | n/a | 17B MMDiT, 32×32 VAE downscale. Scaffolding only. |
| **Anima / OmniGen 2 / Lumina 2.0 / HiDream / Kandinsky 5 / F-Lite** | ❌ | n/a | n/a | Not started. |

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

- **Per-block weight streaming** for Flux.2 Dev (32B) and stock-FP8 Qwen-Image — would
  let large models run on 12 GB cards by uploading-running-freeing one block at a time.
  Requires a substantial refactor of the GPU weight cache. Cost: PCIe-bound runtime.
- **Anima / Lumina 2.0 / OmniGen 2** — unique architectures, no scaffolding yet. Out of
  scope for the current image-model push (defer to a follow-up phase).
