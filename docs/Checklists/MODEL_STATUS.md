# Model Implementation Status

Snapshot of where each non-LLM image-generation model stands in SharpInference.
Last updated: 2026-05-02 session — AuraFlow / Chroma / ERNIE-Image push.

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
| **Qwen-Image** | 🚧 | stub | n/a | 20B MMDiT. Needs custom block + complex RoPE + Wan-style 3D causal-conv VAE + Qwen2.5-VL text encoder + GGUF reader. ~2500-3000 lines. |
| **Chroma** | 🔧 | done | scaffolded | Full impl green (~2350 lines): `ChromaApproximator` (5-layer SiLU+RMSNorm MLP) + `ChromaCombinedTimestepEmbeddings` + pruned-AdaLN double + single block variants + `ChromaTransformer` + `ChromaCheckpointConverter` (BFL→diffusers) + T5-only `ChromaPipeline` (with "first padding token unmasked" mask + true-CFG dual-pass) + `ChromaDebugDump` + `ChromaGenerationTests`. All `Forward` methods implemented (zero `NotImplementedException`). Awaits `chroma_v1.safetensors` download. |
| **ERNIE-Image** | 🔧 | done | scaffolded | Full impl green (~2090 lines): non-interleaved 3-axis `ErnieImageRope` (axes 32/48/48, theta=256, image-position offset by `text_lens`) + `ErnieImagePatchEmbed` (1×1 Conv2d) + `ErnieImageBlock` (single-stream w/ shared modulation) + `ErnieImageTransformer` + `ErnieImageCheckpointConverter` (diffusers folder layout) + `ErnieImagePipeline` (Flux2-style BN-unnormalize + standard CFG) + `IErnieTextEncoder` interface + `ErnieImagePlaceholderTextEncoder` (throws `NotSupportedException` with "plug in real encoder" message) + `ErnieImageLlamaTextEncoder` (Llama-shaped fallback wrapping `LlamaStyleEncoder` with `hidden_states[-2]`) + `ErnieImageDebugDump` + `ErnieImageGenerationTests`. Awaits `baidu/ERNIE-Image` download + verification of the actual text encoder architecture (placeholder/Llama fallback may need replacing). |
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
   `Models/Stable-Diffusion/ErnieImage/v1/`.
2. **Critical**: `text_encoder/config.json` will reveal the custom Baidu encoder
   architecture. The C# pipeline currently has a placeholder `IErnieTextEncoder`
   interface — implement the concrete class once the architecture is confirmed.
3. Reuse the existing Flux2-style 128-channel VAE (`VaeConfig.Flux2`).
4. `dotnet test --filter ErnieImageGenerationTests` → iterate via
   `dump_ernie_image_full_forward.py` + `diff_ernie_image_layers.py`.

## Pending unblockers

- **GGUF K-quant reader** (Q4_K_M, Q5_K_M, Q8_0). Would unlock 12-GB-fitting variants
  of Qwen-Image, AuraFlow, Chroma, ERNIE-Image, Flux.2 Dev, and SD3.5 Large. Estimated
  1-2 days of focused work — implement K-quant block dequant kernels in
  `SharpInference.ModelHandler` mirroring `ggml-quants.c`.
- **Per-block weight streaming** for Flux.2 Dev (32B) — requires a substantial
  refactor of the GPU weight cache to support eviction-on-demand per layer.
- **Anima / Lumina 2.0 / OmniGen 2** — unique architectures, no scaffolding yet.
