# Image Models — status

Concise status for every image-generation (diffusion / DiT T2I) model. Build detail, deviations, and
per-model task lists live in [PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md) and
[PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend is defined in [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

These produce clean visual output on real weights, confirmed end-to-end.

| Model | Status | Notes |
|---|---|---|
| **SD 1.5** | ✅ | Clean astronaut-on-horse output. |
| **SDXL** | ✅ | Clean 1024×1024 (5.5 s/step on RTX 3060 F16). |
| **SD3.5 Medium** | ✅ | Clean photorealistic output; 5 pipeline bugs fixed (PHASE_3_DEVIATIONS #31-35). |
| **Flux Dev / Schnell / Krea** | ✅ | Photoreal across all three. |
| **Z-Image Turbo / Base** | ✅ | Clean photoreal; 8 plumbing bugs fixed (PHASE_3_DEVIATIONS #25-30). |
| **Flux.2 Klein 4B** | ✅ | Clean astronaut. |
| **AuraFlow v0.3** | ✅ | Clean on-prompt horse+rider @1024 (`calcuis/aura` fp8). Two fixes: Pile-T5-XL attn scale 1.0 + correct `pile_t5xl_spiece.model` tokenizer. See PARITY §Bugs. |

## Numerically verified, full e2e pending (🔬)

| Model | Status | Notes |
|---|---|---|
| **Ideogram 4** (9.3B DiT) | 🔬 | ~1e-7 parity on the 3060 after the GPU-residency rewrite (`dit_f32.ptx`). A100 timing + visual e2e pending. |

## Built, validation-pending (🔧)

All implemented end-to-end (pipeline + converter + tests), green and structurally tested; each awaits a
checkpoint download + a Python layer-diff pass (and several are gated on VRAM). See PHASE_4 for the
per-model architecture notes and build plans.

| Model | Notes |
|---|---|
| **Flux.2 Dev (32B)** | Needs GGUF Q4 + per-block streaming to fit 12 GB. |
| **Qwen-Image** | Dual-stream DiT + 3-axis RoPE + Qwen2.5-VL encode; awaits ≥22 GB VRAM or Q4_K GGUF. |
| **Chroma / ChromaRadiance / ZetaChroma** | T5-only pipelines; await `chroma_v1.safetensors` + variants. |
| **ERNIE-Image** | Ministral-3B encoder via `LlamaStyleEncoder`; awaits ≥14 GB VRAM or Q4_K GGUF. |
| **Hunyuan Image 2.1** | 17B; needs ≥36 GB (A100/H100) or future Q4_K GGUF. |
| **Lumina-Image-2.0** | NextDiT family (Z-Image sibling); 2B fits 12 GB. |
| **HiDream i1 (Full / Dev)** | Quad-encoder; MoE FFN is single-expert fallback (full routing pending). |
| **Kandinsky 5.0 Lite** | Dual Qwen2.5-VL + CLIP-L embeds. |
| **Anima (Cosmos-Predict2)** | Image-only invariant; img2img/inpaint + LoRA added; DiT ControlNet/IP-Adapter deferred engine-wide. |
| **OmniGen 2** | Joint-stream RoPE; t2i only. |
| **F-Lite (Freepik / Fal.ai)** | T5-XXL layer-17 encode; ~29.4 GB checkpoint. |
| **Krea 2 (Base / Turbo)** | 12.9B single-stream MMDiT; sigmoid output-gate attn + 6-way modulation + text-fusion. SwarmUI wired. |
| **Boogu-Image 0.1 (Base / Turbo / Edit)** | 10B OmniGen2/Lumina-2 lineage + Qwen3-VL-8B vision tower (built). |
| **Microsoft Lens / Lens-Turbo / Lens-Base** | 3.8B dual-stream MMDiT + GPT-OSS MoE encoder + Flux.2 VAE. |
| **Lance (ByteDance) image** | Unified multimodal 3B-active (MoT + MaPE); shares backbone with Lance video. |

## How to promote a 🔧 to ✅

Download the checkpoint, point the test paths at it, run the model's generation test, then iterate with
its `*DebugDump` hooks against the Python `dump_*_full_forward.py` + `diff_*_layers.py` harness until the
first layer with `avg_err > 1e-3` is fixed. Step-by-step unblock recipes per model are in the
"What to do next" section of [PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md).
