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
| **Qwen-Image** (20B MMDiT) | ✅ | Clean photoreal astronaut-on-horse @1024 (Q4_K GGUF + Qwen2.5-VL fp8 TE). 4 bugs fixed (final-layer scale/shift, conditioning template+drop, GGUF shape relabel, weight-cast OOM) + GPU-residency perf rewrite. See PARITY §Bugs. |
| **Anima** (Cosmos-Predict2 2B) | ✅ | Clean on-prompt anime @512 on the 3060 (Qwen3-0.6B embeds). |
| **Lumina-Image 2.0** (2B NextDiT) | ✅ | Clean on-prompt mountain-lake @512 (53s). Needs the DIFFUSERS-format weights (`Alpha-VLLM/Lumina-Image-2.0` transformer+vae), not the original AlphaVLLM single-file. |
| **Chroma** (8.9B fp8) | ✅ | Clean on-prompt astronaut-on-horse @512 (painterly/ink style). Transformer numerically verified vs diffusers (corr ≥0.999 all components). The earlier noise was a bad experimental checkpoint (symlink → `do_not_use/…exp`); re-pointed. |
| **Krea 2 Turbo** (12.9B MMDiT, fp8) | ✅ | Sharp photoreal astronaut-on-horse @1024, 8-step (std 66.5, grid 0.042). Impl fully cross-checked vs ComfyUI `krea2/model.py` (RoPE/modulation/sigmoid-gate attn/text-fusion/scheduler all match). Fix: `Krea2CheckpointConverter` renamed fp8 `.weight` keys but not their `.weight_scale` companions → scales dropped → weights ~250-900× too large → noise; added a scale-suffix pre-pass. Base/CFG path shares the same converter+transformer (untested; CFG anchoring `Krea2Pipeline.cs:100` validation-pending). |
| **Kandinsky 5.0 Lite** (BF16) | ✅ | Sharp on-prompt snow-leopard-on-peak @512 (std 90, grid 0.084), 64s on 4090. 3 fixes: OOM (12GB BF16 cast to F32 → cast to **F16** instead, fits 24GB), then BLANK (F16 fixed weights-not-applied), then NOISE = **`CudaBackend.LayerNorm` F32-input path didn't cast F16/BF16 affine weights** → text_proj/pooled_proj collapsed to ~0 → conditioning dead (same dtype-mismatch class as GroupNormSilu; GENERAL fix). Block was already GPU-resident. |
| **Boogu-Image 0.1 Base** (10B, fp8) | ✅ | Sharp photoreal astronaut-on-horse @1024-cfg (std 97.8, grid 0.038), **~6 min** (under the 10-min bar). Qwen3-VL-8B final-hidden-state conditioning (Boogu T2I system, no drop) + Flux VAE. Fixes: VAE bare-ldm key remap (`ConvertVaeKey`) + **GPU-residency rewrite of the 8 double-stream blocks** (CPU glue `LuminaRmsNormZero`/`AffineScaleShift` → GPU ops; GPU util 7%→72%; single blocks were already GPU-resident). |
| **ERNIE-Image** (8B fp8) | ✅ | Sharp full-contrast astronaut-on-horse @512-cfg (std 60.9, grid 0.069). 4 bugs fixed: flat-black (BF16-BN-cast NaN), SDPA mask-drop (general fix), VAE banding (non-tiled decode), and the WASHOUT = `CudaBackend.GroupNormSilu` F32 path didn't cast BF16 affine weights → wrong VAE GroupNorm → 4-5× low contrast. Transformer was byte-perfect on a std-ratio diff. See PARITY §ERNIE. |

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
| **ChromaRadiance / ZetaChroma** | Chroma variants (base Chroma now ✅). T5-only pipelines; await variant checkpoints. |
| **Hunyuan Image 2.1** | 17B; **BLOCKED on 24GB** — 35GB bf16 + fp8/GGUF repacks use incompatible original-Tencent keys. Needs an fp8-diffusers-naming quant or a GGUF/K-quant reader. |
| **HiDream i1 (Full / Dev)** | Quad-encoder; MoE FFN is single-expert fallback (full routing pending). |
| **Kandinsky 5.0 Lite** | Dual Qwen2.5-VL + CLIP-L embeds. |
| **OmniGen 2** | Joint-stream RoPE; t2i only. |
| **F-Lite (Freepik / Fal.ai)** | T5-XXL layer-17 encode; ~29.4 GB checkpoint. |
| **Boogu-Image 0.1 Turbo** | Base is ✅ (above); Turbo shares the identical code path (same config/TE/VAE, 4-step, tg=1.0) — just needs a verification run. **Edit** variant needs the Qwen3-VL vision tower (ref-image conditioning) — not yet wired. |

## Edit / image-conditioned variants (🔧 — to download + e2e test)

Instruction/edit models reuse the base transformer with image-conditioning slots. These need their own e2e tests + weights:

| Model | Notes |
|---|---|
| **Qwen-Image-Edit** (20B) | Image-conditioned editing branch of Qwen-Image (diffusers `forward_edit`). Was omitted from the t2i scaffold (PHASE_4 #601) — needs the edit pipeline path + e2e test. A **Q5_K_M GGUF is already local** (SwarmUI `qwen-image-edit-2511-Q5_K_M.gguf`); reuse the now-✅ Qwen-Image transformer + the conditioning/final-layer fixes. |
| **Boogu-Image 0.1 Edit** | Image-edit variant of Boogu-Image (same 10B backbone + reference-image conditioning). Needs the edit conditioning path + weights (fp8/GGUF) + e2e test. |
| **Flux.1 Kontext** | Already built (`FluxToolsConfig` Kontext, `flux1-dev-kontext_fp8_scaled.safetensors` local) — edit/instruction path exists; e2e visual verification pending. |
| **Microsoft Lens / Lens-Turbo / Lens-Base** | 3.8B dual-stream MMDiT + GPT-OSS MoE encoder + Flux.2 VAE. |
| **Lance (ByteDance) image** | Unified multimodal 3B-active (MoT + MaPE); shares backbone with Lance video. |

## How to promote a 🔧 to ✅

Download the checkpoint, point the test paths at it, run the model's generation test, then iterate with
its `*DebugDump` hooks against the Python `dump_*_full_forward.py` + `diff_*_layers.py` harness until the
first layer with `avg_err > 1e-3` is fixed. Step-by-step unblock recipes per model are in the
"What to do next" section of [PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md).
