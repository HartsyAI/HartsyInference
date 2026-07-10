# Image Models — status

Concise status for every image-generation (diffusion / DiT T2I) model. Build detail, deviations, and
per-model task lists live in [PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md) and
[PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend is defined in [MODEL_STATUS.md](MODEL_STATUS.md).

## Current performance vs ComfyUI (RTX 4090, warm median, 1024², 2026-07-09, `alpha.44.35-local` dev)

Authoritative living copies: [`docs/PERFORMANCE.md`](../PERFORMANCE.md) §5 and
[`benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md`](../../benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md).

| Model | Hartsy | ComfyUI | Status |
|---|---:|---:|---|
| Z-Image-Turbo (8 st) | **2.76 s** | 3.1 s | Faster than Comfy (free −0.2s from the SDXL round-2 fleet changes) |
| **SDXL (20 st)** | **2.93 s** | 3.7 s | **Faster than Comfy** (rounds 1+2 `44.24/44.26-local`: 33.9→3.69→2.93; Reshape round-trips + fused loop, then cuDNN conv + Lt epilogue + VAE-attn fix) |
| Krea2-Turbo (8 st) | **4.52 s** | 6.5 s | Faster than Comfy |
| Boogu-Turbo (4 st) | 3.26 s | 2.54 s | 1.28× (round 2 `44.22-local`: 48.9→5.05→3.26; D=120 cuDNN fused + packed loop) |
| Flux-Schnell (4 st) | 10.5 s | — | first bench |
| Flux.2 Klein 4B (4 st) | **3.45 s** | — | loader FIXED 07-09 (comfy-quant mixed TE: never pre-cast encoder dicts) + params corrected: DISTILLED variant = 4 steps/CFG 1 official; 10-step runs over-stepped it → crunchy artifacts (was benched 10 st since 44.15) |
| Ideogram4 (20 st) | 19.5 s | 17.0 s | 1.15× |
| **ERNIE (20 st)** | **20.0 s** | 23.9 s | **Faster than Comfy** (round 1: 49.6→20.0, seed-777 A/B clean) |
| Flux-Dev (20 st) | 31.0 s | 12.5 s | grind in progress (was 72.4) |
| Qwen-Image (20 st) | **40.9 s** | 54.8 s | Faster than Comfy |
| Boogu-Base (20 st, cfg 4) | 26.5 s | 17.8 s | 1.49× (round 2: ~6 min→43.2→26.5) |
| Chroma1-HD (20 st) | 28.5 s | 16.6 s | round 3 executed: F16 blocks + persistent CFG-pair CUDA graph + context trim (was 550 → 61.1 → 28.3); batched CFG queued |
| AuraFlow (20 st) | **13.93 s** | 14.0 s | Tied with Comfy (round 1 `44.28-local`: 31.4→13.93; packed token-space loop + recipe; NOTE proj_out emits (py,px,c) tokens ≠ patchify's (c,py,px) — velocity must be permuted for token-space Euler) |

## Verified end-to-end (✅)

These produce clean visual output on real weights, confirmed end-to-end.

| Model | Status | Notes |
|---|---|---|
| **SD 1.5** | ✅ | Clean astronaut-on-horse output. |
| **SDXL** | ✅ | Clean 1024×1024. Perf round 1 2026-07-09 (`alpha.44.24-local`): warm 1024²/20 steps **3.69s vs ComfyUI 3.7s (TIED; was 33.9s = 9.2×)** — THE find was `Tensor.Reshape`/`.To()` on GPU activations forcing D2H sync + H2D re-upload (~457 multi-MB misses/step across the attention stack + skip clones); fixed by passing un-reshaped tensors to dim-explicit ops, allocating outputs in final shape, and device skip clones. Plus batched-CFG single forward, drain-free `CfgEulerStep` loop (Euler/epsilon only), cached ADM embedding, dual-CLIP prompt cache, KEEP_MODELS. Seed A/B corr 0.9990. SD1.5/Refiner/Inpaint inherit the block fixes. Round 2 (`44.26-local`): **2.93s (1.26× faster than Comfy)** — cuDNN conv forward (new `CudnnConv`, default-on, fleet-wide), cuBLASLt bias-epilogue promoted to standard profile, VaeAttention Reshape purge (VAE 377→235ms), cross-attn K/V cache. GPU-bound at 100% util ≈63% of F16 peak; **sub-2s needs fp8 UNet GEMMs (output-changing — a quality decision) — see the benchmark log round-3 menu**. Remaining F16 levers: GroupNormSilu grid (~−0.15s), fused QKV (~−0.1s), cold TE 4.4s (ClipTextEncoder Reshape audit). |
| **SD3.5 Medium** | ✅ | Clean photorealistic output; 5 pipeline bugs fixed (PHASE_3_DEVIATIONS #31-35). |
| **Flux Dev / Schnell / Krea** | ✅ | Photoreal across all three. |
| **Z-Image Turbo / Base** | ✅ | Clean photoreal; 8 plumbing bugs fixed (PHASE_3_DEVIATIONS #25-30). |
| **Flux.2 Klein 4B** | ✅ | Clean astronaut. |
| **AuraFlow v0.3** | ✅ | Clean on-prompt horse+rider @1024 (`calcuis/aura` fp8). Two fixes: Pile-T5-XL attn scale 1.0 + correct `pile_t5xl_spiece.model` tokenizer. See PARITY §Bugs. |
| **Qwen-Image** (20B MMDiT) | ✅ | Clean photoreal astronaut-on-horse @1024 (Q4_K GGUF + Qwen2.5-VL fp8 TE). 4 bugs fixed (final-layer scale/shift, conditioning template+drop, GGUF shape relabel, weight-cast OOM) + GPU-residency perf rewrite. See PARITY §Bugs. Perf grind 2026-07-08 (`alpha.44.8-local`): warm 1024²/20 steps **40.9s vs ComfyUI 54.8s (1.34× FASTER; was ~355s = 6.5×)** — device joint RoPE (was per-block host loop), cuDNN flash SDPA, fused device CFG+Euler drain-free loop, TE prompt cache, KEEP_MODELS. Next lever (documented in the benchmark doc): Q4_K→fp8 requant at load (est. →~30s). |
| **Anima** (Cosmos-Predict2 2B) | ✅ | Clean on-prompt anime @512 on the 3060 (Qwen3-0.6B embeds). |
| **Lumina-Image 2.0** (2B NextDiT) | ✅ | Clean on-prompt mountain-lake @512 (53s). Needs the DIFFUSERS-format weights (`Alpha-VLLM/Lumina-Image-2.0` transformer+vae), not the original AlphaVLLM single-file. **Swarm-wired 07-10** (`Lumina2Loader`, live Gemma-2-2B `hidden_states[-2]` encode, coherent @1024/25st/cfg4 — but 650s: F32 + host-glue, residency/perf pass still pending). Core detection trap: the diffusers transformer matches `isOmniGen` → fix via `EditModelMetadata type=lumina-2`. |
| **Chroma** (8.9B fp8) | ✅ | Clean on-prompt astronaut-on-horse @512 (painterly/ink style). Transformer numerically verified vs diffusers (corr ≥0.999 all components). The earlier noise was a bad experimental checkpoint (symlink → `do_not_use/…exp`); re-pointed. |
| **Krea 2 Turbo** (12.9B MMDiT, fp8) | ✅ | Sharp photoreal astronaut-on-horse @1024, 8-step (std 66.5, grid 0.042). Impl fully cross-checked vs ComfyUI `krea2/model.py` (RoPE/modulation/sigmoid-gate attn/text-fusion/scheduler all match). Fix: `Krea2CheckpointConverter` renamed fp8 `.weight` keys but not their `.weight_scale` companions → scales dropped → weights ~250-900× too large → noise; added a scale-suffix pre-pass. Base/CFG path shares the same converter+transformer (untested; CFG anchoring `Krea2Pipeline.cs:100` validation-pending). |
| **Kandinsky 5.0 Lite** (BF16) | ✅ | Sharp on-prompt snow-leopard-on-peak @512 (std 90, grid 0.084), 64s on 4090. 3 fixes: OOM (12GB BF16 cast to F32 → cast to **F16** instead, fits 24GB), then BLANK (F16 fixed weights-not-applied), then NOISE = **`CudaBackend.LayerNorm` F32-input path didn't cast F16/BF16 affine weights** → text_proj/pooled_proj collapsed to ~0 → conditioning dead (same dtype-mismatch class as GroupNormSilu; GENERAL fix). Block was already GPU-resident. |
| **Boogu-Image 0.1 Base** (10B, fp8) | ✅ | Sharp photoreal astronaut-on-horse @1024-cfg (std 97.8, grid 0.038), **~6 min** (under the 10-min bar). Qwen3-VL-8B final-hidden-state conditioning (Boogu T2I system, no drop) + Flux VAE. Fixes: VAE bare-ldm key remap (`ConvertVaeKey`) + **GPU-residency rewrite of the 8 double-stream blocks** (CPU glue `LuminaRmsNormZero`/`AffineScaleShift` → GPU ops; GPU util 7%→72%; single blocks were already GPU-resident). |
| **Boogu-Image 0.1 Turbo** (10B, fp8) | ✅ | Sharp photoreal astronaut-on-horse @1024, 4-step tg=1.0. Shares Base's config/TE/VAE/converter. Perf grind 2026-07-09 (`alpha.44.22-local`): warm **48.9s→3.26s vs ComfyUI 2.54s (1.28×)**, Base **~6min→26.5s vs 17.8s (1.49×)** — round 1: device RoPE port (OmniGen2Rope.Apply was a host loop draining Q/K per block), cached rope tables + context-refined caption, drain-free CfgEulerStep, KEEP_MODELS + loader TE prompt cache w/ evict staging; round 2: cuDNN head-dim gate widened to admit D=120 (fused flash attention now engages), allowF16 on both block SDPAs, packed-latent loop (bit-equality unit-tested). Swarm routing unblocked via `EditModelMetadata type=boogu` (stale cached metadata — core already had the detector). |
| **ERNIE-Image** (8B fp8) | ✅ | Sharp full-contrast astronaut-on-horse @512-cfg (std 60.9, grid 0.069). 4 bugs fixed: flat-black (BF16-BN-cast NaN), SDPA mask-drop (general fix), VAE banding (non-tiled decode), and the WASHOUT = `CudaBackend.GroupNormSilu` F32 path didn't cast BF16 affine weights → wrong VAE GroupNorm → 4-5× low contrast. Transformer was byte-perfect on a std-ratio diff. See PARITY §ERNIE. Perf grind 2026-07-08 (`alpha.44.19-local`): warm 1024²/20 steps **49.6→~20.3s vs ComfyUI 24.0s** — step-invariant 77 MB attention-mask/RoPE/text-proj caches, masked cuDNN flash SDPA (`allowF16`), drain-free CfgEulerStep loop, TE prompt cache + KEEP_MODELS, device unpatchify/rgb. Seed-777 A/B clean. |
| **HiDream i1 Dev** (17B fp8, quad-encoder + MoE) | ✅ | Sharp photoreal astronaut-on-horse @1024, verified full 25-step. Residency round 1 (07-10): ~29s/step → **~1.4s/step (≈20×)** — GPU MoE gate+combine kernels (`dit_moe_topk_gate_f32`/`dit_row_gated_accum_f32`), device attention glue (flat-QK-norm caveat: norm precedes head-split), GPU rope + sig-cached Precompute, KEEP_MODELS + quad-encoder prompt cache. **1024²-CFG OOMs on step 1** (F32 activations + 17 GB resident fp8) — F16-activation port is the next lever; no bench row until it lands. 2 bugs found via numerical diff vs diffusers `transformer_hidream_image.py`: (1) **caption_projection loaded only 2 of 49** (`CaptionChannels.Length`) → every Llama layer through `caption_projection[0]` + T5 through `[1]` = garbage conditioning (brown cloud); fix = load all 49, per-block `caption_projection[i]`, T5→`[-1]` (`t5_proj` relL2 17.86 pinned it). (2) **FFN inner-dim** computed `4·hidden=10240` vs weights 6912/3584 → SwiGLU buffer overflow; fix = derive from `w1.Shape[0]`. GPU-residency block rewrite (52→29s/step). 1024-CFG path functional (smoke-tested). Caveat: intermittent fp8-load flake under memory pressure — retry. |
| **OmniGen 2** (fp16, Qwen2.5-VL-conditioned MMDiT) | ✅ | Coherent astronaut-on-horse @512-nocfg AND @1024-CFG. All 3 bugs (wrong subject, blocky bottom-third, 1024-CFG illegal-address) were ONE root cause: `ComputeFfnInnerDim` used Llama `8/3·dim`=6912 but the checkpoint FFN weight is `4·dim`→10240 → SwiGLU buffer overflow (out-of-bounds GEMM writes corrupted tail image tokens + adjacent memory). Fix = `4·HiddenSize`. Found via numerical diff vs cloned `VectorSpaceLab/OmniGen2` (attn matched 0.009, MLP-out bottom rows 1.0). Precomputed embeds were fine.**Swarm-wired 07-10** (`OmniGen2Loader`, live Qwen2.5-VL-3B encode with ComfyUI-template parity, coherent @1024/20st/cfg4, 98.8s pre-optimization). |

## Numerically verified, full e2e pending (🔬)

| Model | Status | Notes |
|---|---|---|
| **Ideogram 4** (9.3B DiT) | ✅ | e2e coherent + prompt-faithful (structured-JSON prompt), visually verified. Perf grind 2026-07-08 (`alpha.44.7-local`): warm 1024²/20 steps **19.5s vs ComfyUI 17.0s (1.15×; was 2.5×)** — step-invariant conditioning caches, cuDNN flash attn @ head_dim 256, `HARTSY_DIT_F16` (o/w3 1/64 sandwich damp), drain-free loop, banded-im2col full-res Flux.2 VAE decode, KEEP_MODELS + prompt cache. Remaining menu in `benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md`. |

## Built, validation-pending (🔧)

All implemented end-to-end (pipeline + converter + tests), green and structurally tested; each awaits a
checkpoint download + a Python layer-diff pass (and several are gated on VRAM). See PHASE_4 for the
per-model architecture notes and build plans.

| Model | Notes |
|---|---|
| **Flux.2 Dev (32B)** | Needs GGUF Q4 + per-block streaming to fit 12 GB. |
| **ChromaRadiance / ZetaChroma** | Chroma variants (base Chroma now ✅). T5-only pipelines; await variant checkpoints. |
| **Hunyuan Image 2.1** | 17B; **BLOCKED on 24GB** — 35GB bf16 + fp8/GGUF repacks use incompatible original-Tencent keys. Needs an fp8-diffusers-naming quant or a GGUF/K-quant reader. |
| **F-Lite (Freepik / Fal.ai)** | T5-XXL layer-17 encode; ~29.4 GB checkpoint. |

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
