# Image models — Python baselines for the 6 previously un-compared models (2026-07-18)

Fills the "❓ no Comfy baseline" rows in the image perf table. These six had a HartsyInference
warm number but **no Python reference**, so "faster than Python" was unproven. This run produces a
real Python number for each, on the **RTX 4090** (SM 8.9), matching the existing image-bench protocol.

## Method

- **ComfyUI-native models** (Krea2-Base, Lumina2, HunyuanImage-2.1, Flux.2-Dev, ChromaRadiance): routed
  through the **SwarmUI API → ComfyUI backend** — the exact same harness/toggle methodology as every other
  Comfy number in `image_comfy-vs-hartsy_2026-07-05.md`. ComfyUI 0.28.0 (2026-07-17) supports all five
  arches natively; the two GGUF models needed the **city96/ComfyUI-GGUF** node (installed this session).
  Harness: `benchmarks/swarm_image_bench/bench_t2i.py` + `models_comfy_missing{,2}.json`.
- **F-Lite** (no ComfyUI arch): standalone **diffusers** reference via the `f_lite` package
  (`benchmarks/python-baseline/bench_flite.py`).
- Protocol (both): **1 warmup + 3 timed gens** (F-Lite: 2 — sequential-offload is slow), **random seed each**,
  warm = median, same astronaut prompt, per-model step/cfg matching the Hartsy measurement. Peak 4090 VRAM
  sampled during each gen.

## Results — RTX 4090, warm median (Ratio = Hartsy ÷ Python; <1.0 = Hartsy faster)

| Model | Params | **Python (4090)** | Hartsy (4090) | **Ratio** | State |
|---|---|---:|---:|---:|---|
| **F-Lite** | 1024²/30st/cfg6 | **122.98 s** (diffusers, seq-offload) | 61.5 s | **0.50×** | ✅ Hartsy 2× faster \* |
| **Krea2-Base** | 1024²/28st/cfg4.5 | **42.13 s** (Comfy) | 30.3 s | **0.72×** | ✅ Hartsy faster |
| **Flux.2-Dev (Q4_K_S)** | 1024²/20st/cfg1 | **54.37 s** (Comfy) | 52.6 s | **0.97×** | ✅ Hartsy ~tied/faster |
| **HunyuanImage-2.1 (Q4_K_M)** | 2048²/20st/cfg3.5 | **48.08 s** (Comfy) | ~50.0 s | **1.04×** | ✅ matched (see note) |
| **Lumina-Image 2.0** | 1024²/25st/cfg4 | **10.05 s** (Comfy) | 17.7 s | **1.76×** | ⚠️ Hartsy slower — perf pass |
| **ChromaRadiance** | 1024²/20st/cfg3.5 | **24.68 s** (Comfy) | 54.4 s | **2.20×** | ⚠️ Hartsy slower — worst gap |

Peak VRAM (Comfy, 4090): Krea2-Base 19.8 GB, Lumina2 9.8 GB, HunyuanImage-2.1 22.4 GB, Flux.2-Dev 23.3 GB,
ChromaRadiance 23.2 GB. All output images visually verified coherent (ChromaRadiance shows its known dark/
vignette WIP character; F-Lite its signature circular vignette).

\* **F-Lite caveat — memory-bound, not compute-bound.** F-Lite is 10B (~20 GB bf16) + T5-XXL (~9 GB) ≈ 29 GB,
over the 4090's 24 GB. Fully-resident **and** accelerate model-offload both OOM (the 20 GB DiT + 1024²
activations exceed 24 GB during its forward), so the Python run uses **sequential (layer-level) CPU offload** —
the standard way a 24 GB diffusers user runs a 10B model, but it streams every layer each step (peak VRAM only
2.8 GB, 4090 ~81% util). Hartsy's 61.5 s also streams weights but does so far more efficiently. On a ≥40 GB GPU
diffusers would run F-Lite resident and would likely be much faster than 123 s — so read this as
"Hartsy beats diffusers-on-a-24GB-card," not a compute-parity claim.

## 3060 (RTX 3060, 12 GB) — "if it fits"

Only **Lumina-Image 2.0** fits 12 GB cleanly; the others exceed it at these settings.

| Model | Python-Comfy (3060) | Fits 12 GB? |
|---|---:|---|
| Lumina-Image 2.0 | **54.91 s** | ✅ (11.4 GB peak, tight) |
| Krea2-Base | — | ❌ fp8 DiT+T5 ~20 GB peak |
| Flux.2-Dev (Q4_K_S) | — | ❌ ~23 GB peak |
| ChromaRadiance | — | ❌ ~23 GB peak |
| HunyuanImage-2.1 | — | ❌ 22 GB peak @2048² |
| F-Lite | — | ❌ 10B model |

No Hartsy 3060 baselines exist for these models (all Hartsy image numbers are 4090), so the 3060 column is a
standalone Python reference, not a head-to-head.

## Takeaways for the perf-pass backlog

Perf-pass queue after re-benching on `alpha.61/62` (several handoff numbers were stale):
1. **Lumina-Image 2.0 ~1.76×** (17.7 s vs 10.05 s) — needs a re-bench + profile (already F16; lever unknown).
2. **ChromaRadiance — perf pass done, honest fp8-vs-fp8 = 1.02× (MATCHED).** The 54.4 s was stale (already
   31.2 s). Levers: **fp8 weight requant** (`QuantizeDitBlocksToFp8`, native fp8 tensor-core GEMM) 31.18→22.53;
   CopyInto→re-patchify 22.37; **F16 NeRF head** (F32 hypernetwork BatchedMatMul+GLU → F16 tensor cores; the un-damp
   `param_generator`+L2-normalize stay F32 for F16-overflow safety) 21.46 s — all coherent, peak VRAM 22→14 GB.
   **Honest bench** (both fp8; Comfy `--fp8_e4m3fn-unet --fast fp8_matrix_mult` = 21.07): **1.02×, within run-noise.**
   (Comfy fp8 *without* fp8_matrix_mult = 26.37 — slower than its own BF16 24.68; the first "beats Comfy" was the
   apples-to-oranges our-fp8-vs-Comfy-BF16.) 4090 warm median; GPU-compute-bound (~100 % util).

**HunyuanImage-2.1 is NOT slower** — the 74.1 s in the original handoff was stale (alpha `44.71`); a re-bench on
`alpha.61` shows **~50.0 s = 1.04× of Comfy, matched.** An F16-activation opt-in was tried (2026-07-18, `alpha.61`)
but is **perf-neutral** on this arch: the Q4_K GGUF weights dequant to F32 in the GEMM regardless of activation
dtype. Beating Comfy would require a Q4→F16 dequant-GEMM backend path (a bigger change than the activation flag).

Four models are **already at/ahead of Python**: Krea2-Base (0.72×), Flux.2-Dev (0.97×), HunyuanImage-2.1 (1.04×),
F-Lite (0.50×, memory-bound caveat).

## Repro / environment notes

- ComfyUI backend on the 4090 = SwarmUI backend **#1** (`GPU_ID:0`; ComfyUI default `cuda:0` = 4090). 3060 =
  backend **#0** (`GPU_ID:1`). SwarmUI bound `192.168.10.188:7801` (LAN IP, not localhost).
- GGUF support: `git clone city96/ComfyUI-GGUF` into `dlbackend/ComfyUI/custom_nodes` + `pip install -r requirements`
  (the `gguf` pkg) into ComfyUI's venv.
- F-Lite: `pip install git+https://github.com/fal-ai/f-lite` into ComfyUI's venv (torch 2.11 / diffusers 0.38).
  Pin the 4090 with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (torch's default `cuda:0` grabbed the
  3060 otherwise).
- Weights had been pruned from disk; re-downloaded: `city96/FLUX.2-dev-gguf` (Q4_K_S),
  `QuantStack/HunyuanImage-2.1-GGUF` (Q4_K_M), `Comfy-Org/Lumina_Image_2.0_Repackaged` (bf16),
  `lodestones/Chroma1-Radiance` (`latest_x0_full_20M_dataset_run_1024`), `Freepik/F-Lite` (diffusers).
