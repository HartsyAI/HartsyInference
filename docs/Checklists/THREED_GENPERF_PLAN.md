# 3D (image → mesh) gen-perf — audit + optimization plan (2026-07-14)

Applying the image/video/world gen-perf playbook (`WORLD_GENPERF_PLAN.md`, `VIDEO_GENPERF_PLAN.md`,
memories `radiance-perf-pass`, `cuda-graph-step-capture-recipe`, `vae-host-loops-hidden-20s`,
`image-genperf-host-glue-wins`, `bf16-decode-gemv-kernel`) to the **3D models** (`HartsyInference.ThreeD`).
Both models are already ✅ correctness-verified end-to-end (`MODEL_STATUS_3D.md`); this is a dedicated
**perf** pass. Metric = **seconds per mesh** and the explicit goal: **correct AND faster than the Python
reference**. Method rule (non-negotiable): **benchmark + document first**, then optimize; parity/coherence
preserved at every step (a fast wrong mesh is worthless).

## Baselines (2026-07-14, RTX 4090, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`)

Python reference = `/tmp/benchvenv` (torch 2.13.0+cu130). Engine = Release net10.0, `CudaBackend` ordinal 0.
Input = `examples/chair.png` → foreground-composited-on-gray-0.5 512² (`/tmp/chair_prep.png`). Both produce
the coherent chair; tri-counts differ by iso/mc settings, not correctness.

| Model | Python ref (4090) | Engine (4090) | Gap | Python dtype / notes |
|---|---|---|---|---|
| **TripoSR** | **0.58 s** neural (scene_codes 114.6 ms + density-grid 256³ 469.2 ms) | **26.2 s** | **~45× slower** | F32 (tsr default). Excludes host MC on both sides. |
| **Hunyuan3D-2** | **5.76 s** (30 steps + res-128 VAE decode → 66k tri) | **71.3 s** (67 s = 30 steps ×~2 s + ~4 s VAE/MC) | **~12× slower** | **fp16** (hy3dgen default). Engine is F32. |

Reproduce:
- Python TripoSR: `RES=256 python /tmp/bench_triposr.py` (stubs torchmcubes/rembg; times forward + query_triplane).
- Python Hunyuan3D: `STEPS=30 RES=128 python /tmp/bench_hunyuan3d.py` (real `hy3dgen` shape pipeline, fp16).
- Engine TripoSR: `TRIPOSR_WEIGHTS=.../triposr.safetensors TRIPOSR_IMAGE=/tmp/chair_prep.png dotnet test … TripoSrGenerationTests`.
- Engine Hunyuan3D: `HY3D_MODEL_DIR=/tmp/hunyuan3d HY3D_IMAGE=/tmp/chair_prep.png HY3D_STEPS=30 HY3D_GRID=128 dotnet test … Hunyuan3DGenerationTests`.

Weights: TripoSR `/tmp/triposr/model.ckpt` → converted `tests/python-reference/triposr_reference/triposr.safetensors`
(549 tensors, F32, via `dump_triposr_reference.py`). Hunyuan3D `/tmp/hunyuan3d/{hunyuan3d-dit-v2-0,hunyuan3d-vae-v2-0}/model.fp16.safetensors`.

## Where the time goes (audit + step-timing)

### TripoSR — the whole 26 s is the **host density decode**
Feed-forward LRM: DINO ViT-B/16 → Transformer1D (16 blocks) → triplane → NeRF MLP → 256³ density grid → MC.
- DINO + backbone (GPU-resident-ish): ~1 s.
- **256³ = 16.7M-point density field via `TriplaneNerfDecoder.SampleTriplane` — a HOST per-point loop**
  (`TriplaneNerfDecoder.cs:115` grid-sample, `:68-69` density, `:81-84` colors; chunk 32768). This is the
  ~25 s killer. Python does the identical query on-GPU in 469 ms.
- Backbone also has host head-reshape loops (`Hunyuan3DAttention.ToHeads/FromHeads` `:36/:47`, 32×/gen) and a
  GEGLU host loop (`TripoSrTransformer.cs:199`, 16×).

### Hunyuan3D-2 — the wall is the **DiT denoise loop** (~2 s/step × 30 ≈ 60 s)
- DINOv2-giant cond: ~6 s (one-time; first-step spike).
- DiT: 16 double + 32 single Flux blocks, N=3072 tokens, **2 separate CFG forwards per step**, F32. ~2 s/step.
- VAE decode (res 128) + MC: ~4 s. Grid decode = ~512 sequential geo-decoder cross-attn calls at chunk 4096.

## The levers (order it wins), per the playbook

Confirmed by read-only audit (2026-07-14). **Every SDPA in both 3D models passes `allowF16=false`** → they
run the materialized TF32 path, never cuDNN fused flash. This is the Oasis 900→2.9 ms lever, untapped.

1. **Phase probes FIRST** — add `[triposr-phase]` / `[hy3d-phase]` wall logs (DINO / backbone / DiT-steps /
   VAE-decode / MC). None exist today (only coarse test-side Stopwatch).
2. **GPU-resident density decode (TripoSR)** — port `SampleTriplane` + NeRF density/color to device
   (grid-sample kernel or existing ops). Biggest single win (25 s → target <1 s).
3. **cuDNN fused SDPA (`allowF16: true`)** — 5 ThreeD sites (Hunyuan3D Flux double `Hunyuan3DFluxBlocks.cs:77`
   / single `:179`, VAE self `Hunyuan3DVaeBlocks.cs:54`, geo-decoder cross `:144`; TripoSR `Hunyuan3DAttention.Attend`)
   + 2 conditioner sites. All QK-norm/LayerNorm-bounded → safe. Also halves the 1.6 GB geo-decoder score buffer.
4. **Kill per-step host CFG sync (Hunyuan3D)** — `Hunyuan3DShapePipeline.cs:153` reads `vCond/vUncond.DataPointer`
   every step (forces D→H sync + host Euler). Make CFG combine + Euler GPU-resident; keep latents on device.
5. **Batched CFG** — the pipeline runs 2 separate DiT forwards/step; batch cond+uncond into one batch-2 forward
   (halve launch/glue overhead) where it doesn't break residency.
6. **F16 activations** — none wired (everything `DType.F32`). Python runs Hunyuan3D in fp16; matching it
   halves DiT activation bandwidth. Wire a `DitDtype.Act`-style flag through the (already GPU-resident) Flux blocks.
7. **CUDA-graph** the fixed-shape 30-step DiT forward (absent today) once the step is drain-free (lever 4).
8. **GPU-resident ShapeVAE grid decode** — remove per-chunk host Fourier/occupancy copies
   (`Hunyuan3DVaeBlocks.cs:160`, `Hunyuan3DShapeVae.cs:103`); larger chunks once F16 shrinks the score buffer.

## Host-glue surface (`DataPointer` density audit, 2026-07-14)

| File | Hot-path DataPointer reads | Note |
|---|---|---|
| `TriplaneNerfDecoder.cs` | `:115` (per-point grid-sample), `:68/:81` (density/color) | **the 25 s TripoSR wall** |
| `Hunyuan3DShapePipeline.cs` | `:153-154` per-step CFG combine + Euler | 30× D→H sync |
| `Hunyuan3DAttention.cs` | `:36/:47` ToHeads/FromHeads | TripoSR-only head reshape |
| `TripoSrTransformer.cs` | `:59/:91/:98/:114/:199` | single-forward host loops + GEGLU |
| `Hunyuan3DVaeBlocks.cs` | `:160` occ→host, `:199` Fourier host write | per grid chunk (~512×) |
| `Hunyuan3DShapeVae.cs` | `:103` per-chunk occ copy to host `values[]` | grid assembly |

Hunyuan3D DiT double/single blocks (`Hunyuan3DFluxBlocks.cs`) + VAE resblock/geo-decoder attention math ARE
already GPU-resident (all glue via `IBackend`, no mid-forward host reads) — the residency claim holds there.

## Per-model plan

### TripoSR — RUN FIRST (verified, ungated, small; the density-decode proof case)
1. `[triposr-phase]` probes (DINO / backbone / density-grid / MC) → confirm the ~25 s is the host decode.
2. GPU-port the triplane density/color decode (kill `SampleTriplane` host loop) — bit-exact vs the current
   host path, parity-guarded. Target: beat Python's 0.58 s neural.
3. `allowF16` on the two TripoSR attns + host head-reshape → device. F16 backbone if it holds parity.
4. Re-run gen; coherence + tri-count unchanged; record ratio vs Python.

### Hunyuan3D-2 — the flagship (30-step DiT is the wall)
1. `[hy3d-phase]` probes (cond / per-step DiT / VAE-decode / MC).
2. `allowF16` on all 4 Hunyuan3D SDPA sites + 2 DINOv2 sites (Oasis-class win on the joint attention).
3. Kill per-step host CFG sync; batched CFG; keep latents device-resident across steps.
4. F16 activations through the Flux blocks (match Python fp16), per-stage relL2 gate.
5. CUDA-graph the 30-step forward.
6. GPU-resident VAE grid decode.
7. Re-run at steps=30/grid=128; coherent chair; record ratio vs Python 5.76 s.

## Results log

> _(pending — optimization rounds start after this baseline doc lands)_

## Method rules (from the war)
- Phase probes before any op-level work; wall ≠ op-profile means an un-instrumented host phase.
- **Correctness before perf** — parity/coherence gate every round; bit-identical residency ports first,
  numeric-risk (F16/graph) behind flags with per-stage relL2.
- GPU-shared box: hard-gate every run on `nvidia-smi` idle wait-loop; prefer 3060 for fits, take turns.
- The bar is **faster than Python** with a coherent mesh — not just "faster than before".
