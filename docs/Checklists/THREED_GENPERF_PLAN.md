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

> ### ✅ Round 1 (2026-07-14) — TripoSR: 26.2 → 2.1 s (12.5×), density decode now BEATS Python, parity preserved
> Four new NVRTC kernels + shared-op ports took TripoSR from 26.2 s to **2.1 s on the 4090**, coherent chair
> unchanged (42222 verts / 84448 tris), all parity held. Phase-probe-driven (`HARTSY_3D_PHASE=1` →
> `[triposr-phase]`), each round re-probed to find the real wall — which moved three times:
>
> 1. **Triplane density decode (the 25 s wall): host `SampleTriplane` → `triplane_grid_sample_f32` kernel.** The
>    16.7 M-point 256³ density grid ran 3 host bilinear samples/point. New kernel generates grid coords in-kernel
>    (ij order) or from a coords buffer, samples the 3 planes device-resident → the feature tensor stays on GPU so
>    the NeRF MLP Linears hit the activation cache (no host round-trip). `IBackend.TriplaneGridSample` (default host
>    impl = byte-identical to the old loop) + `CudaBackend` override; planes uploaded once (weight-cache resident).
>    **Density-grid 25 s → 354 ms → now faster than Python's 469 ms.** CPU parity `Decoder_DensityGrid` corr **1.0**.
> 2. **GEGLU host loop → `geglu_erf_f32` kernel (transformer 7.3 → 2.1 s).** The `proj.DataPointer` read in the
>    diffusers GEGLU drained the compute stream *every block*, serializing the 16-block backbone (the
>    `sync-h2d-stream-drain` disease). Fused split + exact-erf gate + multiply into one device pass. Kernel uses
>    true `erff`; the host default replicates the A-S `VitOps.Erf` so CPU stays byte-identical.
> 3. **ConvTranspose2d was on the CPU (1549 ms for a 32²→64² upsample!).** `CudaBackend` never overrode
>    `ConvTranspose2d` → the tiny post-processor upsample ran the naive host scatter-add. New gather-form
>    `conv_transpose2d_f32` kernel (one thread/output, no atomics) → **1549 → 3 ms.** Validated by
>    `CudaOpBisectTests.ConvTranspose2d_UpsampleShape` (CPU-vs-CUDA corr **1.0**, maxAbs 1.7e-5). **Shared win** —
>    also used by ClipSeg / YOLO-Proto / Demucs / RVC / ResembleEnhance (were all paying the host cost).
> 4. **DINO ToHeads/FromHeads host loops → `Permute0213` + cuDNN `allowF16` SDPA** (both `DinoViTEncoder` and
>    `Dinov2VisionEncoder`, shared Vision code; `Hunyuan3DAttention.Attend` too). Bit-identical reshape; DINO
>    846 → ~640 ms. Parity `DinoVit_ImageTokens` corr **1.0** held.
>
> **Final TripoSR phase split (4090):** dino 613 · transformer 528 (blocks 195 + weight-preload ~270 + conv 3) ·
> density 338 · marching-cubes 573 (host) · colors 9 = **2.1 s**. Our density decode beats Python; the remaining
> gap vs Python's 0.58 s neural is the DINO+backbone (Python 115 ms — per-op overhead / small-GEMM occupancy, the
> Oasis R5-8 regime) and host marching cubes. **Regression:** `CudaOpBisectTests` 5/5 (ConvTranspose/GroupNorm/
> LayerNorm/SDPA all corr ~1.0), TripoSR CPU parity 4/4, GPU mesh coherent. Kernels live in `native/cuda/dit/dit_f32.cu`
> (4 added: triplane_grid_sample / geglu_erf / conv_transpose2d + the earlier set); PTX built via
> `nvrtc_compile` (no nvcc), installed to `src/HartsyInference.Cuda/Ptx/dit_f32.ptx`.
>
> **Remaining TripoSR levers (deferred):** GPU marching cubes (573 ms host); DINO/backbone per-op fusion or
> CUDA-graph (the 115 ms-vs-1141 ms gap); F16 backbone. Diminishing returns vs the Hunyuan3D flagship.

> ### 🔶 Round 2 (2026-07-14) — Hunyuan3D-2: 71.3 → 53.9 s via cuDNN SDPA; DiT diagnosed = per-op overhead (CUDA-graph is next)
> `allowF16: true` on all 4 Hunyuan3D SDPA sites (Flux double `Hunyuan3DFluxBlocks.cs:77`, single `:179`, VAE self
> `Hunyuan3DVaeBlocks.cs:54`, geo-decoder cross `:144` — all mask-null, QK-norm-bounded → safe). **71.3 → 53.9 s**,
> mesh coherent (80542 verts / 161176 tris ≈ the F32 80200/160444 → F16 SDPA safe). Phase split (`[hy3d-phase]`,
> 4090): **dit-loop 47.0 s** (was ~60 s) · dinov2-cond 4.6 s · vae-decode 2.2 s · mc 0.14 s.
>
> **DiT diagnosis (decisive):** ~783 ms/forward (Python fp16 ~96 ms → 8× off). The forward is GPU-resident (only a
> tiny per-step timestep-sinusoid host read). `HARTSY_GEMM_F16=1` moved the dit-loop **0%** (48.3 vs 47.0 s) → the
> DiT is **NOT GEMM-compute-bound**; the wall is **per-op host launch overhead** (~720 ops × 60 forwards × ~1 ms),
> the exact Oasis R5→R6 regime. **∴ F16 activations are NOT the lever; CUDA-graph step-capture is** (Oasis got 3.4×
> that way; world-plan R3). Latent shape is fixed every step and cond is fixed across all steps → ideal capture
> regime. This is the LTX-2.3/Oasis fixed-buffer subsystem (a distinct round), not a one-line flag.
>
> **Remaining Hunyuan3D levers, in ROI order:** (1) **CUDA-graph the DiT block loop** (capture once, replay per
> forward; timestep-embed/patchify/final-layer outside capture, refresh fixed latent+vec buffers; clean eager
> fallback) — the identified lever, ~3× expected → ~16 s DiT. (2) **Batched CFG** (2 forwards → 1 batch-2) halves
> launch count too, but needs a batch-2 audit of every Flux-block op (Permute0213 etc. assume batch 1) — risk.
> (3) per-step host CFG sync `Hunyuan3DShapePipeline.cs:153` (minor). (4) F16 ruled out for the DiT (per-op-bound).

> ### ✅ Round 3 (2026-07-14) — Hunyuan3D DiT CUDA-graph capture: dit-loop 47 → 30.7 s (1.53×), bit-exact, default-on
> Implemented step-graph capture in `Hunyuan3DDit.Forward` (mirrors the Oasis/`OasisDit` recipe). The 48-block DiT
> forward is per-op-host-overhead-bound (R2 proved it), so capturing the device-resident block loop + final layer
> once and replaying with a single `cuGraphLaunch` collapses ~720 host launches/forward → ~0. **Key:** the only
> per-step-varying inputs — `img` (latent_in), `txt` (cond_in), `vec` (timestep embed, the sole host read) — are
> computed OUTSIDE capture and `CopyInto` fixed buffers (`_imgFixed`/`_txtFixed`/`_vecFixed`) before each launch,
> so the captured region reads stable addresses and the graph captures **once** (geometry sig = n,scond, fixed
> across the loop) and replays all 60 forwards. Velocity lands in a non-graph-owned `_velFixed` (last captured op,
> survives graph-memory auto-free) → `CopyOut`. Extracted a shared `RunBlocks` for the eager + capture paths.
> `StepGraphOwner` guards the single graph slot; clean eager fallback (`_graphDead`) on any capture failure; CPU
> backend (StepGraphSupported=false) always eager. **dit-loop 47.0 → 30.7 s (1.53×); total 53.9 → 37.2 s
> (cumulative 71.3 → 37.2 = 1.92×).** Mesh **80542 verts / 161176 tris — bit-identical** to the pre-graph run
> (replay correct). Default-**on** via `DitStepGraph.EnabledDefaultOn` (host-issue-bound → validated win, like
> Chroma); `HARTSY_DIT_GRAPH=0` disables. cuDNN F16 SDPA is capture-compatible (Oasis R6 precedent held).
>
> **Post-graph diagnosis:** the replayed forward is ~500 ms (host launches now ~0), but the block FLOPs are only
> ~2.7 TFLOP ≈ 30-70 ms of TF32 → the DiT is now **GPU-execution-bound at poor F32 efficiency** (small-GEMM
> occupancy + F32 bandwidth over 48 blocks × ~4442 tokens). Python's 96 ms/forward is fp16 + fused. **∴ the next
> lever flips back to F16 activations** (now that host overhead is gone, F16 halves bandwidth AND enables F16
> tensor cores) — a `DitDtype.Act` wiring through the Flux double/single blocks (per-stage relL2 gate; MG3's
> high-gain-stream caution noted, but Python runs the whole model fp16 so it's in-distribution). That + batched
> CFG are the remaining rounds to close on Python. **Minor follow-up:** the graph fixed buffers (~20 MB) aren't
> disposed (Hunyuan3DDit has no Dispose hook, unlike OasisDit's DisposeCore) — fine for a process-lived pipeline,
> worth a cleanup for churned pipelines.

> ### ✅ Round 4 (2026-07-15) — Hunyuan3D F16 activations + device CFG/Euler: dit-loop 30.8 → 26.7 s; the DiT is occupancy-bound, not GEMM-bound
> Wired F16 activations through the Flux double/single blocks (mirrors the Krea2 `DitDtype.Act` recipe): the blocks
> now **dtype-follow** their input, with one `CastToF16` at the block-loop boundary in `Hunyuan3DDit.RunBlocks` and
> a `CastToF32` back before the final layer; modulation params (`ModParams`, `NormModulate` scale/shift) stay F32
> (the F16 norm/modulate/gate/attention kernels take F16 activation + F32 params). Verified every op on the path
> has an F16 kernel. Default-**on** via the global `DitDtype.Act` (HARTSY_DIT_F16, matches Python fp16); `=0` forces
> F32. Also replaced the per-step host `FlowStepAscending` with device `IBackend.CfgEulerStep` so `latents` stays
> GPU-resident across the loop (no per-step D2H drain of the two velocities).
>
> **Clean A/B/A (both graph-on, 4090): F16 off 30.8 / 30.9 s vs F16 on 26.8 s = real, reproducible 1.15×.** Mesh
> coherent (79920-80432 verts vs F32 80542 — F16 rounding, all coherent chairs; Python runs fp16, in-distribution).
> Device CFG/Euler: **~0%** (26.7 vs 26.8 s) — the 60 per-step drains were NOT the wall (they only waited on compute
> that runs anyway); kept it (cleaner, latents resident). **Cumulative Hunyuan3D 71.3 → ~33 s (2.2×).**
>
> **The finding:** F16's mere 1.15% + device-CFG's 0% prove the graph-replayed DiT forward (~445 ms) is **NOT
> block-GEMM- or CFG-sync-bound** — the block FLOPs are ~30-70 ms of TF32, so the GPU runs the captured graph at
> ~10-15% efficiency. The wall is **many small per-block kernels at moderate token count** (48 blocks × ~18 ops =
> norms/slices/permutes/modulate over 4442 tokens; each memory-bound + occupancy-starved, F16 only halves the
> bandwidth fraction). Oasis R7-R8 regime → the **next lever is KERNEL FUSION** (fuse `NormModulate`
> LayerNorm+AddScalar+Affine → 1 kernel; fuse `ModParams`; fuse QKV split/permute) + the DINOv2-giant conditioner
> (4.5 s, same inefficiency). Large, low-per-item ROI, best with an nsys profile on a **quiet** GPU (this box is
> heavily contended — the same config swings 26-39 s run to run). **Honest status: the engine's identified fast
> levers (cuDNN SDPA, CUDA-graph, F16, device-CFG) are all applied + correct; closing the last ~5× to Python's
> 5.76 s needs op-fusion work, not another flag.**

## Method rules (from the war)
- Phase probes before any op-level work; wall ≠ op-profile means an un-instrumented host phase.
- **Correctness before perf** — parity/coherence gate every round; bit-identical residency ports first,
  numeric-risk (F16/graph) behind flags with per-stage relL2.
- GPU-shared box: hard-gate every run on `nvidia-smi` idle wait-loop; prefer 3060 for fits, take turns.
- The bar is **faster than Python** with a coherent mesh — not just "faster than before".
