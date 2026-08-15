# Video models — HartsyInference vs ComfyUI scoreboard

Canonical, single-source-of-truth scoreboard for video (T2V) diffusion models. Consolidates the
`video_comfy-vs-hartsy_*` campaign write-ups and the per-model bring-up benchmarks that formerly lived
as separate dated files in [`benchmarks/results/`](../results/) (now retired — this table is the
successor) into one table. Where multiple source runs covered the same model, the **freshest scoreboard
run wins** (07-11 over 07-08 over 07-03), unless a later per-model or per-feature result gave a more
precise number for that specific model — see Notes below for the one case where that applies (Wan2.2
TI2V-5B step-cache).

**Hardware:** RTX 4090 24 GB only — no video benchmarks have been run on the RTX 3060.
**Methodology:** end-to-end wall-clock through the **SwarmUI API** — the identical generation request
routed to the ComfyUI backend, then to the HartsyInference backend, on the same GPU, same request, warm
(model resident). This is the user-perceived latency gap, not an isolated kernel/pipeline timing. See
[`README.md`](README.md) for the engine's default performance profile and
how to reproduce these numbers. Standard workload (unless noted): 25 frames, 512×320, h264-mp4,
`videoresolution=Image`, seed randomized per gen to defeat SwarmUI's identical-params result cache.

## Results — warm generation (model resident)

| Model | GPU | HartsyInference | ComfyUI | Ratio | Date | Source |
|---|---|---:|---:|---:|---|---|
| Wan 2.1 T2V 14B (fp8, 15 steps) | RTX 4090 | 30.58 s | 30.62 s | 1.00× — tied (parity) | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.1 T2V 1.3B (fp16, 20 steps) | RTX 4090 | 11.22 s | **6.28 s** | 1.79× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-0.9 2B (fp16, 20 steps) | RTX 4090 | 4.59 s | **2.84 s** | 1.62× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| Wan 2.2 TI2V-5B (fp16, 20 steps) | RTX 4090 | 15.5 s | **4.52 s** | 3.4× slower | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-2.3 22B (video+audio, 20 steps) | RTX 4090 | 42.3 s | n/a — no comparable Comfy workflow | n/a | 2026-07-11 | video_comfy-vs-hartsy_2026-07-11.md |
| LTX-2.5 22B dev (video+audio, int8-convrot, 30 steps)† | RTX 4090 | **47.40 s** | **42.48 s** | **1.12× slower** | 2026-08-14 | bench_ltx25.py |
| HunyuanVideo 13B T2V (fp8, 20 steps) | RTX 4090 | 1m26s e2e (~2.15 s/step) | n/a — no Comfy Hunyuan T2V workflow benched yet | n/a | 2026-07-02 | hunyuanvideo_e2e_2026-07-02.md |
| Kandinsky-5.0 T2V Lite (2B, 30 steps) | RTX 4090 | 102.0 s e2e (~2.9 s/step) | n/a — not yet wired through SwarmUI (in-engine text encoders pending) | n/a | 2026-07-02 | kandinsky5_t2v_e2e_2026-07-02.md |

Row count: 8. Bold marks the faster (lower-wall-clock) side of each head-to-head comparison; rows with no
ComfyUI baseline are left unbolded.

The LTX-2.5 row is the last SwarmUI-warm measurement. Two changes have landed since, both measured on the CLI
harness with interleaved 4-rep campaigns: **grouped Linear −41.6 ms/step** and the **fused int8 mma GEMM −10.5
ms/step** (1459.6 → 1406.8 ms/step, ≈ −1.6 s over 30 steps). Neither is folded into the row: extrapolating one
harness's delta onto another's wall clock is how a scoreboard starts lying. Re-run `bench_ltx25.py` (after
checking the deployed extension DLL against engine HEAD) to refresh it.

**Before adding or trusting any row or delta here, read "The harness's noise floor" below.**

## LTX-2.5 diffusion decoder — temporal tiling, and the 40× that remains (2026-08-14)

**The geometry ceiling is gone.** Every pass in the decoder now runs over halo-padded temporal chunks sized
against a VRAM budget (`LtxVideo25TemporalChunks`, `ChunkWorkspaceBytes`; 0 = derive from free VRAM). 768×512×97f
decodes where it previously OOMed (9312 MB needed vs 7981 free).

| geometry | before | after |
|---|---:|---:|
| 768×512×97f | **OOM** | **115.5 s** (8 chunks of 13 frames + 5 halo) |
| 512×320×97f | 38.4 s | **35.99 s** (2 chunks — faster despite halo overhead) |

**It is EXACT, not an approximation** — this is the part that matters and it was verified three ways: chunked vs
un-chunked decode is **bit-identical on CPU (max-abs diff 0.0)** and **bit-identical on CUDA with the real
checkpoint** at 512×320×25f, and the ComfyUI layer-diff still passes at every stage. Two design choices buy that:
a **deferred-write FIFO** holds each chunk's residual add until no unprocessed chunk's halo still needs the
pre-attention value, and the rope is built at **global** frame positions so the tables reproduce the untiled ones
exactly. Halo is `kernelT/2` per side **per block** (5 output frames at stage 5) — chunking per block rather than
around the whole stack is what makes a small halo sufficient. Note the padded window must also be floored at the
kernel width, since `Na3d` collapses to `min(kernel, frames)` and a short end window would silently change every
window inside it.

Not comparable to ComfyUI's `temporal_overlap 16`: that is a **blend** overlap for an approximate scheme, ours is
an **exact** halo. Deliberately not matched. Whole-decode tiling could not have been made seam-free here anyway —
the stage-5 stack's receptive field is ±40 output frames, ~18 latent frames per side, wider than the entire
13-latent-frame clip.

Seam check (the expected failure mode, and the reason "it didn't OOM" is not success): adjacent-frame mean|diff|
across all 97 frames gives boundary transitions 2.47/2.83/2.43/2.01/2.81/2.84/2.58 against a series mean of 2.499
(sd 0.400) — every boundary inside noise, and the series *minimum* is a boundary. The top spikes are wave motion.
8×-amplified difference images at three boundaries are indistinguishable from three motion controls.

### The na3d kernel is now query-tiled — and it was never the whole gap (2026-08-14)

**The previous edition of this section said the ~40× gap to the conv decoder "is almost entirely this kernel."
That was arithmetic, not measurement, and it was wrong.** Timed directly around `IBackend.Na3d` with a sync on
each side, through the decoder class with no pipeline in the picture, on the 4090:

| na3d's measured share of the decode, BEFORE the rewrite | decode | na3d | share |
|---|---:|---:|---:|
| 512×320×25f | 9.13 s | 2.43 s | **27 %** |
| 768×512×97f | 109.0 s | 40.6 s | **37 %** |

So even an infinitely fast na3d left ~68 s at 768×512×97f against the conv decoder's 2.878 s. **The other
63-73 % of the decode is now the top perf item, and nobody has attributed it yet.**

**What was built.** `ltx25_na3d_tiled8x8_f32` / `ltx25_na3d_tiled4x8_f32`: one block per (batch, t, h-tile,
w-tile, head), staging the tile's *union* window through shared memory once instead of re-reading a full
kt·kh·kw window per query. Softmax is online (flash-style) because the union does not fit as a score matrix;
positions outside a given query's own window are masked to -FLT_MAX and contribute an exact zero. Selected
automatically at head_dim 64; kill switch `HARTSY_LTX25_NA3D_TILED=0` restores the per-query kernel, which stays
in the file as the fallback for every other shape.

**The tile is 1 deep in T on purpose.** Temporal chunking must stay bit-identical, and a tile spanning T would
put the same global query into differently-aligned tiles in the two arms. Measured after the change: chunked
(4 frames/chunk) vs un-chunked at 512×320×25f is **0 of 12 288 000 elements different**.

Matched-budget A/B, N=2 per arm, arms interleaved in one process with a fixed `ChunkWorkspaceBytes` so the chunk
count cannot differ between them (it derives from free VRAM otherwise, and the halo re-attention scales with it):

| 4090, direct decoder harness, real checkpoint | untiled | tiled |
|---|---:|---:|
| 512×320×25f — whole decode | 9.127 / 8.774 s | **6.933 / 6.281 s** |
| 512×320×25f — na3d only | 2.433 / 2.420 s | **0.366 / 0.367 s** |
| 768×512×97f — whole decode | 109.046 / 109.966 s | **75.914 / 75.711 s** |
| 768×512×97f — na3d only | 40.615 / 40.715 s | **6.429 / 6.445 s** |
| stage-5 trunk kernel, per diff block, 25×80×128×4 heads, window 11³ | 298 ms | **45.4 ms** |

External baseline for the whole decode: the **conv** decoder is 2.878 s at 768×512×97f.

Measured through the decoder class directly, **not** through SwarmUI: SwarmUI runs the *deployed* extension's
DLL and PTX, so `ltx25_bench.sh` cannot see a kernel change without a redeploy.

Where the stage-5 kernel now sits: an 8×8 query tile stages an 11×18×18 = 3564-position union instead of 64
separate 1331-position windows, so it executes **2.68× more FLOP** than the window arithmetic requires and reads
**~24× less**. At 45.4 ms/block that is **~20.5 TFLOP/s raw** (≈25 % of the 4090's ~82 TFLOPS fp32, up from
~1 %) and **~0.65 TB/s** of union traffic — the kernel is now issue/shared-memory bound, not bandwidth bound.

**Do not re-chase** (stage-5 trunk shape, best of 5-6 launches, each arm against the 8×8/256-thread control at the
*same* source revision, 4090):

- **A smaller 4×8 tile is slower**, 50.0 ms vs its control's 45.7 ms, despite 22 % less arithmetic (union 2772 vs
  3564). Once tiled, this kernel is bound by shared-memory loads per FMA, and the bigger tile's fatter register
  block wins by more than the extra FLOPs cost. Do not "optimize" by shrinking the tile.
- **128 threads per block is slower**, 51.5 ms vs its control's 44.7 ms, even though it halves shared-memory loads
  per FMA (PPT 2→4, DPT 4→8). The occupancy loss dominates. 256 is the setting.
- The claim that the untiled kernel was **DRAM-bandwidth bound is also wrong**: it achieved ~2 TB/s of
  *naive-model* traffic on a ~1 TB/s part, i.e. L2 was already absorbing half the redundancy. The win came from
  the shared-memory staging and the register-blocked score/AV inner loops, not from saving DRAM reads.

**Where both wrong claims came from, because the mistake is reusable.** Neither was a typo; they came from one
step of bad reasoning written into this file earlier the same day. The roofline was computed as: naive window
traffic 5.58 TB → ~5.6 s at ~1 TB/s, against a **measured whole-VAE decode of 9.8 s** — and that agreement was
read as "the kernel sits near its traffic bound." **The 9.8 s was never the kernel's time.** Timing directly
around `IBackend.Na3d` puts it at 2.43 s of that 9.8 s. So the traffic model actually over-predicted the kernel
by ~2.3×, which is exactly the L2 reuse the measurement later found — the evidence that the premise was wrong
was already in the arithmetic, hidden by comparing a component's model against the whole pipeline's clock.

The lesson is the one this file keeps re-learning in different costumes: **a component's model must be compared
against that component's measured time, never against the wall-clock of the thing containing it.** The same
shape produced the "int8 GEMM is at the hardware wall" error (bare GEMM 537-620 TOPS vs 350 in-chain — always
time the kernel alone) and the luminance-not-semantics metric in the retracted conditioning finding. Cheap
insurance: an NVTX range or a stopwatch around the op costs minutes and would have killed both claims before
they were written down.

Untried, in rough order of expected value: a 64-position chunk (score inner loop 1.33 → 2.0 FMA per shared-memory
load) needs 66 KB, so it wants the `MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in and drops to 1 block/SM — worth one
measurement; and the deterministic stages 0-3 also route through the tiled kernel when their dims allow, but they
are ~1 % of the decode and not worth tuning.

### Landmine: the diffusion env var alone is not enough

`Models/Stable-Diffusion/LTX-2.5/` carries the **conv** VAE, and the engine loads the **folder**, not a file.
`HARTSY_LTX2_DIFFUSION_VAE=1` on its own therefore **silently falls through to the conv decoder** — a
768×512×97f run "succeeded" in 2.9 s and proved nothing. Worse, you must **swap** the symlink, never add a
second one: `IsDiffusionVideoVae` is one boolean over the *merged* key set, so having both VAEs present flips it
true and routes the conv decoder's `decoder.*` keys into the diffusion bucket, corrupting both. Always confirm
the log line `HARTSY_LTX2_DIFFUSION_VAE set — … (310 tensors)`. Full procedure in
`benchmarks/swarm_video_bench/DIFFUSION_VAE_HEADTOHEAD.md`.

## LTX-2.5 diffusion decode: 79.2 s → 15.5 s, and now FASTER than ComfyUI's (2026-08-14)

The decode was **~85% host-transfer tax, not GPU work.** A profiling pass (stopwatch around GPU-synced scopes
plus A/B, not a roofline guess) attributed all but 0.4% of it, and the answer was that real GPU compute was only
~10 s of 74.9 s. Three host loops were the cost:

| target | measured cost | fix |
|---|---:|---|
| `LtxVideo25NaBlock.Modulate` — host loop over every token | **47.2 s (63%)** | `IBackend.AffineBroadcastLastDim` already existed with matching indexing; host loop deleted |
| `attended.Reshape(...)` — `Tensor.Reshape` forces a full D2H for a **metadata-only** change | **~11.5 s** | `Na3d` relaxed to accept a rank-2 output of identical bytes, so the view disappears |
| `LtxVideo25PixelShuffleUpsample` — host shuffle | **4.25 s** (re-measured; 3.71 predicted) | new `ltx25_pixel_shuffle_f32`; host loop kept as the numerical reference |

| geometry | before | after |
|---|---:|---:|
| 768×512×97f | 79.16 s | **15.52 s** |
| 512×320×25f | 6.92 s | **1.51 s** |

### ⚠️ RETRACTED: this did NOT flip the head-to-head. Warm-vs-warm, we are SLOWER.

An earlier revision of this section claimed ~2.2× faster than ComfyUI on the diffusion decode. **That was
wrong, and the error was mine, not an agent's.** ComfyUI's "cost of the diffusion decoder" was computed as
**+27.45 s** by differencing its **cold** diffusion run (68.92 s, which included loading the 1.5 GB VAE)
against its **warm** conv run (41.47 s). Cold against warm. That inflated Comfy's decode cost by roughly a
whole model load, and every conclusion drawn from it inherited the inflation.

Re-measured properly — both engines through the SwarmUI API, both on the **diffusion** VAE, both **warm**,
768×512×97f/30 steps, VAE identity confirmed in every submitted workflow, DiT residency confirmed 48/48:

| | warm wall | evidence |
|---|---:|---|
| **ComfyUI** | **42.1 s** (42.22 / 42.09) | `Prompt executed in`, `vae_name: ltx-2.5-video-vae-bf16` on every submission |
| **Hartsy** | **60 s** (cold 76 s) | decode 12.96 s, plan `6706 MB → 29 of 97 f/chunk`, h264+AAC, 4.0417 s = 97/24 |

**Hartsy is 1.43× SLOWER on the quality decoder** — worse than the 1.12× on the conv path, because Comfy's
warm diffusion decode costs it almost nothing over conv (42.1 vs 41.5) while ours costs ~10 s. The decoder work
in this section is still real and large (79.2 → 12.9 s decode, geometry ceiling gone, visibly cleaner output);
it closed a far worse gap, but it did not close it.

**Two methodology rules this cost us, both now mandatory for any row in this file:**
1. **Never difference a cold measurement against a warm one.** A cold arm carries model load; here that was
   ~27 s, larger than the effect being claimed.
2. **Record DiT residency with every SwarmUI number.** A first attempt at the Hartsy arm returned 68.8 s purely
   because the DiT was in streaming mode (`resident prefix 18, streamed 30`), which starves the decode's
   VRAM-derived chunk budget AND cripples the denoise. Nothing in the harness flags it. **A run without
   `resident prefix 48, streamed 0` in its log is void** — including, potentially, rows taken earlier today.

Correctness held throughout, which matters because this decoder has already shipped three silent bugs: the
ComfyUI layer-diff passes at every stage (1e-7–5e-4 relL2 against a 5e-3 threshold, verified by dumping our side
and diffing in numpy rather than trusting a green test), chunked-vs-un-chunked stays bit-identical, 46 CUDA +
26 diffusion tests pass, and the decoded frames are **bit-identical to the pre-change run at both geometries**,
97/97 and 25/25, plus a visual check.

### Two bugs in the `((IBackend)this).X()` fallback idiom

A backend override calling `((IBackend)this).SomeOp(...)` to reach the managed default is **infinite recursion**,
not a fallback: the class method implicitly implements the interface member, so interface dispatch re-enters the
override. One instance was caught before shipping (in the new pixel-shuffle CUDA fallback, proven with a
standalone repro and fixed via a static `Ltx25PixelShuffleReference`). **A second is live in shipped code and is
now fixed here**: `VulkanBackend.AffineBroadcastLastDim` used it in its dtype-mismatch branch — a stack overflow
on any non-F32/F16 input on Vulkan. The managed body is now `IBackend.AffineBroadcastLastDimReference`, matching
the `Na3dReference` pattern, whose doc comment says exactly why it exists.

### Still open

~15.5 s remains against the profile's ~10 s compute estimate. The unmeasured candidates are the remaining host
loops — `PatchifyPixels`/`UnpatchifyPixels` (~1.4 s combined at the old scale) and the slice-scatter staging in
`AddContextPass`/`FlushAttention`. Also **recommended as a project, not a patch**: making `Tensor.Reshape`
metadata-only for a device-resident tensor. It needs a *view-aware device cache*, because the cache is keyed by
tensor identity and a view's writes are invisible to its parent — this codebase has been bitten by that exact
gap three times (`ApplyKeyframesAbsPos`, the q/k-norm path, and a `Linear` into a reshaped `modulation` that
left it all zeros and degenerated every AdaLN). Worth 11.5 s here and much more across every model.

## LTX-2.5 — diffusion decoder vs ComfyUI's, measured as a DELTA (2026-08-14)

The matched-quality end-to-end row **could not be run**, for a reason worth recording (below). What could be
measured is the thing that actually matters: **what each engine pays to use the diffusion decoder instead of the
conv one**, at 768×512×97f/30 steps. Taking it as a delta against each engine's own conv run cancels the harness,
so the two sides are comparable even though they were not measured the same way.

| | conv | diffusion | **delta** |
|---|---:|---:|---:|
| **ComfyUI** (prompt-execution, same session, warm) | 41.47 s | 68.92 s | **+27.45 s** |
| **Hartsy** (decode phase, direct harness, tiled kernel) | 2.878 s | 75.9 s | **+73.0 s** |

**We are ~2.7× behind ComfyUI on the diffusion decode.** Not the 40× the earlier scoreboard implied, and not
parity either. ComfyUI's arm is proven from the submitted workflow JSON — `"vae_name":
"LTX-2/ltx-2.5-video-vae-bf16.safetensors"` decoded through `VAEDecodeTiled` at tile 2048 / temporal 64 /
temporal_overlap 16 — and its conv control in the same session used
`ltx-2.5-video-vae-conv-bf16.safetensors`, so the delta is a within-engine, within-session difference.

Treat the 2.7× as indicative, not a scoreboard row: Comfy's figure is whole-prompt execution (so its delta
includes any decode-related overhead beyond the VAE itself) while ours is the decode phase alone. The comparison
is honest about direction and rough magnitude; it is not a like-for-like wall-clock row, and it is not labelled
as one.

### Why the end-to-end row is blocked: the deployed EXTENSION refuses the diffusion VAE

Not the engine — the engine decodes with it correctly. The **SwarmUI extension** validates the model folder and
refuses outright:

> `LTX-2.5 bundle is incomplete — '…/LTX-2.5' contains the LTX-2.5 diffusion video VAE, which the engine's
> LTX-2 pipeline does not decode with yet. Stage 'ltx-2.5-video-vae-conv-bf16.safetensors' instead…`

That message is **stale** — the pipeline does decode with it now — but it is compiled into the deployed
extension assembly, which `deploy_extension.sh` does not rebuild (it deploys the *engine* DLLs and PTX). Two
things make patching it non-trivial, and neither should be attempted casually:

- The check reads the file's **keys**, not its name: staging the diffusion VAE under the conv filename was tried
  and still refused. So there is no configuration-only workaround.
- **The deployed extension's source is not on disk.** Its DLL is `…-36c19cfa.dll`, matching the extension repo's
  detached HEAD `36c19cf`, but the refusal string does not appear anywhere in that checkout. The deployed binary
  contains code no source tree has — including, per the runbook, the folder-loading (`split bundle`) behaviour
  the working conv path depends on. Rebuilding from the current source would therefore risk **losing** working
  behaviour to fix a stale message. Reconstruct deliberately, with the conv path re-verified after, or not at all.

Everything touched for this measurement was reverted and verified: `Backends.fds` and `Settings.fds` byte-identical
to their timestamped backups, the VAE symlink back to conv, the systemd drop-in removed, hartsy backends 7/8
enabled and running, comfy 0/1 disabled, and a normal generation confirmed working afterwards.

## LTX-2.5 — conv-vs-conv head-to-head, both arms re-measured (2026-08-14)

Standard workload (768×512, 97f, 30 steps, cfg 3.0, **conv** decoder on both sides), SwarmUI API,
`bench_ltx25.py`, N=5 warm + 1 cold, random seed per gen, 4090 (nvidia-smi index 1). **Both arms measured the
same day against the same weights** — nothing carried forward, per this file's no-splicing rule.

| | Hartsy | ComfyUI |
|---|---:|---:|
| **Warm mean (N=5)** | **47.40 s** | **42.48 s** |
| Warm spread | 47.13–47.56 (sd 0.17) | 42.08–42.73 (sd 0.29) |
| Cold | 87.39 s | 82.92 s |
| Peak VRAM | 23730 MiB | 24050 MiB |

**Ratio: 1.12× slower** (was 1.18× on the 08-12 row). Hartsy improved 50.54 → 47.40 s across the 08-14 fixes.

**The decoder question is CLOSED, and closed by evidence rather than by assumption.** Both arms are proven to
have decoded with `LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors`, read out of the workflow JSON SwarmUI logs
before submission (`Data/Logs/2026-08/14-08-47.log`): `VAELoader` → the conv VAE, `CLIPLoader` → the
comfy-format `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot` at type `ltxv`, `UNETLoader` →
`ltx-2.5-22b-dev-transformer-int8_lean_convrot` at `weight_dtype: default` (**no fp8 cast** — the `.swarm.json`
carries a stale `special_format: fp8_scaled` tag on an int8 file, and SwarmUI happens to map that to
`dtype=default`, which is correct here by luck rather than by design). Workload re-confirmed from the graph
(`EmptyLTXVLatentVideo 97/512/768`, `SwarmKSampler steps 30 cfg 3`) and by ffprobe on a warm output
(768×512, 97 frames, 24 fps, h264 **+ aac**, so both engines produced dual-stream A/V).

**The 08-12 42.76 s row was ALSO conv** — `Data/Logs/2026-08/13-09-56.log` holds exactly that campaign's 6 gens,
all at this workload, all on the same conv VAE. So today's number **confirms** that row rather than replacing it,
and the "which decoder did Comfy use" caveat is retired.

⚠️ **One real asymmetry, in Comfy's favour and worth keeping in mind:** Comfy decodes through `VAEDecodeTiled`
(tile 2048, overlap 256, **temporal 64, temporal_overlap 16**) — spatially a single tile at 768 wide, but
temporally chunked. Hartsy's conv decode is untiled. Same weights and same decoder, so the row is fair; but it
means Comfy is not paying a peak-memory penalty we would pay, and it is the same tiling capability we lack on
the diffusion decoder.

**No repack rebuild was needed after all.** The flat model name is recreated by a one-line setting —
`SDModelFolder: Stable-Diffusion` → `Stable-Diffusion;diffusion_models` — because the DiT already lives in
`Models/diffusion_models/`, and `IsDiffusionModelsFormat` then routes it to `UNETLoader` with
`LoadingVAE == null`, which is exactly what selects Comfy's separate-VAE loader branch.
`CommonModels.Known["ltx2-5-video-vae"]` already points at the conv VAE on disk. No symlinks, no copies, no
downloads. Setting reverted after the run (both `Settings.fds` and `Backends.fds` restored byte-identical to
their timestamped backups; service left active with the hartsy backends enabled).

### Two landmines this campaign surfaced

- **A backend's `GPU_ID` is a CUDA ordinal, not an nvidia-smi index, and they disagree on this box.** Backend
  **id=0 has `GPU_ID: 1` and runs on the 3060**; backend **id=1 has `GPU_ID: 0` and runs on the 4090**, because
  `CUDA_VISIBLE_DEVICES` enumerates fastest-first. The first attempt launched ComfyUI on the 3060 and had to be
  thrown away. Every historical Comfy benchmark in the logs ran as `ComfyUI-1` — **use backend id=1**, and
  confirm from the log line `[ComfyUI-1] Device: cuda:0 NVIDIA GeForce RTX 4090` before trusting a number.
- **A crashed ComfyUI writes a multi-GB systemd coredump, and this box has no headroom for one.** The aborted
  3060 run produced an **8.7 GB** core, which took the root filesystem to **zero bytes free**. That ENOSPC then
  landed mid-write on `Backends.fds` and truncated it to 0 bytes; SwarmUI restarted with no backends and
  crash-looped, producing five more cores, until the file was restored from backup and rewritten atomically.
  Recovered fully — but ~11 GB of root-owned dumps remain in `/var/lib/systemd/coredump` and need root to clear.
  **Cap it** (`Storage=none` or a small `MaxUse=` in `/etc/systemd/coredump.conf`) before running another
  campaign on a near-full disk.

⚠️ **The deployed extension was 21 hours stale when this campaign started** — DLLs from Aug 13 10:26, and
`ltx25_na_decoder.ptx` absent from its `Ptx/` entirely. So the SwarmUI backend had been running without the new
attention kernel and without any of the 08-14 correctness fixes, and **any SwarmUI-side number taken earlier on
08-14 describes old code**. Redeployed (14 DLLs + 78 PTX, unit restarted, deployed `HartsyInference.Cuda.dll`
md5-matching a fresh net8.0 build) before the numbers above were taken. This is the **second** scoreboard-grade
measurement lost to this trap, so it is now a **hard guard rather than a memory note**:
`bench_ltx25.py` md5-compares every deployed `HartsyInference.*.dll` and every `Ptx/*.ptx` against the local
net8.0 build and **refuses to run** on a mismatch (`LTX25_SKIP_DEPLOY_CHECK=1` to override deliberately).
Verified both ways — corrupting one deployed PTX byte makes it exit 2 naming that file; restoring it passes.

### Why the matched-decoder head-to-head is still open

The point of the campaign was a like-for-like comparison, and it is blocked in two different ways:

- **Diffusion-vs-diffusion cannot run at the standard workload — we OOM.** 768×512×97f needs **9312 MB** for the
  decode against **7981 MB** free: stage 5 there is ~2.4M tokens and we have **no tiling**. It runs at
  512×320×97f (decode 38.4 s) and 512×320×25f (9.8 s). ComfyUI decodes this VAE **tiled**
  (`VAEDecodeTiled [512, 64, 64, 16]`), which is why it has no such ceiling. **Temporal tiling is the sized next
  feature**, and the reference is built for it — `forward_pre_diffusion` already takes `drop_leading_frame` /
  `pad_trailing` precisely so a tiled caller can decode a later temporal chunk. Expect Hartsy to lose that row on
  wall-clock when it can finally be run: an untiled 38.4 s decode against a tiled one. **That gap is the tiling
  prize, measured.**
- ~~Conv-vs-conv could not be run.~~ **DONE — see the head-to-head section above.** It needed no repack rebuild
  at all, only the one-line `SDModelFolder` addition.

## LTX-2.5 diffusion VAE decoder on the GPU (2026-08-13)

The `NADiffusionDecoder` — the decoder Lightricks' own templates use, credited with "sharper faces, legible
text, fewer smears" — was unusable: **`IBackend.Na3d` had no CUDA override on any backend**, so its 3D
neighborhood attention ran as a six-deep scalar host loop. At 512×320×25f that is ~2.8 TFLOP single-threaded
(256k stage-5 tokens × 4 heads × an 11×11×11 = 1331 window × head_dim 64, over 8 diff blocks). The rope was
host-side too, but it is the minor term (~8.4 GB of PCIe, seconds). Two new kernels in
`Kernels/ltx25vae/ltx25_na_decoder.cu`; workload `ltx25_bench.sh`, 4090, 30 steps, seed 1.

**To reproduce you must set `HARTSY_LTX2_DIFFUSION_VAE=1`** — the decoder is opt-in (`e3e702a8`) precisely
because its output is still wrong, and a model directory carrying only the diffusion VAE is otherwise refused.
Both arms of that gate were verified by generation: without the flag the refusal fires and names the conv file;
with it the run logs `output is KNOWN NOT CORRECT` and decodes in 9.67 s.

| geometry | decode, host | decode, CUDA | speedup |
|---|---:|---:|---:|
| 512×320×25f | **>22 min — timed out, never finished** | **9.67–10.24 s** | ≥100× |
| 256×160×9f | 173.22 s | **1.45 s** | **119.9×** |

The 512×320×25f decode has now been measured four times across two sessions and two builds — 12.89 s before the
q/k fix below, then 10.24 / 9.8 / 9.67 s after it, the 9.8 s on an independent run by another session at a
different seed. Read the post-fix figure as ~10 s, not as a number resolved to 0.1 s.

DiT step time is unchanged (262.8 vs 264.6 ms/step at 512×320×25f — within this harness's ~17-20 ms noise
floor), and the shipping conv-decoder path is untouched (decode 376 ms), as expected since it uses no NA.

**The decode is now fast AND correct (2026-08-14).** It first came back fast and wrong — right composition
under heavy full-frame noise — and a ComfyUI layer-diff found three separate defects. High-frequency energy
on the same frame: **11.8 (broken) → 2.06, against the conv decoder's 3.10**, so the diffusion decode is now
the cleaner of the two, which is what the model card claims for it.

### Diffusion vs conv decoder — quality pass (2026-08-14)

Four prompts chosen to test the model card's specific claims, each decoded **both** ways from the same prompt and
seed, so the latent is identical and the ONLY difference is the decode. 512×320×25f, 30 steps, seed 1, 4090.

| scene | grain conv | grain diff | decode conv | decode diff |
|---|---:|---:|---:|---:|
| lighthouse | 2.91 | **2.06** | 355 ms | 9610 ms |
| portrait (faces) | 1.63 | **0.93** | 320 ms | 9554 ms |
| neon street (legible text) | 4.83 | **4.47** | 362 ms | 10053 ms |
| desert car (fast motion) | 1.90 | **1.20** | 357 ms | 9230 ms |

Grain (mean local deviation from a 3×3 neighbourhood) is lower on the diffusion decode in all four, by
**7–43%** — 43% on the portrait, 37% on the car, 29% on the lighthouse, and only **7.5%** on the neon street.

**Read that last figure the right way round: neon has the SMALLEST grain delta and by far the LARGEST visual
difference.** The conv decode renders the shop sign as illegible smear and streaks the whole frame horizontally;
the diffusion decode renders distinct characters. Conv's failure there is *smearing*, which a local-deviation
metric barely registers — so the table understates exactly the case the model card is about. The conv desert-car
frame likewise carries a cross-hatch artifact across the entire image that the diffusion one does not. **Look at
the frames; the table is a floor on the difference, not a measure of it.**

⚠️ **A sharpness metric was tried and is NOT reported as evidence.** p90 gradient magnitude comes out slightly
*higher* for the conv decode in 3 of 4 scenes — because conv's own cross-hatch and streaking are high-gradient,
so the metric scores its artifacts as edges. That is the same failure as the retracted "conditioning is inert"
pixel-delta metric: real number, wrong quantity. Grain is reported because a local-deviation measure over a
matched latent is meaningful; sharpness is left to the frames.

**Cost:** the diffusion decode adds **~9.3 s** per 25-frame generation over conv (~0.35 s). Unmeasured and
relevant before defaulting it on: it drops the whole resident DiT prefix (`dropping the resident prefix for the
diffusion VAE decode`), which in a long-lived SwarmUI process makes the *next* generation pay a re-preload. The
CLI harness runs one generation per process, so it cannot see that cost.

### The three defects, and how they were found (2026-08-14)

Structural inspection found **none** of them. Block wiring, AdaLN chunk order, `scale_shift_table`, patchify
channel packing (`c·16 + r·4 + q`, W-sub outer — the unusual bit), pixel-shuffle grouping, the timestep
embedding (4e-5 vs the reference), and the NATTEN window rule (~1e-7 vs `comfy_kitchen.na3d` in every sliding
and clamped regime) were all read and all correct. What found the bugs was a **numerical layer-diff against
ComfyUI's `NADiffusionDecoder`** — `LtxVideo25ReferenceLayerDiffTests` plus `benchmarks/ltx25_layerdiff/`, both
sides consuming the same file-supplied latent and noise so the pipeline is out of the picture.

1. **q/k reached attention un-normalized on CUDA.** `Forward` RMS-normed them through a `Reshape` view in a
   `using` block; the result landed in the view's device cache, which `Dispose` frees without write-back. Passes
   on `CpuBackend`. Fixed by norming the rank-6 tensor directly — `RmsNorm` already rows by the last dim, so the
   view bought nothing. **Real, but not the main one:** hi-freq noise moved only 12.02 → 11.78.
2. **The AdaLN modulation was ALL ZEROS.** Same defect class, other direction: `Linear` writing *into* a
   `Reshape` view of `modulation`. Every diff block's scale/shift degenerated to its `scale_shift_table` alone,
   so the stage-5 denoise was effectively untimed. The layer-diff caught it instantly — `modulation` ref std
   12.26 against ours **0.00000** — where no amount of code reading had.
3. **The one that mattered, and it is NOT an LTX bug — see the core section below.**

**Why the no-PTX A/B proved less than it looked like.** Running the same geometry with the new PTX removed puts
`Na3d` and the rope back on the managed reference, and the arms agree to mean |Δ| 0.0023/255. That exonerates
**the two new kernels** — not the decoder's other CUDA ops, which ran identically in *both* arms. Nothing ever
established this decoder was numerically right on any backend: the prior 1×2×2 test asserts key mapping and
geometry, never output values. Bugs 1–3 all sat in that blind spot.

### ⚠️ CORE BUG, all models: a device write to an auto-promoted tensor was silently discarded

`GpuTransferHelper.CopyToDevice` auto-promotes a host tensor to a **resident weight** on its second upload once
it is ≥ `AutoPromoteMinBytes` (1 MB), on the assumption that a tensor uploaded twice unchanged is a weight. The
weight cache is checked **before** the activation cache. `CacheActivation` did demote a promoted tensor, but
only when the op wrote *through* the promoted pointer (`promotedPtr == gpuPtr`). An op that writes a **fresh**
buffer and rebinds the tensor — the normal `Linear`/`Add` shape — left the stale promotion in place, so every
later read returned the **pre-op value and the device write vanished**.

In this decoder that hit `x` in `LtxVideo25NaBlock.Forward`: `RmsNorm(norm1)` uploads `x` (miss 1),
`Add(x, x, attended)` uploads it again (miss 2 → promoted), the add's result is cached as an activation, and
the *second* `RmsNorm` then read the promoted pre-add copy. Reproduced with no LTX code at all in
`Ltx25InPlaceAddThenNormTests`: `RmsNorm → in-place Add → RmsNorm`, CPU-vs-CUDA relL2 **0.335 at 160 rows and
clean at 80** — because 160×2048×4 = 1.25 MB crosses the 1 MB promote threshold and 80 rows (0.64 MB) does not.

**Fixed** in `CacheActivation`: a device write demotes the promotion regardless of which buffer it lands in.
The demoted buffer comes from `cuMemAlloc`, so it cannot be parked in `PendingOrphans` (that frees against the
async pool) — it goes to a new `PendingPersistentFrees`, swept with `cuMemFree` by the next op, because freeing
it inline double-frees: it is usually that same op's input, whose `finally` still has to run.

**This was silently corrupting any model whose ≥1 MB activation is read twice from host and then written
on-device.** It is size-gated, which is why it hid: the same code is correct at smaller shapes. Kill switch
`HARTSY_NO_AUTOPROMOTE=1` disables promotion entirely and was used to prove the pre-existing test failures
below are unrelated to this fix.

**Regression evidence.** The shipping conv-decoder generation is **byte-identical** across the fix — all 25
frames and `audio.wav` md5-match — and its step time is unchanged. 544/550 CUDA tests pass; the 6 that fail are
this session's own diagnostics with a 1e-4 threshold below the TF32 GEMM floor (relaxed to 1e-3, all pass).
In the Diffusion suite, `MiniMaxH3Assets`, `VideoFeatureDeclaration` and `QwenImageVaeParity` fail **identically
with `HARTSY_NO_AUTOPROMOTE=1`**, i.e. with the new code path unreachable, so they are pre-existing; the two
step-cache real-weight failures are the documented `CUDA_DEVICE_ORDER=PCI_BUS_ID` landmine.

Verification: `Ltx25NaDecoderKernelTests` diffs both kernels against the managed reference (the declared
numerical authority) on non-square shapes, on every face of the volume — `Na3dWindowStart`'s slide-inward rule
is exactly where a window kernel diverges and an interior-only test would miss it — against an independently
computed dense attention, and with identity rope tables. **Every case asserts the PTX actually loaded.** That
guard matters here more than usual: both entry points fall back to the managed reference when the PTX is
missing, and the reference is what the tests compare against, so without it the whole file would pass
vacuously. Confirmed by hiding the PTX and watching all 11 fail, then restoring it and watching all 11 pass.

## LTX-2.5 22B dev — first Comfy-vs-Hartsy head-to-head (2026-08-12)

† Off-standard workload, not the table's usual 25f/512×320/20-step smoke test — LTX-2.5 is a 22B model
worth a "decent length/quality" pass instead: 768×512, **97 frames** (~4.0 s @ 24 fps), 30 steps, cfg 3.0,
`ltx-2.5-22b-dev-transformer-int8_lean_convrot` (dev, non-distilled, comfy-kitchen int8-convrot on both
DiT and Gemma-4 TE, joint video+audio latent). N=5 warm reps + 1 cold, random seed per gen, same script
family as `swarm_video_bench/bench_t2v.py` (`bench_ltx25.py`, both backends routed through the SwarmUI API
one at a time, model resident). Cold: 79.14 s (Hartsy) vs 71.94 s (Comfy). Peak VRAM 21.5 GiB (Hartsy) vs
24.0 GiB (Comfy) — fully DiT-resident, no block-swap streaming (22B fits in int8 on this 24 GB card). Comfy
needed the `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot` TE Swarm auto-downloads (14.3 GB); Hartsy uses
its own staged files directly.

**Both sides are prompt-faithful — this row IS quality-matched.** An earlier revision of this row carried a
caveat that every Hartsy frame ignored the prompt and blamed the connector bug `5ad864c2` for still being
live at `06fb26c8`. That was wrong, and the cause was a **stale deployment, not a live defect**: the engine
DLLs in the extension's output folder had been built at 14:50, before both `aa8e6bc7` (the Gemma-4 tower
wired into the pipeline, 17:18) and `06fb26c8` (the `prompt_adaln` timestep-scale fix, 19:02). Rebuilt at
HEAD, the same prompt/seed renders an actual lighthouse-at-sunset scene — verified by inspecting frames from
both the CLI and the SwarmUI-produced mp4. **Check the deployed DLL's build time against the engine's HEAD
before attributing a bad generation to a code bug.**

### What moved it from 117.13 s to 56.62 s (2026-08-12 perf pass)

Measured one change at a time on the CLI at the same workload (cold wall 139.69 s → 82.56 s), then confirmed
end-to-end through the SwarmUI API. Every step carried its own correctness gate:

| change | effect | correctness gate |
|---|---|---|
| Rebuild at HEAD (deployment was a 14:50 build) | — | frames prompt-faithful |
| Reflect spatial pad folded into `wan_vae_build_padded` | VAE decode 38.14 s → 3.94 s | all 97 frames + audio **bit-identical** |
| `ApplyKeyframesAbsPos` moved on-device | −82 MB D2H per forward | output changes — the 2.5 keyframe marker had been silently dropped on CUDA |
| F16 block activations (`DitDtype.Act`) + new `ltx2_split_rope_f16` | step 2.563 s → 1.756 s | SSIM 0.9957–0.9966 vs the F32 build across the clip |
| Trim the pool before sizing the resident prefix | warm reps stop alternating 62.6/93.5 s | n/a (residency only) |
| BF16 conv-VAE decode (2 new kernels) | VAE decode 3.94 -> 2.89 s | SSIM 0.9983-0.9986 vs the F32-VAE build |
| Fused QK-norm+RoPE+head-major, and the per-head gate (4 new kernels) | step 1.743 -> 1.629 s | fused vs unfused chain pinned by unit test; SSIM 0.992-0.994 |
| Fused block RMS-norm + AdaLN shift/scale (2 new kernels) | step 1.629 -> 1.593 s | same |
| GELU folded into the int8 dequant epilogue (`LinearGelu`) | step 1.593 -> 1.572 s | fused vs Linear-then-Gelu, max abs err 3.9e-3 |
| Pin the per-generation RoPE tables as resident weights | step 1.572 -> 1.522 s | all 97 frames + audio **bit-identical** |

The reflect-pad one is the headline: `CausalConv3d.ReflectPadSpatial5D` was a scalar per-element C# loop that
read `DataPointer` (draining the whole activation D2H and freeing its device copy) and rebuilt a full-size
host tensor that the next op re-uploaded. Every LTX-2 VAE conv is built with `spatialReflectPad: true`, so all
42 conv forwards paid it, the last up-stage over ~313 M elements. It was invisible to `HARTSY_PROFILE`
because 126 of 189 `CudaBackend` ops had no `NvtxRange` — 40 have been instrumented since, and the rule stands:
**if a profile's op totals do not sum to the phase wall-clock, look for un-scoped ops before concluding anything.**

The pool-trim one is a residency trap worth remembering: `FreeMemoryBytes()` counts pool-retained blocks as
used, so a generation that sized its resident prefix straight after the previous one's VAE decode measured
~5 GB of phantom pressure, pinned 22 of 48 blocks and streamed the rest. Because that generation then never
filled VRAM it did not evict, so the next one pinned all 48 — a stable two-generation ping-pong that cost
~30 s on every other request and is invisible in a single-generation CLI run.

### The remaining gap is glue, not math — measured, not inferred

`Int8ConvRotGemmThroughputTests` times the **whole** resident int8-ConvRot `Linear` (activation ConvRot →
per-row dynamic int8 quant → cuBLASLt IMMA → dequant epilogue) at LTX-2.5's real DiT shapes:

| shape | per call | achieved | vs the 4090's ~330 TOPS dense INT8 peak |
|---|---:|---:|---:|
| FFN up, 4992×16384×4096 | 1.992 ms | 336.4 TOPS | **102%** |
| FFN down, 4992×4096×16384 | 2.003 ms | 334.5 TOPS | **101%** |
| attn q/k/v/o, 4992×4096×4096 | 0.564 ms | 296.8 TOPS | 90% |

> ⚠️ **The conclusion below ("at the hardware wall", "101–102% of peak") is WRONG and was corrected on
> 2026-08-13 — see "Where Comfy is actually faster" further down.** Two errors compounded: the ~330 TOPS
> reference is roughly HALF the 4090's dense INT8 rate (660.6 TOPS: INT8 runs at 2× the 330.3 FP16/FP16-accum
> rate), and timing the *whole chain* hid that the surrounding passes — not the GEMM — own the time. The bare
> cuBLASLt GEMM at these shapes measures **536–620 TOPS**. Kept here because the do-not-retry entries derived
> from it are still valid; the headline claim is not.

**The GEMM path is at the hardware wall**, epilogue included — so the int32 IMMA-accumulator round trip is
already paid for inside those numbers, and a custom fused-dequant IMMA kernel has nothing left to win.
Summed over 48 blocks × 2 CFG branches these shapes come to ~0.81 s/step of irreducible GEMM against a
1.74 s step, so the reachable rest is **non-GEMM glue**. Six fused kernels have since taken the step to
1.593 s by collapsing that glue: `ltx2_qk_norm_rope_headmajor` (RmsNorm → split-RoPE → `Permute0213`, three
full passes over each [S, inner] tensor down to one), `ltx2_head_gate` (the per-head gate was expanded to a
full `[seq, inner]` tensor through a constant 0/1 GEMM and then multiplied — now one in-place broadcast), and
`ltx2_rms_modulate` (the affine-free RMS + AdaLN shift/scale pair each block runs six times). What remains is
~4,600 launches/step of `Modulation` row-slicing and the two surviving permutes per attention.

ComfyUI is doing the same arithmetic on the same silicon — `comfy/samplers.py:610` only skips the uncond
pass at `cond_scale == 1.0`, so at cfg 3 it runs the same 60 forwards — which puts its non-GEMM glue at
roughly 0.4 s/step against our 0.93 s. That difference is the whole remaining gap.

**Where the 1.572 s step stands after the fusion work** (profiled per-op, `HARTSY_PROFILE_SYNC`, so ~19% high):
`Linear` 1064 ms (55%) and `SDPA` 333 ms (17%) are BOTH at their hardware roofline — the GEMM per the table
above, and attention at ~47 TFLOP/step against the 4090's ~165 TFLOPS FP32-accumulate FP16 tensor-core rate.
Together that is ~1.17 s of the 1.593 s real step. The reachable remainder is ~0.42 s: `Ltx2QkNormRopeHeadMajor`
148, `GatedResidual` 65, `H2D_MISS_SMALL` 65 ms over 971 calls/step — **diagnosed and fixed**: `HARTSY_H2D_TRACE=1` (now logging small
misses too, not just megabyte-scale ones) showed the audio RoPE cos/sin tables `[101, 1024]` missing the cache
**1536 times per step**, ~620 MB of pure PCIe traffic. They are built on the HOST, so they are neither a
preloaded weight nor any device op's output, and at ~0.4 MB they are too small for auto-promote to rescue — the
big video tables at `[4992, 2048]` missed only twice. Pinning all six tables with `PreloadWeights` when they are
built (and freeing them on a grid re-size) took the step 1.572 -> 1.522 s with **bit-identical** output, `Permute0213` 50 (the two survivors per attention: V-in and
the output), `SliceRows` 31 (the ~3,850 `Modulation` row-slices), `Ltx2RmsModulate` 33, `Ltx2HeadGate` 27.

### 2026-08-13 pass — residency, and the measured ceiling on the rest

Warm mean **56.62 s → 50.54 s** (N=5, SwarmUI API, same workload/harness); cold 79.14 → 71.08 s. Ratio to
Comfy (re-measured 42.76 s): 1.34× → **1.18×**. Every change here is residency-only or bit-identical, so the
output is unchanged by construction — confirmed by inspecting the rendered mp4 (prompt-faithful lighthouse
clip, coherent motion across frames 0/32/64/96, no artifacts) and by analysing its audio track (4.05 s, peak
14.5% FS, −30.3 dBFS RMS, 0% clipped, no silent stretch, broadband ~1.4k zero-crossings/s consistent with the
prompt's surf). Peak VRAM 21992 → 23932 MiB of 24564: the prefix now survives the decode, and that is the price.

| change | effect | correctness gate |
|---|---|---|
| VAE decode evicts only the prefix TAIL it is short of (+1 block margin), not all 48 | −5.3 s warm (combined) | residency only — output unchanged |
| `_onesV`/`_onesA` added to `LtxVideo2Block.EnumerateWeights` | `H2D_MISS_SMALL` 3571 → 2419 calls | residency only — output unchanged |
| Fused `convrot_quant_rowwise` (rotation + per-row int8 quant in one pass) | step 1510 → 1496 ms | **bit-identical** vs the unfused pair |
| Token-major (BSHD) attention — deletes both `Permute0213`s per attention | step **1475.2 → 1435.7 ms** (same-build control) | all 97 frames **pixel-identical**, mean\|diff\|=0 |

**Token-major attention (2026-08-13).** cuDNN accepts token-major strides `[S·H·D, D, H·D, 1]` on Q/K/V/O,
bit-identical and at **zero** cost (2.755 vs 2.756 ms/call at h=32,s=4992,d=128 — it stays on the fused flash
engine). So Q/K now come out of a new `ltx2_qk_norm_rope_tokenmajor_f16/f32` writing contiguously, V feeds
SDPA straight from its Linear, and the output needs no permute back. `HARTSY_LTX2_TOKENMAJOR=0` reverts.

**The projected win was ~135 ms/step and the real one is 39.5 — the difference is the lesson.** The claim
that the head-major kernel's *scattered write* was its bottleneck is **false**: measured at LTX's real shape,
head-major 0.1939 ms vs token-major 0.1927 ms (0.6%, noise). Both already run at ~840 GB/s of ~1008 because
**half of that kernel's traffic is the F32 cos/sin tables**, not the activation. The entire realized win is
the two deleted `Permute0213`s (~31 ms) plus allocation/launch savings. **Remaining headroom in that kernel is
F16 cos/sin tables, not the store pattern.** (Also: the 1496.2 figure carries ~20 ms of build/run drift; 1475.2
is the same-build control, which is the honest comparison.)

The two were measured **together**, never separately; the `DiT preload+prime` drop (3494 → ~457 ms) accounts for
~3.0 s of the 5.3 s and the remainder is unattributed. The `ones` fix is also not fully closed: ~403 small misses
per step survive it, from a source not yet identified — a thread worth pulling.

**The +1 block of margin is load-bearing, and finding it needed the journal.** Freeing exactly the computed
deficit hit the warm mean 0.24 s faster (51.09 s) but logged `OOM on async first attempt: requested=594.0 MB,
free=42.9 MB` once per generation, mid-decode. Those retries appear **zero** times in the 2026-08-12 warm-rep
journal for the identical workload, so they were introduced by the change, not pre-existing: `decodeNeed`'s
104 B/px is a bracket that under-estimates the real peak, and the old free-everything eviction had been hiding
that error under ~21 GB of slack. With the extra block, warm reps log none and only the cold generation still
retries. A single-generation CLI run shows neither the win nor the retries — **check
`journalctl --user -u swarmui.service` across warm reps before believing a residency change is clean.**

Also verified, since the TE-miss path is the one 5 identical warm reps never exercise: two generations with
fresh prompts evict the prefix for the 14.2 GB Gemma encode, re-squeeze to 46 blocks, and settle there with no
OOM and no prefix ping-pong.

The eviction one is the same class of trap as the pool-trim above, and bigger. `decodeNeed` is
`max(3 GiB, frames·h·w·104 B)` = 3783 MiB here against 3392 MiB free — a **391 MiB** deficit that was being
paid by freeing the entire 21.5 GB resident prefix, so the next generation spent **3.5 s** in
`DiT preload+prime` re-uploading all 48 blocks to have reclaimed less than one block's worth. Freeing from the
END of the prefix (2 of 48 blocks here) leaves the pin at `_residentPrefixBlocks` describing the survivors, and
the next generation's idempotent `PreloadWeights` tops up only the freed tail. **Invisible in a
single-generation CLI run** — the CLI's `preload+prime` is a cold first load either way; only warm reps through
the API show it.

The `ones` one: the unit weights for the affine-free RMS pre-norms are built on the HOST in the block
constructor and were absent from `EnumerateWeights`, so — exactly like the RoPE tables above — they were
neither a preloaded weight nor any device op's output, and every one of the 8 norms per block per CFG branch
re-uploaded them (768 reads/step).

Per-op at the 1.510 s step (`HARTSY_PROFILE_SYNC`, 1800 ms profiled vs 1510 ms real = 1.19× high; deflated),
the accounting closes exactly: `Linear` ~850 + `SDPA` ~350 + glue ~311 = 1511 ms.

**An earlier revision of this section concluded from those three numbers that "glue is the entire addressable
budget" and that parity was the asymptote of eliminating all of it (~41.8 s). That was wrong**, because it
took the "`Linear` is at 101–107% of peak" claim at face value. `Linear` is at ~60% of what the same GEMM
delivers when timed without its surrounding passes, so the largest addressable item is inside `Linear`, not
in the glue. See "Where Comfy is actually faster" below. The lesson generalizes: **a whole-chain throughput
number cannot tell you the kernel is at the wall — time the kernel alone before concluding anything is
irreducible.**

Glue is still worth ~106 ms/step of the gap, and its largest single cluster is layout, not math:
`Ltx2QkNormRopeHeadMajor`'s scatter plus both surviving `Permute0213`s are ~135 ms/step spent purely converting
token-major ↔ head-major for SDPA. `CudnnSdpa` already sets per-tensor strides (it builds Kᵀ by swapping them),
so a strided BSHD descriptor could delete that cluster without a transpose — untested, with a measured mechanism.

Verified like-for-like while establishing that: `ffprobe` on both backends' benchmark mp4s shows h264 97
frames **plus** an aac track, so Comfy is running the same dual-stream video+audio DiT, not a video-only path.

### Where Comfy is actually faster — measured, not inferred (2026-08-13)

Warm **50.54 s** vs ComfyUI **42.76 s** re-measured the same day on the same box (1.18×). ComfyUI was benchmarked
directly for the first time instead of being inferred: its own tqdm reports **1.17 s/step** against our
**1.496 s**, so the gap is **326 ms/step**. It is NOT one thing, and it is NOT what the section above claimed.

**Step decomposition, both engines** (ours profiled; Comfy's Linear from its measured chain throughput, its
SDPA assumed equal since both run flash-class kernels on identical shapes):

| | Hartsy | Comfy | gap |
|---|---:|---:|---:|
| `Linear` (int8 chain) | 850 ms | ~629 ms | **221 ms** |
| `SDPA` | ~350 ms | ~350 ms | ~0 |
| glue | ~256 ms | ~190 ms | ~66 ms |
| **step** | **1435.7 ms** | **1170 ms** | **266 ms** |

**Attention is NOT the gap, and Comfy is not using a special attention here.** Its `attention_comfy_kitchen_int8`
is opt-in behind `--use-ck-attention` and is unused in this run; it takes plain PyTorch SDPA on the flash
backend. Our cuDNN fused flash measures 2.755 ms/call at b=1,h=32,s=4992,d=128 = ~148 TFLOPS against the
4090's ~165 TFLOPS FP16/FP32-accum peak (**~90%**), and summing every attention in a step lands at ~347 ms,
matching the profiled ~350. Both engines are at the same wall. **83% of the remaining gap is `Linear`.**

**Our INT8 GEMM is excellent — it is the passes around it that cost.** `Int8GemmEpilogueProbeTests.
GemmVersusChainSplit` times the bare `Int8GemmExecutor` at the same shapes as the whole-chain numbers:

| shape | bare cuBLASLt GEMM | our full chain | comfy-kitchen full chain |
|---|---:|---:|---:|
| ffn_up 4992×16384×4096 | 1.228 ms = **545 TOPS** | 1.891 ms = 354 | 1.244 ms = 538 |
| ffn_down 4992×4096×16384 | 1.081 ms = **620 TOPS** | 1.970 ms = 340 | 1.576 ms = 425 |
| attn q/k/v/o 4992×4096×4096 | 0.312 ms = **537 TOPS** | 0.564 ms = 297 | 0.418 ms = 401 |

comfy-kitchen's numbers come from calling `torch.ops.comfy_kitchen.int8_linear` in ComfyUI's own venv at these
shapes with `convrot=True`. **Comfy's entire chain costs about what our bare GEMM costs** — because it fuses
the dequant into a CUTLASS epilogue (`_C.cutlass_int8_dequant`) and fuses the Hadamard rotation into the
row-wise quantizer (`_C.quantize_int8_rowwise_convrot64`), while we run them as separate kernels around an
INT32 accumulator that goes to HBM and comes back.

**That round trip is the single biggest remaining item: 203 GB/step ≈ 203 ms/step ≈ 6 s** (summed
`m·n·4` bytes written + read over every DiT GEMM). It accounts for ~92% of the Linear gap and ~62% of the whole
gap. Arithmetic that closes: 283 TFLOP/step ÷ 545 TOPS = 519 ms of actual GEMM, against a profiled `Linear` of
850 ms — 331 ms of overhead, of which 203 ms is the accumulator.

**cuBLASLt provably cannot fuse it.** `Int8GemmEpilogueProbeTests.WhichInt8OutputConfigsDoesCublasLtAccept`
asks the driver directly: of `{32I,32F} compute × {INT32,F16} D × {scalar, device-vector} alpha`, only
**INT32 D with scalar alpha** is accepted — every other combination returns rc=15, no algos. So removing the
round trip needs our own INT8 mma kernel with a dequant epilogue (or an N-chunked GEMM whose INT32 tile stays
resident in the 72 MB L2 — untested; note the refuted M-chunking below is a *different* axis).

Also settled: ComfyUI **does not batch the CFG pair** — `LTXAV.extra_conds` wraps the text embedding as
`CONDRegular`, whose `can_concat` demands exact shape equality, and LTX-AV trims embeddings to each prompt's
true token count, so positive and negative differ in length and `batch_chunks` stays 1. Both engines run 60
batch-1 forwards. And Comfy is *streaming* the 21.5 GB checkpoint per forward (`20484MB Staged`, dynamic VRAM
loading) while we hold all 48 blocks resident — it wins the GEMM despite paying PCIe we do not.

### The fused-dequant int8 mma GEMM — built, correct, and NOT fast enough (2026-08-13)

Since cuBLASLt provably cannot fuse the dequant, the epilogue has to live in our own kernel. One was written:
`Kernels/dequant/int8_mma_gemm.cu`, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, block tile
128×128×64, 8 warps as 2×4 (warp tile 64×32, 64 int32 accumulators/thread), two `cp.async` stages, shared row
stride padded to BK+16 (at the natural 64 B stride the 8 rows a warp reads for one A fragment collide 4-way;
80 B spreads them across 8 distinct bank groups and keeps 16 B `cp.async` alignment), and the full
`actScale[m]·wScale[n]·acc + bias[n]` + gelu applied **in registers** so the int32 accumulator never exists in
memory.

It is **bit-identical** to cuBLASLt + `w8a8_dequant_bias` (max abs 0 on every shape, including ragged-M and
the gelu path — `Int8MmaGemmTests`). It is also **slower than the pair it replaces**:

| shape | fused mma | cuBLASLt + dequant | verdict |
|---|---:|---:|---|
| ffn_up 4992×16384×4096 | 2.498 ms = 268 TOPS | 1.749 ms = 383 TOPS | −30% |
| ffn_down 4992×4096×16384 | 2.289 ms = 293 TOPS | 1.190 ms = 563 TOPS | −48% |
| attn 4992×4096×4096 | 0.602 ms = 278 TOPS | 0.492 ms = 341 TOPS | −18% |

**The epilogue fusion is worth ~0.2–0.7 ms per call; a hand-rolled inner loop gives back more than that.**
Break-even is ~383 TOPS on ffn_up and the kernel sits at 268. The deficit is the GEMM inner loop, not the
epilogue: the epilogue's entire store traffic is 163 MB ≈ 0.18 ms on ffn_up, while the kernel is 1.27 ms above
an ideal-GEMM floor. Getting from 268 to 550 needs the parts CUTLASS has and this does not — a 3–5 stage
`cp.async` pipeline (needs >48 KB dynamic shared, so `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
opt-in), `ldmatrix` instead of 48 scalar `ld.shared.b32` per k-tile per warp (currently 1.5 shared loads per
mma), register double-buffering of fragments, and per-shape tile tuning.

**The kernel is left in the tree but is NOT wired into any inference path** (nothing calls
`LaunchInt8MmaGemmDequant`) — it is a correct, unit-pinned foundation for that work, not a regression risk.
Anyone resuming: the target to beat is the *pair*, not the bare GEMM.

### What is actually inside `Linear` — measured, and the next lever (2026-08-13)

`Linear` is one profiler label over FOUR kernels (ConvRot → per-row int8 quant → cuBLASLt IMMA → dequant
epilogue), which is why every estimate of its internals this session was guesswork. `Int8.Quant` /
`Int8.Gemm` / `Int8.Dequant` sub-scopes now exist (`HARTSY_PROFILE_FINE=1`; thousands of pushes per step, so
they are off in a normal profile run). Measured per step (sync-inflated 1.23×; deflated in the last column):

| sub-op | calls/step | ms/step | deflated |
|---|---:|---:|---:|
| `Int8.Gemm` | 3021 | 586.8 | ~477 |
| `Int8.Dequant` | 3021 | 202.8 | ~165 |
| `Int8.Quant` | 3021 | 199.9 | ~162 |

`Int8.Dequant` at 202.8 ms independently confirms the 203 GB/step int32 round-trip model, and `Int8.Gemm` at
~477 ms matches 283 TFLOP/step ÷ 545 TOPS. The GEMM is at the hardware wall; the other ~327 ms is scaffolding.

### The fused mma GEMM: three limiter hypotheses tested and refuted (2026-08-13)

Target, from the isolated timings (`Int8MmaGemmTests`, min-of-batches):

| shape | fused mma | pair total | cuBLASLt GEMM alone | dequant share of pair |
|---|---:|---:|---:|---:|
| ffn_up 4992×16384×4096 | 288 | 398 | **565** | 0.50 of 1.69 ms (30%) |
| ffn_down 4992×4096×16384 | 316 | 596 | **642** | 0.08 of 1.12 ms (7%) |
| attn_qkvo 4992×4096×4096 | 287 | 392 | **569** | 0.13 of 0.43 ms (31%) |

The prize is ~30% on the two shapes that matter, but only if the fused kernel MATCHES cuBLASLt's ~569 TOPS —
the epilogue saving is worth nothing while the mainloop runs at half speed. (`ffn_down` is not worth chasing at
all: its GEMM is already at 642 TOPS and the dequant is only 7% of the pair.) Every change below stayed
bit-exact against cuBLASLt+dequant (max abs 0), so this is purely a throughput story.

- **Pipeline depth — refuted.** STAGES 2→3 moved attn_qkvo 288→297 and made ffn_down *worse*, 316→284: 3 stages
  is 60 KB of shared, which drops the block count per SM from two to one. Reverted to 2.
- **Shared-load instruction count — refuted.** `ldmatrix.x4`/`.x2` replaced the scalar `ld.shared.b32` fragment
  loads, 24 per k-step down to 8, and bought 1–6%. At peak the LSU is only ~9% of the issue budget, so the
  48-loads-against-32-mma ratio was never the constraint. Kept anyway: bit-exact, fewer instructions, no cost.
- **Occupancy — refuted.** 120 registers/thread, zero spill, and **two blocks per SM** by both registers and
  shared. It was already fine.

**Measurement traps hit while establishing that, both worth remembering.** (1) `CU_FUNC_ATTRIBUTE_NUM_REGS` is
**4**; attribute **0** is `MAX_THREADS_PER_BLOCK`. Reading 0 returns 256 for any 256-thread kernel, which reads
exactly like "the compiler took all 256 registers" and supported a completely wrong occupancy diagnosis for
several experiments. The tell was that forcing `CU_JIT_MAX_REGISTERS=32` changed nothing and still reported zero
spill — impossible if the cap were real. (2) `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 99 KB)` "to leave
room" is not free: that budget is what the driver sizes occupancy against, so it forces one block per SM. Ask for
exactly what the kernel uses (`CudaKernels.Int8MmaSharedBytes`).

### ⚠️ Re-baselined and profiled 2026-08-14 — five claims below are STALE. Read this first.

A fresh measurement pass (cold-L2 trio, `HARTSY_PROFILE_FINE=1`, 4-rep `ltx25_ab.sh`, and Nsight Compute against
a like-for-like control) corrected five things recorded in this section:

1. **The "203 ms/step INT32 round trip" prize is already ~62% collected.** `Int8.Dequant` now measures **80.2
   ms/step sync-inflated → 67.8 deflated** over 2005 calls, not 202.8 over 3021 — because the fused kernel
   already took the 960 square video-width projections (960 × 0.132 ms = 127 ms/step; 202.8 − 127 = 76 against
   80.2 measured, which closes). **Remaining round-trip prize ≈ 68 ms/step ≈ 2.0 s over 30 steps**, now dominated
   by ffn_up (96 calls × 0.523 ms ≈ 50 ms/step).
2. **`Linear` is ~53% of the step gap, not 83%.** Our int8 chain is 750 ms/step deflated against Comfy's ~629,
   so ~121 ms/step of a ~230 ms/step gap. (Comfy's side is carried from 08-13, not re-measured; ours is fresh.)
3. **The kernel ships at 3 stages / 92.16 KB shared, not 2 / 60 KB.** `#define STAGES 3` in the source and ncu
   confirms 92.16 KB. "Shipping at 2 stages, 60 KB" below is wrong. **Deeper staging is already taken.**
4. **The Stream-K note is based on a misread symbol.** `nm -DC` on comfy-kitchen finds **zero** `StreamK`
   symbols; all 576 Sm80 int8 kernels use `GemmIdentityThreadblockSwizzle<1>`. Drop Stream-K from consideration.
5. **"We beat comfy-kitchen's chain on two shapes" was apples-to-oranges** — it compared our fused GEMM+dequant
   (no quant) against their full chain (with quant). Against **our own pair**, cold: attn_qkvo **+7.9%**,
   ffn_up **−2.2%** (a net loss), ffn_down **−26.4%**.

Fresh trio, cold-L2, GPU idle. Note bare cuBLASLt is *faster* cold than the old warm reference (537/545/620 →
568.7/575.3/668.0), so the mainloop deficit is slightly larger than recorded:

| shape | fused mma | cuBLASLt+dequant pair | bare cuBLASLt | fused vs pair |
|---|---:|---:|---:|---:|
| attn_qkvo | 424.1 TOPS | 393.0 | **568.7** | **+7.9%** |
| ffn_up | 388.4 | 397.0 | **575.3** | −2.2% |
| ffn_down | 438.6 | 595.9 | **668.0** | −26.4% |

Live baseline `ltx25_ab.sh HARTSY_INT8_FUSED_MMA`, 4 reps interleaved, private CLI snapshot: **1397.3 ms/step**
shipping vs 1417.0 fused-off. The fused kernel is worth **−19.7 ms/step** (all four pairs same sign, paired
t = 5.2) — better than the −10.5 recorded below.

#### The profile: it is ONE instruction, and cuBLASLt is a like-for-like control

cuBLASLt's kernel demangles to `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16` — **cuBLASLt *is*
CUTLASS at the same tile, warp, instruction, swizzle and stage count as ours.** So the gap is implementation
quality, not configuration, and there is nothing left to copy.

| | ours | cuBLASLt (CUTLASS) |
|---|---:|---:|
| duration | 427.8 µs | 308.5 µs |
| tensor pipe % of peak | **58.3** | **80.9** |
| total L2 sectors | **47.36M** | **33.25M** (storing 2× our bytes) |
| global LD sectors/request | **18.88** | **16.00 = ideal** |
| shared ST bank conflicts | **2,236,416** | **0** |
| stall mio_throttle | **3.48** | 0.33 |
| stall long_scoreboard | 0.17 of 15.8 | 0.36 |
| achieved occupancy | 16.66% | 16.34% |

Duration ratio 1.386× ≈ L2 sector ratio 1.424× — a tight fit for an L2-bound kernel. **The predicted 77% tensor
utilization was wrong; it is 58.3%.** Per-SASS attribution localizes **100% of the excess L2 traffic to a single
instruction**, `LDGSTS.E.BYPASS.128` (the cp.async operand loads): 1.50× ideal sectors and 3.75× ideal shared
wavefronts, with the epilogue STS slab at 8.00×. Everything else is perfect — `ldmatrix` 1.00×, epilogue stores
1.00×, scale/bias loads 1.00× (the fused epilogue is nearly free, adding 119,808 requests for 0.6% of sectors).
ffn_up shows the identical defect (18.882 sectors/request).

**Next step (R1): replace the `BK+16` pad with CUTLASS's unpadded XOR-swizzled shared layout**, epilogue slab
included. Two caveats: the causal link from the 80-byte stride to the *global*-side 1.50× is the strongest
hypothesis with an empirical control, not a proven mechanism — the swizzle is itself the deciding experiment;
and it must preserve the `ldmatrix` conflict-freedom the pad currently buys. **Do NOT** add a 4th stage
(`long_scoreboard` is 0.17 of 15.8 — there is no global latency to hide, and cuBLASLt wins at 3) or try 128×128
(cuBLASLt wins at identical occupancy; 128×128 already measured 288 vs 413). The kernel comment's "BK=128 next
experiment" is refuted — the cuBLASLt control runs BK=64 at exactly 1.00× sectors.

**Honest prize, and it does NOT flip the headline row.** If the mainloop reached cuBLASLt's rate: ~71 ms/step on
already-fused calls plus ~60 ms/step from newly-admissible ffn_up = **ceiling ≈ 130 ms/step ≈ 3.9 s over 30
steps**, which equals the entire measured Linear gap (self-consistent). Against this file's documented 2–3×
optimism, **realistic ≈ 45–70 ms/step ≈ 1.4–2.1 s**. The conv row is 47.40 s against Comfy's 42.48 s, so even
the *ceiling* lands at ~43.5 s — still behind. This work narrows the gap; it does not close it.

### R1 done: the XOR swizzle deletes ALL the excess L2 traffic and buys 4% (2026-08-14)

The unpadded XOR-swizzled operand layout shipped. It did exactly what the profile predicted to the traffic
counters and **almost nothing to the clock**, which is the finding: it removes the entire measured L2 excess and
the kernel gets 4% faster, so **L2 traffic was never what the kernel was waiting on.**

`int8_mma_gemm.cu` now carries two entry points — the swizzled one and the padded one it replaced, kept verbatim
as an A/B control (`HARTSY_INT8_MMA_SWIZZLE=0`). Bodies are duplicated, not templated: a template parameter
threaded through one body perturbs register allocation on *both* instantiations, and the control's whole job is
to reproduce the shipped SASS. It does — 195 regs / 0 spill, unchanged, verified with `ptxas -v`.

**The layout.** `physical byte offset of vector v of row r = r*64 + ((v ^ ((r>>1) & 3)) << 4)`. This is CUTLASS's
`TensorOpMultiplicandCrosswise<8,64>` permutation: that layout's `kFactor` is
`kTileShapeContiguous*kElementsPerAccess/Crosswise = 8*16/64 = 2`, it folds two strided rows into each 128-byte
line, and permutes with `partition_contiguous_residual ^ (partition_strided_residual % 4)` — at this shape,
exactly `v ^ ((r/2) & 3)`. **`v ^ (r & 3)`, which is what you write first, is wrong**: a 64-byte row makes the
bank pattern repeat every *two* rows, so it has period 4 and collides r=0 with r=4. That mistake produces a
kernel that is correct, conflicted, and slower — i.e. a confusing null result, not a crash.

| ncu @ attn_qkvo | padded (control) | **swizzled** | cuBLASLt (CUTLASS) |
|---|---:|---:|---:|
| duration | 437.4 µs | **417.2** | 307.8 |
| tensor pipe % peak | 59.82 | **62.09** | 85.83 |
| **total L2 sectors** | **47,356,502** | **32,023,088** | 33,242,204 |
| global LD sectors/request | 18.88 | **15.12** | 16.00 |
| global LD sectors | 38,458,368 | **30,790,656** | 30,670,848 |
| **L2 throughput %** | **85.25** | **50.42** | 73.01 |
| shared ST bank conflicts | 2,236,416 | 2,236,416 | 0 |
| stall mio_throttle | 3.57 | 2.27 | 0.33 |
| achieved occupancy | 16.69 | 16.66 | 16.75 |

**The premise the change was commissioned on was wrong, and the change worked anyway.** R1 was framed as "the
pad forces a thread map whose 128-bit *global* loads split cache lines." It does not: `load_tile`'s global
address is `gr*K + k0 + (tid&3)*16` and **never involves the shared stride at all**, K and k0 are both multiples
of 64, so every 64-byte run is 64-byte aligned across exactly 2 sectors of one line — already ideal, and the
thread map (4 threads along K × 8 rows per warp) already *was* CUTLASS's `PitchLinearWarpRakedThreadMap`
arrangement. Not one global address changed in this commit.

What the pad actually split was the **instruction**. `LDGSTS` is one op whose *destination* addressing is the
shared side; L1TEX serializes it into wavefronts by shared-bank conflict, and each wavefront runs its own global
tag lookup, so the same sectors are re-tagged and counted again. The banked numbers already implied this and it
went unnoticed: 18.88/16 = **1.18×** per request against **1.50×** on total sectors, with the loads 97.5% L2
hits throughout — the residual is extra *requests*, not extra bytes. Hence the falsifiable prediction, and it
held: **7.7M global sectors and 15.3M L2 sectors vanished without a single global address changing.** Total
global sectors now match cuBLASLt to 0.4%; per-request lands at 16.06 once the epilogue's 119,808 scale/bias
requests are excluded (2,036,736 − 1,916,928 = 119,808 exactly, as recorded above).

**And the 2,236,416 shared-store conflicts were never the operand loads — they are 100% the epilogue slab.**
The profile presented them as part of the same `LDGSTS` defect; they are a second, unrelated one. The arithmetic
closes to the digit: 624 blocks × 8 warps × (4 i × 2 half × 8 j) = 319,488 ideal wavefronts, ×8 = 2,555,904
measured, conflicts = 7/8 = 2,236,416. `LDGSTS` does not retire through the LSU shared-store path these counters
watch, which is why the operand conflicts showed up as global replays instead.

**Trio, cold-L2, all three arms in one process** (the control reproduces its banked 424.1/393.0/568.7):

| shape | **swizzled** | padded (control) | pair | bare cuBLASLt | sw vs pair | **sw vs padded** |
|---|---:|---:|---:|---:|---:|---:|
| attn_qkvo 4992×4096×4096 | **437.5** | 420.1 | 391.0 | 565.7 | +11.9% | **+4.1%** |
| ffn_up 4992×16384×4096 | **411.0** | 403.3 | 397.1 | 575.3 | **+3.5%** | +1.9% |
| ffn_down 4992×4096×16384 | 465.1 | 439.8 | 596.4 | 664.9 | −22.0% | +5.8% |
| audio_attn 256×2048×2048 | 52.1 | 51.1 | 166.5 | 218.1 | −68.7% | +1.9% |
| audio_ffn_up 256×8192×2048 | 205.5 | 202.0 | 256.4 | 325.6 | −19.9% | +1.7% |
| text_kv 512×4096×4096 | 221.7 | 218.6 | 322.4 | 372.0 | −31.2% | +1.4% |

Bit-exact (max abs **0**) against cuBLASLt+dequant at all five shapes including ragged-M, on **both** arms; the
correctness gate was tightened from `maxRel < 2e-2` to `maxAbs == 0` to match what the kernel actually
guarantees. 46/46 `Ltx2` CUDA tests pass.

**Criteria met and missed.** Shared-ST conflicts → ~0: **missed, and the criterion was misattributed** — they
are the epilogue, see (b) below. Sectors/request → ~16: **met** (15.12 raw, 16.06 excluding scale/bias). Tensor
pipe → ~80%: **missed**, 59.82 → 62.09 against cuBLASLt's 85.83. attn_qkvo ≥ 520 TOPS: **missed**, 437.5.
ffn_up ≥ 480: **missed**, 411.0 — but its sign against the pair **did flip**, −2.2% → **+3.5%**.

**CONFIRMED −18.4 ms/step end-to-end** (`ltx25_ab.sh HARTSY_INT8_MMA_SWIZZLE 0 1 4`, arms interleaved, private
CLI snapshot verified to carry both entry points in the PTX *and* the Cuda DLL). All four pairs same sign
(−15.6, −15.7, −25.7, −16.6), **paired t = 7.53**, and the delta is larger than either arm's own spread:

| | mean ms/step | median | range | spread |
|---|---:|---:|---:|---:|
| `HARTSY_INT8_MMA_SWIZZLE=0` (padded) | 1387.8 | 1387.5 | 1381.3–1395.2 | 13.9 |
| **swizzled (ships)** | **1369.5** | 1369.0 | 1365.7–1374.0 | 8.3 |

That is **−0.55 s over 30 steps**. The commissioning bar was −40 ms/step: **missed, by better than 2×.** This is
banked on top of the fused kernel's own −19.7 ms/step, not a re-count of it.

**Why the clock barely moved, stated plainly.** The 08-13 profile read `duration ratio 1.386× ≈ L2 sector ratio
1.424×` as a tight fit for an L2-bound kernel. That was a coincidence. We removed 32% of L2 sectors and 41% of
L2 throughput utilization and got **4.6%** of kernel wall clock; the kernel is now at 50% L2 utilization — less
loaded than cuBLASLt's 73% — and still 1.36× slower than it. **Whatever the remaining gap is, it is not L2
bandwidth, and the "moving fewer bytes through L2 is the only way this goes faster" thesis is falsified by its
own experiment.** The surviving asymmetry is `mio_throttle` 2.27 vs 0.33 with `long_scoreboard` *lower* than
cuBLASLt's and occupancy identical — an issue-side stall on the memory-IO pipe, not bandwidth and not latency.
**That is the open question for whoever goes next**, and it is a different one from the one this file has been
chasing since 08-13. Note the honest bound on the whole line of work: the R1 ceiling was priced at ~130 ms/step
assuming the mainloop reached cuBLASLt's rate; the actual delivery is 18.4, i.e. **7× under its own estimate**,
which is worse than this file's documented 2–3× optimism.

### (b) The epilogue slab swizzle: mechanically perfect, worth exactly nothing (2026-08-14)

Measured separately, on top of R1, and **reverted**. Permuting the slab's 16-byte vector index by its row
(`physical = logical ^ (slabRow & 7)`) is conflict-free by construction and measured exactly that:

| ncu @ attn_qkvo | (a) only | (a)+(b) | cuBLASLt |
|---|---:|---:|---:|
| shared ST bank conflicts | 2,236,416 | **0** | 0 |
| shared ST wavefronts | 2,555,904 | **319,488** (ideal) | 638,976 |
| stall mio_throttle | 2.27 | 2.21 | 0.33 |
| duration | 417.2 µs | 423.0 | 307.8 |

Every conflict gone, wavefronts at the theoretical floor and **half cuBLASLt's**, and the kernel did not get
faster — attn_qkvo 437.5 → 435.9 TOPS, ffn_up 411.0 → 403.2, all against a control arm stable to 0.2% in the
same process. `mio_throttle` moved 2.27 → 2.21, which also says the epilogue's conflicts were not what was
throttling the MIO pipe. Reverted; the shipped PTX is byte-identical to the (a)-only build that was measured.

**This is the useful half of the result.** A bank-conflict counter is not a stall. L1/TEX has headroom here (SM
throughput 60%, L2 50% after the operand fix) to absorb an 8-way store conflict in an epilogue worth a few
percent of the kernel. The 08-13 entry made the mirror-image error in the other direction — it ruled conflicts
out *because* L1/TEX had headroom, which was wrong reasoning that reached the right answer for (b) and the wrong
one for (a). Neither utilization headroom nor a conflict count localizes a stall; only per-SASS attribution and
an A/B do.

### Widening the gate to admit ffn_up: NOT worth measuring, and the ~60 ms/step estimate was inflated

`HARTSY_INT8_MMA_WIDE_GATE=1` exists (loosens `n <= 2k` to `n <= 4k`, admitting ffn_up 4992×16384×4096 and
nothing else; `k <= 2n` still excludes ffn_down) and is **default OFF, deliberately unmeasured**.

The ceiling note above priced newly-admissible ffn_up at **~60 ms/step**. That number assumed the fused kernel
reaching bare cuBLASLt's 575.3 TOPS. It does not — it reaches 411.0. The gate decision is only ever fused-vs-pair
at the rate we actually have:

| | ms/call | × 96 calls/step |
|---|---:|---:|
| fused (swizzled) | 1.662 | |
| cuBLASLt + dequant pair | 1.687 | |
| **actual gain** | −0.025 | **−2.4 ms/step** |
| hypothetical, if fused matched bare cuBLASLt | 1.165 | −50.1 ms/step |

**−2.4 ms/step is ~8× below this harness's 17–20 ms between-run sd**, so a campaign could not resolve it, and
the sign is not safe either: the padded arm reads ffn_up at +1.6% vs the pair here against −2.2% recorded on
08-14, i.e. ffn_up sits on the break-even line within ±4 points of run-to-run. Widening the gate on a margin
that thin is exactly the error that cost +38.7 ms/step when the fused path was first wired in without a floor.
**Leave it off until the mainloop is actually faster** — the gain is a function of the kernel's rate, so this
reopens only if someone closes the `mio_throttle` gap.

### The rewrite: 128×256 tile + shared-memory epilogue — the fused GEMM now WINS (2026-08-13)

Two changes, both taken from comfy-kitchen's shipped configuration, each verified bit-exact (max abs 0) against
cuBLASLt+dequant at every shape including ragged-M:

1. **128×256×64 block tile** (warp tile 64×64, 4×8 mma tiles, 128 accumulator registers, 195 regs/thread with no
   spill, one block/SM). Arithmetic intensity was the limiter, as predicted: 288→325, 316→417, 287→341 TOPS.
2. **Shared-memory epilogue.** mma's fragment layout scatters the output — a warp covered 8 rows × 16 contiguous
   bytes per store, half of a 32-byte sector. Staging a 16×256 slab through shared (8 KB, reusing the dead
   mainloop buffer) and re-emitting it as 32 bytes per lane makes every store fully coalesced: 341→**413 TOPS**
   on attn_qkvo.

| shape | before | **after** | vs pair (3 runs) |
|---|---:|---:|---|
| attn_qkvo 4992×4096×4096 | 288 | **413** | **+5.2 / +5.3 / +5.2%** |
| ffn_up 4992×16384×4096 | 288 | **402** | +2.6 / +4.2 / +0.2% |
| ffn_down 4992×4096×16384 | 316 | **436** | −27.1 / −25.8 / −26.2% |

**The store pattern mattered and I argued it wouldn't.** The reasoning that deferred it — "cuBLASLt writes twice
these bytes as int32 and is still faster, so stores cannot be the gap" — compared *bytes moved* when the right
comparison is *sectors touched*: cuBLASLt's int32 store is perfectly coalesced, this one was not. Between the
two epilogue changes it was worth ~20% on attn_qkvo, more than the tile rewrite.

Also: pipeline depth was re-tested at the new tile and is still not a limiter (2 stages 331/410/343 vs 3 stages
325/417/341, a wash), so the XOR swizzle that 4 stages would have required is not worth building. Shipping at
2 stages, 60 KB.

The gate is `m >= 1024 && k <= 2n && n <= 2k` (`HARTSY_INT8_FUSED_MMA=0` reverts). The K/N split: the fused kernel's win is
the int32 round trip it deletes, which scales with the OUTPUT, while its mainloop still runs ~30% behind
cuBLASLt's — so a deep-K shape spends all its time in the part we are worse at and has almost nothing to save
(ffn_down's dequant is 11% of the pair against attn's 31%). This is a throughput crossover, not an invariant;
re-measure it on a different card.

#### …and end-to-end it is a REGRESSION (2026-08-13)

| | mean ms/step | median | range | spread |
|---|---:|---:|---:|---:|
| `HARTSY_INT8_FUSED_MMA=0` | **1417.8** | 1418.0 | 1409.4–1425.7 | 16.3 |
| `HARTSY_INT8_FUSED_MMA=1` | 1456.4 | 1457.2 | 1442.9–1468.4 | 25.5 |

**+38.7 ms/step SLOWER**, all four paired reps negative (+36.4, +27.2, +33.5, +57.6) — outside the spread, so
real. A kernel that is +5.2% on three hand-picked shapes made the model 2.7% slower.

**A microbenchmark over selected shapes is not evidence about the workload.** The `k <= 2n` gate also admits
every SMALL-m call — audio attention, audio FFN, the text-side k/v projections — where m is a few hundred rather
than 4992. Measured directly rather than assumed:

| shape | m | fused | pair | vs pair |
|---|---:|---:|---:|---|
| audio_attn 256×2048×2048 | 256 | 51.6 | 173.3 | **−70.2%** |
| text_kv 512×4096×4096 | 512 | 219.6 | 331.6 | **−33.8%** |
| audio_ffn_up 256×8192×2048 | 256 | 204.8 | 266.3 | **−23.1%** |
| attn_qkvo | 4992 | 412.1 | 393.5 | +4.7% |
| ffn_up | 4992 | 405.7 | 396.8 | +2.2% |

`audio_attn` is **3.5× slower per call** (0.042 vs 0.012 ms). A 128×256 block tile at m=256 launches two
M-blocks, so the grid covers a fraction of one wave across 128 SMs — structurally the wrong shape for a big-tile
kernel, while cuBLASLt's heuristic picks a small-m tile. Those calls are numerous enough per step to swamp the
video-width gain. This is the scoreboard's own "don't extrapolate one regime's delta onto another" rule,
violated by the gate rather than by the harness.

Fix: `FusedMmaMinRows = 1024` on `UseFusedMmaGemm`, admitting only the regime actually measured (every win is at
m ≥ 1543 — ffn_up's smaller row chunk; audio and text sit at 256–512).

#### With the floor: the regression is gone, and it still does not pay

| | mean ms/step | median | range | spread |
|---|---:|---:|---:|---:|
| fused off | **1410.6** | 1408.4 | 1404.1–1421.6 | 17.5 |
| fused on, m ≥ 1024 | 1417.5 | 1416.3 | 1409.6–1427.9 | 18.3 |

**+6.9 ms/step**, all four pairs same-sign (+5.5, +6.7, +9.1, +6.3). The floor recovered 32 of the 38.7 ms, so
the small-m diagnosis was right — but +4.7%/+2.2% on the eligible shapes should have netted roughly −10 ms/step,
and it nets +7. That gap is what sent the harness to cold L2 (below), which found the remaining bad shape.

#### With the symmetric gate: −10.5 ms/step, and it SHIPS ON

| | mean ms/step | median | range | spread |
|---|---:|---:|---:|---:|
| fused off | 1417.4 | 1417.2 | 1407.7–1427.4 | 19.7 |
| **fused on** (`m ≥ 1024`, `k ≤ 2n && n ≤ 2k`) | **1406.8** | 1403.0 | 1401.3–1419.9 | 18.6 |

**−10.5 ms/step**, all four pairs same-sign (−7.5, −6.1, −17.5, −11.1), **paired t = 4.15**. Default ON;
`HARTSY_INT8_FUSED_MMA=0` reverts.

**Read the arc, not just the result** — the same kernel measured +38.7 / +6.9 / −10.5 ms/step under three gates.
The kernel never changed; what changed was which shapes it was allowed to serve, and each correction came from
measuring a regime instead of extrapolating into it. The final win is also smaller than the ~14 ms predicted from
per-call microbenchmark deltas, so even the corrected harness still over-predicts in situ.

**Why isolated and deployed disagreed, and the harness fix.** `BestMs` reran the kernel against the SAME resident
operands, so the weight stayed hot in the 4090's 72 MB L2 across every rep — and this kernel is precisely the
more L2-hungry one, which was the whole arithmetic-intensity diagnosis. A real step touches each weight once,
cold. The harness now rotates over enough weight buffers to exceed L2 (`ColdBuffers`, ≥192 MB working set), and
that immediately changed a conclusion:

| shape | warm | **cold** |
|---|---|---|
| attn_qkvo 4992×4096×4096 | +4.7% | **+6.1%** |
| **ffn_up 4992×16384×4096** | **+2.2%** | **−7.0%** |
| ffn_down | −26.8% | −28.1% |
| audio_attn / text_kv / audio_ffn_up | −70 / −34 / −23% | −69 / −33 / −23% |

**ffn_up flipped from a win to a loss** — its 64 MB weight is the one warm L2 flattered most, and it accounts
for the +6.9 ms/step: the gate was admitting a loser. **Min-of-batches over fixed buffers systematically
flatters cache-hungry kernels.** Only the roughly-square attn shape survives cold, so the gate became symmetric
(`k <= 2n && n <= 2k`): a wide-N shape carries proportionally more weight traffic, which is what this kernel is
worst at. It also sharpens the bar for the mainloop — cuBLASLt's ~569 TOPS under COLD L2, not 413 under warm.
The upside if it gets there is the full ~30% dequant share, which is why the kernel is kept.

**A bug the bit-exactness gate caught, worth repeating:** the first shared-epilogue draft was *faster* (attn
422 TOPS) and wrong — `int4` is 8 halves, and the write-out loop gave each thread a 16-half span, so one `int4`
store silently left half of every slab row unwritten. It would have shipped as subtly corrupted video. Always
gate a kernel rewrite on exact agreement, not on a tolerance.

**And a second one only the Linear-level test could catch.** `Int8MmaGemmTests` calls the kernel directly and
never passed a bias, so it reported max abs 0 at 20M elements while the two paths actually disagreed whenever
bias was present: nvrtc contracted `v = d·rowScale·wScale; v += bias` into an FMA differently in each kernel,
giving 1 ulp in F32 and flipping ~2 outputs per 500k across an F16 rounding boundary. Spelling the epilogue as
an explicit `fmaf(acc·rowScale, wScale, bias)` matches the reference's contraction and restores exact equality.
Two lessons: **test the path the caller takes, not just the kernel**, and **identical source-level operator
order does not imply identical floating point across separately compiled kernels.**

**Verified end-to-end:** with the fused path forced on, all 97 frames and `audio.wav` are byte-identical to the
control run — which covers every shape the unit tests do not enumerate. Frames inspected (coherent lighthouse
at sunset, waves progressing, no artifacts); audio 4.04 s, peak 1.2% FS, −54.1 dBFS, 0 clipped, matching every
other run today.

### Host-side overhead is real, measured, and does not reach the wall clock (2026-08-13)

Timing the driver calls the resident-int8 chain makes per `Linear`, in isolation (`Int8ResidentHostCostTests`):
`cuMemGetInfo` **5.229 µs/call** and a pool alloc+free pair **1.208 µs**. At ~2,700 resident-int8 Linears per
step that is ~14 ms/step of `cuMemGetInfo` and ~16 ms/step of pool churn — apparently free money.

It is not. Caching cuBLASLt's descriptor, layouts and heuristic pick (a bigger target: ~45k host calls/step)
measured, 4 interleaved reps per arm:

| | mean ms/step | median | range | sd |
|---|---:|---:|---:|---:|
| per-call objects (as shipped) | 1428.5 | 1427.7 | 1410.3–1448.3 | 15.5 |
| cached descriptors | 1428.4 | 1432.3 | 1408.0–1440.9 | 14.3 |

**Paired mean +0.1 ms, t = 0.02.** A dead null, and the mechanism explains it: `nvidia-smi dmon` has always shown
SM at 99–100% on this path, so host time spent *queuing* work is hidden behind GPU execution and never reaches
the wall clock. The cache was reverted rather than carried — it added a lock, two dictionaries and object
lifetime management for zero gain. The free-VRAM coast and the persistent-scratch idea (the other two legs of the
same plan) were dropped unbuilt for the same reason.

**Do not re-chase host-side launch/allocation cost on this path while SM occupancy is saturated.** The way to
falsify this conclusion is to show occupancy is *not* saturated, not to re-measure a different host call. Note
also that an earlier do-not-re-chase note in `Int8GemmExecutor` asserted this same null from one run per arm —
which established nothing. The table above is the evidence.

### comfy-kitchen's int8 GEMM, read off its shipped binary (2026-08-13)

The gap on `Linear` has been attributed to "their CUTLASS beats our cuBLASLt" without knowing *what* they run.
It is knowable: `comfy_kitchen/backends/cuda/_C.abi3.so` (in the ComfyUI venv) exports 1,169 symbols from
`cutlass_gemm_int8.cu`, and the mangled names carry the whole template configuration. Demangled, the LTX shapes
route to:

```
cutlass::gemm::kernel::DefaultGemmWithVisitor<
  int8_t, RowMajor, align 16,  int8_t, ColumnMajor, align 16,  bf16|half|float, RowMajor, align 8,
  int32_t accum, float compute, OpClassTensorOp, Sm80,
  ThreadblockShape<128,256,64> | <128,128,64>,  WarpShape<64,64,64>,  InstructionShape<16,8,32>,
  epilogue::threadblock::TreeVisitor2x< VisitorAuxStore, VisitorCompute<plus>,
      VisitorCompute<multiplies>, VisitorAccFetch, VisitorColBroadcast, VisitorRowBroadcast >,
  ThreadblockSwizzleLeanStreamK,  Stages = 3 | 4,  OpMultiplyAddSaturate >
```

Everything we inferred is confirmed, and the remaining unknowns are now specifics rather than guesses:
- **The dequant IS the epilogue.** `VisitorColBroadcast` is the per-output-column weight scale,
  `VisitorRowBroadcast` the per-row activation scale, `VisitorCompute<multiplies>` then `<plus>` the bias, and
  `VisitorAuxStore` writes bf16/half/float straight out. **The int32 accumulator never reaches HBM** — which is
  precisely the ~165 ms/step round trip we cannot remove with cuBLASLt.
- **The same `mma.m16n8k32` instruction** our own kernel already uses. The instruction is not the gap.
- **Where our kernel actually differs**: they run **3–4 pipeline stages** (ours, at the time: 2), a
  **128×256×64** tile as well as 128×128×64 (ours: 128×128×64 only), and **alignment-16 vectorized operand
  loads**. They also ship a full tile ladder — 16×64×64, 16×128×64, 32×64×64, 32×128×64, 64×64×64, 64×128×64,
  64×256×64, 128×128×64, 128×256×64, 64×64×128, 128×256×128 — and select per shape, where ours has one tile for
  everything.

That was the roadmap for `Kernels/dequant/int8_mma_gemm.cu` (then 268 TOPS, wired to nothing, needing ~383 to
beat the cuBLASLt+dequant pair and ~550 to match them). **Stages and the 128×256 tile have both been taken —
see the sections above; the rest of this list is stale, and one item of it is refuted.**

> **CORRECTION — `ThreadblockSwizzleLeanStreamK` was a misread symbol, and Stream-K is dropped entirely.**
> `nm -DC` on comfy-kitchen finds **zero** `StreamK` symbols; all 576 Sm80 int8 kernels use
> `GemmIdentityThreadblockSwizzle<1>` — the same fixed swizzle we already run. The follow-on claim that
> "Stream-K matters at LTX's shapes because a fixed swizzle leaves SMs idle on the tail wave" therefore rested
> on nothing, and is withdrawn: cuBLASLt beats us at an **identical achieved occupancy** (16.34 vs 16.66), which
> is the opposite of a tail-wave problem. Do not re-propose it.

### Profiling the fused GEMM: the limiter is L2 BANDWIDTH, not compute (2026-08-13)

**Nsight Compute is installed on this box** — `~/.local/cuda-tools/nsight-compute/opt/nvidia/nsight-compute/2026.2.1/ncu`,
not on PATH — and profiling counters are already unlocked for non-root (`/etc/modprobe.d/nvidia-profiling.conf`
sets `NVreg_RestrictProfilingToAdminUsers=0`; `/proc/driver/nvidia/params` shows `RmProfilingAdminOnly: 0`). No
sudo, no install. Several hours of this file's guesswork could have been one profiling run.

At attn_qkvo (4992×4096×4096):

| metric | value |
|---|---:|
| Memory (L2) throughput | **85.2%** |
| Compute (SM) throughput | 57.6% |
| DRAM throughput | 17.9% |
| L1/TEX throughput | 33.7% |
| register / shared spilling | 0 |

**The kernel is L2-bandwidth bound.** That single fact retroactively explains why every instruction-level and
occupancy change in this file's history came back at 1–6%: they were all secondary to a constraint nobody had
measured. It goes faster only by moving fewer bytes through L2.

**The traffic accounting closes exactly**, and it says where the remaining gap is:

| | sectors | bytes |
|---|---:|---:|
| ideal operands (A read by N/BN blocks, B by M/BM) | 30.7M | 981 MB |
| ideal output | 1.3M | 41 MB |
| **ideal total** | **31.9M** | |
| **measured** | **47.4M** | |
| **excessive (uncoalesced)** | **15.3M** | **491 MB wasted** |

Two consequences. First, **the tile is already at the hardware ceiling**: 256×256 would cut ideal traffic to
663 MB, but it needs 256 accumulator registers per thread, and at 512 threads the register file caps you at 128
— the accumulators alone are 128. 128×256 at 256 threads is the maximum, so the 981 MB floor cannot be lowered
by tiling. Second, that makes the **491 MB of *wasted* traffic the entire remaining gap** — eliminating it at
constant bandwidth lands near 620 TOPS, past cuBLASLt's 569.

**Two claims in the kernel's own comments were wrong, and only the profiler could show it.** The header asserted
the 80-byte shared stride was "conflict-free" for ldmatrix (it is not — 45% of shared wavefronts are excessive)
and that the epilogue was "fully coalesced" (it was writing 32 B per lane as two 16-byte stores at 32-byte
stride, half-using every sector). Fixing the epilogue to one `int4` per lane, with a warp covering a whole
512-byte output row, is bit-exact and moved **attn_qkvo 413.5 → 422.4 TOPS (+6.1% → +8.0% vs the pair)** and
**ffn_up 367.7 → 384.9 (−7.0% → −1.3%)**. It removed only 1.3M of the 16.6M excessive sectors though — a real
instance, not the main one, which is still unlocalised.

**Also do NOT act on ncu's occupancy advice here.** It reports 16.7% occupancy and an "83% local speedup"
against it. The 128 accumulator registers that pin this to one block per SM are the same thing that buys the
arithmetic intensity; raising occupancy means a smaller tile, which raises L2 traffic, which is the actual
constraint. A profiler names the limiter — it does not know the design.

### The per-head gate, folded into the activation quantization (2026-08-13) — SHIPPED, e2e UNMEASURED

`Ltx2HeadGate` was a full read AND write of an attention output — 81.8 MB per video-width call at ~581 calls
per step — purely to scale it, immediately before `to_out`'s Linear read the same tensor again to quantize it.
The gate now folds into that quantization pass (`convrot_quant_rowwise_gated_f16`), so it costs no traffic of
its own; `IBackend.LinearHeadGated` carries it, with a default that runs the old gate-then-Linear sequence.
Kill switch `HARTSY_LTX2_GATEFUSE=0`.

**Bit-identical** to the separate pass (0 of 163,840 F16 words differ, `GroupedLinearTests`). That is by
construction and deliberately fragile-looking: the fused kernel reproduces the separate pass's **f16 store** and
keeps its `(x · 2) · sig` multiply association rather than the algebraically equal `x · (2 · sig)`. Either
shortcut would silently change every attention output in the model.

**CONFIRMED −21.6 ms/step** (4 interleaved reps, all pairs same-sign: +16.0, +22.3, +23.1, +25.1; **paired
t = 11.0**), which matches the traffic argument of ~23 GB/step almost exactly — the rare case this file's
arithmetic did NOT over-predict.

| | mean ms/step | median | range | spread |
|---|---:|---:|---:|---:|
| `HARTSY_LTX2_GATEFUSE=0` | 1427.6 | 1428.2 | 1410.9–1443.3 | 32.4 |
| **fused** | **1406.0** | 1405.5 | 1394.9–1418.2 | 23.3 |

**The first attempt at this campaign reported it as null**, from the single pair that completed before a
concurrent session rebuilding the shared `bin/Release` mid-run corrupted it. The change was shipped as
explicitly UNVERIFIED rather than claimed, and the re-run against a private CLI snapshot
(`LTX25_BENCH_CLI`) is what recovered the real result. Two lessons: a corrupted campaign can produce a
plausible null as easily as a plausible win, and "ship it but say it is unmeasured" was the right call over
both "claim the win" and "drop the change".

⚠️ These absolute numbers sit on a working tree carrying another session's uncommitted `AudioCfgEulerStep`
edit, so they are **not continuous** with the 1417.4 / 1410.6 baselines earlier in this file. Both arms share
it, so the delta is clean; the absolutes are not comparable across those campaigns.

**Concurrent-session hazard, now closed.** Two sessions sharing this repo collide on BUILD OUTPUT as well as on
the GPU, and the build collision is the quieter one: `HartsyInference.Video.dll` was rebuilt between rep 1 and
rep 2 of a campaign, silently swapping the binary under an A/B. The free-VRAM guard in `ltx25_bench.sh` catches
GPU contention loudly; nothing caught this. `ltx25_bench.sh` now takes `LTX25_BENCH_CLI` to point at a private
snapshot of the Release output — use it whenever anyone else might be building.

### ⚠️ The harness's noise floor — read before trusting any delta in this file (2026-08-13)

**The same build, same seed, run back to back, spreads ~25 ms/step; sd across 4 reps is 17–20 ms.** Measured
directly: two consecutive ungrouped runs gave 1459.6 and 1435.2, and a 4-rep arm ranged 1437.7–1473.7.

The variance is **per-process, not thermal**: steps *within* one run vary by only ~±5 ms (1452/1453/1456/1460/1461
in sequence), so it is not drift during a run — each process settles into its own level and stays there. Prime
suspect is transient-pool address layout changing L2 set conflicts for the big activations; cuBLASLt algo choice
is ruled out, since the heuristic never sees pointers and is deterministic for a shape.

Consequences, which apply retroactively:
- **A single run per arm cannot resolve anything under ~50 ms.** Several deltas recorded earlier in this file
  were accepted on one run per arm — notably fused ConvRot+quant (−14 ms) and the F16 rope tables (−5 ms), both
  of which are *inside* the noise band and are therefore unestablished, not disproven. Token-major (−39.5 ms) is
  marginal at n=1; grouped Linear re-measured at n=4 came out **larger** than its n=1 estimate, not smaller.
- **`ltx25_ab.sh` now exists** and is the way to measure a change: alternating arms, N reps each, reporting
  mean/median/range per arm so the delta can be compared against the spread. It also holds `swarmui.service`
  down for the whole campaign — restarting it per rep costs ~40 s each and left the unit `failed` when systemd's
  stop timeout fired while the GPU was busy.
- Budget accordingly: a credible 4-rep campaign is ~20 minutes. Do not spend it on a change predicted to be
  worth less than the spread.

### Grouped Linear — the redundant quantize pass, collected (2026-08-13)

The quantize pass was structurally redundant, not inefficient: `LtxVideo2Attention` issued `to_gate_logits`,
`to_q`, `to_k` and `to_v` as four separate `Linear` calls, and in SELF-attention all four take the *same*
`qInput`, so the same `[4992, 4096]` activation was ConvRot'd and row-quantized **four times** for one answer.

**Shipped:** `IBackend.LinearMulti(input, ops)` — several projections of one input. The CUDA override resolves
every op's device pointers up front, runs the rotate+quant **once** per row chunk, then loops GEMM+dequant per
weight against the shared `(pAct8, pRowScale)`. `LtxVideo2Attention` groups by input: self-attention takes all
four in one group, cross-attention (text, a2v/v2a) still shares gate+q and k+v. Ops the resident int8 chain
cannot serve — an F16 weight, a different ConvRot group, a different k — fall out to an ordinary `Linear` each,
so a mixed group is partitioned, never refused. Kill switch `HARTSY_GROUPED_LINEAR=0`.

Measured with the interleaved protocol (`ltx25_ab.sh`, 4 reps per arm, arms alternating), NOT one run per arm:

| | mean ms/step | median | range | sd |
|---|---:|---:|---:|---:|
| ungrouped (`HARTSY_GROUPED_LINEAR=0`) | 1463.3 | 1470.9 | 1437.7–1473.7 | 17.1 |
| grouped | **1421.7** | 1420.3 | 1401.4–1444.7 | 20.4 |

**−41.6 ms/step (2.8%)**, and all four paired reps favour grouped (+38.5, +29.0, +69.2, +29.7; paired t = 4.4).
Quant launches/step 3021 → 2243 (−26%), and `Int8.Quant` 199.9 → 165 ms/step in the sync-inflated profile, so
the win lands exactly where predicted and by the predicted mechanism.

An earlier revision of this section reported **−31 ms** from one run per arm. That was not wrong so much as
unresolvable: the harness's between-run spread is larger than the effect (see below). The projection was still
optimistic — ~65 ms predicted against 41.6 delivered — because the saved passes are on average narrower than
the ones kept (audio- and text-width groups are cheap), so counting redundant *launches* overstates redundant
*bytes*. Treat a launch count as an upper bound and a byte count as the real one.

Route note: the two candidates written up before implementation were a quantize-once cache and QKV weight
fusion at load. **Weight fusion was killed by traffic arithmetic before any code was written** — the three
outputs of a fused `to_q|to_k|to_v` need a split copy (read 123 MB + write 123 MB per self-attention) to save
122 MB of quant traffic, a net loss. Grouping by input keeps the whole win with no weight-layout change, no
output slicing, and no cache-invalidation hazard, since the shared buffer lives exactly as long as the call.

Verified bit-exact, twice over: `GroupedLinearTests` asserts byte-identical F16 outputs against per-op `Linear`
(including a mixed group and the n=32 gate shape), and the end-to-end control run is byte-identical too — all
97 PNG frames and `audio.wav` md5-match the ungrouped run. Frames re-inspected: coherent lighthouse at sunset,
keeper visible, waves progressing naturally, no artifacts.

The kill-switch control shares the refactored `RunResidentInt8`, so it does not by itself prove the *refactor*
byte-exact against pre-refactor code. It is covered transitively: the run's `audio.wav` md5 (`2c7a96c6`) matches
the `frames_tokenmajor` run from before the refactor, and audio and video decode from the same joint latent.

**This was the last INPUT-side lever inside `Linear`** — the redundancy that remained was in the activation
preprocessing, and it is now collected. What remains is ~477 ms of GEMM at the hardware wall, ~165 ms of dequant
(the int32 round trip, cuBLASLt-blocked), and ~137 ms of quant that is no longer redundant. Output-side fusion
is still open and is NOT cuBLASLt-blocked, because the dequant epilogue is our own kernel: it writes
`[s, inner]` f16 to HBM and `Ltx2QkNormRopeTokenMajor` reads it straight back for q and k, so folding norm+rope
into the epilogue would delete that round trip (~80 MB per tensor per attention). Feasible only while N-tiling
stays off, since RMSNorm needs a whole row. Size it at ~30 ms by traffic and then halve it — every projection
in this pass ran 2-3× optimistic.

### Two more int32-round-trip attacks, both refuted (2026-08-13)

- **N-tiling the GEMM so the int32 accumulator stays in L2** — the obvious cheap alternative to a fused
  epilogue, and the axis the earlier *row*-chunk experiment did NOT test (tiling M shrinks the GEMM's m; tiling
  N does not). Built it: `w8a8_dequant_bias_strided_f16/f32` writes a column slice into a wider destination row,
  and the resident-int8 path loops over N with the weight sliced by pointer offset ([N,K] row-major makes an
  output column slice a contiguous weight row range). **Monotonically worse:** no tiling **1457.2** ms/step,
  N=2048 **1474.0**, N=1024 **1596.3** — even at 1024, where the tile is 20 MB and unambiguously L2-resident.
  Extra launches plus smaller-n GEMM efficiency cost more than the round trip. Left in, defaulted OFF
  (`DefaultInt8ColChunk = int.MaxValue`, override `HARTSY_INT8_N_CHUNK`).
- **F16 RoPE cos/sin tables** — half the fused QK kernel's traffic is its F32 tables, so halving that looked
  free. It does speed the kernel: **0.189 → 0.156 ms (−17.5%)**. But that kernel is only ~2.5% of the step, so
  end-to-end it is **1461.0 → 1456.0 ms (−5 ms, inside run noise)** for a **real output change** (SSIM 0.9956
  across the clip, frame inspected and coherent). Not worth a numerics change; left in, **OPT-IN** via
  `HARTSY_LTX2_ROPEF16=1`.

Both were projected at 20–25 ms and delivered ~0 and ~5. Together with the mma kernel above, that is three
attacks on the int32 accumulator: cuBLASLt refuses to fuse it, tiling for L2 makes it worse, and a
hand-written fused GEMM is 18–48% slower than cuBLASLt+dequant. **The ~203 ms/step is real but currently
unreachable without a CUTLASS-class fused int8 GEMM.** That is the honest state of the biggest lever.

⚠️ **Unresolved measurement drift.** The token-major result was measured at **1435.7** ms/step; after the two
changes above landed (both now defaulted off) the same workload measures **1472.3**. The GPU was idle and cool
(41 °C, no throttle), so it is not thermal. The prime suspect is the F16-table work's refactor of
`dit_f16.cu`'s two rope kernels into one shared `template<TabT, bool HeadMajor>` body — the F32-table path may
have compiled worse. **Check that before trusting any step number against 1435.7.**

**Do not re-chase**, all closed by measurement this session:
- **cuBLASLt algo selection for int8** — caching the per-shape descriptors + heuristic result (removing ~45k
  redundant host cuBLASLt calls per step) is within run noise, and autotuning its 16 heuristic candidates on
  real buffers is *worse* (1543 vs 1510 ms/step): a 3-rep timing is noisy enough to lock in a bad algo for the
  process lifetime. Its first pick is already its best. Reverted.
- **Fusing ConvRot into the row-wise quantizer** — done and kept (`convrot_quant_rowwise_f16/f32`, 7 bytes per
  element down to 3, **bit-identical**, pinned by `ConvRotFusedQuantTests` against the unfused pair). Worth
  only **−14 ms/step**: the quantized activation is reused across the q/k/v/gate Linears that share an input,
  so this path runs far fewer times than a naive per-Linear count suggests. Do not expect more here.
  NB `group` must be a power of FOUR (the Hadamard is `kron(h4,…,h4)`); 128 silently produces garbage in the
  rotation stage loop, which is why `HasFusedConvRotQuant` rejects it.
- **Batching the CFG pair into one batch-2 forward** — the premise was that m=4992 is too small and the
  attention projections' 90%-of-peak is an m artifact. It is not. `Int8ConvRotGemmThroughputTests` at
  m=9984: `attn_qkvo` **1.244 ms / 269.2 TOPS / 82%** against 2×0.564 = 1.128 ms at m=4992 — batching makes
  the attention projections **10% slower**, not faster. The FFN shapes are a wash (107%/107% vs 107%/103%).
  The other two claimed wins do not survive either: row-wise batching leaves every elementwise glue kernel's
  total bytes unchanged (it halves launch *count*, not work — and `GatedResidual` at 62 ms/step is already
  within ~20% of its bandwidth roofline), and halving weight HBM traffic cannot be banked on top of a GEMM
  chain already measuring >100% of compute peak. Large refactor (batch-2 SDPA, `s % S` rope, batch-strided
  head-major emit, VRAM at 23878/24564), negative-to-negligible payoff.
- **Making `ltx2_qk_norm_rope_headmajor` cheaper per-element** — rewritten to hold the RoPE partner lane in a
  register (one thread per pair) and reduce via warp shuffles, dropping the 17 KB dynamic-shared request that
  capped it at ~5 blocks/SM. Passed the unfused-sequence test at both dtypes and bought **7% on the kernel
  (0.075 → 0.069 avg) and 0 ms of step time** (1509.4 vs 1510.6). Reverted. At ~428 GB/s of the 4090's ~1008
  it is not bandwidth-bound: the limiter is the **scattered write** `out[(h·seq+s)·headDim+d]`, which sprays
  32 separate 256 B regions across a 40 MB tensor per token. Anything that does not fix the head-major
  scatter will not move this kernel.
- **L2-sizing the int32 accumulator** — shrinking the row chunk so it fits L2 is monotonically *slower*
  (256/64/32/16 MB → 1768/1854/1920/2070 ms per step); the extra chunks cost more in launches and small-m
  GEMM efficiency than the round trip costs in bandwidth.
- **SageAttention INT8 at this sequence length** — `HARTSY_SAGE_F16_MIN_SKV=1024` to force it on at
  skv 4992 gives 1777 ms/step vs 1749 ms for the cuDNN fused flash path. The default 12288 gate is right.
- **CUDA graphs / launch-overhead work** — the step is GPU-bound, not host-launch-bound (`nvidia-smi dmon`:
  SM 99–100%, memory controller 78–80% throughout the denoise).

## SeedVR2-3B restoration — bring-up baseline vs Python reference (2026-08-01)

Not a T2V row: restoration (`hartsy restore`), measured at the E2E-parity operating point — 9-frame
Big Buck Bunny 360p clip, 640×360-area output, 4090, N=5, 95% CI (Student-t df=4). Correctness is
settled separately (C# output ≡ Python at SSIM 0.99950 with injected reference noises — see
`PARITY_VERIFICATION.md`); this row is the SPEED baseline for the future perf pass.

| Impl | Shape | Wall (9 frames) | s/frame | Peak VRAM |
|---|---|---|---|---|
| Python reference | **warm in-process**, bf16, causal slicing, dit-offload | 1.45 s ± 0.09 | 0.161 | 17.6 GiB |
| HartsyInference (bring-up) | **cold CLI e2e** (process + 13.6 GB fp32 mmap load + ffmpeg decode/mux), fp32, host-math DiT | 44.00 s ± 0.27 | 4.89 | ~16 GiB |

**Read the caveats before quoting a ratio:** the runs differ in warmth (in-process warm vs full CLI
cold start), dtype (bf16 vs fp32), and DiT execution (torch device kernels vs the deliberate host-math
bring-up shape — window gather/scatter, RoPE, qk-norm, AdaSingle all CPU-side). From the E2E gate run,
pipeline-only C# time at this shape is ~52.7 s *including first CUDA touch*; the perf-pass levers
(device window pack/unpack, GPU RoPE à la `HunyuanImageRope.ApplyGpu`, F16 activations, graph capture)
are enumerated in `MODEL_STATUS_VIDEO.md` §SeedVR2 follow-ups. Matrix-scale numbers (25f, 960×540-area):
~14 s/frame, 17.1 GB peak, 7/7 clips green.

## MiniMax-H3 fl2va — DiT quantization builds (2026-08-12)

Not a T2V row: in-engine **step time**, not SwarmUI e2e. Same weights published at three precisions by
Comfy-Org, so this is a build-vs-build comparison, not an engine-vs-engine one. Workload is
`benchmarks/minimax_h3/h3_bench.sh`'s gold baseline — 141 frames @ 512×288, seed 1, 4090 (nvidia-smi
index 1), SwarmUI stopped, mean of steps 4..N.

| DiT build | File | Step time | Residency | Date |
|---|---|---:|---|---|
| `pruned_int8_convrot` | 20.97 GB | **5.807 s** (n=27, 5.781–5.868) | fully resident, 20.96 GB weights + 1.78 GB reserve vs 24.22 GB free | 2026-08-12 |
| `pruned_fp8_scaled` | 20.96 GB | 8.6 s | fully resident, 22.5 GB | 2026-08-05 |
| `fl2va_bf16` | 66.28 GB | ~90 s | streams per call | 2026-08-05 |

**Caveat on the fp8 row:** it is carried forward from its own bring-up session, not re-measured beside the
int8 run. Both used the same script and gold workload, but not the same day or the same driver/VRAM state,
so read the int8-vs-fp8 gap as indicative until the two are run back to back.

The int8 build is INT8 tensor-core (IMMA) work — activation ConvRot, per-row dynamic int8 quant, cuBLASLt
IMMA, dequant epilogue — against a weight that never leaves int8. Correctness is settled separately: the
GEMM chain matches comfy-kitchen's eager reference at relL2 5.1e-8–2.7e-7 with F32 activations
(`Int8ConvRotCudaParityTests`), and the generation's frames were inspected. Note that step time is
**insensitive to the row-chunk size** the path picks from free VRAM: int32 accumulation is exact and
order-independent, so chunking changes neither the result nor the arithmetic, only the transient footprint.

## Notes

- **LTX-2.5 22B dev is the first LTX-2.5 Comfy-vs-Hartsy row** — 1.34× slower, Hartsy 56.62 s vs Comfy
  42.25 s, quality-matched (both prompt-faithful, frames inspected on both sides). The first measurement of
  this row read 117.13 s and 2.77×; that was a stale deployed build, and the perf pass that followed it is
  broken down in the dedicated section above. Benchmarking it required updating the live ComfyUI backend (was v0.28.0, no LTX-2.5
  support at all) to v0.32.0, and a temporary SwarmUI `SDModelFolder` root addition + two service
  restarts to route the split DiT/TE/VAE repack through Comfy's separate-VAE loader path instead of its
  bundled-checkpoint assumption — reverted after the benchmark, no lasting server config change.
- **Wan 2.1 T2V 14B is the only video model at parity with ComfyUI** (30.58 s vs 30.62 s) — first video
  model to catch Comfy. Per the campaign write-up it has reached its
  fp8 compute floor (CUDA-graph and batched-CFG closed out as dead ends with evidence), so parity is
  where it is expected to stay absent a fundamentally faster fp8 GEMM.
- **ComfyUI column is carried forward from the 2026-07-03 head-to-head** for every model that has one —
  ComfyUI's own performance did not change across engine versions, only the Hartsy side did (per the
  07-11 file), so reusing the 07-03 Comfy numbers against the 07-11 Hartsy numbers is valid.
- **Wan2.2 TI2V-5B step-cache is opt-in and NOT the shipped-default number in the table above.**
  `2026-07-22_accel_stepcache_wan_4090.md` measured
  1.18–1.55× speedups (44.1–57.7 s vs a 68 s warm baseline) via `HARTSY_STEP_CACHE`, but on a *different*
  workload (832×480, 33 frames, 50 steps — not the standard 512×320/25f/20-step scoreboard workload, so
  the 68 s baseline there isn't directly comparable to the 15.5 s row above). More importantly, the
  benchmark's own verdict is negative for the pinned gate: no threshold holds SSIM ≥ 0.95 (best case 0.88
  at 1.18×), because Wan's 50-step UniPC trajectory is chaotically sensitive to any reuse — outputs stay
  coherent and prompt-faithful but diverge from the un-cached seed. The engine ships this **default OFF**
  as a "fast non-reproducible sampling" opt-in, not a transparent accelerator; `PERFORMANCE.md` (retired) §1's
  default-on feature table and §6 experimental-switch table both omit `HARTSY_STEP_CACHE` entirely,
  confirming it is not part of the standard profile.
- **HunyuanVideo 13B and Kandinsky-5.0 T2V Lite have no ComfyUI baseline yet** — per the 07-11 scoreboard
  these are still open rows pending a Comfy Hunyuan T2V workflow and in-engine text-encoder wiring
  (Kandinsky-5) respectively. The numbers shown are engine-side e2e wall-clock only, from their
  2026-07-02 bring-up benchmarks (not re-measured on a later engine build in these sources).
  HunyuanVideo runs at ~2.15 s/step via
  fp8-resident weights + GPU RoPE + `HARTSY_FP8_NATIVE`.
- **LTX-2.3 22B has no comparable Comfy workflow on this box**, so its row is internal-progress-only:
  451 s (2026-07-03) → 95.5 s (07-08) → 42.3 s (07-11), a 10.7× cumulative improvement, block-swap-bound
  (streams ~19 GB/forward on a 24 GB card).
