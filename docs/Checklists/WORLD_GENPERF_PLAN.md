# World / Interactive gen-perf — audit + optimization plan (2026-07-12)

Applying the image/video/audio gen-perf playbook (`VIDEO_GENPERF_PLAN.md`, memories `radiance-perf-pass`,
`cuda-graph-step-capture-recipe`, `vae-host-loops-hidden-20s`, `image-genperf-host-glue-wins`) to the
**interactive world models**. Same principles, but the metric is **step latency / FPS** (real-time AR
loops), not seconds-per-clip. Target for interactive models is 25–40 FPS (Phase 10 gate:
MG2 25 FPS @540p / MG3 ≥10 FPS @720p on a 4090).

## The playbook, distilled (order it wins)

1. **Wall-clock phase probes FIRST** — `[world-phase]` logs (VAE-encode seed / per-step DiT / ActionModule /
   VAE-decode / history-roll+FOV / integrate+Plücker). No world model has probes yet.
2. **GPU-residency ports** — replace every host `DataPointer` loop in DiT blocks / ActionModule / VAE with
   existing device ops (`AffineBroadcastLastDim`, `GatedResidualLastDim`, `SliceRows`, `WanRmsNormChannel`,
   `Modulate`, `LayerNormNoAffine`). Bit-identical, zero numeric risk, biggest wins.
3. **cuDNN fused SDPA** (`HARTSY_SDPA_CUDNN=1`, `allowF16`) — mask-null only, D∈{64,128}. Spatiotemporal
   axial attn + ActionModule mouse-self / keyboard-cross attn are the sites.
4. **F16 activations** (`HARTSY_DIT_F16`) — halve bandwidth on norms/elementwise; per-arch opt-in.
5. **CUDA-graph step capture** (`HARTSY_DIT_GRAPH`) — AR loops are FIXED-shape, single-forward (distilled
   1–4 step), resident → **the best graph-capture regime in the whole engine**. Needs a drain-free step
   (device sampler, fixed boundary buffers).
6. **Caching + residency** — `HARTSY_KEEP_MODELS`, RoPE-table memo, per-frame timestep-embed cache,
   image-cond/CLIP cache across segments, FOV-selection GPU kernel (reference: single biggest FPS knob).
7. **Async hygiene** — no per-step `TrimMemoryPool`, no pageable per-step H2D, no mid-forward host reads.

## Inventory — what we support

| Model | Params | Status | Fits 4090 (24 GB)? | Weights | Metric regime |
|---|---|---|---|---|---|
| **DIAMOND** (Atari) | ~tiny CNN U-Net | ✅ verified (CUDA+CPU, bit-exact) | yes (CPU too) | ungated `eloialonso/diamond` | pixel-space EDM, 3-step; launch-bound → **graph** |
| **Oasis-500m** | 500M DiT-S/2 | ✅ verified (CUDA/3060, corr 1.0) | yes | ungated mirror `camenduru/oasis-500m` + `vit-l-20` VAE | DDIM v-pred 10-step AR; small → launch-bound |
| **Matrix-Game 2.0** | 1.8B Wan2.1 | 🔧 DiT fwd parity only (ActionModule pending) | yes | Wan2.1 DiT + 16ch VAE + CLIP-ViT-H (DL + `.pth` conv) | DMD 3-step AR; glue+compute |
| **Matrix-Game 3.0** | 5B Wan2.2-TI2V | 🔧 canned-action built, numerics pending | yes (24 GB min) | Wan2.2 DiT + VAE + umT5 (DL) | FlowUniPC/DMD; ActionModule + FOV memory |
| **Hunyuan-GameCraft 1.0** | 12.5B HYVideo | 🔧 structural, numerics pending | **NO** (40 GB+) | user `.pt` | PCM+CFG 8-step; big-GPU only |
| **Cosmos-Predict1 V2W** | 13B AR | ❌ not started | NO | — | FSQ tokenizer + AR transformer |

## Host-glue surface (audit, `DataPointer` density = the Krea2 disease)

| File | DataPointer reads | Note |
|---|---|---|
| `MatrixGame3ActionModule.cs` | 23 | mouse self-attn / keyboard cross-attn — **novel surface, not shared with video** |
| `OasisSpatioTemporalBlock.cs` | 12 | axial spatial + causal temporal, each with adaLN-zero |
| `OasisDit.cs` | 12 | per-frame `c[t]=TimestepEmbed+Linear(action)` |
| `MatrixGame3Pipeline.cs` | 13 | segment loop, history roll, integrate/Plücker |
| `MatrixGame2Pipeline.cs` | 12 | AR 3-frame blocks, DMD renoise |
| `OasisPipeline.cs` | 10 | AR frame loop, Diffusion Forcing |
| `MatrixGame3Transformer.cs` | 8 | memory-augmented sequence, readout |
| `Diamond/*.cs` | ~17 total | tiny; U-Net + EDM precond + sampler |

## Results log

> ### ✅ Round 1 (2026-07-12) — Diamond + Oasis-500m real perf runs on the 4090 (verified, parity preserved)
> Both ungated verified models downloaded, references re-dumped, parity reconfirmed on CUDA (Diamond 3/3, Oasis
> 4/4), then residency-ported with **existing device ops only** (zero backend/kernel edits), each guarded by a
> parity assert in a new gated FPS harness (`DiamondGenPerfTests`, `OasisGenPerfTests`). Metric = the AR per-frame
> compute, 4090.
>
> **DIAMOND (Atari, tiny CNN U-Net + EDM, 3-step):** AR-rollout **223 → 36 ms/frame (6.2×), 4.5 → 27.8 FPS**
> — past the 25-FPS interactive bar. Bit-exact (coherence mean/std unchanged; parity 3/3 held). Wins, in order:
> `DiamondOps.AdaGroupNorm` host loop → `GroupNorm(x, γ=1+scale, β=shift)` (44 loops/forward, 223→80.6 ms, 2.77×);
> `ConcatChannels`/input `cat` host memcpy → `backend.Concat` (12/forward, 80.6→36 ms, 2.19×). Residency floor
> reached — remaining ~12 ms/step is pure launch overhead (many small kernel launches ×3 steps + the pixel-quantize
> D2H syncs). **Remaining lever (not built):** CUDA-graph the fixed-shape 3-step denoise; blocked only by the
> lack of a device pixel-quantize op (clamp→256-level truncation), which forces the one host readback/step. A tiny
> quantize PTX kernel unlocks a drain-free step → graph capture (the launch-bound regime where it wins wall).
>
> **Oasis-500m (DiT-S/2, 16 blocks, 576 tokens, dim 1024):** per-DiT-forward **2342 → 1327 ms (1.77×)**, parity
> corr 0.99999929 unchanged (block0/blockLast taps corr ~1.0). Ported the per-frame adaLN glue in
> `OasisSpatioTemporalBlock` bit-exactly: `Clone` → `backend.CopyInto`; host `DiTUtils.LayerNormNoAffine` →
> `backend.LayerNormNoAffine`; `ModulatePerFrame` → `SliceLastDim`+`AddScalar`+`AffineBroadcastLastDim`;
> `GatedAddPerFrame` → `SliceLastDim`+`GatedResidualLastDim` (working tensors reshaped to rank-3 `[T,sp,dim]` so
> the broadcast ops index the frame axis). **Dominant remaining lever (not built):** the attention island —
> `SplitHeads`×3 / `MergeHeads` / `RotateBhsd`×2 per attention (192 host round-trips/forward). It's harder than the
> broadcast ports: Oasis uses **interleaved (2i,2i+1) + PARTIAL axial** RoPE on a `[b,h,s,d]` layout, which the
> stock `backend.ApplyRope` (rotate-half) and `ApplyRopeInterleaved` (full-dim, `[b,s,h,d]`) do NOT match →
> needs a dedicated device axial-rope apply + `Permute0213` head split/merge (or a small kernel). Expected once
> device-resident: the Krea2/Chroma-class 4–20× (the host attention loops are the floor). FinalLayer modulation +
> patchify/unpatchify + `BuildCondition` host add are smaller follow-ups.
>
> Harness recipe (both): env-gated (`DIAMOND_PERF=1`/`OASIS_PERF=1` + `*_WEIGHTS`/`*_DIT`+`*_REF` **absolute**
> paths + `PARITY_BACKEND=cuda`); warmup then timed window with `backend.Sync()`; parity assert vs reference each
> run. Weights: Diamond `eloialonso/diamond` Breakout.pt (54 MB), Oasis `camenduru/oasis-500m` (`.pt`→safetensors).

> ### ✅ Round 2 (2026-07-12) — kernels + Python GPU baselines; Diamond BEATS reference, Oasis fully device-resident
> Established a **no-nvcc PTX build path**: `native/cuda/nvrtc_compile.c` (build `cc -O2 -o /tmp/nvrtc_compile
> nvrtc_compile.c -ldl`; run `LD_LIBRARY_PATH=~/.local/lib/cuda13 /tmp/nvrtc_compile in.cu out.ptx compute_80`) —
> the box has `libnvrtc.so.13` but no `nvcc`. Wrote **4 new PTX kernels in `native/cuda/dit/dit_f32.cu`**:
> `oasis_split_heads_f32` / `oasis_rope_interleaved_f32` / `oasis_merge_heads_f32` (device ports of the block's
> host attention loops — interleaved+partial axial rope on `[b,h,s,d]`) and `dit_pixel_quantize_f32` (DIAMOND EDM
> output). Wired each through `IBackend` (default host impl = numeric reference) + `CudaBackend` override +
> `CudaKernels` launch. All parity-guarded.
>
> **Python GPU reference baselines (torch cu124 eager, 4090):** Oasis DiT forward **67.9 ms**; Diamond 3-step
> frame **47.5 ms (21 FPS)**. (Script `/tmp/bench_world.py`, bench venv `/tmp/benchvenv`, `CUDA_VISIBLE_DEVICES=0`
> = 4090 in torch fastest-first order.)
>
> **DIAMOND — now BEATS the reference.** Ported the whole EDM step to device (input/obs `Scale`, combine
> `Scale`+`Add`, `PixelQuantize` kernel) + device Euler in the sampler → the 3-step denoise is now fully
> device-resident **and drain-free (graph-capturable)**. Parity 3/3 bit-exact, coherence identical. **223 → 33 ms
> (6.8×), 4.5 → ~30 FPS — vs torch eager 47.5 ms / 21 FPS = 1.4× FASTER than reference.** Remaining: wire
> `StepGraph` capture (the step is now ready); marginal since Diamond already beats reference.
>
> **Oasis — attention island now device-resident (new kernels), bit-exact.** `SplitHeads`/`MergeHeads`/`RotateBhsd`
> host loops → the 3 new kernels. Parity corr 0.99999928, all block taps corr ~1.0. **1327 → 923 ms; cumulative
> 2342 → 923 ms (2.54×).** The gap to torch (68 ms) is **NOT algorithmic**: weights are resident (cached), and
> Release == Debug (957 vs 923 ms) → it's the engine's **per-op host overhead** across ~960 op-launches/forward.
> **The one remaining lever is CUDA-graph step capture** (record the 960-op sequence once, replay eliminates the
> per-op host cost → expected ~torch-parity). This is the LTX-2.3-scale fixed-buffer recipe over 16 blocks — a
> distinct subsystem, not a kernel. Secondary: fuse the tiny adaLN slice/addscalar ops (a single Oasis-adaLN
> kernel from `mod[T,6·dim]`) to cut op count ~20%.

> ### ✅ Round 3 (2026-07-12) — Oasis CUDA-graph capture: 923 → 272 ms (cumulative 2342 → 272 ms, 8.6×)
> Wired step-graph capture into `OasisDit.Forward` (`HARTSY_DIT_GRAPH=1`, opt-in). The now-device-resident 16-block
> loop captures once and replays with a single `cuGraphLaunch` per DDIM step; host patchify/cond-build/final-layer
> stay OUTSIDE capture, refreshing fixed input/cond buffers (`_xFixed`/`_condFixed`) before each launch; RoPE
> tables + causal mask are cached per (t,gh,gw) geometry sig (stable device addresses); the block output lands in
> a non-graph-owned snapshot (`_blockOutFixed`) as the last captured op (the HunyuanVideo graph-owned-free rule).
> Interactive AR geometry is FIXED → sig never flips → captured once, replayed every forward. Clean eager fallback
> on any capture failure (`_graphDead`). **923 → 272 ms (3.4×), and the graph-REPLAY output is bit-exact
> (corr 0.99999928 vs reference — verified in-harness on the 30th forward).** The captured block loop is now at its
> **F32-kernel GPU floor** (~230 ms); the residual gap to torch's 68 ms is F32-vs-F16 kernel efficiency (torch uses
> TF32/fused-flash) → the next lever is `DitDtype.Act` F16 + cuDNN fused SDPA (flagged, numeric-risk), plus porting
> the host patchify/final-layer tail into the captured region. Diamond's step is likewise graph-ready but already
> beats reference, so its capture is deferred. Regression: Diamond parity 3/3, Oasis parity 4/4 (eager) held.

> ### ✅ Round 4 (2026-07-12) — Matrix-Game ActionModule safe residency ports (shared by MG2 + MG3)
> The novel non-Wan perf surface of both Matrix-Game models is `MatrixGame3ActionModule` (mouse self-attn /
> keyboard cross-attn, 23 host loops). Since MG2/MG3 numerics are **unverified with no weights** (correctness
> before perf), only the unambiguously-safe, existing-op host loops were ported to device: `AddInPlace`→
> `backend.Add`, `RmsNormHeads`→`backend.RmsNorm`, `LayerNormRows`→`LayerNormNoAffine`+`AffineBroadcastLastDim`.
> Bit-identical-class; all 13 Matrix-Game structural tests pass (CPU). The deeper attention-layout loops
> (`SplitQkvTemporal`/`MergeTemporal`/`ApplyRopeBatched` — like Oasis's but temporal-batched with grid-indexed
> RoPE) need dedicated kernels AND a **measured run needs real weights + numeric verification of the ActionModule**
> — that download+parity bringup (MG2 is closest: its Wan-backbone DiT forward is already parity-verified) is the
> gating next phase for MG2/MG3/GameCraft, not more speculative porting.

> ### Round 5 (2026-07-13) — Oasis torch-gap characterization: NOT compute-bound; the wall is kernel count
> Chased the 272 ms → torch 68 ms gap. Findings (all measured on the 4090): **(1) TF32 is already ON by default**
> for F32 GEMMs (`_allowTf32 = SM≥8 && !HARTSY_NO_TF32`) — the Linears already use tensor cores. **(2) Oasis
> TOLERATES F16** — forced-F16 SDPA keeps parity (v-pred corr 0.99999930). **(3) But F16 is NOT the lever:** added
> an opt-in `HARTSY_GEMM_F16` (`CUBLAS_COMPUTE_32F_FAST_16F`, F16-mantissa matmul / F32 accumulate — parity held at
> corr 0.99999929) and it moved the wall **0% (273 vs 272 ms)** → the DiT is **not GEMM-compute-bound**. A back-of-
> envelope confirms it: the whole forward is ~250 GFLOP ≈ 3 ms of TF32 GEMM, yet the wall is 272 ms. So the cost is
> **kernel COUNT / GPU occupancy** — ~800 tiny ops/forward (16 blocks × ~50), each underutilizing the GPU even inside
> the graph. **The remaining lever is kernel FUSION** (fused adaLN = LayerNorm+slice+addscalar+affine → 1 kernel;
> fused attention; fused gated-residual), NOT F16/TF32/graph. That's a large effort with diminishing returns on an
> already-8.6×, 500M model — and the op-math vs wall discrepancy (predicted ~20 ms of kernels vs 272 ms measured)
> means the real next step is an **nsys profile of the graph replay** to find where the time actually goes, before
> writing fusion kernels. `HARTSY_GEMM_F16` kept as harmless opt-in infra (default off; may help genuinely
> GEMM-bound models). No regression: Diamond 3/3, Oasis 4/4 default paths unchanged.

> ### ✅✅ Round 6 (2026-07-13) — Oasis SDPA was the wall: cuDNN fused flash → **272 → 28 ms (cum 2342 → 28 ms = 83×, 2.4× FASTER than torch)**
> The Round-5 "occupancy-bound, needs fusion" conclusion was WRONG — a phase probe (`HARTSY_OASIS_PHASE=1`,
> `[oasis-phase]`/`[oasis-block]` logs) proved the 272 ms graph-replay was **~all SDPA**: setup 0.6 ms · replay 260 ms ·
> final 8 ms, and inside the block **sdpa ≈ 900 ms** (eager, sync-timed) vs modNorm 3.4 / mlp 9.6 ms. Oasis's attention
> shapes are pathological for the materialized QKᵀ→softmax→PV path — **temporal is 2304 batch-heads × seq 4**, spatial
> 64 × seq 144. The one-line fix: `allowF16: true` on both SDPA calls → **cuDNN fused flash attention** (mask-null
> spatial + `[1,1,T,T]`-mask temporal, D=64, F16-tolerant — verified corr 0.99999930). **SDPA 900 → 2.9 ms (~300×);**
> whole DiT forward with graph **272 → 28.2 ms** (35.4 fwd/s). Parity: full suite 4/4 bit-exact (block taps corr ~1.0,
> v-pred 0.99999930, VAE 0.99999999); graph-replay output bit-exact. cuDNN SDPA is **capture-compatible** (graph
> captured + replays clean). **Cumulative Oasis 2342 → 28.2 ms = 83×, now 2.4× faster than torch eager (68 ms).**
> LESSON: profile before concluding — F16-GEMM's 0% "proved compute-bound" but the real wall was the SDPA kernel
> choice, invisible until phase-probed. `HARTSY_GEMM_F16` from R5 kept as harmless infra (not needed here).
> Diamond 3/3 regression clean (block change is Oasis-only).

> ### ✅ Round 7 (2026-07-13) — Oasis device FinalLayer + full-forward capture → **19 ms (cum 2342 → 19 ms = 121×, 3.6× faster than torch)**
> After the SDPA fix, a phase probe (`[oasis-phase]`) showed the residual split: setup 0.6 ms · graph-replay ·
> **host FinalLayer 8 ms** (per-frame modulation + unpatchify loops). Ported FinalLayer to device: modulation →
> `SliceLastDim`+`AddScalar`+`AffineBroadcastLastDim` (rank-3), unpatchify → a new `oasis_unpatchify_f32` kernel
> (5th NVRTC kernel; per-frame `[py,px,ci]` gather, IBackend default = host reference). **28 → 19 ms.** Then
> extended the captured graph to cover **blocks + FinalLayer** (one `cuGraphLaunch` = full v-prediction; velocity
> lands in `_velFixed`, returned as a fresh copy so the caller can dispose freely) — architecturally complete,
> perf-neutral (~19 ms). `HARTSY_GEMM_F16` re-tested post-SDPA-fix: still only ~2% (GEMMs are not the wall).
> Parity: full suite 4/4 bit-exact (v-pred corr 0.99999930, replay bit-exact), Diamond 3/3 regression clean,
> solution builds clean. **Final Oasis: 2342 → 19 ms = 121×, 3.6× faster than torch eager (68 ms), bit-exact.**
> Remaining (sub-ms, optional): fuse the block adaLN (LayerNorm+slice+addscalar+affine → 1 kernel) to cut kernel
> count further; the wall is now balanced across small block ops (modNorm/attn-prep/mlp), no single hot spot.

> ### ✅ Round 8 (2026-07-13) — Oasis fused adaLN → **18.8 ms (cum 2342 → 18.8 ms = 124×)**
> Fused the block+final adaLN into one kernel: `oasis_adaln_f32` (6th NVRTC kernel) does
> `LayerNorm(x)·(1+scale[f]) + shift[f]` with scale/shift sliced from `mod` per frame in-kernel — replacing
> LayerNorm + SliceLastDim×2 + AddScalar + AffineBroadcast (5 kernels → 1) at 64 sites/forward (256 launches
> saved). 19.4 → 18.8 ms, parity bit-exact (v-pred corr 0.99999929, block taps ~1.0). IBackend default host impl
> = numeric reference. **6 NVRTC kernels total** (split_heads / rope_interleaved / merge_heads / unpatchify /
> adaln / pixel_quantize). The block loop is now balanced with no single hot spot; further micro-fusion
> (gated-residual slice-fold, split+rope) is available for sub-0.5 ms trims. **Final Oasis: 2342 → 18.8 ms = 124×,
> 3.6× faster than torch eager, bit-exact.** Diamond 3/3, solution builds clean.

> ### ✅ Round 9 (2026-07-13) — Matrix-Game 3.0 CORRECTNESS bringup: core DiT + ActionModule parity-verified on real weights
> The flagship world model (5B Wan2.2-TI2V finetune) is now numerically verified against the Skywork `WanModel.forward`
> reference on the **real ungated `base_model`** (12.9 GB, Apache-2.0; the actual shipped config is dim 3072 / 24 heads /
> 30 layers / ffn 14336 / sigma_theta 0.8 — the repo's 5120/40/40 `config.py` is stale). New oracle
> `dump_mg3_reference.py` (flash→dense-SDPA monkeypatch, `WanCrossAttention` F32 override, namespace-pkg bypass) +
> `dump_mg3_action_reference.py` (isolated ActionModule). Two gated tests: `MatrixGame3ParityTests` (Stage A) +
> `MatrixGame3ActionParityTests` (Stage B).
>
> **Stage A — core memory-mode Wan backbone (no action / no plucker / memory_length=0): PORT VERIFIED at F32.** Per-block
> taps show **corr 1.00000000 through block 15**, gently drifting to 0.99990 by block 29 — pure F32 summation-order /
> transcendental (gelu-tanh/silu/LN) differences vs torch, amplified by MG3's residual stream growing ~1000× over 30
> blocks (a real port bug would corrupt block 0-15, not just the tail). **Found + fixed a real divergence:** MG3's
> memory-mode block builds the cross-attn residual on the NORMED hidden (`x = norm3(x); x = x + cross_attn(x)`) — added
> opt-in `WanVideoBlock.CrossAttnResidualNormed` (default false → the whole video fleet is byte-identical, 54/54 Wan+MG
> tests green). **GPU F16 gotcha:** the F16-GEMM path (bf16→F16 weight casts + allowF16 SDPA) amplifies the same tail
> drift catastrophically on this ill-conditioned synthetic regime (randn latent, t=1000, 8×8 spatial) → block-29 corr
> collapses 1.0→0.35 (F16-SDPA-off only recovers to 0.61). This is precision, NOT a port bug (CPU F32 is the true-math
> gate); real generation runs the model in bf16 like the reference, and F16 (10-bit mantissa) > bf16 (8-bit) so it's
> fine there — but the AR loop compounding warrants an F32-accumulation or per-step-renorm check at gen time.
>
> **Stage B — ActionModule (the novel dual-attention surface): VERIFIED, one real bug fixed.** Isolated single-module
> parity (mouse temporal self-attn + keyboard cross-attn, [8,28,28] θ=256 rope) on block-0's real weights: **mouse-only
> corr 0.99996, both corr 0.99995** (relL2 ~1e-2, the self-cancelling mouse-grid rope-construction diff). **Bug:** the
> per-frame action-window gather was off by one window step — `start = i·ratio − ratio·(window−1)` shifted every window
> 4 raw-frames too recent; correct is `start = i·ratio − ratio·window` (upstream front-pads `pad_t = ratio·window` then
> slices `padded[ratio·i : ratio·i+pad_t]`). Shared by both streams (both call `BuildWindows`). A dedicated subagent
> confirmed the rope interleave/freq/axis conventions already matched.
>
> **Stage C — memory-frame + Plücker paths VERIFIED (2026-07-13).** Both remaining MG3 surfaces now match the reference
> (`dump_mg3_reference.py` MEM/PLK stages; tests `MemoryMode_WithMemoryFrames` + `PluckerCamera_Injection`), block 0-15
> corr ~1.0 (same F32-tight / F16-tail pattern as Stage A). **(1) FOV memory-frame path** (M=2 memory ‖ F=3 pred,
> historical rope [3,4] + pred [5,6,7], memory timesteps 0): the existing `Forward(memoryFrames, ropeFrameIndices,
> outputFrames)` already handled it — the single-`BuildCosSin` over concatenated `[mem_idx…, pred_idx…]` reproduces the
> reference's per-segment split-rope exactly (rope is per-token by frame index). No code change needed; verified. **(2)
> Per-block Plücker camera injection** — the C# `AddPlucker` was a PLACEHOLDER (GELU not SiLU, added to patch-embed once,
> no per-block `cam_scale`/`cam_shift`). Reimplemented: `MatrixGame3Transformer.ProjectPlucker` builds the global camera
> embedding once (`patch_embedding_wancamctrl` + `c2ws` SiLU refine + residual), then each block applies its own affine
> via new `MatrixGame3CamInjector` (`cam_injector` SiLU refine + residual → `cam_scale`/`cam_shift` →
> `x = (1+scale)·x + shift`) through a new generic `WanVideoBlock.postSelfAttnHook` (fires between the self-attn residual
> and cross-attn, matching upstream position; default null → video fleet unaffected). `cam_*` keys pass the Wan converter
> unchanged (no rename collision). PLK final-v corr 0.99947 even on GPU F16 (this path stays well-conditioned).
>
> **MG3 correctness is now COMPLETE for the DiT forward** (backbone + action + memory + Plücker all parity-verified).
> Remaining before/for the perf run: (a) a combined memory+action+Plücker forward + the ActionModule's own memory path
> (subagent note #5, action rope reset) as a belt-and-suspenders check; (b) the real-weight genperf harness (mirror
> `MatrixGame2GenPerfTests`) + phase probe. Method rule held throughout: correctness before perf. Weights:
> `/tmp/mg3_ckpt/base_model`.

## Per-model perf-run plan

### DIAMOND — RUN NOW (tiny, verified, ungated) — the graph-capture proof case
- **Fits, cheap.** Real-time metric. 3-step Karras+Euler, fixed 64×64, 4-frame history → ideal fixed-shape
  single-forward loop.
- Plan: (0) download weights + build reference dump → confirm parity still ✅; (1) `[diamond-phase]` wall
  probes on an AR rollout (per-step U-Net vs sampler vs history); (2) residency ports of the ~17 host loops;
  (3) **CUDA-graph the 3-step denoise** (fixed shape, resident) — expect launch-bound win; (4) FPS before/after.

### Oasis-500m — RUN NOW (small, verified, ungated)
- **Fits.** 10-step DDIM v-pred AR with Diffusion Forcing, 32-frame sliding window, 360×640.
- Plan: (0) download `oasis500m.safetensors` + `vit-l-20.safetensors`, reference dump → reconfirm corr 1.0;
  (1) `[oasis-phase]` probes on a real RGB-in/RGB-out rollout; (2) residency ports (SpatioTemporalBlock 12 +
  OasisDit 12 + pipeline 10); (3) cuDNN SDPA on the axial attns (mask-null? check); (4) graph the 10-step
  per-frame denoise; (5) FPS before/after, byte-exact-vs-baseline gate.

### Matrix-Game 2.0 — CORRECTNESS VERIFIED + F16 FFN (Rounds 11-12, 2026-07-13)
> **B — parity vs Skywork DONE.** `dump_mg2_reference.py` runs the Skywork `CausalWanModel._forward_train` on the
> real distilled checkpoint (Foundation/no-action). Tricks: monkeypatch `flex_attention` → dense-additive-mask SDPA
> (the block mask for a 3-frame single block = full attention minus the 64 pad tokens) and cross-attn's
> `flash_attention` → SDPA (no triton/flash_attn); register `wan`/`wan.modules` as bare namespace packages to skip
> the model-zoo `__init__`; stub flash_attn/xfuser with valid `__spec__`. **MG2 Wan-backbone parity PASSES: v corr
> 0.99999994, taps patch/ctx/block0/blockLast corr 0.99999996+.** The C# MG2 DiT is numerically correct. (Gotcha:
> C# unpatchify is `[C,F,H,W]` natural-Wan, not `[F,C,H,W]`.) Deps: bench venv + diffusers/easydict/ftfy.
>
> **A — F16 FFN SHIPPED (contained, verified).** `WanVideoBlock.FfnDtype` (default `F32` → the whole video fleet is
> byte-identical) runs the big `[s, ffn_dim]` FFN intermediate — the dominant per-block activation — in F16
> (CastToF16 → F16 Linear/Gelu → CastToF32; QK/cross-attn already F16 via `allowF16` SDPA). MG2 opts in via
> `DitDtype.Act`. Gated on a new `IBackend.SupportsF16Activations` (CPU is F32-only → clean F32 fallback; caught by
> the CPU structural tests). **MG2 84.3 → 74.5 ms (1.13×), parity corr 0.99999992** (held vs Skywork). Cumulative
> MG2 **106.7 → 74.5 ms (1.43×)** (materialized-SDPA F32 → cuDNN-SDPA + F16 FFN). **Regression CLEAN:** Wan video
> transformer 8/8, MG CPU 8/8 + 6/6, Oasis 4/4, Diamond 3/3, full solution build (incl Vulkan). **Remaining lever
> (scoped, de-risked — every WanVideoBlock op already has an F16 kernel):** extend F16 to the attention Linears +
> norms (the full ~25-site block), handling the G>1 per-frame-timestep modulation mixing (Mul/Add) + boundary casts
> (input/encoder F16, temb F32, output F32) → a further ~1.3× on the bandwidth-bound blocks; do it with the full
> video-fleet regression since it's shared code.

### Matrix-Game 2.0 / 3.0 — REAL-WEIGHT PERF RUN done (85 ms, cuDNN SDPA 1.28× verified), bottleneck diagnosed
> **Round 10 (2026-07-13): MG2 runs on real Skywork weights.** New gated harness `MatrixGame2GenPerfTests` loads the
> 6.48 GB distilled DiT via `MatrixGame2CheckpointConverter.LoadAndConvert` → `MatrixGame2Transformer` (Universal
> cfg, dim 1536, 30 Wan blocks + 15 action blocks), runs one 3-frame latent block (768 tokens) with live
> mouse/keyboard streams, times ms/forward. **85 ms/forward (12 fwd/s), finite non-flat velocity** (std 1.37).
> **cuDNN SDPA fix verified on real weights: 106.7 → 83.2 ms (1.28×)** (`HARTSY_SDPA_NO_F16=1` toggles it; output
> byte-identical). Bottleneck diagnosis (measured toggles): action module NEGLIGIBLE (84.5 off vs 85.0 on — the
> Oasis-style attention collapse doesn't apply, it's a small fraction here); `HARTSY_GEMM_F16` **0%** (not
> GEMM-math-bound); token-scaling 2.25× tokens → 1.9× time (≈linear = **bandwidth-bound**, not launch-bound like
> Oasis — so CUDA-graph is NOT the lever); `HARTSY_DIT_F16` **0%** (DitDtype.Act not wired for this transformer).
> **The MG2 lever is F16 activations** (halve the 30-block Wan DiT's activation bandwidth) — needs wiring
> `DitDtype.Act` through `MatrixGame2Transformer`/`WanVideoBlock` (touches shared video code; behind a flag with
> per-stage relL2). MG2 weights in `/tmp/mg2_ckpt`. Numeric parity vs a Skywork reference is still the separate
> gate (DiT already corr 0.99999473 in the status doc); the perf harness asserts finite/non-flat, not parity.

### Matrix-Game 2.0 / 3.0 — ActionModule optimized (cuDNN SDPA + residency); real-weight run = next bringup
> **Round 9 (2026-07-13):** Applied the Oasis insight to the **shared `MatrixGame3ActionModule`** (used by MG2 AND
> MG3). Its temporal self-attention (mouse: `[sp, heads, tt, headDim]` — batched over spatial positions × tiny
> frame-seq) is the SAME pathological shape that made Oasis's materialized SDPA catastrophic → added `allowF16:
> true` to both stream SDPA calls (mask-null, D=64) for cuDNN fused flash. Plus the earlier safe residency ports
> (`AddInPlace`→`Add`, `RmsNormHeads`→`RmsNorm`, `LayerNormRows`→device). All 13 Matrix-Game structural tests pass
> (CPU; `allowF16` is a GPU-only kernel-choice, CPU path unchanged). The DiT core is Wan-backbone (`WanVideoBlock`)
> — already carries the video campaign's optimizations (RoPE memo, device final-layer, cuDNN SDPA, fp8 residency).
> **MG2 weights downloaded** (`Skywork/Matrix-Game-2.0`: DiT `base_distill.safetensors` 6.48 GB + `Wan2.1_VAE.pth`,
> ungated, in `/tmp/mg2_ckpt`). **Next (the gating bringup for a MEASURED run):** write `dump_mg2_reference.py`
> (clone the Skywork repo's `WanModel`), reconfirm the DiT parity (status: corr 0.99999473 already recorded) +
> verify the ActionModule numerics, then a real-weight genperf harness mirroring `MatrixGame2Pipeline`'s per-block
> input construction (36-ch composite + per-frame timesteps + action windows). The ActionModule SDPA/residency wins
> are then measurable; expect the same class of collapse as Oasis on the action-attention.

### Matrix-Game 2.0 — STRUCTURAL (fits, but ActionModule numerics unverified — correctness before perf)
- Blocked on: (a) ActionModule parity, (b) weight download + `.pth`→safetensors (Wan2.1 VAE, CLIP-ViT-H).
- Do the **bit-identical residency ports** now (safe regardless of numerics) + probe scaffold; defer wall
  bench until ActionModule parity lands. Inherits every Wan `WanVideoBlock`/`WanDitOps` port already shipped.

### Matrix-Game 3.0 — STRUCTURAL (flagship; fits 24 GB but numerics pending + big download)
- Mirror the LTX-2.3/Wan-14B video directive: residency ports + graph scaffold with clean eager fallback so
  big-GPU users benefit; verify parity on the CPU tiny-config; defer wall bench to real-weight validation.
- Reuses shipped Wan2.2 ports (RoPE memo, device final-layer, temb drain fix). New surface = ActionModule
  (23 host loops) + FOV-memory selection (CPU port today → GPU kernel = biggest FPS knob per reference).

### Hunyuan-GameCraft 1.0 — STRUCTURAL ONLY (12.5B won't fit 4090)
- Same directive: implement residency ports + graph path structurally, verify the capture path falls back
  to eager cleanly, defer wall proof to a 40 GB+ user. Inherits HunyuanVideo ports (already partly done).

### Cosmos-Predict1 V2W — not started; out of scope for this pass.

## Method rules (non-negotiable, from the war)
- Phase probes before any op-level work; wall ≠ op-profile means an un-instrumented host phase.
- **Correctness before perf** — do NOT optimize MG2/MG3/GameCraft's numerically-unverified paths beyond
  bit-identical residency ports; a fast wrong AR loop compounds error every frame.
- GPU-shared box: hard-gate every GPU run on `nvidia-smi` idle wait-loop; prefer 3060 for fits, take turns.
- Bit-identical ports first (residency/caching), numeric-risk (F16/graph) behind flags with per-stage relL2.
- Interactive metric = **step latency / FPS**, and coherence of the rendered rollout (not just finite floats).
