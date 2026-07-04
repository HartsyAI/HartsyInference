# CUDA Graph findings (2026-07-04, RTX 4090, driver 580.159.03)

## Why
LTX-0.9 at small resolution (~640 tokens) is **launch-overhead bound**: sync-profiling showed ~1/3 of wall-clock is
per-kernel CPU launch latency across ~50k tiny kernels/gen (34k Linear, 27k Permute0213, 20k RmsNorm). Device-porting
and dtype tweaks plateaued there. CUDA graphs — capture the repeated kernel topology once, replay with one
`cuGraphLaunch` — is the structural fix for that class. The image pipelines (also resident, many small ops) benefit
identically. It does **not** help LTX-2.3 (block-swap streams changing weight pointers → breaks a frozen graph, and it's
bandwidth-bound anyway).

## What already existed
`src/HartsyInference.Cuda/CudaGraph.cs` — a complete `Capture` / `TryUpdate` / `Launch` wrapper over the driver graph
API — but it was **wired into nothing** and its docstring flagged it "untested locally… validate on hardware."

## Validated on hardware (tests/HartsyInference.Cuda.Tests/CudaGraphTests.cs)
- **`CudaGraph_CaptureReplay_Scale_MatchesDirect` — PASSES.** A 4-op `Scale` chain (1 → ·2·2·2·2 = 16) captured through
  the **backend's high-level ops** and replayed once gives 16, identical to direct execution. So: (a) the driver
  capture→instantiate→launch path works on this GPU/driver; (b) the backend's stream-ordered activation allocations are
  capturable as graph-memory nodes; (c) a single replay reads correct data (the tensor→dptr cache set at capture time
  still points at valid memory after the launch).
- **`CudaGraph_RepeatedReplay_WithPerOpAlloc_ThrowsOnSecondLaunch` — PASSES (documents the blocker).** Replay 0 is
  correct (12); replay **1 throws `CUDA_ERROR_INVALID_VALUE`**.

## The repeated-replay blocker — and the one-line fix
Capturing the backend's high-level ops records each op's per-op `cuMemAllocAsync` as a graph **allocation node with no
matching free inside the captured region** (the activation is cached and freed later, on `Tensor` dispose). A single
launch is fine, but the N-step denoise loop needs **repeated** replay, and the second launch re-runs the alloc node
against graph memory still live from the first → invalid argument.

**SOLVED (2026-07-04): instantiate with `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`.** The graph then frees its
previous launch's allocations before each relaunch, and reuses the same virtual addresses (so pointers cached at capture
time stay valid). `CudaGraph(stream, autoFreeAllocationsOnRelaunch: true)` — validated by
`CudaGraph_RepeatedReplay_AutoFreeOnLaunch_IsStable` (4 replays of a per-op-allocating chain, all correct). **No
persistent-buffer arena / allocator rewrite needed** — the backend's existing per-op allocation model captures fine.
(This works because the graph's output is consumed each step *before* the next relaunch frees it; the persistent latent
lives outside the captured region.)

## Remaining implementation path (much smaller now)
1. Pick a resident model's denoise loop (LTX-0.9 — its forward is already host-excursion-free after this session's
   RoPE/modulation/FinalLayer device-porting — or a resident image pipeline).
2. Split the per-step work: the **captured region** is the sync-free device chain (the DiT forward); keep the CPU-side
   scheduler/CFG/sigma math and any remaining host RoPE-table builds **outside** it (capture forbids host readback/sync).
3. Ensure per-step inputs land in **stable device buffers** the captured graph reads: the latent tensor must keep the
   same device buffer across steps (the euler update writes in place), and per-step scalars (timestep) enter via kernel
   params that are re-recorded, or a small stable buffer.
4. Capture on step 0 (or a warmup step), then `Launch()` per step with `autoFreeAllocationsOnRelaunch: true`.
5. Gate behind `HARTSY_CUDA_GRAPH` + per-model opt-in; fall back to normal execution if capture throws.

Expected payoff: removes the ~1/3 launch-latency tax on every resident (fits-in-VRAM) model — all image pipelines +
small/low-res video. Feasibility + the repeated-replay mechanism are now **both proven on hardware**; what's left is
per-pipeline plumbing (input-buffer stability + hoisting host math), not core-engine surgery.
