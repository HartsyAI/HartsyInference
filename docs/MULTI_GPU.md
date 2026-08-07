# Multi-GPU: Fit, Latency & Throughput

Three different questions get called "multi-GPU", and HartsyInference answers each with a different
mechanism. Find your question first; the config knob follows from it:

1. **"My model doesn't FIT one card"** → [§1 Sharding & placement](#1-make-it-fit--sharding--placement-pooled-vram)
   — pool VRAM across cards (sequential pipeline split; not faster, but it *runs*).
2. **"Make ONE generation FASTER"** → [§2 Latency parallelism](#2-make-one-generation-faster--replicated-weight-parallelism)
   — CFG-branch parallel, context parallelism, decode/denoise overlap (weights replicated or
   components moved; concurrent compute).
3. **"Serve MORE requests"** → [§3 Throughput](#3-serve-more-requests--data-parallel-engines)
   — independent engine instances, one per GPU (or two per GPU), behind your queue.

Everything here works over plain PCIe with **no P2P and no NVLink required** — cross-GPU hand-offs
are host-staged by default (P2P is used when the driver reports it), because mismatched consumer
cards are the primary tested target (the dev rig is an RTX 4090 + RTX 3060). For a
decision-oriented "which strategy on which hardware" walkthrough, see
[`PARALLELISM_GUIDE.md`](PARALLELISM_GUIDE.md). Full measured tables with method notes:
[`benchmarks/results/2026-08-05_multigpu_speeds.md`](../benchmarks/results/2026-08-05_multigpu_speeds.md)
(cited below as "the benchmark doc").

The one-sentence mental model: **sharding pools VRAM, it does not add speed** (a pipeline split runs
stages sequentially; the win is that a model that cannot fit one card now runs at all, or runs
un-quantized where it used to be crushed to fit), while **CFG-parallel, context parallelism, and
component placement can be outright wall-clock wins** — with honestly measured cases where they are
not (see §2's caveats).

---

## 1. Make it FIT — sharding & placement (pooled VRAM)

| Feature | What it does | Win | Enable via | Verified on |
|---|---|---|---|---|
| **LLM layer split** | Splits a text LM's transformer layers across N GPUs, proportional to free VRAM (or explicit ratios). Logits/sampling run on the last stage. | VRAM pooling | CLI `--device "cuda:0+cuda:1"` (text) or `--lm-shard-gpu N`; extension `LmShardGpuId` (or `DitShardGpuId`, which feeds the same list); library `PlacementConfig.ShardDevices` | Qwen3-32B Q4_K_M (19.8 GB): **OOMs on a 4090 alone, runs at 11.8 tok/s split** across 4090+3060 — now a committed regression test (`LlmShardingEngineTests.Sharded_TwoGpus_Qwen3_32B_DecodeTokPerSec`), alongside Llama-3.2-1B **exact token parity** vs single-GPU (`Sharded_TwoGpus_ExactTokenParity_VsUnsharded_GreedyFixedSeed`). |
| **VLM layer split** | Same layer split for vision-language models: mllama's gated cross-attention states peer-copy onto every stage that owns one; splice-style VLMs ride the existing embeds hand-off with no new primitive. | VRAM pooling | Same shard settings as the LLM split | Llama-3.2-11B-Vision (mllama) and Qwen2.5-VL-7B (splice-style): **exact text parity** vs unsharded on a real image question, pooled VRAM rise asserted on both cards (`VlmShardingEngineTests`, benchmark doc "VLM layer split" section). mllama's sharded decode is slower per-token (10.6 vs 14.8 tok/s — per-token vision-state peer copy on a no-P2P box, documented follow-up). |
| **Audio-LM layer split** | Same layer split for codec-token audio LMs (YuE Stage-1 7B, CosyVoice 2's Qwen2.5-0.5B). When sharded, YuE's load-time quantization **defaults to off** — the model runs at checkpoint precision (bf16) pooled across cards instead of being quantized down to fit one. | VRAM pooling → **quality** | Same shard settings as above; precision override `HARTSY_AUDIO_LM_QUANT=q4k\|q8\|off` | YuE Stage-1 bf16 (13.5 GB canonical) pooled at **8.7 + 4.3 GB** across 4090+3060; full-pipeline output verified via a **committed** Whisper-STT content-word-recall test (`YueLmShardingEngineTests`), not a manual session. CosyVoice 2: 24-layer split, fully intelligible cloned speech, **4/4 Whisper content-word recall** (`CosyVoiceLmShardingEngineTests`). |
| **DiT block sharding** | Splits a diffusion transformer's block loop across GPUs (block ranges, asymmetric per-card preload — pooled, not replicated). 2-way on six families; **N-way generalized on Qwen-Image** (`PlacementPlanner.DitSplitPlan` N-stage, `DitShardStage`) — 3-stage verified on this box (2 physical GPUs, 3 backend instances; a ≥3-physical-card run needs other hardware). | VRAM pooling | CLI `--dit-shard-gpu N`; extension `DitShardGpuId`; library `ShardDevices` + `EnableDitSharding` | Krea2, **Qwen-Image 20B** (the "doesn't fit 24 GB resident" case), Flux.1 (plain generations), Chroma, HunyuanImage 2.1, MiniMax-H3 (fp8), SD3.5 Medium — real weights, gated vs single-GPU baselines (see the fp8 cross-device caveat in Limits). Lumina-Image-2.0 + HiDream-I1: synthetic-only (bit-exact same-GPU; no checkpoints on this box yet). |
| **TE / VAE placement** | Runs text encoders and/or the VAE on another GPU, keeping the denoiser's card free of the multi-GB encoder evict/re-upload cycle. Often also a latency win — see §2. | VRAM + often **latency** | CLI `--te-gpu N` / `--vae-gpu N`; extension `TextEncoderGpuId` / `VaeGpuId`; library `TextEncoderDevice` / `VaeDevice` | Wan TI2V-5B **43.7 s → 32.7 s** (umT5 off the main card); SDXL SSIM 0.9998; Flux SSIM 0.8126 (`FluxComponentPlacementEngineTests`) — engine-verified. Wan VAE (was completely unwired) SSIM 0.9999; LTX-1 TE+VAE (VAE was unwired, TE already was) **16.4 s → 10.2 s**, SSIM 0.9943; LTX-2 TE+VAE(+audio) code was already fully wired — its engine test exists (`LtxVideo2ComponentPlacementEngineTests`) but is unrun on this box (checkpoint ~22 GB, not downloaded — disk-constrained). Qwen-Image combined TE/VAE-placement + sharding measured SSIM 0.9876 (`QwenImageCombinedPlacementShardingEngineTests`). Composes with DiT sharding. |

### How sharding actually works

**LLM layer split** (`LlmPlacement` → `GenericTransformer.ForwardEmbedsStaged`): the transformer's
layers are partitioned into contiguous per-GPU ranges planned from live free VRAM minus a reserve
(explicit `ShardRatios` win when set, llama.cpp `--tensor-split` style). Each stage's weights preload
onto its own card only — never replicated. The hidden state crosses stages through a host-staged copy
(~16-32 KB per token at decode); the KV cache allocates each layer's K/V on that layer's stage card
automatically. The final norm, `lm_head`, and the sampler run on the last stage's GPU. CUDA-graph
decode and speculative decode are disabled when staged (a captured graph can't span devices) — decode
runs the eager path.

**DiT block sharding** (`ForwardSharded`): the same idea over a diffusion transformer's block loop —
contiguous block ranges per card, with the joint activation handed across per step (~50-150 MB
host-staged per crossing at 1024², a few ms). Shared weights (embedders, final layer) live on card A.
Step-graphs, step-caching, and block-streaming are disabled while sharded; expect eager per-step
times. The split point comes from live free VRAM, byte-weighted, so the bigger card automatically
takes the larger block range. On Qwen-Image the plan is N-stage (`DitShardStages`); the other
families remain on the 2-way `DitShardBackend`/`DitShardSplitBlock` shape (ROADMAP item tracks
widening them).

**Audio-LM split**: YuE's Stage-1 (a LLaMA-2-7B emitting codec tokens) rides the exact same
`LlmPlacement` machinery through its `Qwen2Model` body. The interesting part is the precision policy:
single-GPU YuE quantizes Stage-1 to Q4_K at load so a 7B fits one card — a *fit* decision, not a math
one. With a shard placement active the default flips to **no quantization**: checkpoint-precision
bf16, pooled. `HARTSY_AUDIO_LM_QUANT=q4k|q8|off` overrides in either direction. CosyVoice 2 is the
second consumer of the same plumbing (`Qwen2Model.ForwardEmbedsStaged`, embeds-driven).

**Why there's no P2P requirement**: every boundary above host-materializes (D2H then H2D on the next
card). `IBackend.CopyFromPeer` upgrades to `cuMemcpyPeerAsync` when the pair reports P2P;
`HARTSY_P2P_DISABLE=1` forces the host-staged path for testing. The consumer no-P2P path is the
primary-tested configuration, not the fallback.

---

## 2. Make ONE generation FASTER — replicated-weight parallelism

These features need the model (or the moved component) to *fit* where it's placed — they trade VRAM
for wall-clock. Each has honestly measured cases where it loses on this box; read the caveats.

| Feature | What it does | Win | Enable via | Verified on |
|---|---|---|---|---|
| **CFG-branch parallel** | Runs the negative-prompt branch on a second GPU **concurrently** with the positive branch (weights **replicated**, needs the model to fit both cards). Falls back to sequential automatically (and observably, via a `[CfgParallel]` log line) when it can't. | **Latency** (~1.8-1.9× per step) | CLI `--cfg-parallel-gpu N`; extension `CfgParallelGpuId`; library `CfgParallelDevice` | Wan T2V/TI2V, Flux true-CFG, SDXL. **SDXL is the honest counterexample**: correct output (SSIM 0.9984) but 2.6× *slower* denoise on this box — `SdxlRecipe` stages the UNet as F32 (~10 GB), leaving the 12 GB second card at 28.8 MB free and OOM-retry-thrashing every step (`SdxlCfgParallelEngineTests`, benchmark doc). Wan at 6 steps: the ~10 GB double-preload eats the per-step win; longer schedules amortize it. Mutually exclusive with DiT sharding and context parallelism by design. |
| **Context parallelism (Wan, v1)** | Splits the video DiT's token sequence across 2 GPUs at latent-frame granularity; each rank runs ALL blocks on its token range with **weights replicated**, exchanging only per-block self-attention K/V (two-phase barrier, host-assembled). Observable `[ContextParallel] active/fell-back(...)` decision. | **Latency** (long sequences) | Library `PlacementConfig.ContextParallelDevices` (≥2 entries; entry 0 = primary). CLI `--cp-gpu` is landing. Mutually exclusive with DiT sharding and CFG-parallel. | Mechanism **byte-exact** on synthetic dual-backend runs (`ContextParallelWanTests`); real-weight Wan TI2V-5B cross-device SSIM **0.9616** gated > 0.90 against the measured cross-ARCH drift ceiling 0.7774 (`WanContextParallelEngineTests` + `WanCrossGpuRegimeDiagnosticTests`). **HONESTLY slower at the tiny 675-token test geometry** (35.7 s vs 22.4 s — exchange/imbalance-bound on a no-P2P heterogeneous pair); the win case is long sequences (≥ ~2.7k tokens; 720p-class is 27k) on balanced links. |
| **TE / VAE placement (latency mode)** | Same knob as §1's row: moving a multi-GB text encoder off the denoiser's card removes the per-prompt evict/re-upload cycle. | **Latency** | `--te-gpu` / `--vae-gpu` etc. (§1) | Wan TI2V-5B **43.7 → 32.7 s**; LTX-1 **16.4 → 10.2 s** (T5-XXL dominates at small geometry). Wan VAE-only placement is a wash on wall time at 9 frames (the win there is headroom, not latency). |
| **World-model VAE/denoise overlap** | With `VaeDevice` set, Oasis decodes a finished frame's latent on the VAE card **concurrently** with the next frame's denoise on the primary (dedicated background thread; unset = byte-identical original path). | **Latency** | Library `PlacementConfig.VaeDevice`; CLI `--vae-gpu` | Oasis-500m: warm baseline 5.20 s → **3.85 s** (~26% faster), SSIM 0.9999, dual-GPU utilization trace confirming genuine overlap (`OasisVaeDeviceOverlapEngineTests`, benchmark doc). |
| **Collective transport (foundation)** | `NcclApi` (runtime-resolved libnccl.so.2), `ICollectiveComm` with `NcclComm` + `HostStagedComm` universal fallback, `CollectiveComm.Create` factory with a logged `[Collective]` decision; `CudaTopology.ProbeLinks()` P2P/NVLink matrix for strategy planning. | Enables tensor/context parallelism | Automatic (transport choice is logged; missing libnccl is a fallback, never a crash); `HARTSY_NCCL_DIR` override | Cross-device AllReduce **bit-exact** (0/1,048,576 mismatches); AllGather 256 MB/rank at **4.79 GB/s** — the honest no-P2P PCIe number for this box; the same code path auto-uses NVLink/P2P where present (`CollectiveCommTests`). |

**Landing now (in progress, not yet verified — do not treat these as shipped):** Qwen-Image context
parallelism, LLM tensor parallelism v1 (splitting individual GEMMs with all-reduce on the seam,
consuming the collectives layer above), and the CLI `--cp-gpu` flag. This doc will gain their
measured rows when their real-weight gates land in the benchmark doc.

---

## 3. Serve MORE requests — data-parallel engines

Throughput needs none of the machinery above: construct **one `InferenceEngine` per GPU** (each
loads its own full copy of the model) and split incoming requests across them behind a queue.
This is plain data parallelism — N cards ≈ N× requests/sec for models that fit one card, with zero
cross-GPU traffic and no numerics caveats (each request runs exactly the single-GPU path).

```csharp
using InferenceEngine engine0 = new InferenceEngine("cuda", 0);
using InferenceEngine engine1 = new InferenceEngine("cuda", 1);
// route request i to engines[i % 2]; Task.WhenAll over the batch
```

- Already possible today — this is the pattern two SwarmUI backends with different `GPU_ID`s use, and
  concurrent one-backend-per-GPU generations run live (they are what surfaced and verified the
  `Tensor.EnsureCpuData` concurrency fix, `TensorConcurrentSyncTests` / ROADMAP §1). Now pinned by a
  dedicated test skeleton: `DataParallelServingEngineTests`
  (`tests/HartsyInference.Diffusion.Tests/`) — two engines on cuda:0/cuda:1, same SDXL checkpoint,
  concurrent requests via `Task.WhenAll`, coherence-gated, wall-vs-sequential and per-card VRAM
  printed. Skeleton awaiting its first real run — no measured throughput number is claimed here yet.
- **Same-GPU dual backends** (multi-tenant cards): two independent engine instances share one
  physical GPU with isolated streams/caches/mempools; generations are serialized per-ordinal by
  `DeviceGate` by default. 8/8 isolation tests pass incl. step-graph capture. Concurrent
  (non-serialized) mode `HARTSY_SAME_GPU_CONCURRENT=1`: the earlier near-VRAM-capacity failure was
  **root-caused 2026-08-06 (a step-graph-capture-abort virtual-address leak, NOT a concurrency
  race), fixed via `GpuTransferHelper.PurgeAbortedCaptureAllocs`**, and
  `SameGpuConcurrentRealWeightTests` is back in the campaign green gate (7 consecutive bit-identical
  passes). Still opt-in pending a longer soak.
- Disaggregated serving (separate prefill/decode pools) is a roadmap item (M5), not built.

---

## Enabling it

### SwarmUI extension (recommended surface)

Per-backend settings (Server → Backends → your HartsyInference backend). All GPU numbers are CUDA
ordinals — **fastest-first, not `nvidia-smi` order**; run `nvidia-smi` during a generation to confirm
which physical card an ordinal is.

| Setting | Effect |
|---|---|
| `GPU_ID` | Primary card. |
| `TextEncoderGpuId` | CLIP/T5/umT5/… on this card. |
| `VaeGpuId` | VAE encode/decode on this card. |
| `CfgParallelGpuId` | Negative CFG branch concurrent on this card (weights replicated). |
| `DitShardGpuId` | DiT block loop split across `GPU_ID` + this card (pooled). Also feeds the LLM/audio-LM shard list. |
| `LmShardGpuId` | LM-only layer split (text + audio LMs) **without** DiT sharding. |

`DitShardGpuId` and `CfgParallelGpuId` are mutually exclusive (two different uses of a second card);
the backend refuses to start with both set. Context parallelism has no extension setting yet (config
+ landing CLI flag only).

### CLI

Every generation command takes the shared placement flags:

```bash
# 32B LLM pooled across two cards (composite device — text command)
hartsy text "..." -m qwen3 --model-path Qwen3-32B-Q4_K_M.gguf --device "cuda:0+cuda:1"

# Un-quantized YuE Stage-1 pooled across two cards (lyrics in the prompt, style in --genre)
hartsy music $'[verse]\n...\n[chorus]\n...' -m yue -g "uplifting pop female vocal" --lm-shard-gpu 1

# Qwen-Image 20B DiT split across two cards
hartsy image "..." -m qwen-image --dit-shard-gpu 1

# Wan video with the text encoder on the second card (wall-clock win)
hartsy video "..." -m wan --te-gpu 1

# CFG branches in parallel (weights replicated, latency win)
hartsy video "..." -m wan --cfg-parallel-gpu 1
```

`--cp-gpu` (context parallelism) is landing; until it ships, CP is library-config only.

### Library

```csharp
using HartsyInference.Core.Backends;
using HartsyInference.Engine;

PlacementConfig placement = new PlacementConfig
{
    ShardDevices = ["cuda:0", "cuda:1"],   // pool VRAM: LLM/audio-LM layer split
    EnableDitSharding = true,              // + DiT block split for diffusion
    TextEncoderDevice = "cuda:1",          // or place components instead
    // CfgParallelDevice = "cuda:1",       // mutually exclusive with EnableDitSharding
    // ContextParallelDevices = ["cuda:0", "cuda:1"],  // mutually exclusive with both of the above
};
using InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement });
```

An all-defaults `PlacementConfig` is byte-identical to single-GPU behavior — placement is pure opt-in.

### Environment variables

| Variable | Meaning |
|---|---|
| `HARTSY_AUDIO_LM_QUANT=q4k\|q8\|off` | Audio-LM precision override (default: Q4_K single-GPU, off when sharded). |
| `HARTSY_P2P_DISABLE=1` | Force host-staged cross-GPU copies even when P2P is available. |
| `HARTSY_KV_F16=1` | Half-precision KV cache (halves LLM context VRAM; opt-in). |
| `HARTSY_SAME_GPU_CONCURRENT=1` | Concurrent generations from two backends on one GPU — capacity bug root-caused + fixed 2026-08-06, back in the campaign gate; still opt-in pending soak. |
| `HARTSY_NCCL_DIR` | Override the libnccl.so.2 probe directory for the collective transport. |
| `HARTSY_FP8_NATIVE=0` | Disable native fp8 GEMM everywhere — matches the GEMM regime across mixed-SM shard stages (see Limits; ~1.7× slower on Qwen-Image). |

---

## Measured results

Full tables with method notes: [`benchmarks/results/2026-08-05_multigpu_speeds.md`](../benchmarks/results/2026-08-05_multigpu_speeds.md).
Highlights (RTX 4090 + RTX 3060, PCIe, no P2P — every hand-off host-staged):

- **Qwen3-32B Q4_K_M**: cannot load on the 4090 alone (OOM at 0.3% free) → **runs at 11.8 tok/s** split 16.7 + 10.2 GB (committed regression test).
- **Qwen-Image 20B fp8**: 13.4 + 6.2 GB pooled. Fidelity gates (2026-08-06): same-device split SSIM **1.0000** (> 0.99 gated), cross-device with matched fp8 regime (`HARTSY_FP8_NATIVE=0`) **0.9929** (> 0.95 gated); the default cross-device SSIM (~0.18) is **informational only** — a per-tensor-fp8 outlier-channel regime difference between SM 8.9/8.6, root-caused as NOT a sharding defect (`QwenImageFp8PrecisionDiagnosticTests`; the previously-recorded 0.9734 was contradicted and withdrawn).
- **MiniMax-H3 fp8**: cross-device sharded mosaic root-caused (scale-blind fp8 fallback dequant in `LinearImpl`) and **fixed** — cross-device SSIM 0.17 → **0.9597** at defaults, a real > 0.90 gate again; same-device 1.0000 throughout; pooling +13,998/+10,086 MiB at no wall-time cost in that run (cross-device sharded 30.7 s vs unsharded 30.8 s warm).
- **Chroma HD**: sharded run was *faster* than baseline (49.8 → 39.0 s) — the baseline paid other costs.
- **Wan TI2V-5B** TE placement: **43.7 → 32.7 s**. LTX-1 TE+VAE: **16.4 → 10.2 s**.
- **Wan context parallelism**: mechanism byte-exact; real cross-device SSIM 0.9616 (> 0.90 gated; whole-model cross-arch ceiling 0.7774 measured as the control); slower at tiny geometry (35.7 vs 22.4 s) — the win case is long sequences on balanced links.
- **Collectives**: AllReduce bit-exact cross-device; AllGather **4.79 GB/s** host-staged/SHM on this no-P2P box.
- **Oasis** `VaeDevice` overlap: 5.20 → **3.85 s** warm, SSIM 0.9999, both GPUs concurrently busy.
- **YuE Stage-1 bf16**: 8.7 + 4.3 GB pooled (vs 13.5 GB + activations on one card, or Q4_K crushing); full pipeline sings supplied lyrics near-verbatim per Whisper STT at both precisions.
- **VLM split**: mllama + Qwen2.5-VL exact text parity sharded vs unsharded; mllama per-token decode slower (10.6 vs 14.8 tok/s, no-P2P vision-state copies).
- **SDXL CFG-parallel**: correct (SSIM 0.9984) but 2.6× slower denoise on this box — the F32-staged ~10 GB UNet replica strangles the 12 GB card.
- **Same-device split bit-exactness**: Flux same-device DiT split is bit-exact over 262k output values; Llama-3.2-1B layer split has exact token parity. Cross-device runs are SSIM/tolerance-gated instead only because mismatched SMs (8.6 vs 8.9) legitimately take different fp8/GEMM paths.

---

## Limits (read before filing a bug)

- **Sharding is not a latency feature.** Pipeline splits are sequential; per-step time is the same or a
  few % slower (boundary copies). If the model fits one card, one card is fastest.
- **DiT sharding is 2-way on most families; N-way landed on Qwen-Image only** (`DitShardStages`).
  The 3-stage shape is verified on this box with 3 backend instances over 2 physical GPUs; a genuine
  ≥3-physical-card run remains untested (no such hardware here). Widening the other five families is
  a tracked ROADMAP item.
- DiT sharding **disables step-graphs, step-cache, and block-streaming** for the sharded model, and on
  Flux it silently falls back to unsharded for ControlNet/Kontext/inpaint/regional requests (log line).
- **Cross-device fp8 fidelity on mixed-SM pairs**: two checkpoints with massive activation residuals
  (Qwen-Image ±10M, MiniMax-H3 ~2.7e6) showed that the SM 8.9 native per-tensor-fp8 GEMM regime and
  the SM 8.6 fallback regime legitimately produce very different images — for Qwen-Image this is a
  property of the checkpoint, present with NO sharding at all, and its default cross-device SSIM gate
  is informational (the same-device and matched-regime gates above are the real bars). MiniMax-H3's
  case turned out to also hide a real bug (scale-blind fallback dequant), now fixed — its cross-device
  gate is real again (> 0.90). Full evidence chain: benchmark doc ‡ section +
  `QwenImageFp8PrecisionDiagnosticTests`. `HARTSY_FP8_NATIVE=0` matches regimes at ~1.7× GEMM cost.
- **LLM split exclusions**: SSM/Mamba families and Gemma-4 per-layer-embedding models throw
  `NotSupportedException` under staged placement (enforced in `GenericTransformer.ForwardEmbedsStaged`,
  not just documented). **mllama and splice-style VLMs are now supported sharded** (2026-08-06 —
  the earlier "vision sidecars are skipped" behavior is gone; mllama pays a per-token vision-state
  peer copy, a measured follow-up). CUDA-graph + speculative decode are off while staged.
- **CFG-parallel replicates weights** — the model must fit both cards *with headroom*; otherwise it
  falls back to sequential (generation still completes; check the `[CfgParallel]` log line for which
  path ran). Known gap: SDXL's preload gate is exception-only, so `active` can be logged even when the
  second card is left thrashing at ~29 MB free (the measured 2.6×-slower case) — a post-preload
  headroom check is a documented follow-up.
- **Context parallelism v1 is Wan-only, 2 ranks, no MoE dual-expert, no step-cache**, and is
  exchange-bound at small geometry (measured slower at 675 tokens). Qwen-Image CP and NCCL-backed
  exchange are landing.
- **Same-GPU concurrent mode** (`HARTSY_SAME_GPU_CONCURRENT=1`): capacity failure root-caused + fixed
  (capture-abort VA leak); back in the campaign gate, still opt-in pending longer soak. Serialized
  (default) remains solid.
- **3D models and most world models don't consume placement yet** (`MeshService` still routes every
  load through the primary backend; `WorldService` now consumes `VaeDevice` for **Oasis** — the
  overlap-decode win above — but GameCraft awaits its multi-checkpoint loader and Hunyuan3D-2's best
  fit is CFG-parallel, not sharding). Frame-paced interactive loops (DIAMOND, Matrix-Game live mode)
  are latency-critical — block sharding's per-step boundary copies don't fit a 25-30 ms/frame budget,
  so sharding there is deliberately out of scope.
- **Tensor parallel is landing (v1 in progress on top of the shipped collectives layer); expert
  parallel is not built.** The collective transport itself (NCCL + host-staged fallback + topology
  probe) **is** built and verified — see §2. Design notes:
  [`docs/Research/MULTI_GPU_PARALLELISM.md`](Research/MULTI_GPU_PARALLELISM.md).
- **SwarmUI extension note**: `DitShardGpuId` was disabled on backend #7 while the MiniMax-H3 mosaic
  was live; the engine-side fix has landed, but re-enabling needs an extension change + DLL-swap live
  verification.

Verification lives in `tests/run-multigpu-campaign.sh` — every placement/sharding/CFG-parallel/CP
class runs against real weights, where a missing checkpoint fails the run instead of silently
skipping.
