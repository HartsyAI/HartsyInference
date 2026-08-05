# Multi-GPU: Sharding, Placement & Parallelism

HartsyInference can spread **one model across several GPUs** (sharding — pooled VRAM), put **different
components of one pipeline on different GPUs** (placement), run **CFG's two branches concurrently**
(replication — a latency win), and run **two independent backends on one GPU** (multi-tenant cards).
Everything here works over plain PCIe with **no P2P and no NVLink required** — cross-GPU hand-offs are
host-staged by default (P2P is used when the driver reports it), because mismatched consumer cards are
the primary tested target (the dev rig is an RTX 4090 + RTX 3060).

The one-sentence mental model: **sharding pools VRAM, it does not add speed** (a pipeline split runs
stages sequentially; the win is that a model that cannot fit one card now runs at all, or runs
un-quantized where it used to be crushed to fit), while **CFG-parallel and component placement can be
outright wall-clock wins**.

---

## Feature matrix

| Feature | What it does | Win | Enable via | Verified on |
|---|---|---|---|---|
| **LLM layer split** | Splits a text LM's transformer layers across N GPUs, proportional to free VRAM (or explicit ratios). Logits/sampling run on the last stage. | VRAM pooling | CLI `--device "cuda:0+cuda:1"` (text) or `--lm-shard-gpu N`; extension `LmShardGpuId` (or `DitShardGpuId`, which feeds the same list); library `PlacementConfig.ShardDevices` | Qwen3-32B Q4_K_M (19.8 GB): **OOMs on a 4090 alone, runs at ~12 tok/s split** across 4090+3060. Llama-3.2-1B split has **exact token parity** vs single-GPU. |
| **Audio-LM layer split** | Same layer split for the big codec-token music LMs (YuE Stage-1 7B). When sharded, the load-time quantization **defaults to off** — the model runs at checkpoint precision (bf16) pooled across cards instead of being quantized down to fit one. | VRAM pooling → **quality** | Same shard settings as above; precision override `HARTSY_AUDIO_LM_QUANT=q4k\|q8\|off` | YuE Stage-1 bf16 (13.5 GB canonical) pooled at **8.7 + 4.3 GB** across 4090+3060; full-pipeline output verified via a **committed** Whisper-STT content-word-recall test (`YueLmShardingEngineTests`, real run: >=50% recall on real `[verse]/[chorus]` lyrics), not a manual session. |
| **DiT block sharding** | Splits a diffusion transformer's block loop across exactly 2 GPUs (block ranges, asymmetric per-card preload — pooled, not replicated). | VRAM pooling | CLI `--dit-shard-gpu N`; extension `DitShardGpuId`; library `ShardDevices` + `EnableDitSharding` | Krea2, **Qwen-Image 20B** (the "doesn't fit 24 GB resident" case), Flux.1 (plain generations), Chroma, HunyuanImage 2.1, MiniMax-H3 (fp8) — real weights, SSIM-gated vs single-GPU baselines. |
| **TE / VAE placement** | Runs text encoders and/or the VAE on another GPU, keeping the denoiser's card free of the multi-GB encoder evict/re-upload cycle. | VRAM + often **latency** | CLI `--te-gpu N` / `--vae-gpu N`; extension `TextEncoderGpuId` / `VaeGpuId`; library `TextEncoderDevice` / `VaeDevice` | Wan TI2V-5B **43.7 s → 32.7 s** (umT5 off the main card); SDXL SSIM 0.9998; Flux SSIM 0.8126 (`FluxComponentPlacementEngineTests`) — all three have a real engine test. Qwen-Image, Chroma, HunyuanImage, LTX-1/2 are wired code paths but **UNVERIFIED** — no `ComponentPlacementEngineTests` class exists for them yet. Composes with DiT sharding. |
| **CFG-branch parallel** | Runs the negative-prompt branch on a second GPU **concurrently** with the positive branch (weights **replicated**, needs the model to fit both cards). Falls back to sequential automatically (and observably, via a `[CfgParallel]` log line) when it can't. | **Latency** (~1.8-1.9× per step) | CLI `--cfg-parallel-gpu N`; extension `CfgParallelGpuId`; library `CfgParallelDevice` | Wan T2V/TI2V, Flux true-CFG. Mutually exclusive with DiT sharding by design. |
| **Same-GPU dual backends** | Two independent engine instances share one physical GPU with isolated streams/caches/mempools (multi-tenant cards). Generations are serialized per-GPU by default. | Multi-tenancy | Two backends with the same `GPU_ID` (extension warns + co-fits) | 8/8 isolation tests incl. step-graph capture. Concurrent (non-serialized) mode exists behind `HARTSY_SAME_GPU_CONCURRENT=1` but has a **known allocator bug near VRAM capacity** — leave it off. |

---

## How sharding actually works

**LLM layer split** (`LlmPlacement` → `GenericTransformer.ForwardEmbedsStaged`): the transformer's
layers are partitioned into contiguous per-GPU ranges planned from live free VRAM minus a reserve
(explicit `ShardRatios` win when set, llama.cpp `--tensor-split` style). Each stage's weights preload
onto its own card only — never replicated. The hidden state crosses stages through a host-staged copy
(~16-32 KB per token at decode); the KV cache allocates each layer's K/V on that layer's stage card
automatically. The final norm, `lm_head`, and the sampler run on the last stage's GPU. CUDA-graph
decode and speculative decode are disabled when staged (a captured graph can't span devices) — decode
runs the eager path.

**DiT block sharding** (`ForwardSharded`): the same idea over a diffusion transformer's block loop —
blocks `[0, split)` on card A, `[split, N)` on card B, with the joint activation handed across per
step (~50-150 MB host-staged per crossing at 1024², a few ms). Shared weights (embedders, final
layer) live on card A. Step-graphs, step-caching, and block-streaming are disabled while sharded;
expect eager per-step times. The split point comes from live free VRAM, byte-weighted, so the bigger
card automatically takes the larger block range.

**Audio-LM split**: YuE's Stage-1 (a LLaMA-2-7B emitting codec tokens) rides the exact same
`LlmPlacement` machinery through its `Qwen2Model` body. The interesting part is the precision policy:
single-GPU YuE quantizes Stage-1 to Q4_K at load so a 7B fits one card — a *fit* decision, not a math
one. With a shard placement active the default flips to **no quantization**: checkpoint-precision
bf16, pooled. `HARTSY_AUDIO_LM_QUANT=q4k|q8|off` overrides in either direction.

**Why there's no P2P requirement**: every boundary above host-materializes (D2H then H2D on the next
card). `IBackend.CopyFromPeer` upgrades to `cuMemcpyPeerAsync` when the pair reports P2P;
`HARTSY_P2P_DISABLE=1` forces the host-staged path for testing. The consumer no-P2P path is the
primary-tested configuration, not the fallback.

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
the backend refuses to start with both set.

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

### Library

```csharp
using HartsyInference.Core.Backends;
using HartsyInference.Engine;

PlacementConfig placement = new PlacementConfig
{
    ShardDevices = ["cuda:0", "cuda:1"],   // pool VRAM: LLM/audio-LM layer split
    EnableDitSharding = true,              // + DiT block split for diffusion (exactly 2 devices)
    TextEncoderDevice = "cuda:1",          // or place components instead
    // CfgParallelDevice = "cuda:1",       // mutually exclusive with EnableDitSharding
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
| `HARTSY_SAME_GPU_CONCURRENT=1` | Concurrent generations from two backends on one GPU — **known-buggy near VRAM capacity, leave off**. |

---

## Measured results

Full tables with method notes: [`benchmarks/results/2026-08-05_multigpu_speeds.md`](../benchmarks/results/2026-08-05_multigpu_speeds.md).
Highlights (RTX 4090 + RTX 3060, PCIe, no P2P — every hand-off host-staged):

- **Qwen3-32B Q4_K_M**: cannot load on the 4090 alone (OOM at 0.3% free) → **runs at ~12.1 tok/s** split 16.7 + 10.2 GB.
- **Qwen-Image 20B fp8**: 13.4 + 6.2 GB pooled, SSIM 0.9734 vs single-GPU, 21.0 → 24.9 s (the honest sharding cost).
- **Chroma HD**: sharded run was *faster* than baseline (49.8 → 39.0 s) — the baseline paid other costs.
- **Wan TI2V-5B** TE placement: **43.7 → 32.7 s**.
- **YuE Stage-1 bf16**: 8.7 + 4.3 GB pooled (vs 13.5 GB + activations on one card, or Q4_K crushing); full pipeline sings supplied lyrics near-verbatim per Whisper STT at both precisions.
- **Same-device split bit-exactness**: Flux same-device DiT split is bit-exact over 262k output values; Llama-3.2-1B layer split has exact token parity. Cross-device runs are SSIM/tolerance-gated instead only because mismatched SMs (8.6 vs 8.9) legitimately take different fp8 GEMM paths.

---

## Limits (read before filing a bug)

- **Sharding is not a latency feature.** Pipeline splits are sequential; per-step time is the same or a
  few % slower (boundary copies). If the model fits one card, one card is fastest.
- **DiT sharding is 2 GPUs exactly** (the LLM split takes N). >2-way DiT is planned.
- DiT sharding **disables step-graphs, step-cache, and block-streaming** for the sharded model, and on
  Flux it silently falls back to unsharded for ControlNet/Kontext/inpaint/regional requests (log line).
- **LLM split exclusions**: SSM/Mamba families, Gemma-4 per-layer-embedding models, and mllama
  cross-attention layers all throw `NotSupportedException` under staged placement (enforced in
  `GenericTransformer.ForwardEmbedsStaged`, not just documented — fixed 2026-08-05 for mllama, which
  previously silently produced wrong output instead of throwing); VLM vision sidecars are skipped with
  a warning at load time (single-device for image questions). Full VLM sharding is a planned Phase 5
  item, not abandoned. CUDA-graph + speculative decode are off while staged.
- **CFG-parallel replicates weights** — the model must fit both cards; otherwise it falls back to
  sequential (generation still completes; check the `[CfgParallel]` log line for which path ran).
- **Same-GPU concurrent mode** (`HARTSY_SAME_GPU_CONCURRENT=1`) has a known allocator double-free near
  VRAM capacity; serialized (default) is solid.
- **3D and world models don't consume placement yet** (`MeshService`/`WorldService` use the primary
  backend only). Surveyed and planned: Hunyuan-GameCraft's 12.5 B DiT is the genuine pooling case
  (a near-mechanical port of the existing HunyuanImage sharding, blocked on its `.pt` multi-checkpoint
  loader); chunked world models get `VaeDevice` decode-overlap; Hunyuan3D-2's best fit is CFG-parallel,
  not sharding. Frame-paced interactive loops (DIAMOND, Matrix-Game live mode) are latency-critical —
  block sharding's per-step boundary copies don't fit a 25-30 ms/frame budget, so sharding there is
  deliberately out of scope.
- **Tensor parallel (NCCL), expert parallel, and sequence parallel are not built** — `TensorParallelDegree`
  in `PlacementConfig` is reserved config. Design notes: [`docs/Research/MULTI_GPU_PARALLELISM.md`](Research/MULTI_GPU_PARALLELISM.md).

Verification lives in `tests/run-multigpu-campaign.sh` — every placement/sharding/CFG-parallel class
runs against real weights, where a missing checkpoint fails the run instead of silently skipping.
