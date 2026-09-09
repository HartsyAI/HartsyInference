# Multi-GPU operation

Choose placement by purpose and measure on the actual topology. Engine PlacementConfig is the contract; CLI and extension settings translate into it. [Open work](Checklists/ROADMAP.md) tracks hardware/family gaps.

| Purpose | Library settings | CLI |
|---|---|---|
| Fit LM layers across cards | ShardDevices, optional ShardRatios | --device "cuda:0+cuda:1" (text), --lm-shard-gpu |
| Fit diffusion blocks | ShardDevices + EnableDitSharding | --dit-shard-gpu |
| Place text encoder or VAE | TextEncoderDevice / VaeDevice | --te-gpu / --vae-gpu |
| Run CFG branches concurrently | CfgParallelDevice | --cfg-parallel-gpu |
| Split attention context | ContextParallelDevices (primary first) | --cp-gpu |
| Serve independent requests | One InferenceEngine per ordinal | Separate processes or library routing |

DiT sharding, CFG parallelism and context parallelism are mutually exclusive. Explicit settings take priority over ParallelPlanner suggestions (--parallel auto); inspect the logged decision. Confirm ordinal-to-physical-GPU mapping; CUDA_DEVICE_ORDER=PCI_BUS_ID makes it explicit.

## Contracts and limits

- Layer/block sharding places contiguous ranges using available memory or explicit ratios. It pools weights, but sequential boundaries add transfers. LLM final norm/head/sampler live on the final stage; stage-local KV follows layer placement.
- Staged LM decode disables graphs/speculation. SSM/Mamba and Gemma-4 per-layer embeddings are excluded. mllama and splice-style VLMs have sharded coverage; vision-state copying remains a performance concern.
- Most diffusion families use two stages. Qwen-Image supports N stages; three backend instances on two physical cards do not verify three physical GPUs. Sharding disables step graphs/cache/streaming; inspect family-specific fallback logs for unsupported compositions.
- CFG replicas must fit both cards with operating headroom. Exception-only preload success can still leave the secondary thrashing; an active log alone does not prove a speed benefit.
- Wan context parallelism has two-rank/no-dual-expert/no-step-cache restrictions. Qwen-Image CP exists, but the documented unequal-VRAM pair verified fallback, not active two-card execution. LLM tensor parallelism has exact-token correctness coverage; expert parallelism remains open.
- Transfers can use peer/NCCL transport or host staging depending on operation/topology. Missing NCCL has a fallback; do not extrapolate PCIe measurements to NVLink.
- One engine per card keeps independent requests local; this is a consumer routing pattern, not an automatic API fleet scheduler. DataParallelServingEngineTests records 1.71× for four requests in the August 7 campaign; the old “skeleton only” claim was obsolete.
- Same-GPU engines serialize by default. Concurrent mode is opt-in pending soak; the historical capture-abort allocation leak was fixed. Re-run capacity/cancellation tests for changes.
- Mesh and most world-model placement remain limited; Oasis VAE overlap is implemented. Frame-paced worlds need latency validation before adding boundary transfers.
- Precision differences across SM generations require controls. Same-device split parity does not establish cross-device equality; matched-SM physical pairs still need testing.

YuE/CosyVoice use shared LM placement. The historical audio-LM policy chooses Q4_K for single-card fit and unquantized weights when sharded; inspect the resolved family policy and settings before comparing quality. Current knobs/defaults are declared by EngineKnobs; legacy names and disposition are in [ENV_VARS](ENV_VARS.md).

Extension settings historically include GPU_ID, TextEncoderGpuId, VaeGpuId, DitShardGpuId, LmShardGpuId and CfgParallelGpuId. Verify the extension's pinned version for its current surface; do not infer deployment state from this repository.

## Recorded evidence

These are retained August 2026 measurements, not fresh runs. The previously linked local results/2026-08-05_multigpu_speeds.md is absent from the tracked repository. Preserve the numbers with this limitation; re-run tests/run-multigpu-campaign.sh with its required checkpoints for current evidence.

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


Large Wan CP geometry (832×480×25 frames) also lost: 4m09s versus 1m12s single-card. Longer sequences alone therefore do not establish a win on the heterogeneous no-P2P pair. Qwen matched-regime testing remains necessary; never dismiss cross-device corruption as hardware drift without a control.
