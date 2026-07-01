# Video Models — status

Concise status for every video-generation (T2V / I2V) model. Build detail lives in
[PHASE_9_VIDEO.md](PHASE_9_VIDEO.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **LTX-Video 0.9 (2B)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.safetensors` + standalone fp8 T5-XXL → 25-frame 704×480 clip of a cinematic sunlit garden, prompt-faithful, temporally varying. Three bugs fixed to get from garbage→coherent (all in [PARITY_VERIFICATION.md]/memory `ltx-video-e2e-bugs`): (1) **VAE latent denormalization** was missing — decoder now applies `raw = latent·std_of_means + mean_of_means` from `per_channel_statistics` (was a blue lattice); (2) **caption projection** read `l = encoder.Shape[0]` = the batch dim (1) on the rank-3 `[1,L,4096]` T5 tensor, collapsing the caption to one token + a GPU OOB write — now derives `l` from element-count/last-dim (was blank output); (3) **T5 padding** attended unmasked in cross-attn — the generation entry now truncates to real tokens via `CfgHelper.SliceBatchElementPrefix` (was a dark, weakly-conditioned scene). RoPE / AdaLN order / final-norm / flow-match schedule+sign / timestep all verified to match diffusers and were correct. Remaining polish: a faint vertical-striping artifact in some frames; numeric layer-diff vs a Python reference still pending; perf is host-glue-bound (~5.7 s/step at 1320 tokens). |

## Built, validation-pending (🔧)

All built end-to-end (transformer + VAE + pipeline + converter), structural tests pass; numeric parity
against a Python reference is pending for every one.

| Model | Notes |
|---|---|
| **Wan 2.2 TI2V-5B** (T2V / I2V) | umT5 entry + Wan2.2 3D causal VAE incl. encoder (RGB-input I2V works); TI2V VAE == decoder. **Real-weight output now works** (2026-07-01): first e2e run was all-black — root-caused to `WanDitOps.TextEmbed` reading `l=Shape[0]=1` on the rank-3 `[1,L,H]` umT5 encoder → undersized Linear output buffer → NaN text context → NaN model → black frames; fixed by deriving rows from `ElementCount/lastDim`. Now generates a real, NaN-free image (trajectory-stable through 20-step tiny run on the 3060). VAE decoder confirmed correct in isolation. **Full-res now runs (2026-07-01):** 832×480×33f (14,040 tokens) completes e2e on the 4090, NaN-free, real 33-frame output — `CudaBackend.ScaledDotProductAttention` routes the plain F32/no-mask case to the existing online-softmax flash kernel when the score matrix would exceed ~half free VRAM (numerically equivalent, relL2 5e-4 vs GEMM), eliminating the ~19 GB matrix. Slow (~94 s/step; monolithic flash kernel is a perf target) but no longer OOMs. **Still 🔧, not ✅:** numeric parity vs a Python Wan reference is pending; Vulkan/AMD full-res still OOMs (no SPIR-V flash shader yet — CUDA-only fix). Closest video model to ✅. |
| **LTX-Video 0.9.5 / 0.9.7 (13B)** | Shares the now-✅ 0.9 DiT/pipeline; adds the **timestep-conditioned VAE decoder** (V097 config, `VaeTimestepConditioned=true`) — that decode path is built but not yet exercised on real weights (needs the 0.9.5/13B checkpoint download). |
| **LTX-2** (22B) | Dual-stream audio+video; Gemma 49-layer wiring; SwarmUI loader wired (blocked on engine NuGet republish). |
| **WanAnimate / WanS2V / WanVace** | Wan-lineage variants on the shared backbone. |
| **Lance (ByteDance) video** | Shared Lance backbone + Wan2.2 3D causal VAE; brought up the reusable 3D-video foundation. |
| **Kandinsky-5 video** | Built on the Kandinsky-5 backbone. |

## Not started (❌)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W (5B / 13B)** | NVIDIA AR video-continuation. FSQ tokenizer + AR transformer substrate (reusable for AR-token world models). `.pt` pickle weights, T5-11B encoder. |

## Notes

The reusable 3D-video foundation (CausalConv3d, streaming Wan VAE, frame encoders) was brought up by the
Lance and Wan builds and is shared across video + world models. The fastest path to the first ✅ here is a
single Python layer-diff pass on Wan 2.2 (the most complete, and now producing real NaN-free output —
weights are staged, no download needed). Note the `TextEmbed` rank-3 fix (2026-07-01) also covers any
Wan-family text path fed a `CfgHelper.SliceBatchElement` (rank-3) encoder — Lance, Matrix-Game, WanS2V/Vace.
