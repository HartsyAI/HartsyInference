# Video Models — status

Concise status for every video-generation (T2V / I2V) model. Build detail lives in
[PHASE_9_VIDEO.md](PHASE_9_VIDEO.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **LTX-Video 13B (0.9.7 dev)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltxv-13b-0.9.7-dev-fp8.safetensors` (fp8-resident, ~15 GB, no OOM on 24 GB with `CacheWeightCasts=false`) → sharp photorealistic 704×480×25f at 30 steps. Reuses the 0.9.5 timestep VAE (identical config) + `V097` transformer (48 layers, head_dim 128, cross 4096). fp8 velocities NaN-free; 8 steps under-denoises (dev model needs ~30). Prompt adherence looser than 2B (cfg/STG tuning item, not a pipeline bug); numeric parity pending; ~25 s/step (fp8 dequant per GEMM, host-glue-bound). |
| **LTX-Video 0.9.5 (2B)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.5.safetensors` → coherent cat-in-sunlit-garden (better prompt adherence than 0.9). Shares the 0.9 28-layer transformer; validates the **timestep-conditioned VAE decode path** (decode t=0.05 / noise 0.025) end-to-end. Required: a **0.9.5 VAE converter rename table** (`VAE_095_RENAME_DICT` — the 0.9 up_block regrouping would corrupt the 0.9.5 layout; selected via `IsTimestepVae`) + generalizing `LtxVideoVaeDecoder` to the residual channel-changing pixel-shuffle upsamplers (`upsampleFactor`/`upsampleResidual`, `time_embedder = 4·outC`, decoder_block_out_channels (256,512,1024)). Same faint striping artifact as 0.9; numeric parity pending. This VAE architecture is shared by the 13B (0.9.7). |
| **LTX-Video 0.9 (2B)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.safetensors` + standalone fp8 T5-XXL → 25-frame 704×480 clip of a cinematic sunlit garden, prompt-faithful, temporally varying. Three bugs fixed to get from garbage→coherent (all in [PARITY_VERIFICATION.md]/memory `ltx-video-e2e-bugs`): (1) **VAE latent denormalization** was missing — decoder now applies `raw = latent·std_of_means + mean_of_means` from `per_channel_statistics` (was a blue lattice); (2) **caption projection** read `l = encoder.Shape[0]` = the batch dim (1) on the rank-3 `[1,L,4096]` T5 tensor, collapsing the caption to one token + a GPU OOB write — now derives `l` from element-count/last-dim (was blank output); (3) **T5 padding** attended unmasked in cross-attn — the generation entry now truncates to real tokens via `CfgHelper.SliceBatchElementPrefix` (was a dark, weakly-conditioned scene). RoPE / AdaLN order / final-norm / flow-match schedule+sign / timestep all verified to match diffusers and were correct. Remaining polish: a faint vertical-striping artifact in some frames; numeric layer-diff vs a Python reference still pending; perf is host-glue-bound (~5.7 s/step at 1320 tokens). |

## Built, validation-pending (🔧)

All built end-to-end (transformer + VAE + pipeline + converter), structural tests pass; numeric parity
against a Python reference is pending for every one.

| Model | Notes |
|---|---|
| **Wan 2.2 TI2V-5B** (T2V / I2V) | umT5 entry + Wan2.2 3D causal VAE incl. encoder (RGB-input I2V works); TI2V VAE == decoder. **Real-weight output now works** (2026-07-01): first e2e run was all-black — root-caused to `WanDitOps.TextEmbed` reading `l=Shape[0]=1` on the rank-3 `[1,L,H]` umT5 encoder → undersized Linear output buffer → NaN text context → NaN model → black frames; fixed by deriving rows from `ElementCount/lastDim`. Now generates a real, NaN-free image (trajectory-stable through 20-step tiny run on the 3060). VAE decoder confirmed correct in isolation. **Full-res now runs (2026-07-01):** 832×480×33f (14,040 tokens) completes e2e on the 4090, NaN-free, real 33-frame output — `CudaBackend.ScaledDotProductAttention` routes the plain F32/no-mask case to the existing online-softmax flash kernel when the score matrix would exceed ~half free VRAM (numerically equivalent, relL2 5e-4 vs GEMM), eliminating the ~19 GB matrix. Slow (~94 s/step; monolithic flash kernel is a perf target) but no longer OOMs. **Still 🔧, not ✅:** numeric parity vs a Python Wan reference is pending; Vulkan/AMD full-res still OOMs (no SPIR-V flash shader yet — CUDA-only fix). Closest video model to ✅. |
| **LTX-Video 0.9.5 / 0.9.7 (13B)** | Shares the now-✅ 0.9 DiT/pipeline; adds the **timestep-conditioned VAE decoder** (V097 config, `VaeTimestepConditioned=true`) — that decode path is built but not yet exercised on real weights (needs the 0.9.5/13B checkpoint download). |
| **LTX-2.3 22B** (dual-stream A/V) | **Runs fully e2e (2026-07-01, 4090):** `ltx-2.3-22b-dev-fp8.safetensors` loads cleanly against the code (DiT 4186 + connectors 262 + vae 170 + audioVae 102 + vocoder 1227 keys). Gemma-3-12B fp8 encode → dual-stream connectors → **block-swap** streams the ~19 GB fp8 DiT with a ~1.2 GB resident window (fits 24 GB, no OOM; wired via `IStreamingBlock`/`BlockStreamingController`, Flux pattern) → dual-stream denoise → video VAE decode (latent denorm wired) → 25 temporally-varying, prompt-responsive frames. **Not yet ✅:** output is prompt-responsive (garden foliage colors) + temporally varying but has a **persistent spatial grid/lattice artifact** that is **identical at 8 and 30 steps** — so it is NOT under-denoising but a real dual-stream spatial-structure bug (token packing/patchify, `LtxVideo2Rope` grid, or the video VAE unpatchify). Needs a numerical layer-diff vs the vendored diffusers `LTX2VideoTransformer3DModel` / VAE to localize (each run ~6 min at 8 steps). Audio decode deferred — BigVGAN vocoder needs grouped `ConvTranspose1d` (groups=768) which CudaBackend lacks (pipeline catches it → video-only). Memory: `ltx2-19b-vs-23-divergence`. |
| **LTX-2 (19B, superseded)** | The earlier 19B dev checkpoint is architecturally divergent from the code (2.3) — no prompt-mod/gated-attn (both since made optional so they no-op on it), and a single shared `aggregate_embed` (49·3840→3840) + two 3840-dim `{video,audio}_embeddings_connector`s vs 2.3's separate video-4096/audio-2048 connectors. Deleted in favor of the 22B (memory `ltx2-19b-vs-23-divergence`). |
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
