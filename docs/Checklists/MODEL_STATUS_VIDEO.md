# Video Models — status

Concise status for every video-generation (T2V / I2V) model. Open work is in the
[Remaining work](#remaining-work) section below; bring-up debugging notes live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **MiniMax-H3** ("Hailuo 03", omni: text/image/video/audio -> video + jointly generated stereo audio) | **Real-weight coherent video AND audio confirmed 2026-08-03, on a 12 GB RTX 3060.** Weights landed 2026-08-02; the full port (DiT + video VAE + audio VAE + Qwen3-VL nvfp4 text encoder + recipe/pipeline) was built and verified end-to-end the same night. ([details](#minimax-h3)) |
| **SeedVR2-3B (video/image RESTORATION — `Modality.Restore`, not T2V)** | **Full parity chain + 7-clip real-footage matrix verified (2026-08-01, 4090).** Per-stage gates: window partition EXACT (40 grids / 2,490 slices, Unit-tier fixture `SeedVr2Tests` (windowing facts)); preprocessing maxAbs 2.3e-6 (`SeedVr2Tests`, env `SEEDVR2_PRE_REF`) — caught 2 … ([details](#seedvr2-3b-videoimage-restoration--modalityrestore-not-t2v)) |
| **SeedVR2-7B (restoration)** | **✅ v1 NaDiT ported + real-weight smoke (2026-08-02).** The smoke run of 2026-08-01 had surfaced that `configs_7b/main.yaml` builds `models.dit.nadit` (**v1**), NOT the `models/dit_v2` tree the 3B uses — key names coincide, so it loaded and silently produced mud (GT SSIM 0.71; `Detect` then threw on the plain-MLP+no-tail signature). ([details](#seedvr2-7b-restoration)) |
| **Wan 2.1/2.2 mainstream family** (TI2V-5B, T2V-1.3B fp16, T2V-14B fp8, I2V-14B CLIP fp8, I2V-A14B MoE, T2V-A14B MoE, FLF2V) | **All validated e2e on real weights (2026-07-01/02, 4090): coherent output.** The backbone is numerically de-risked: a Python layer-diff (faithful `comfy/ldm/wan/model.py` port, fp8-dequantized weights) proved the C# transformer matches end-to-end — patch_embed exact, all 40 blocks ~1e-3 (teacher-forced), autoregressive output relL2 4e-3 = the fp8 noise floor (memory `wan-14b-fp8-divergence`). ([details](#wan-2122-mainstream-family)) |
| **Wan 2.1 VACE-1.3B** (control video) | **Real-weight coherent control-conditioned output confirmed (2026-07-02, 4090):** `wan2.1_vace_1.3B_fp16.safetensors`, 25f 480×320, 20 steps, moving-square control clip → coherent prompt-styled clip (`WanVace_Gpu_E2E`). ([details](#wan-21-vace-13b)) |
| **HunyuanVideo 13B (T2V, 720p)** | **Real-weight coherent output confirmed (2026-07-02, 4090); production config is now the fp8 checkpoint at 2.15 s/step, full clip in 1m26s** (25f 512×320, 20 steps, embedded-guidance 6.0; day-one arc was ~20 min). ([details](#hunyuanvideo-13b-t2v-720p)) |
| **LTX-2.3 22B** (dual-stream video+audio) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-2.3-22b-dev-fp8.safetensors` + Gemma-3-12B fp8 → coherent cats-in-a-sunlit-garden clip (25f, temporally varying) **plus a decoded 48 kHz waveform**. ([details](#ltx-23-22b)) |
| **LTX-Video 13B (0.9.7 dev)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltxv-13b-0.9.7-dev-fp8.safetensors` (fp8-resident, ~15 GB, no OOM on 24 GB with `CacheWeightCasts=false`) → sharp photorealistic 704×480×25f at 30 steps. ([details](#ltx-video-13b-097-dev)) |
| **LTX-Video 0.9.5 (2B)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.5.safetensors` → coherent cat-in-sunlit-garden (better prompt adherence than 0.9). ([details](#ltx-video-095-2b)) |
| **Kandinsky-5.0 T2V Lite (2B)** | **Real-weight coherent output confirmed (2026-07-02, 4090, FIRST attempt):** `Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` (transformer BF16→F16 via `Kandinsky5CheckpointConverter.LoadDiffusersFolder`, config = `Kandinsky5Config.VideoLite2B` exact match) + the repo's shared HunyuanVideo 3D VAE → prompt-faithful, temporally-coherent snow-leopard clip (25f 512×512, 30 steps, CFG 5.0; `Kandinsky5_Gpu_T2V_ShortClip`). ([details](#kandinsky-50-t2v-lite-2b)) |
| **Lance (ByteDance) video (3B)** (T2V) | **Real-weight coherent output confirmed (2026-07-21, 4090), CLI catalog-path verified:** `hartsy video -m lance-video` (`bytedance-research/Lance` repo, `Lance_3B_Video/model.safetensors` + `tokenizer.json`, ~13.2 GB downloaded and sha256-pinned this pass; the Wan2.2 VAE resolves as a side model, already staged). ([details](#lance-bytedance-video-3b)) |
| **LTX-Video 0.9 (2B)** (T2V) | **Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.safetensors` + standalone fp8 T5-XXL → 25-frame 704×480 clip of a cinematic sunlit garden, prompt-faithful, temporally varying. ([details](#ltx-video-09-2b)) |

## Built, validation-pending (🔧)

All built end-to-end (transformer + VAE + pipeline + converter), structural tests pass; numeric parity
against a Python reference is pending for every one.

| Model | Notes |
|---|---|
| **AnimeGen-T2V (aidealab)** | Drop-in checkpoint for the ✅ Wan2.2 T2V-A14B MoE path — a full bf16 finetune of Wan-AI/Wan2.2-T2V-A14B (Apache-2.0), shipped as the standard dual-expert pair (`high_noise.safetensors` + `low_noise.safetensors`, 28.6 GB each, original Wan key naming, diffusers `WanTransformer3DModel.from_single_file`-compatible). ([details](#animegen-t2v-aidealab)) |
| **Wan family — remaining items** | The family itself is ✅ above; still open: per-variant numeric parity beyond the T2V-14B layer-diff; full-res TI2V flash path is slow on CUDA (~94 s/step, monolithic kernel is a perf target). ([details](#wan-family--remaining-items)) |
| **Native fp8 GEMM — VALIDATED on Ada (2026-07-02, first exercise)** | `HARTSY_FP8_NATIVE=1` on the 4090: **~3× faster steps on T2V-14B fp8 (1.65 s/step vs ~5.0 s at 320×192/20 steps)** via `Fp8GemmExecutor` (cublasLtMatmul, fp8 weights consumed directly + dynamic absmax e4m3 input quant). ([details](#native-fp8-gemm--validated-on-ada-2026-07-02-first-exercise)) |
| **LTX-Video 0.9.5 / 0.9.7 (13B)** | Shares the now-✅ 0.9 DiT/pipeline; adds the **timestep-conditioned VAE decoder** (V097 config, `VaeTimestepConditioned=true`) — that decode path is built but not yet exercised on real weights (needs the 0.9.5/13B checkpoint download). |
| **LTX-2 (19B, superseded)** | The earlier 19B dev checkpoint is architecturally divergent from the code (2.3) — no prompt-mod/gated-attn (both since made optional so they no-op on it), and a single shared `aggregate_embed` (49·3840→3840) + two 3840-dim `{video,audio}_embeddings_connector`s vs 2.3's separate video-4096/audio-2048 connectors. ([details](#ltx-2-19b-superseded)) |
| **Wan2.2-S2V-14B** (speech-to-video) | **Real-weight e2e PASSED (2026-07-02, 4090, `WanS2V_Gpu_E2E`)** after the faithful rewrite to ComfyUI `WanModel_S2V` (CausalAudioEncoder + AdaIN injector ×12 + cond-mask + per-frame timesteps + reference token-append; Wav2Vec2-large stable-layer-norm front-end, legacy `weight_g/v` pos-conv fallback). ([details](#wan22-s2v-14b)) |
| **Wan2.2-Animate-14B** | **Real-weight e2e PASSED (2026-07-02, 4090, `WanAnimate_Gpu_E2E`)** — the healthiest output profile of the variant fleet (per-frame means 128–155, clip mean 139, 20 steps @ 12.5 s/step, fp8 KJ v2 checkpoint). ([details](#wan22-animate-14b)) |
## In progress (🔬 / 🔧)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W (5B / 13B)** | 🔬 **Bringup started 2026-07-17 (engine-only — no Swarm world-model surface).** **DV tokenizer ENCODER real-weight VERIFIED** (`Cosmos-Tokenize1-DV8x16x16`): real arch recovered from `model.pt` `network.*` — factorized causal 3D VAE (base 128, ch_mult `[1,2,4,4]`, 2 resblocks/stage, down.0 spatiotemporal + down.1 spatial-only downsample, mid = 2 resblocks + spatial self-attn + **causal-temporal** self-attn), band-major 2-level Haar patcher (RGB 3→192 ch), 16→6 `quant_conv` → FSQ `[8,8,8,5,5,5]` (64000 vocab, `atanh`+eps 1e-3). Parity vs `encoder.jit`: continuous pre-FSQ **max\ | Δ\ | =2.86e-5**, FSQ tokens **31/32 bit-exact** (the 1 flip is a provable F32 half-integer rounding tie).** ([details](#cosmos-predict1-v2w-5b--13b)) |

## Notes

The reusable 3D-video foundation (CausalConv3d, streaming Wan VAE, frame encoders) was brought up by the
Lance and Wan builds and is shared across video + world models. The fastest path to the first ✅ here is a
single Python layer-diff pass on Wan 2.2 (the most complete, and now producing real NaN-free output —
weights are staged, no download needed). Note the `TextEmbed` rank-3 fix (2026-07-01) also covers any
Wan-family text path fed a `CfgHelper.SliceBatchElement` (rank-3) encoder — Lance, Matrix-Game, WanS2V/Vace.

## Remaining work

Distilled from the retired PHASE_9_VIDEO / VIDEO_CLI_CATALOG_HANDOFF / LTX_PARITY_REPORT plans.
`kandinsky5-video` and HunyuanVideo are now CLI-surfaced + verified above, so they are omitted.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Numeric validation
- [ ] All models are built structurally; numeric parity vs a Python reference is pending for every one not already ✅ (LTX 0.9 / 0.9.5 / 13B and LTX-2 22B are verified e2e).

### Wan / LTX open items
- [x] **Wan `EndFrame` real wiring on the non-concat path — DONE 2026-08-11 for `wan-22-5b`.**
  `GenerateFromEmbeddings`/`GenerateFramesAsync`/the internal `RunDenoise` now take an optional
  `lastFrameLatent` alongside `firstFrameLatent`, symmetric per-frame-timestep-0 pinning at latent-frame
  `T_lat-1` (`WriteLastFrame`, mirroring the existing `WriteFirstFrame`). `WanVideoRecipePipeline` VAE-encodes
  `VideoRequest.VideoEndFrame` the same way `InitImage` already was. Real-weight verified against the local
  TI2V-5B checkpoint: solid-red init / solid-blue end synthetic colors, frame 0 and the final decoded frame
  visually confirmed to match their respective conditioning colors (`WanEndFrameRealWeightTests.cs`). The
  2026-08-09 `Supports`/`SupportsFor` narrowing is reverted for `wan-22-5b` only. `wan-21-1_3b` shares the
  identical non-concat code path (same `ResolveConfig` branch shape) so the mechanism should cover it too,
  but **stays narrowed** — no local 1.3B checkpoint exists to verify against, and this doc's own verification
  rule is a real generation actually looked at, not "should work by symmetry." Revisit once a 1.3B checkpoint
  is available.
- [ ] **LTX has no VAE encoder at all** (confirmed — only decoders exist for both LTX-Video and LTX-2), no
  image-conditioning parameter anywhere in `LtxVideoTransformer`, no image argument on `LtxVideoPipeline`'s
  methods. LTX image-to-video needs a new VAE encoder class, a transformer-side conditioning input, and
  pipeline plumbing — genuinely new machinery, use Wan's `Wan22VaeEncoder.EncodeRgbFrame` →
  `firstFrameLatent` → `GenerateFromEmbeddings` shape as the template, not a shortcut.

### SeedVR2 restoration follow-ups
- [x] ~~fp32 whole-clip VAE activation ceiling~~ **BF16 VAE activations landed (2026-08-02)** — reference
      precision, CUDA default, `HARTSY_SEEDVR2_VAE_F32=1` reverts; pixel/latent boundaries stay F32 and the
      mid-block attention runs F32 (the recorded F16-attention all-black class). New BF16 variants of the
      five `wan_vae` glue kernels + BF16 `SliceRows`; `SeedVr2VaeConfig.ActivationDType` plumbs the dtype.
      **Measured (4090, BBB 25f, `--clip-frames 5`): 960×540-area peak 17.1 → 9.0 GB; 720p-area now runs
      whole-clip at 13.3 GB peak** (fp32 needed ~30 GB); output vs the f32 path SSIM 0.9998 / PSNR 46.9.
      **3060 (12 GB) now runs 960×540-area end-to-end: 201 s / 25 f, peak 7.8 GB**, output identical to the
      4090's. Gates: `Vae_Bf16Activations_MatchF32_OnCuda` (staged conv→encoder→decoder, relL2 ≲1.4e-2,
      passes against BOTH the f32 and numz-fp16 checkpoints) + the f32 real-weight parity re-verified
      unchanged (enc 2.9e-6 / dec 2.6e-6). Bug found on the way (fixed in `CudaBackend`): the BF16/F16
      GroupNorm paths fed a NON-F32 affine (the fp16 checkpoint's) to the kernel raw — F16 bits read as
      BF16 → flat-gray output; `CastAffineDownIfF32` now converts any affine dtype to the kernel's.
      REMAINING: 1080p-area still exceeds 24 GB even in bf16 (measured OOM 2026-08-02) — needs tiled or
      temporal-slab conv (`convDt`/`padded` are the peak) before 1080p+ targets work.
- [x] ~~DiT perf pass~~ **Device-resident DiT landed (2026-08-02)** — tokens stay `[L,C]` backend tensors
      end-to-end; fused `QkvSplitNorm` (per-head RMS qk-norm), window pack/unpack via
      `RowGather`/`RowScatterAdd` over per-geometry I32 index tensors, rope via `WanRopeInterleaved` with
      per-token tables (identity-padded beyond RotDim), AdaSingle as precombined emb+ada constant vectors
      (timestep is fixed at 1000), all cached in `SeedVr2DevicePlan` + block mod caches across chunks.
      Plus GPU-resident `SeedVr2PixelShuffle`/`SeedVr2PadBottomRight` (the VAE's last host round-trips;
      IBackend defaults keep the host reference for CPU). **Measured e2e (4090, BBB 25f): 960×540-area
      14.5 → 2.7 s/frame (362 → 68 s); 720p-area 25.7 → 8.4 s/frame (pre-shuffle-kernel).** Per-chunk
      phase split at 960×540 (`HARTSY_LOG_LEVEL=Debug` prints it): 4090 encode 1.0 s / dit 6.3 s /
      decode 2.5 s; 3060 5.2 / 6.2 / 13.8 (169 s wall — compute-bound in the decoder convs). Tiny-config
      parity unchanged (blocks ≤8.9e-4, output 1.05e-4); output vs the host-math implementation SSIM
      0.9998. Weight residency across chunks was measured a wash (uploads are not the bottleneck;
      resident peaked 23.98/24 GB for ~0% wall) — phase staging stays, which is also what 12 GB cards
      need. Remaining lever toward sub-2 s/frame: the DiT phase is now HOST-LAUNCH-BOUND (~6.3 s for
      ~0.4 s of GEMM math — ~2k small op dispatches across 32 blocks), i.e. the CUDA-graph block path
      (KERNEL.md priority list: graphs only help when launch-bound — that is now measured true here).
- [x] ~~Publish converted weights~~ Catalog repointed at `numz/SeedVR2_comfyUI` fp16 safetensors
      (verbatim original keys — verified 2026-08-01 via catalog download → convert → bbb9 restore; fp16
      output visually equivalent to fp32, delta is generative HF repaint; 3B DiT + VAE Sha256s pinned).
      REMAINING: upload the 1.2 MB `seedvr2_embeddings.safetensors` to `HartsyAI/SeedVR2-safetensors`
      (upstream ships pos/neg emb as torch-pickle `.pt` only — unloadable in pure C#); until then local
      file under `Models/Video/SeedVr2/` serves it. Pin 7B DiT Sha256 after its v1 port verifies.
- [x] ~~7B v1 NaDiT port~~ **DONE 2026-08-02** — see the SeedVR2-7B row above (parity 8.96e-4 + real-weight
      smoke GT SSIM 0.8797 at 640×360-area; ≤640×360 is the 24 GB envelope for the 16.4 GB fp16 DiT).
      Catalog 7B Sha256 still unpinned.
- [ ] SwarmUI full generation+restore run. Progress 2026-08-01: alpha.6 PUBLISHED; extension compiles
      clean against the pure NuGet package; stock Swarm build verified live — backend registers and all
      six Video Restore params appear in ListT2IParams. Remaining: one video generation with the group
      toggled on (LTX-2 checkpoint is staged in Swarm's model root). Landmine fixed on the way: stray
      2.0.0-alpha.64/65-local packs in the dev feed (~/.local/share/hartsy-local-nuget) SemVer-outranked
      alpha.6 and broke restore with NU1605 downgrades — quarantined to hartsy-local-nuget-quarantine/;
      future publishes must stay above any local pack version or the feed must be kept clean.
      **2026-08-01 update: DONE for video generation.** The extension pins `2.0.0-alpha.8` (alpha.7 was tagged
      but never reached nuget.org — the index goes alpha.6 → alpha.8; do not pin it). Verified end-to-end through
      stock SwarmUI on the published package (no `UseLocalHartsy`): extension restore + build clean, backend #7
      live on the 4090, `POST /API/GenerateText2Image` with `LtxVideo2/ltx-2.3-22b-dev-fp8` at 512×320,
      `textvideoframes` 25, 20 steps, cfg 3.0, seed 42 → **`h264` 512×320 25 frames @24fps + a muxed AAC stereo
      48 kHz track**, frames coherent (cat walking through a sunlit garden). Note Swarm ran on port 7802 bound to
      the LAN IP, not `localhost:7801`. Restore group still not exercised in the same pass.
      For local work ahead of a publish, build with `-p:UseLocalHartsy=true -p:HartsyRepo=<engine repo>` against a
      Release net8.0 CLI bin.

### Shared-infra gaps
- [x] **Generated-audio return path on the video contract** (`TODO(E-IMG-4/5)`) — **DONE 2026-08-01.**
      `IVideoRecipePipeline.Generate` now returns `VideoGenerationResult` (frames + `AudioBuffer?`) and
      `IVideoService.GenerateAsync` returns it as a `Task` (the old `IAsyncEnumerable<VideoFrame>` never streamed —
      it awaited the whole list first). `VideoAudioResolver` centralises precedence: pipeline-attached audio beats
      `VideoAudioInput` pass-through, trimmed to clip length; `VideoAudioReference` stays conditioning-only.
      LTX-2.3 attaches its soundtrack instead of logging-and-dropping it, and Wan2.2-S2V attaches the driving
      speech at source quality. The extension's ffmpeg mux (`VideoOutputEncoder.AudioTrack`) was orphaned dead
      code — never constructed — and is reconnected. Gate: 9 Unit-tier `VideoAudioContractTests` + a synthetic
      ffmpeg mux check (rgb24 stdin + f32le stereo → mp4 with a 48 kHz AAC stereo stream, durations aligned).
      **Verified on real LTX-2.3 weights 2026-08-01**, both surfaces: `hartsy video -m ltx-2` writes 25 coherent
      frames + `audio.wav` (2ch/48 kHz/1.0417s), and stock SwarmUI produces an mp4 carrying video **and** a muxed
      AAC stereo track. The split-VAE decode bug listed above is fixed, so the video is coherent too.
      Caveat: the soundtrack is real but ~80 dB too quiet — see the LTX-2.3 audio section below.
- [ ] `IBackend.PackedAttention` (varlen FlashAttention).
- [ ] `DenoiseKvCache` (~2-3× denoise speedup).
- [ ] `DistilledFlowMatchEuler` (DMD / CM / Lightning few-step schedules).
- [ ] `IDiscreteVideoTokenizer` (Cosmos).
- [ ] LTX 0.9.5 timestep-conditioned VAE variant.
- [ ] Wan multi-frame encode.
- [ ] Native 3D-conv / temporal-attn PTX.

### LTX-2.3 audio is ~80 dB too quiet (open, localized 2026-08-01)
Real generation (512×320×25f, seed 42) produces a stereo soundtrack with correct duration, true L/R
decorrelation and real temporal structure — but at **peak −43.9 dBFS / RMS −59.4 dBFS**, effectively
inaudible. `HARTSY_LTX2_PROBE=1` now covers the audio stages (`ProbeTensor` in `LtxVideo2Pipeline`) and
localizes the loss to **at or before the audio VAE output**:

| stage | measured |
|---|---|
| audio latent (raw, pre-denorm) | min −7.30 max 4.36 rms 2.47 |
| audio latent (unpacked, post-denorm) | min −7.18 max 4.74 rms 3.25 |
| **audio VAE out (log-mel)** | **min −11.73 max −4.71 mean −10.10** |
| vocoder out (waveform) | peak 0.0019 rms 0.00022 |

Confirmed through SwarmUI on the muxed mp4 as well (`volumedetect`: mean −56.6 dB, max −45.1 dB), so this is
the engine's output, not a CLI-only artifact.

The log-mel sits essentially on the `log(clamp(mel,1e-5))` silence floor (−11.51). A real 16 kHz speech
reference through the same convention measures **min −11.51 max +4.54 mean −1.33** — our mean is 8.8 nats
low and our max 9.25 nats (~80 dB in magnitude) low. The vocoder is faithfully rendering near-silence, so
it is not the culprit.

**Ruled out this pass** (do not re-check these first):
- Latent denorm is applied and correct — `MapAudioVae` strips the `audio_vae.` prefix so
  `ReadStats(conv.AudioVae, …)` finds `per_channel_statistics.*`; our `v*std+mean` matches diffusers'
  `_denormalize_audio_latents`. Magnitude cannot explain it anyway: `std-of-means` averages 1.17
  (range 0.74–2.05), worth ~1.4 dB. This is NOT the video `per_channel_statistics` bug's twin.
- `norm_out` type — the reference picks affine GroupNorm for `norm_type="group"` vs parameter-free
  `LTX2AudioPixelNorm` for `"pixel"`; the checkpoint ships **zero** norm weights, so our hardcoded
  pixel-norm is right.
- Vocoder STFT config matches the reference exactly (filter 512, hop 80, window 512, 16 kHz in,
  natural-log clamp 1e-5), and the chain order `denorm → unpack → audio_vae.decode → vocoder` matches
  `pipeline_ltx2.py`.

**The audio VAE is EXONERATED (2026-08-02).** A one-off diagnostic (since removed in the 2026-08-06 suite
cleanup) decoded synthetic latents through the real decoder and compared log-mel levels:

| latent fed to the VAE | log-mel mean | max |
|---|---|---|
| drawn from the checkpoint's own `per_channel_statistics` | **−4.34** | +0.58 |
| drawn from the observed generation's stats (mean −1.48, σ 2.89) | −6.14 | +2.23 |
| unit normal | −4.42 | −1.21 |
| **the actual generation** | **−10.10** | −4.71 |

Given a latent from the training distribution the decoder produces healthy levels. **The fault is upstream:
the audio latent itself is off-distribution** — mean −1.48 / σ 2.89 against the checkpoint's +0.018 / 1.17
(2.47× over-dispersed, far beyond sampling noise at n=3328).

**CFG is what disperses it, and dispersion tracks quietness** (8 steps, seed 42, video cfg 3.0):

| audio CFG | latent σ (pre-denorm) | log-mel mean | waveform peak |
|---|---|---|---|
| 3.0 (= video, shipped behaviour) | 2.22 | −10.10 | −43.9 dBFS |
| 7.0 (the authors' recommendation) | 2.69 | −11.09 | −60.5 dBFS |

**Do not "fix" this by raising `audio_guidance_scale` to the authors' 7.0** — measured, it makes the
soundtrack ~17 dB QUIETER. Their recommendation assumes the guidance rescale + STG + modality-isolation
guidance that accompany it in `pipeline_ltx2.py`, none of which this port implements. `AudioGuidanceScale`
exists on the config (null = follow the video scale, the reference default) with `HARTSY_LTX2_AUDIO_CFG`
for A/B, but the default is deliberately NOT 7.0.

**PARTIAL FIX SHIPPED — guidance rescale (2026-08-02).** `AudioGuidanceRescale` (default **1.0**, knob
`HARTSY_LTX2_AUDIO_RESCALE`) applies diffusers' `rescale_noise_cfg` to the audio stream: the guided velocity
is pulled back to the conditional prediction's mean/σ. Implemented as an affine transform of the guided
velocity (`v_final = A·v_cfg + B`) so only four scalars reach the host and the Euler step stays on device —
`LancePipelineCommon.CfgCombineRenormInPlace` was NOT reused directly because it writes host-side and the
audio latent is GPU-cached (a host write would go stale against its device copy).

Effect at 8 steps: latent σ **2.216 → 1.141** (checkpoint target 1.17), log-mel mean **−10.10 → −6.60**.
**Verified on a real 20-step generation** (512×320×25f, seed 42, cfg 3.0): soundtrack
**peak −43.9 → −28.2 dBFS, RMS −59.4 → −45.4 dBFS** (~14 dB recovered), still true stereo, 25 frames intact,
video coherent and visually unchanged (the video Euler step is untouched).

**Still not fully fixed.** −28 dBFS peak is audible but roughly 20 dB below a healthy soundtrack, and the
log-mel (−6.60) remains short of the −4.34 a training-distribution latent produces in the VAE diagnostic.
The gain at 20 steps (~14 dB) is also smaller than the 8-step log-mel delta suggested (~30 dB), so step
count interacts with the residual. Remaining suspects, in order: the STG (`stg_scale` 1.0, blocks `[28]`)
and modality-isolation guidance (`modality_scale` 3.0) that the reference pairs with LTX-2.3 and this port
does not implement — those add extra DiT forward passes with modified conditioning, so they are a real
piece of work; then a parity dump of the audio VAE/vocoder, which still have no captured reference
activations.

Still unvalidated independently: the audio VAE and vocoder have no reference activations captured (the
BigVGAN anti-alias filters are recomputed rather than loaded), so a parity dump remains worthwhile even
though the level defect is now traced upstream of them.

### MiniMax-H3
- [x] **Borrowed-view-as-output sweep — done 2026-08-03, no other site affected.** H3's mosaic bug was a
      `View`/`RowView` passed as an in-place op's OUTPUT tensor (CUDA binds the result to the view and drops the write
      on dispose, silently; the CPU path is unaffected, so unit tests pass). Swept every borrowed-pointer
      `new Tensor(void*, ...)` and `RowView`/`View` construction outside `HartsyInference.Core`: the only ones are
      `MiniMaxH3Transformer` (fixed), `MiniMaxH3VideoVaeDecoder` tile blending, and the LLM stacked-weight slicers
      (`Qwen35Model`, `GgufLanguageModel`). The latter two are safe — the VAE views feed `VaeTiling.BlendVertical`,
      a pure host `float*` loop, and the LLM views are read-only GEMM inputs. Re-run the grep when adding a model that
      slices activations rather than weights.
- [x] **fl2va DONE and controlled-experiment verified 2026-08-05.** The two-timestep hardcode is gone: `MiniMaxH3Pipeline`
      now derives rows per step via `MiniMaxH3Conditioning.BuildTimestepRows` — the reference's sorted-dedup of
      `{t_v, t_a}` ∪ `max(t_v, 0.999)` (cond/ref_img) ∪ `max(t_a, 1.0)` (ref_audio) — and splices conditioning rows in
      fresh each step rather than letting the sampler integrate them. `MiniMaxH3RecipePipeline` VAE-encodes
      `InitImage`/`VideoEndFrame` into keyframe rows and presents them to Qwen3-VL as `<Picture i>` (matching
      `minimax.py`'s fl2va path, which uses the *same* presentation as ref images). **Proof, seed-controlled:** the
      keyframes came from a seed-1 clip, so generation was re-run at seed 7 — T2VA scored 0.2198/0.2358 against them,
      fl2va scored **0.9920/0.9827**, the conditioning being the only variable. CLI: `--init-image` / `--end-frame`.
      *Non-obvious:* `FinalLayer` already emits target-rows-only, so the latent state must NOT grow — only the
      transformer's input does; and ref segments **interleave by kind** (a `video_audio` block emits RefAudio before
      RefImage), so row assembly must walk `layout.Segments`, never group by kind. Weight-free gates in
      `MiniMaxH3ConditioningTests`; T2VA stayed bit-identical (`h3_rell2.py` exactly 0.000e+00) across every change.
- [x] **ref2va COMPLETE 2026-08-06 — images, standalone audio, AND reference videos.** All three are wired end-to-end
      (typed `VideoRequest.ReferenceImages`/`ReferenceVideos`/`ReferenceAudios`, CLI `--ref-image`/`--ref-video`/
      `--ref-video-audio`/`--ref-audio`, SwarmUI `H3 Reference Images`/`Videos`/`Audio`). **The old "needs a 2-frame
      path" note rested on a wrong premise:** a 2-frame vision block is **`gridT = 1`**, not 2 — the two frames *fill*
      the temporal patch a still image fills by duplicating itself (`comfy/text_encoders/minimax.py:35-68` returns
      `grid_thw = [1, grid_h, grid_w]`). So a video block costs exactly a still image's token count, and
      `Qwen3VlVisionEncoder`, `Qwen3VlVisionConfig`, `MiniMaxH3TextEncoding` and `MiniMaxH3PackedLayout` needed **no
      change at all**; only `Qwen3VlImageProcessor` gained a frame-stack overload (the single-frame path re-reads one
      plane because its `tp` counter never appears in the source index). **Adherence is now measured, not assumed** —
      the ref2va checkpoint is downloaded and hash-verified: control vs reference video is mean |diff| **99.3** with
      **0.0% identical pixels**, while two clips built from the same source frames differ by only **2.9** from each
      other, so the conditioning is doing real work rather than perturbing the seed. A paired soundtrack
      (`video_audio`) shifts output a further **2.8**, confirming the RefAudio rows reach the packed layout.
      Reference geometry follows the node exactly: `adapt_canvas` with the shrink guard (a 256² clip stays 256², it is
      never upscaled to 768²), truncate to the output frame count **then** snap **down** to `17k+5`, and Qwen sees the
      already-resized frames at 2 fps with pair-mean timestamps. fl2va and ref2va are still rejected in combination —
      the layout restarts its position cursor for ref blocks, so their coordinates would overlap.
      **Known ceiling:** `BuildCausalMask` is a dense `[S,S]` F32, so three 1344×768 clips (~18k tokens) costs ~1.3 GB
      for the mask alone; at the reference's 3-clip cap this now raises a clear error naming token count and size
      rather than OOMing.
- [x] **LoRA on the fp8 build DONE 2026-08-06 (ComfyUI's approach).** `LoraStack.ApplyTo` no longer rejects fp8: it
      dequantizes (`CastTo(F32)` already folds `Fp8ScaleFactor` in), merges in F32 via the unchanged quant-unaware
      `AccumulateDelta`, then requantizes with a recomputed `absmax/448` scale and **seeded stochastic rounding** —
      matching `comfy/model_patcher.py:patch_weight_to_device` + `quant_ops.py` `scale="recalculate"`. Reuses the
      existing `QuantizeToFp8Scaled`. **No backend or GEMM change**, so the weight stays packed fp8 and both the native
      fp8 GEMM and the static-input-scale path survive: measured **165/123 ms per step with the LoRA vs 168/122 ms
      without**, and output mean |diff| **8.5** with **0.0% identical pixels**. **The trap that makes this subtle:**
      `Fp8InputScaleFactor` is a *separate* propagation from `Fp8ScaleFactor` and lives on the tensor object; a fresh
      tensor defaults it to `0f`, which silently fails the `> 0f` gate in `MiniMaxH3Transformer.Modulate` and costs
      ~188 ms/step with **correct output and no failing test**. It is carried across explicitly and pinned by a test
      that fails if the carry is removed. The old guard also tested `Fp8ScaleFactor != 1` rather than `DType.IsFp8`,
      so an fp8 weight at identity scale slipped through — now fixed.
- [x] **Both VAE encoder halves built 2026-08-05**, from weights the checkpoints already shipped (no converter change
      needed). `MiniMaxH3VideoVaeEncoder` (3D causal CNN; reuses the existing `CausalConv3d`, which already had H3's
      reflect-spatial + causal-zero-temporal modes, and `VaeTiling`; `SplitTiles` hoisted onto the config so encoder and
      decoder share one grid) — round-trip correlation **0.9636 untiled / 0.9733 tiled**. `MiniMaxH3AudioVaeEncoder`
      (DAC stack + causal-attention posterior head pooling 2048→32) — see the stereo item below. H3's encoder group norms
      take **per-frame** statistics, unlike the clip-wide `GroupNormSilu3d` the other VAEs use. Both gates are GPU-only:
      on CPU the video encoder is minutes/frame and the audio encoder ran >11 min for 3 s of audio (1 s on CUDA).
- [ ] int8 `convrot` quantization (Comfy's shipped quant: block-diagonal Hadamard over 256-channel groups, QuaRot/
      SpinQuant family) is detected and rejected by `MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot`, not implemented.
      This is the lever that would take the DiT off the mmap-thrash path — see the perf note in the status row.
- [ ] Block streaming (`IStreamingBlock`/`BlockStreamingController`, as LTX-2.3 uses) is not wired for H3's 50 blocks.
- [x] **Stereo channel identity confirmed 2026-08-05** by encoding a real stereo source rather than inspecting generated
      audio: left 440 Hz / right 1320 Hz through `MiniMaxH3AudioVaeEncoder` → `MiniMaxH3AudioVaeDecoder` gives left
      **0.25133 @440 / 0.00012 @1320** and right **0.00005 @440 / 0.23821 @1320** — ~2000:1 separation, and 0.25 is the
      exact DFT magnitude of the 0.5-amplitude input, so the round trip is quantitatively accurate, not just ordered.
      This is what the old caveat asked for: a swap is inaudible on generated content but obvious on an encoded reference.
- [ ] Sampling with very few steps is pathological at shift 12 (a 4-step schedule puts ~80% of the denoising in the
      final jump). Decide whether to clamp the shift for low step counts or just document a floor.
- [x] **Catalog `Assets` + sha256 DONE 2026-08-05** — `minimax-h3` is now `ValidationPending` + CLI-drivable with four
      assets (fp8 DiT, nvfp4 text encoder, video VAE, audio VAE) from `Comfy-Org/MiniMax-H3`. Verified by a
      catalog-only run (`hartsy video -m minimax-h3`, no `--model-path`): resolved all four, downloaded nothing, and
      produced frames + audio. **HuggingFace's LFS `oid` is the file sha256** — confirmed by hashing local copies, so
      `curl .../api/models/<repo>/tree/main/<dir>` sources catalog hashes without downloading (query per directory; the
      top-level `?full=true` response reports size 0). *Provenance trap:* the audio VAE staged here is a hardlink to the
      **vendor** `FL2VA/audio_vae/model.safetensors` and is byte-different from the Comfy-Org repack (605,429,308 vs
      605,254,808) despite the same weights, so that one hash is the published oid, not a locally-verified one; the
      other three match byte-for-byte. Harmless because `ModelDownloader` returns early on `File.Exists` and verifies
      only fresh downloads.
- [x] **LoRA wired 2026-08-05 — H3 is the first video model with LoRA end-to-end.** `MiniMaxH3Recipe.LoadTransformer`
      now hands back the converted dict so the merge lands before `LoadWeights`; the stack is owned by
      `MiniMaxH3RecipePipeline` and disposed after the transformer. **Corrected 2026-08-08 against the real published
      LoRA** (`larryvrh/MiniMax-H3-Turbo-Lora`): the `transformer.blocks.` passthrough below was only ever exercised by
      a hand-built LoRA carrying that prefix. The actual H3 LoRA has **no wrapper prefix at all** — its roots are the
      checkpoint's own keys (`blocks.0.attn.qkv_proj.lora_A.weight`) — so it hit `LoraFormat.Unknown` and was rejected
      at load. Fixed by a `DiffusersBareDit` detection arm (last in precedence, so a prefixed file never falls into it).
      **It also targets the UNPRUNED adaln projection** `[96768, 2688]` while every pruned build stores the curve-table
      form `[96768, 8]`, so 51 of its 259 modules (50 block adaln + `final_layer`) cannot merge on a pruned checkpoint:
      measured **208 of 259 merged, 51 skipped**. `LoraStack` now shape-checks each delta and names the skip rather
      than handing a mismatched delta to `backend.Add`. Still true: the **fused `qkv_proj` merges** because a PEFT
      export targets it as one Linear
      (verified against the real bf16 checkpoint: `blocks.0.attn.qkv_proj.weight` `[21504, 5376]` BF16). Only a
      kohya-*underscored* export would need work (`LoraKeyTransformer`'s allowlist lacks `q_norm`/`k_norm`/`adaln_proj`).
      **LoRA works on EITHER build** — an fp8 target is dequantized, merged in F32, and requantized back to fp8 with a
      recomputed scale (ComfyUI's approach), so the weight stays on the native fp8 GEMM path instead of being rejected.
      *Landmine:* `Fp8InputScaleFactor` must be carried onto the merged tensor or the weight silently drops off the
      static-input-scale fast path (~188 ms/step). *Landmine:* merged weights are written back at the checkpoint dtype,
      so a BF16 base resolves only ~0.4% of magnitude — a probe delta below ~1e-3 rounds away entirely.
- [ ] **int8 `convrot` is the last unsupported variant that matters.** Comfy-Org publishes **five DiT builds per task**
      (ten total — `bf16`, `int8_convrot`, `pruned_bf16`, `pruned_fp8_scaled`, `pruned_int8_convrot`, for each of fl2va
      and ref2va), not four; the
      engine loads bf16 (66 GB) and **`pruned_fp8_scaled` (21 GB), both verified on real weights** — the fp8 build
      is the production choice at **8.6 s/step / 22.5 GB fully resident on a 4090 vs ~90 s/step for bf16**, and its
      run is also the only exercise of the pruned `adaln_t_table` curve path (`curves=True`);
      `MiniMaxH3Assets` ranks formats so an
      unloadable `int8_convrot` file never wins over a loadable sibling. The two `int8_convrot` DiTs (34/21 GB) and
      the `int8_convrot` text encoder still throw.
- [ ] **SwarmUI path: extension side done, blocked only on the engine NuGet publish.** SwarmUI core added native
      H3 support (`T2IModelClassSorter`, PR #1469): it owns the `minimax-h3` compat class AND the `minimax-h3`,
      `minimax-h3/vae`, `minimax-h3/audio-vae` model classes, detected on `video_patch_proj`+`audio_patch_proj`.
      **The extension must NOT register any of these** — `Register`/`RegisterCompat` are backed by `Dictionary.Add`,
      so a duplicate ID throws at pre-init and would break extension load. The extension therefore only maps core's
      compat class to the engine family in `ModelSupport` (same shape as `lightricks-ltx-video-2`, which likewise
      shares its compat class with a `/vae` class). Builds clean; **unverified end-to-end** until a NuGet release
      carrying the H3 recipe ships. Run a Swarm generation as the first check after publishing.

### CLI catalog
- [ ] `cosmos-predict1-5b` / `cosmos-predict1-13b`: `IVideoRecipe` wrappers (also blocked on the DV pixel-decoder port, not just plumbing).
- [ ] LTX-2 CLI split-VAE decode: root-cause the transformer-only-split checkpoint's decode magnitude blow-up (bundled-ckpt path is ✅; split path outputs noise, Sha256 left null).
- [ ] Assets / sha256 wiring + verification runs for the remaining catalog entries.

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### MiniMax-H3

**Real-weight coherent video AND audio confirmed 2026-08-03, on a 12 GB RTX 3060.** Weights landed 2026-08-02; the full port (DiT + video VAE + audio VAE + Qwen3-VL nvfp4 text encoder + recipe/pipeline) was built and verified end-to-end the same night. Two runs, both first-try-correct after the fix below: 256x256 / 5 frames / 30 steps ("a cat walks through a sunlit garden") -> photoreal tabby in sunlit grass, temporally stable; **512x288 / 39 frames / 30 steps ("a golden retriever puppy splashing through a shallow stream in a forest") -> tracking-shot dog with physically plausible water splashes, motion-blurred background, continuous gait, plus a 1.625 s 32 kHz stereo soundtrack peaking at -12.5 dBFS.** **THE bug (a repo-wide hazard class, not an H3 quirk): a borrowed `View`/`RowView` passed as the OUTPUT of an in-place op.** CUDA binds the result to the view and the dispose callback skips the D2H, so the write is silently discarded. RoPE, adaLN modulation, and the gated residual were therefore all no-ops across all 50 blocks -- `h` never left the patch embedding and every frame decoded to a regular-grid magenta/green mosaic. Fixed by forcing the read-back (`Flush(view) => _ = view.DataPointer`) at the three sites; CPU-vs-CUDA parity 0.246 -> 4.75e-4, step-0 velocity rms 7.90 -> 2.24. **Other denoisers/VAEs have not been swept for this pattern -- see Remaining work.** **Reference-verified, not guessed:** the packed layout, dual-shift schedule, `pack_audio`/`unpack_audio` and the final-layer modulation rows are byte-identical to upstream ComfyUI master (`comfy/ldm/minimax/{model,vae,audio_vae}.py`, diffed against `raw.githubusercontent.com`), and the audio VAE decoder matches the reference module the checkpoint itself ships (`audio_vae/minimax_h3_audio_vae.py`) at **relL2 4.9e-6 CPU / 1.3e-3 CUDA** (`MiniMaxH3AudioVaeParityTests`, dump `tests/python-reference/minimax_h3_audio_vae_ref.py`). **Two traps worth knowing.** (1) *Audio latent statistics are useless as a health signal*: encoding through the real VAE, digital silence gives normalized rms 0.580, a -15 dBFS 440 Hz tone gives 0.612, white noise 0.552 -- everything lands near 0.58, so "rms looks low" proves nothing. Conversely a random N(0,1) latent decodes ~30 dB LOUDER than any real content because it is far out of distribution (which is why the pre-fix broken DiT produced clipping noise). Parity dumps must therefore use an *encoded* latent, not sampled noise. (2) *Short clips read as silent*: the model ramps audio in over roughly the first second (gen6 measured -54.8 dBFS at t=0 rising to -32 dBFS by t=1.5 s), so a sub-second test clip lands entirely inside the onset and looks broken when it is not. **Geometry is on coarse, non-obvious grids** (`MiniMaxH3Geometry`, transcribed from the reference nodes and pinned by `MiniMaxH3GeometryTests`): frame counts snap UP onto `17k+5`; video latent frames are `(frames-5)/17*5 + 2`, **NOT `frames/4`** (each latent token spans the `{1,4,4,4,4}` cycle, so 5 tokens cover 17 frames); audio length derives from the ALIGNED frame count; and each pixel axis rounds to **32, not 16** — a multiple of 16 that is not a multiple of 32 leaves an odd latent axis and the 2x2 patchifier drops its last row/column with no error. All three were live defects at the recipe's own declared defaults (1360x768x121f delivered 1344x768 and 102 frames against ~5.0 s of audio); fixed 2026-08-03 and verified by a run that asked for 22 frames at 272x208 and got exactly 22 frames at 256x192 with audio and video both 0.9167 s. **Memory:** `SafeTensorsLoader` mmaps, so the 66 GB bf16 DiT loads at 943 MB RSS and the whole run fits 10.3 GB of the 3060's 12 GB. **Perf: 1.94 s/step at 512x288x141f (2026-08-04), against ComfyUI's 1.67 s/step on the identical fp8 checkpoint and GPU.** Full 30-step clip: 129 s end-to-end, 58.2 s of that sampling **The cause was host round-trips, not weight residency.** `View`/`RowView` were built as `new Tensor((void*)t.DataPointer, ...)`, and `ActivationCache` is keyed by Tensor object reference (`GpuTransferHelper.cs:38`), so a view can never alias its parent's device buffer -- and merely CONSTRUCTING one calls `DataPointer` -> `EnsureCpuData` -> cache-evict + `cuStreamSynchronize` + D2H. The worst site was `SplitPart`, which host-copied a `[seq, 21504]` QKV tensor 3x per attention (~473 MB D2H + 473 MB H2D per block, ~47 GB/step). Fixed by restructuring so views are never needed: `SliceLastDim` for the QKV/adaln splits, q/k allocated 4-D up front so `RmsNorm` runs in place and `ApplyRopeSingle` consumes them directly, the residual stream shaped `[seq, 1, hidden]` so `AffineBroadcastLastDim`/`GatedResidualLastDim` modulate the whole sequence in one launch off a `RowGather`ed table, and `Concat` for segment assembly. **Acceptance metric: D2H syncs per forward 74 -> 0** (`IBackend.GetD2hSyncCount`, whose doc states a GPU-resident loop must stay at ~0); parity unchanged at video relL2 4.752E-004 / audio 6.555E-004. No new kernels were needed. Secondary: the nvfp4 text encoder's per-call host dequant now narrows to BF16 with one shared scratch buffer (~97 GB -> ~48.8 GB of H2D per prompt encode). **Ruled out, do not retry:** `CacheWeightCasts`; F32 activations blocking native fp8 (the guard at `CudaBackend.cs:958` accepts F32); weight residency/streaming (with a genuinely free GPU the 19.4 GB preload succeeds and the step does NOT get faster); cache eviction before preload (reclaimed 0 MB, cost ~15 s/step). **Benchmark hygiene:** check `nvidia-smi --query-compute-apps` first -- `swarmui.service` holds ~6.7 GB on the 4090 and silently invalidated a full round of numbers; and the op profiler aggregates the WHOLE run, so the one-time text-encode masquerades as denoise cost. Guarded harness: `scratchpad/h3_bench.sh`. Full spec: [`docs/Research/MINIMAX_H3.md`](../Research/MINIMAX_H3.md). **Multi-GPU (2026-08-05)**: DiT block-range sharding verified on the fp8 build (`MiniMaxH3DitSharding{,Vram}Tests` — 19.76 GB pooled 13.92+5.84 across 4090+3060 at the 34/50 split, finite video AND audio output); fp8 only — the 66 GB bf16 DiT exceeds any 2-consumer-card pool and is excluded from sharding.

### SeedVR2-3B (video/image RESTORATION — `Modality.Restore`, not T2V)

**Full parity chain + 7-clip real-footage matrix verified (2026-08-01, 4090).** Per-stage gates: window partition EXACT (40 grids / 2,490 slices, Unit-tier fixture `SeedVr2Tests` (windowing facts)); preprocessing maxAbs 2.3e-6 (`SeedVr2Tests`, env `SEEDVR2_PRE_REF`) — caught 2 real bugs (torchvision AA bicubic is a=−0.5 PIL-kernel not −0.75, and ATen computes resize weights in float32: double-math drifts 3.4e-5 by output index ~1000); VAE enc+dec relL2 ≤2.9e-6 vs REAL weights (`SeedVr2Tests`, `SEEDVR2_VAE`+`SEEDVR2_VAE_REF`); tiny-config DiT per-block relL2 ≤8.9e-4 / output 1.05e-4 (`SeedVr2Tests`, `SEEDVR2_PARITY_DIR`, dump `Parity/seedvr2_transformer_parity_dump.py` w/ flash_attn SDPA shim); **E2E vs Python real-weight restoration: mean SSIM 0.99950 / PSNR 56.6 dB** (`SeedVr2Tests`, `SEEDVR2_DIT/VAE/EMB/E2E_REF/FRAMES/AREA`, staged driver `run_seedvr2_e2e_reference.py` — reference noises injected via `NoiseHook`; torch RNG unmatchable). Reference quirks ported deliberately (SEEDVR2_ARCHITECTURE.md §2.5): tail-ada cache-collision (attn emb slice), last-layer vid_only (plain-normed txt K/V + ungated residual + txt self-doubling), per-frame VAE GroupNorm, (0,1,0,1) downsampler pad, MAGViT (x y z c) shuffle dropping output frame 1. **Matrix (25f clips, 960×540-area, `--clip-frames 5 --overlap 1`): Reagan USIA '87 / Apollo 11 / JFK '61 / Steamboat Willie / Prelinger '62 / BBB ground-truth / still (t==1) — 7/7 rc=0, ~14 s/frame, peak 17.1 GB, zero OOM** (`Models/TestAssets/restore/run_matrix.sh`, log `matrix_results.log`). Ground truth is honest: pixel metrics prefer bicubic (SSIM 0.877 vs 0.926 extreme; strength 0.7 ≈ unchanged — the loss lives in repainted high frequencies) but **LPIPS wins 26–28%** (extreme 0.735→0.541, mild 0.448→0.324) and Reagan crowd faces visibly resolve — the paper's own generative perception-over-distortion profile. CLI catalog path verified (`hartsy restore`, PNG frames + ffmpeg-subprocess MP4).

### SeedVR2-7B (restoration)

**✅ v1 NaDiT ported + real-weight smoke (2026-08-02).** The smoke run of 2026-08-01 had surfaced that `configs_7b/main.yaml` builds `models.dit.nadit` (**v1**), NOT the `models/dit_v2` tree the 3B uses — key names coincide, so it loaded and silently produced mud (GT SSIM 0.71; `Detect` then threw on the plain-MLP+no-tail signature). The v1 port landed 2026-08-02: pixel-basis rope (`linspace(1,128,10)·π` — matches the checkpoint's `rope.rope.freqs` [10]; 60 of 128 head dims; positions `linspace(−1,1)` per WINDOW axis, `steps=1→[-1]`; VIDEO ONLY — text never rotated), plain GELU-tanh MLP w/ biases (hidden 12288), all 36 blocks fully split (`MmLayers==NumLayers`, zero `.all.` keys), no tail norm/ada, and NO v2 last-layer text shortcut (block 35 computes text in full). `SeedVr2Config.Detect` now returns a v1 config (`PixelRope`/`LastLayerVidOnly`) instead of throwing. **Gates: tiny-config parity vs ByteDance's `models.dit.nadit` blocks ≤9.1e-4 / output 8.96e-4** (`Dit_TinyConfigV1_ForwardMatchesReference_PerBlock`, dump `Parity/seedvr2_transformer_v1_parity_dump.py`); **real-weight smoke on the 4090: BBB GT clip + Reagan at 640×360-area, both rc=0, full-range structured output, GT SSIM 0.8797** — right at the 3B's generative profile (0.877), vs the mis-architecture 0.71. VRAM: the fp16 DiT alone is 16.4 GB — 640×360-area works with ~1 GB to spare; 960×540 grinds the OOM-retry path and 720×405 is marginal, so treat ≤640×360 as the 24 GB envelope until weights are quantized or streamed. F16 weights staged at `Models/Video/SeedVr2/seedvr2_7b_dit_f16.safetensors`; catalog Sha256 still unpinned (pin on first catalog-download verification).

### Wan 2.1/2.2 mainstream family

**All validated e2e on real weights (2026-07-01/02, 4090): coherent output.** The backbone is numerically de-risked: a Python layer-diff (faithful `comfy/ldm/wan/model.py` port, fp8-dequantized weights) proved the C# transformer matches end-to-end — patch_embed exact, all 40 blocks ~1e-3 (teacher-forced), autoregressive output relL2 4e-3 = the fp8 noise floor (memory `wan-14b-fp8-divergence`). **fp8 dark-output fix:** CFG amplifies an fp8 velocity DC bias → `LancePipelineCommon.CfgCombineRenormInPlace` with `WanVideoConfig.CfgRescale` auto-set 0.7 for fp8 by `WanConfigDetector` (fp16 stays plain CFG, byte-identical). **MoE (A14B):** expert-swap keeps only the active 14 GB expert GPU-resident. Generalized harness `WanVariant_Gpu_E2E` (env-driven, auto-detect). **Production entry:** `WanVideoLoader` (checkpoint+VAE paths → auto-detected ready pipeline incl. MoE/VACE routing + `EncodePrompts` umT5 helper), e2e-validated on T2V-1.3B and VACE-1.3B. Numeric parity beyond the 14B layer-diff (per-variant) still open. **I2V-14B perf SOLVED (2026-07-11, 44.78-local): warm 234/218 → 52.6/51.9 s (4.3×), steps 13.1 → 1.82 s/step = the T2V floor, seed-42 frames byte-identical** — the ~160 s I2V-vs-T2V overhead was the CLIP-image branch of `WanVideoBlock.CrossAttention` (host `AddInPlace` D2H-summing both ~92 MB cross-attn branch outputs + host `SliceRows` fresh-tensor cache-miss drains, per block per forward ×40×2 CFG); GPU-ported with existing ops (`backend.SliceRows` + `GatedResidualLastDim` add), I2V loop gained per-step `FreeActivations` + `[wan-phase]` probes. Warm profile now: cond encode+evict 7.4 s · TE/CLIP ~11 s (umT5 prompt cache = next lever) · DiT preload 2.2 s · steps 27.4 s · decode 3.8 s. Animate inherits the block fix. **Warm-path residency shipped (2026-07-11, 44.80-local): same-prompt+image I2V warm 52.6 → 31.9 s, T2V warm ~37 → 30.3 s** — three cross-generation caches + KEEP_MODELS DiT residency: umT5 prompt cache per (pos,neg) token key + CLIP image cache per init-image SHA-256 (extension `WanVideoLoader`; a MISS gates DiT coexistence on measured free VRAM — 8.4 GB umT5 never fits beside the resident 16.4 GB fp8 DiT → logged evict + 1.9–3.4 s re-upload) + **I2V conditioning cache** per (image-hash, geometry) in `WanVideoPipeline` (a same-image repeat skips the whole-padded-clip VAE encode, whose REAL conv peak is ~7.5 GB at 25f 512×320 ≈ 153 F32 copies/frame — measured via the 44.79 OOM; `EnsureVaeEncodeHeadroom` estimate corrected ×24→×160 + pool-trim before the free-VRAM read) + `ReleaseOrKeepTransformer` (single-expert DiT resident across gens unless the decode estimate `max(3 GB, f·h·w·160)` doesn't fit; warm `DiT preload: 0 ms`; MoE always frees; LoRA gens free the base up front; `DisposeCore` evicts on model switch — I2V→T2V switch verified, next load at free 22.58 GB). **gen-1/gen-2/44.78-baseline seed-42 mp4s md5-IDENTICAL** (byte-exact residency path); warm all-HIT profile = steps 27.4 s + decode 3.8 s + mux; peak VRAM 23.0 GB unchanged. **Step-floor = fp8 COMPUTE FLOOR, confirmed (round 10, 2026-07-11, `44.82-local`, no code shipped).** Profiled T2V-14B at 25f 512×320 under `HARTSY_PROFILE_SYNC=1` (a `cuStreamSynchronize` after every op): the step stayed at **~1.79 s/step — identical to the un-profiled 1.82 s baseline**, the definitive compute-bound signature (no async overlap lost, launch overhead ~0% → **CUDA-graph capture would NOT help** and was not attempted). True-GPU-time op table: **Linear (native fp8 GEMM) 4930 ms + SDPA (cuDNN fused flash) 2003 ms = ~68% of GPU time**; the AdaLN/norm/rope/permute glue is already device-resident (~0.4 s/step, ~22%); `H2D_MISS_BIG` 1335 ms is one-time cold-start (umT5 8.4 GB encode + DiT 16.4 GB preload), NOT per-step. The host `scheduler.Step`/CFG-combine round-trip is a tiny 1.1 MB latent (a few ms/step, <0.3%) — a device-Euler port saves nothing and does not unlock graph capture, so it was NOT shipped. **fp8 GEMM coverage audit CLEAN:** `NativeFp8Gemm=True`, every Wan GEMM shape clears the `K%16 && N·outBytes%16` guards (`CudaBackend.cs:511-515`; dims 5120/13824), and the profile shows NO `Cast`/dequant-to-F16 op → zero fp8→F16 recasts; no guard-widening needed (shared kernel untouched). Round 9 already ruled out batched-CFG (fully-resident DiT = no stream to fetch once; per-tensor fp8 activation quant breaks bit-exactness). Only remaining theoretical lever = F16 activations (`HARTSY_DIT_F16`) for the ~0.4 s/step glue — minority, fp8-CFG-risky, out of scope. Decode 3.8 s. **CLI catalog-path verified 2026-07-21:** `hartsy video -m wan` (TI2V-5B, `wan2.2_ti2v_5B_fp16.safetensors`) — catalog `Assets` lists only the DiT file; umT5-XXL and the Wan2.2 VAE resolve as side models inside `WanVideoRecipe` (already staged on disk, zero download). Ran at the declared default (832×480, 33 frames, 50 steps, cfg 5.0): sampled frames 1/17/33 all show a coherent red fox trotting across a snowy field at dawn, correct subject, clear continuous leg/gait motion across the whole clip, no bug found — unlike LTX-Video, `WanVideoRecipePipeline`'s zero-pad-to-512-context umT5 conditioning matched the validated `WanVideoLoader` production path on the first try. **Multi-GPU (2026-08-05)**: TE placement (`PlacementConfig.TextEncoderDevice`, extension `TextEncoderGpuId`, CLI `--te-gpu`) wired + verified through `WanVideoRecipePipeline`; CFG-parallel synthetic-verified (`CfgBranchParallelWanTests`) with a preload-OOM→sequential fallback added, decision observable via `DiffusionPipelineBase.LastCfgParallelDecision` + the `[CfgParallel]` log line; the real-weight e2e class (`WanCfgParallelEngineTests`) is written but pending the checkpoint re-download (campaign phaseB).

### Wan 2.1 VACE-1.3B

**Real-weight coherent control-conditioned output confirmed (2026-07-02, 4090):** `wan2.1_vace_1.3B_fp16.safetensors`, 25f 480×320, 20 steps, moving-square control clip → coherent prompt-styled clip (`WanVace_Gpu_E2E`). Control context rebuilt to the ComfyUI reference (`WanVaceToVideo` + `WAN21_Vace.extra_conds`): inactive/reactive = control·(1−m)/control·m in [-1,1], each VAE-encoded + latent-normalized, plus the 8×8 space-to-depth pixel mask (64 ch, nearest-exact temporal resample) → 96-ch context; per-block hints `proj_out(block(c))·scale` verified structurally identical to `VaceWanAttentionBlock`. Converter gained the `before_proj/after_proj → proj_in/proj_out` renames; `WanConfigDetector` detects VACE from `vace_patch_embedding` + evenly-spread `vace_blocks`. Reference-image (identity) conditioning not modeled yet; 14B VACE untested (34 GB fp16 only). **Control causality proven** (2026-07-02): identical seed at control-scale 0 vs 1 → completely divergent outputs (collapsed dark vs bright control-tracking clip), so the hint branch demonstrably steers generation.

### HunyuanVideo 13B (T2V, 720p)

**Real-weight coherent output confirmed (2026-07-02, 4090); production config is now the fp8 checkpoint at 2.15 s/step, full clip in 1m26s** (25f 512×320, 20 steps, embedded-guidance 6.0; day-one arc was ~20 min). Conditioning: LLaVA-Llama-3-8B (fp8, layer −3, template+crop-95) + CLIP-L pooled. MMDiT (20 double + 40 single blocks) **numerically parity-verified** vs diffusers `HunyuanVideoTransformer3DModel` (per-stage relL2 ~1e-6). Perf chain, each step verified output-identical: (1) blocks on the **GPU-resident Qwen recipe** 75→19 s/step; (2) **fp8-resident DiT** — Kijai `hunyuan_video_720_cfgdistill_fp8_e4m3fn` (13.2 GB identity-scale; converter `NormalizeTencentRaw` for the raw Tencent key scheme; `HunyuanVideoPipeline` picks resident-vs-stream by weight size + 6 GB headroom); (3) **GPU RoPE** — `HunyuanImageRope.ApplyGpu` reuses `WanRopeInterleaved` pre-permute (tables cached per grid) 16.5→7 s/step; (4) per-step/per-tile `FreeActivations(trimPool:false)` (pool trim only at stage transitions) 7→6.1 s; (5) **`HARTSY_FP8_NATIVE=1`** (dynamic e4m3 activation quant → fp8 tensor cores, `Fp8NativeGemmTests`) 6.1→**2.15 s/step**, quality clean (no CFG here to amplify fp8 noise). **VAE decode 9 min → 9.6 s**: the batched `CausalConv3d` fast path extended to replicate padding (`wan_vae_build_padded` w/ replicate-first + spatial edge-clamp; parity corr=1.000000 vs the per-frame reference, Wan guard byte-identical; LTX/Kandinsky decoders share the win) + GPU `Vae3dLayout` in `Upsample` + `DecodeTiled` row-sequential blend. **THE blank-output bug: `CudaBackend.GroupNorm` F32-input path didn't upcast the F16/BF16 VAE affine** → fixed with `CastOnGpu` affine→F32 (memory `groupnormsilu-f32-bf16-affine`; shared VAE ⇒ also fixes Kandinsky-5). Full-res decode >24 GB → `DecodeTiled` feather-blend (corr 1.0). Open (non-blocking): VAE numeric parity vs Python (visual-only today), 720p + I2V unexercised, SDPA unfused (~1 s/step), fp8 bias-add not yet fused into the cublasLt epilogue, bf16-vs-fp8 quality A/B (bf16 ckpt was disk-pruned). **CLI catalog-path verified 2026-07-21:** new `HunyuanVideoRecipe`/`HunyuanVideoRecipePipeline` (registered in `VideoRecipeRegistry`) wrap the already-proven generation construction path (that test was removed 2026-08-06) — `hartsy video -m hunyuan-video` auto-downloads the bf16 (not fp8) `hunyuan_video_t2v_720p_bf16.safetensors` DiT (25.6 GB, sha256-pinned this pass) via catalog `Assets`, then casts BF16→F16 in `HunyuanVideoRecipe.Construct`; LLaVA-Llama-3-8B, CLIP-L, and the 3D VAE (bf16, ~493 MB, sha256-pinned, kept on disk as a `SideModels.HunyuanVideoVae3D` entry) resolve as side models. Ran at the declared default (512×320, 25 frames, 20 steps, cfg 6.0, "a cat walks on the grass, realistic style"): sampled frames 1/13/25 show a coherent, correctly-posed photorealistic cat with continuous walking motion (head bob, leg placement shifting frame to frame) and no late-clip drift — correct on the first attempt, no bug found. Multi-GB DiT download deleted after verification per disk hygiene (sha256 pinned, re-fetchable); the VAE was kept (small, shared-architecture side model). **SwarmUI production path verified 2026-07-23:** the last missing link was one extension-side `ModelSupport` compat-class mapping (`hunyuan-video` → family `hunyuan-video`, Video) — recipe + registry were already live, so Swarm refused the architecture purely for lack of the table entry. Verified e2e through `/API/GenerateText2Image` (512×320, 25f, 20 steps, cfg 6.0, seed 42): decoded frames 1/13/25 show a red vintage convertible on a coastal road w/ ocean waves, coherent motion (background pan + car bob + motion blur). Warm-loop steps ~1.6 s wall each with Sage default-on vs the 2.15 s/step pre-Sage record at this geometry; sampling 20 steps ≈ 78 s + VAE decode; DiT load+convert ~91 s cold. First attempt died disk-full mid-download of the LLaVA side model (9.1 GB need vs 7.7 GB free, root disk at 100%) — fixed by symlinking the existing `HartsyInference/Models/text_encoders/llava_llama3_fp8_scaled.safetensors` into Swarm's `Models/text_encoders/` (same pattern as the DiT symlink; no re-download). Flagship regression gates re-checked after the extension redeploy: Krea2-Turbo 4.41 s, Z-Image-Turbo 2.69–2.90 s engine-internal — both hold. **720p through Swarm exercised same day** (1280×720/17f/10st): ~6.5 s/step, tiled VAE decode ~29.5 s, no OOM, frames crisp + motion-coherent — the "720p unexercised" open item is closed for T2V (I2V remains recipe-TODO).

### LTX-2.3 22B

**Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-2.3-22b-dev-fp8.safetensors` + Gemma-3-12B fp8 → coherent cats-in-a-sunlit-garden clip (25f, temporally varying) **plus a decoded 48 kHz waveform**. **Block-swap** streams the ~19 GB fp8 DiT in a ~1.2 GB resident window (fits 24 GB, no OOM; `IStreamingBlock`/`BlockStreamingController`). The DiT is **numerically parity-verified** vs the vendored diffusers `LTX2VideoTransformer3DModel` (per-block relL2 ~1e-7, tiny matched config). Grid bug fixed: LTX-2.3 uses **`rope_type=split`** (NEOX per-head `(i,i+headDim/2)`), not base-LTX interleaved — implemented split in `LtxVideo2Rope`. Audio fixed: added **grouped `ConvTranspose1d`** to the CUDA `conv_transpose1d_f32` kernel (BigVGAN depthwise upsampling) + audio-latent denorm. See memory `ltx2-19b-vs-23-divergence`. **Perf (VIDEO_GENPERF_PLAN Phases 0–2 + 4-video, 2026-07-08/11):** steps 30 → **~2.0 s** (block GPU-residency port + pinned staging ring + VRAM-gated resident prefix + bitwise-proven CFG pairing), **video VAE decode 0.77 s (44.74-local)** — decode tail GPU-ported with existing ops (`WanRmsNormChannel`, `Permute0213` shuffle chain, `UnpatchifyVae`; exact-parity unit tests `LtxVaeDevicePortParityTests`); e2e 25f 512×320/20st ≈ **97 s**. **Phase 5 Gemma prompt cache shipped (44.75-local, 2026-07-11):** all four paired-CFG embeddings cached per (pos,neg) token key — repeat-prompt gens skip the whole TE phase (Gemma ×2 + connectors + ~12 GB TE upload, 5–22 s), warm same-prompt gen **48.3 s** with frames byte-identical to the miss gen; different-prompt miss path re-encodes correctly (`LTX2_CACHE_ROUNDTRIP=1` roundtrip in `LtxVideo2_Gpu_T2VA_ShortClip`). **Phase 5b resident-prefix persistence shipped (44.76-local, 2026-07-11):** shared weights + the block prefix stay device-resident across gens (Flux KEEP_MODELS idiom; count pinned, kills the 12→10→9 drift; TE-miss evicts only when Gemma doesn't fit — measured — then squeezes this gen and tops back up next gen; `DisposeCore` frees on model switch). Engine 4-gen roundtrip: HIT gens preload+prime 7.8 s → 0.15–0.3 s, walls 48.3 → **44.5–45.0 s**, gen-4 frames byte-identical to gen-3 (asserted). **Phase 4 audio half shipped (44.77-local, 2026-07-11): audio decode 0.13 s** — all vocoder + audio-VAE host loops GPU-ported with existing ops (`LtxAudioDeviceOps` transpose/slice/concat replicate-pads + crops, device MRF sums, `WanRmsNormChannel` pixel-norm w/ folded `eps′=sqrt(C·eps)`, batched-`Permute0213` mel flatten; the BWE transpose-then-flatten proved an identity relabel of the log-mel memory and was eliminated); exact-parity `LtxAudioDevicePortParityTests`; warm same-prompt Swarm gens **39.1/38.7 s**, video frames byte-identical to 44.76, audio waveform vs 44.76 cos 0.9999 / rms+peak dB within 0.03 dB. Remaining: F16 activations (Phase 3) — absmax probe recorded (`HARTSY_LTX2_PROBE=1`, real weights): video stream plateaus **15k–18.2k absmax over blocks 37–45** (audio ≤2k; no stream >60k so no hard F16 overflow, but only ~3.6× headroom to 65,504, and steps are weight-stream-bound at this geometry so the F16 wall win is capped — deferred with full numbers in the worklog). **Split-checkpoint path FIXED + verified (2026-08-01).** Root cause of the 2026-07-21 checkerboard: the split VAE file carries BARE keys, and the converter's bare-key router only sent `decoder.`/`encoder.`/`latents_` to the VAE bucket — `per_channel_statistics.{mean-of-means,std-of-means}` fell through to the Transformer bucket, so `ReadStats` found nothing and latent denormalization was an identity no-op. std-of-means goes as low as 0.074, so the decoder received channels up to ~13× too hot → the documented up-stack blow-up (±943) → RGB clamp saturation. One-line fix in `LtxVideo2CheckpointConverter.RouteKey` (route bare `per_channel_statistics` to `MapVae`); the mid-stack magnitude growth itself is normal (pixel_norm renormalizes — post-fix probe: denorm ±2.06, conv_out −1.80..1.42 ≈ [-1,1]). Verified: `hartsy video -m ltx-2` 512×320×25f 20 steps seed 42 → coherent convertible-at-sunset frames through the previously-broken path; transformer Sha256 now pinned. Bundled single-file checkpoints were never affected (their stats arrive as `vae.per_channel_statistics.*`). Swarm picks the fix up with the next NuGet publish. **Multi-GPU (2026-08-05)**: TE placement (incl. the Gemma-vs-prefix evict skip — the biggest single win) plus video AND audio VAE/vocoder on `VaeDevice` wired (2026-08-04 wave), awaiting checkpoint verification; pattern authority is MULTI_GPU_COMPONENT_PLACEMENT.md.

### LTX-Video 13B (0.9.7 dev)

**Real-weight coherent output confirmed (2026-07-01, 4090):** `ltxv-13b-0.9.7-dev-fp8.safetensors` (fp8-resident, ~15 GB, no OOM on 24 GB with `CacheWeightCasts=false`) → sharp photorealistic 704×480×25f at 30 steps. Reuses the 0.9.5 timestep VAE (identical config) + `V097` transformer (48 layers, head_dim 128, cross 4096). fp8 velocities NaN-free; 8 steps under-denoises (dev model needs ~30). Prompt adherence looser than 2B (cfg/STG tuning item, not a pipeline bug); numeric parity pending; ~25 s/step (fp8 dequant per GEMM, host-glue-bound).

### LTX-Video 0.9.5 (2B)

**Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.5.safetensors` → coherent cat-in-sunlit-garden (better prompt adherence than 0.9). Shares the 0.9 28-layer transformer; validates the **timestep-conditioned VAE decode path** (decode t=0.05 / noise 0.025) end-to-end. Required: a **0.9.5 VAE converter rename table** (`VAE_095_RENAME_DICT` — the 0.9 up_block regrouping would corrupt the 0.9.5 layout; selected via `IsTimestepVae`) + generalizing `LtxVideoVaeDecoder` to the residual channel-changing pixel-shuffle upsamplers (`upsampleFactor`/`upsampleResidual`, `time_embedder = 4·outC`, decoder_block_out_channels (256,512,1024)). Same faint striping artifact as 0.9; numeric parity pending. This VAE architecture is shared by the 13B (0.9.7).

### Kandinsky-5.0 T2V Lite (2B)

**Real-weight coherent output confirmed (2026-07-02, 4090, FIRST attempt):** `Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` (transformer BF16→F16 via `Kandinsky5CheckpointConverter.LoadDiffusersFolder`, config = `Kandinsky5Config.VideoLite2B` exact match) + the repo's shared HunyuanVideo 3D VAE → prompt-faithful, temporally-coherent snow-leopard clip (25f 512×512, 30 steps, CFG 5.0; `Kandinsky5_Gpu_T2V_ShortClip`). Text conditioning via the pre-computed Qwen2.5-VL/CLIP-L embeddings from the T2I dump (in-engine dual-encoder wiring still open). The clean first run is the HunyuanVideo dividend: the shared VAE inherited the GroupNorm-affine fix + replicate-pad fast path, and the pipeline was pre-hardened with `DecodeTiled` + per-step `FreeActivations(trimPool:false)` before first launch. **Perf: 2.9 s/step, 30-step clip in 102 s**, frames identical to the first-run golden (mean px Δ 0.038/255): (1) GPU RoPE — `Kandinsky5Rope.ApplyGpu` reuses `WanRopeInterleaved` (same GPT-J interleaved pairing) pre-permute with memoized preloaded tables; (2) **THE dominant cost was a single host `AddInPlace(temb, pooled)`** — the `DataPointer` read evicted `temb` from the GPU cache, and every block's modulation Silu re-uploaded the 2 KB tensor via a synchronous pageable H2D that **drained the whole queued stream per block** (memory `sync-h2d-stream-drain`; fixed with `backend.Add`). Remaining: SDPA unfused (~1.2 s/step at 7168 tokens), numeric parity vs the diffusers pipeline, I2V path (`EncodeFirstFrame`) unexercised, 121-frame/NABLA envelope untested. **In-engine dual-encoder wiring SHIPPED + CLI catalog-path verified 2026-07-21:** new `Kandinsky5VideoRecipe`/`Kandinsky5VideoRecipePipeline` (registered in `VideoRecipeRegistry`) replace the pre-computed-embeddings-only path with LIVE Qwen2.5-VL-7B + CLIP-L encoding, reusing the exact encode logic the CLI-verified T2I `Kandinsky5Recipe` uses (hoisted into a new shared `Kandinsky5TextEncoding` helper — the "promt engineer" ChatML template, layer-slice, CLIP pooling, and CLIP-L key-stripping are now one implementation for both T2I and T2V instead of two copies). `hartsy video -m kandinsky5-video` — catalog `Assets` list the T2V-Lite-5s-Diffusers `transformer/` + `vae/` shards (already staged, zero download; the bundled `vae/` IS the shared HunyuanVideo VAE in diffusers naming). Ran at the declared default (512×512, 25 frames, 50 steps, cfg 5.0, "a snow leopard walking across a snowy mountain ridge at dusk"): sampled frames 1/13/25 show a coherent, correctly-posed snow leopard with a continuous walking gait — correct on the first attempt, no bug found.

### Lance (ByteDance) video (3B)

**Real-weight coherent output confirmed (2026-07-21, 4090), CLI catalog-path verified:** `hartsy video -m lance-video` (`bytedance-research/Lance` repo, `Lance_3B_Video/model.safetensors` + `tokenizer.json`, ~13.2 GB downloaded and sha256-pinned this pass; the Wan2.2 VAE resolves as a side model, already staged). Ran at the declared default (512×512, 25 frames, 30 steps, cfg 4.0, "a cinematic shot of a cat walking through a sunlit garden, shallow depth of field"): sampled frames 1/13/25 show a coherent cat continuously walking across a flower garden lawn — correct subject and real motion on the first attempt, no bug found. `LanceVideoRecipePipeline` defaults to a real (non-empty) negative prompt unlike the then-existing empty-negative smoke test; that divergence did not manifest as a problem in the output. Multi-GB download deleted after verification per disk hygiene (sha256 pinned, re-fetchable).

### LTX-Video 0.9 (2B)

**Real-weight coherent output confirmed (2026-07-01, 4090):** `ltx-video-2b-v0.9.safetensors` + standalone fp8 T5-XXL → 25-frame 704×480 clip of a cinematic sunlit garden, prompt-faithful, temporally varying. Three bugs fixed to get from garbage→coherent (all in [PARITY_VERIFICATION.md]/memory `ltx-video-e2e-bugs`): (1) **VAE latent denormalization** was missing — decoder now applies `raw = latent·std_of_means + mean_of_means` from `per_channel_statistics` (was a blue lattice); (2) **caption projection** read `l = encoder.Shape[0]` = the batch dim (1) on the rank-3 `[1,L,4096]` T5 tensor, collapsing the caption to one token + a GPU OOB write — now derives `l` from element-count/last-dim (was blank output); (3) **T5 padding** attended unmasked in cross-attn — the generation entry now truncates to real tokens via `CfgHelper.SliceBatchElementPrefix` (was a dark, weakly-conditioned scene). RoPE / AdaLN order / final-norm / flow-match schedule+sign / timestep all verified to match diffusers and were correct. Remaining polish: a faint vertical-striping artifact in some frames; numeric layer-diff vs a Python reference still pending; perf is host-glue-bound (~5.7 s/step at 1320 tokens). **CLI catalog-path verified 2026-07-21:** `hartsy video -m ltx-video` — catalog `Assets` auto-resolve the already-staged checkpoint + T5, `CliDrivable=true`. Found and fixed a real recipe-vs-test divergence: `LtxVideoRecipePipeline` was slicing the full 128-token T5 batch and zero-padding past the real tokens (`VideoRecipeUtils.ZeroPaddedRows`) instead of truncating via `CfgHelper.SliceBatchElementPrefix` like the proven test/PARITY fix above — zeroing still attends the pad rows (softmax spends mass on them) and produced a completely off-prompt, near-static scene (a room with flowers, no cat) at both 25 and would-be 97 frames. Switched to `SliceBatchElementPrefix`; re-verified at the declared-default full clip (97 frames, 704×480, 50 steps, cfg 3.0): sampled frames 1/25/50/75/97 all show a coherent, correctly-lit cat walking through sunlit garden foliage with continuous subtle motion (tail sway, gait) and no late-clip drift/collapse. `VideoDefaults`/`VideoRequest` also gained per-family `Width`/`Height`/`Frames`/`Fps` (previously video always ran at a hardcoded 704×480/25f/25fps regardless of model); a `ParamState` bug that pre-seeded non-empty video tunable defaults (defeating the "unset → family default" contract every other modality follows) was fixed alongside. **Multi-GPU (2026-08-05)**: TE/VAE placement wired for the LTX-1 family (Wan-style ctor threading, 2026-08-04 wave), awaiting checkpoint verification.

### AnimeGen-T2V (aidealab)

Drop-in checkpoint for the ✅ Wan2.2 T2V-A14B MoE path — a full bf16 finetune of Wan-AI/Wan2.2-T2V-A14B (Apache-2.0), shipped as the standard dual-expert pair (`high_noise.safetensors` + `low_noise.safetensors`, 28.6 GB each, original Wan key naming, diffusers `WanTransformer3DModel.from_single_file`-compatible). No engine changes needed: `WanVideoCheckpointConverter` + `WanConfigDetector` + `WanVideoPipeline` (expert swap) handle it as-is; load via `WanVideoLoader.Load` with `WanVideoLoadOptions.LowNoiseDitPath`, or through the SwarmUI extension (high-noise as Model, low-noise in the **Refiner Model** slot — extension MoE pair wiring shipped 2026-07-16). Real-weight e2e validation pending a cloud-GPU run (weights not staged locally; ~24 GB VRAM with the single-resident expert swap).

### Wan family — remaining items

**Streaming video decode (Tier 3.5, 2026-08-11)**: `IVideoService.GenerateFramesAsync` + `WanVideoRecipePipeline.GenerateFramesAsync` stream frames to the caller as the VAE decodes them (`Wan22VaeDecoder.DecodeStreaming`'s temporal groups) instead of buffering the whole clip, and the SwarmUI extension pipes them straight to ffmpeg's stdin (`VideoOutputEncoder.EncodeStreamingAsync`). Scoped tight and honestly to plain T2V/I2V-TI2V (no init image/end frame, no concat-I2V, and the extension only takes this path when the request has no boomerang/trim/restore/audio — those need the full buffered clip). Real-weight verified byte-identical to the buffered path plus genuine incremental delivery (inter-frame arrival timestamps), and live-verified through the running service (valid 480x480/9-frame h264 mp4, visually inspected). LTX/Cosmos/Lance's own `GenerateFramesAsync` are NOT wired into this — Ltx/Cosmos fully materialize their decoded tensor before iterating it (no genuine streaming benefit today), and Lance was not exercised despite sharing Wan's `DecodeStreaming` pattern (no local checkpoint tested).

The family itself is ✅ above; still open: per-variant numeric parity beyond the T2V-14B layer-diff; full-res TI2V flash path is slow on CUDA (~94 s/step, monolithic kernel is a perf target). **Vulkan/AMD full-res OOM is FIXED** (2026-07-29): `sdpa_flash.comp.glsl` is a fused online-softmax SPIR-V kernel (`VulkanBackend.ScaledDotProductAttention`/`FlashAttention`) that never materializes the `[Sq,Skv]` score matrix — the exact previously-OOMing shape (B=1,H=24,S=16384,D=128, ~25 GB score matrix on the old 3-pass path) now completes; see `benchmarks/scoreboards/VULKAN.md` and `docs/Checklists/TROUBLESHOOTING.md`. Still gated to head dims <= 128 (falls back to the materialized path above that) and doesn't support softcap/sink/ALiBi — none of which Wan uses, so this closes the OOM for Wan specifically; AMD/Intel hardware validation itself remains 🔒 (no hardware available, see ROADMAP.md §3). **fp8-CFG low-res darkening**: T2V-14B at 320×192/cfg5 is dark even at 20 steps on BOTH GEMM paths (baseline clip mean 12.5 = native 8.9) — same family as the S2V CFG sensitivity; part of the parity follow-up.

### Native fp8 GEMM — VALIDATED on Ada (2026-07-02, first exercise)

`HARTSY_FP8_NATIVE=1` on the 4090: **~3× faster steps on T2V-14B fp8 (1.65 s/step vs ~5.0 s at 320×192/20 steps)** via `Fp8GemmExecutor` (cublasLtMatmul, fp8 weights consumed directly + dynamic absmax e4m3 input quant). Numerics match the F16-cast baseline at the same config (near-identical per-frame profiles, slightly dimmer — within the config's CFG-dark regime). **Production recipe for Ada+: `WAN_NO_CACHE_CASTS=1 HARTSY_FP8_NATIVE=1`.** Stays opt-in until the fp8-CFG parity item resolves; then evaluate default-on for SM≥8.9.

### LTX-2 (19B, superseded)

The earlier 19B dev checkpoint is architecturally divergent from the code (2.3) — no prompt-mod/gated-attn (both since made optional so they no-op on it), and a single shared `aggregate_embed` (49·3840→3840) + two 3840-dim `{video,audio}_embeddings_connector`s vs 2.3's separate video-4096/audio-2048 connectors. Deleted in favor of the 22B (memory `ltx2-19b-vs-23-divergence`).

### Wan2.2-S2V-14B

**Real-weight e2e PASSED (2026-07-02, 4090, `WanS2V_Gpu_E2E`)** after the faithful rewrite to ComfyUI `WanModel_S2V` (CausalAudioEncoder + AdaIN injector ×12 + cond-mask + per-frame timesteps + reference token-append; Wav2Vec2-large stable-layer-norm front-end, legacy `weight_g/v` pos-conv fallback). Audio injector + cond adds fully GPU-resident. **Memory profile clean** (free VRAM flat across 20 steps — every earlier "leak"/OOM was GPU contention). **Numeric parity PROVEN (2026-07-02, `tests/python-reference/s2v_reference`):** full layer-diff vs a faithful ComfyUI `WanModel_S2V` CPU port on the real fp8 checkpoint — BOTH CFG branches match at all 34 stages (velocity relL2 ~2e-3 = fp8 noise floor) and the **guidance direction matches** (relL2 6.6e-3, DC mean 0.0642 vs ref 0.0640). The cfg-darkening reproduces in the reference math on the same inputs → it is the synthetic e2e conditioning (gradient "face", sine "speech") sitting off-distribution, NOT a code defect. Detector default stays **cfg 2.0 + CfgRescale 1.0** as a conservative production default. **REAL-CONTENT SWARM VALIDATION PASSED (2026-07-09, engine 44.19-local):** real speech (jfk.wav 16 kHz) + a real identity portrait through the SwarmUI extension → identity-faithful talking head with clear mouth articulation (gallery 2259001). Extension fixes that unblocked it: `WanConfigDetector`-sourced config (the loader's every-Nth inject-layer guess was WRONG vs the parity layout), VAE encoder + Init-Image→referenceRgb24 wiring, `ZeroPaddedRows` on the umT5 embeds, SigmaShift→FlowShift (default 8, ComfyUI parity), `Wav2Vec2Large` side-model re-pointed to the Comfy-Org `wav2vec2_large_english_fp16` repackage (old facebook URL 404'd) with the `wav2vec2.` key-prefix strip. **The driving speech is muxed into the mp4** (`VideoOutputEncoder.AudioTrack`, trimmed to clip length) at Wan's native **16 fps default** (audio features are bucketed at 16 fps — any other rate desyncs the mouth; explicit user Video FPS still wins). **CORRECTION (2026-08-01): that mux REGRESSED and was silently absent from this date until now.** When the extension was thinned into a wrapper over `IInferenceEngine.Video`, the frame-only pipeline contract severed the supply: `VideoOutputEncoder.AudioTrack` was never constructed at any call site (dead code), and `VideoRequest.VideoAudioInput` was documented as a mux track with no consumer anywhere in the Engine. Restored today — `WanS2VRecipePipeline` attaches the speech it consumed to its `VideoGenerationResult`, now at the clip's **source rate/channels** rather than the 16 kHz mono conditioning downmix this sentence implies. Not yet re-verified on a real S2V run. **Perf SOLVED (2026-07-09, 44.21): 5.5 s/step (output bit-identical), production 49f 480² in 2.66 min through Swarm w/ sound (gallery 0204001; ComfyUI same job 90 s warm).** Root cause was `WanVideoBlock`'s multi-group (per-frame-timestep) modulation/affine/gate HOST loops — the branch only S2V/TI2V/Animate take — bouncing the hidden tensor 4×/block/forward; now GPU-resident (Permute+SliceRows+AffineBroadcast for modulation, UpsampleNearest2D group-expand + Mul/Add for scale/shift/gate, CPU fallback for non-divisible ref-token layouts). TI2V/Animate inherit it. **Round 7 (2026-07-11, 44.81-local): the LAST per-step host glue ported** — `WanDitOps` multi-group `ConditionTimeGroups` (per-group host copies of temb/proj = 2G stream drains/forward; now device `Concat`, values bit-exact) and `FinalLayer` G>1 (full-hidden D2H + CPU modulate + re-upload x2 CFG/step; now device LayerNormNoAffine/AffineBroadcastLastDim, host fallback when G does not tile S). Steps 4.15 -> 3.44 s/step (17%) in the 8-step harness A/B; dump parity = bit-identical through block 39, velocity relL2 2.5e-4 (10x under the fp8 noise floor); per-step `TrimMemoryPool` PROVEN a win (3.44 vs 4.23 s/step trim-off) and stays default-on (`WAN_S2V_TRIM=0` knob). Swarm 49f 4802 production: 2.38 min gen, back-to-back gens md5-identical, sound muxed + speech-level, frames viewed identity-faithful. NEXT: `WanS2VLoader` has NO warm cache (same-model gen 2 rebuilds the full pipeline + re-encodes TE, ~9 s + encode per gen) — transplant the round-6 `WanVideoCacheEntry` pattern. FramePackMotioner (multi-clip extend) still TODO.

### Wan2.2-Animate-14B

**Real-weight e2e PASSED (2026-07-02, 4090, `WanAnimate_Gpu_E2E`)** — the healthiest output profile of the variant fleet (per-frame means 128–155, clip mean 139, 20 steps @ 12.5 s/step, fp8 KJ v2 checkpoint). Faithful rewrite to `AnimateWanModel` (StyleGAN motion encoder w/ QR direction basis, FaceAdapter fusers every 5th block, pose latent add at frames 1.., reference latent frame + inverted mask concat, ref token-prepend variant, i2v CLIP). **Perf fix that unblocked e2e:** the face clip is denoise-constant, so `EncodeMotion` runs ONCE per CFG branch (`WanAnimatePipeline`) instead of inside every forward — the per-forward host-side StyleGAN encode made a 20-step run exceed 30 min with zero steps completed. **✅ CHECKERBOARD SOLVED (2026-07-11, round 11, `44.84-local`).** The uniform ~16 px halftone tile (period = exactly one DiT token = a 2×2 latent patch × 8× VAE) was NOT an Animate-arch bug — it was a **dropped fp8 weight scale in the checkpoint converter**. `CheckpointConvertUtils.ApplyFp8ScaledDequant` folded the per-tensor `.scale_weight` companion into `Tensor.Fp8ScaleFactor` **only if the companion was F32** (`scaleT.DType == DType.F32`). Every earlier-validated Wan fp8 checkpoint (t2v/i2v/s2v_14B) ships **F32** scales, but the **Kijai `wan2.2_animate_14B_fp8_scaled_KJ_v2`** checkpoint ships them as **BF16** — so the guard silently dropped all 514 block/attn/ffn scales (~0.13–0.46), the raw fp8 weights (±448) ran ~5× hot, block-0 activations exploded (stage-dump `HARTSY_ANIMATE_DUMP`: patchified rms 0.17 → **block-0 out rms 4.4e5**, runaway to 2.7e6 by block-1), and by the head every token collapsed to the dominant singular direction (interTokenStd/rms **0.005**) → identical patches everywhere = the tile. Fix = read the scalar scale as F32 for any F32/F16/BF16 companion (F32 path byte-identical, so **zero regression** to the F32-scale Wan variants — verified all use F32). Post-fix stage dump: block-0 rms **0.72**, head interTokenStd/rms **0.92**, velocity rms 0.65; the tile is gone. Second fix: `WanAnimateTransformer.BuildMotion` now runs the StyleGAN motion encoder in **8-frame chunks** (comfy `encode_bs=8`) — the all-frames-at-once 512² conv stack OOM'd beside the resident 16.4 GB fp8 DiT. **Swarm real-input validation (44.84-local): 17f 480² e2e completes, smooth checkerboard-free video** (Z-Image red-hair portrait as reference + a Wan-T2V dancing clip as the driving video; content is abstract because a raw clip ≠ a pose-rendered skeleton+face-crop, which the loader documents as the user's responsibility). Diagnostics kept env-gated: `HARTSY_ANIMATE_DUMP=<dir>`, `HARTSY_ANIMATE_NO_POSE/NO_FACE=1`. Flagships re-verified on 44.84 (Z-Image 2.82 s, Krea2 4.52 s, coherent); Wan T2V (ti2v-5B) regen coherent. **Update 2026-08-07:** the pose-rendered driving-input recipe shipped natively — `VideoRequest.DrivingVideo`/`DrivingPoseVideo`/`DrivingFaceVideo`/`DrivingAutoPreprocess` (gated by `VideoFeatures.DrivingVideo`), YOLO11-pose skeleton render + face crop ported from the extension preprocessors into `Recipes/Video/WanAnimate*` (extension's dead copies deleted), CLI `--driving-video`/`--pose-video`/`--face-video`/`--no-auto-preprocess`; single-still tiling kept as the no-driving-video fallback. Real-weight e2e with a driving clip still pending (checkpoint not on the dev box). Still open (non-blocking): replace-mode background/mask conditioning, `continue_motion` chunked extension.

### Cosmos-Predict1 V2W (5B / 13B)

=2.86e-5**, FSQ tokens **31/32 bit-exact** (the 1 flip is a provable F32 half-integer rounding tie). `CosmosArTransformer` (3D RoPE T/H/W + per-layer non-causal T5 cross-attn + additive 3D abs-pos), `CosmosV2WPipeline` (prefill→KV AR loop→detokenize→DV-decode→ffmpeg), `T5_11B` preset, `.pt` `CosmosArCheckpointConverter`, CLI catalog — all BUILT; structural CPU tests pass; runs e2e on synthetic weights. **OPEN:** DV **decoder** (pixel render) arch recovered but not yet ported; AR-backbone real-weight layer-diff + full V2W e2e **blocked on disk** (5B 9 GB + T5-11B ~22 GB ≈ 31 GB, local box full); 6 arch assumptions (`rope_theta`, `fuse_qkv`, cross-attn dims, abs-pos name, RoPE split, FSQ convention) flagged in code pending the AR `.pt` key dump. **CLI catalog wiring explicitly DEFERRED (2026-07-21, video catalog pass):** no `IVideoRecipe` wrapper is registered for either `cosmos-predict1-5b-v2w` or `cosmos-predict1-13b-v2w` — even a perfect wrapper couldn't produce pixels today because the DV decoder isn't ported, so `hartsy video -m cosmos-predict1-*` would need a decoder-port implementation, not catalog plumbing. Out of scope for a catalog-wiring pass; do not attempt a decoder port to "complete" this — track it as its own bringup task. The two catalog entries stay short-form/`ValidationPending`/not-CLI-drivable until the decoder lands.
