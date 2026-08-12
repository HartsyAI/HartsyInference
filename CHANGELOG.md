# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N` (the scheme moved
up from `1.0.0-alpha.N`; entries below that pre-date the change and keep their original numbers). The single
source of truth is `<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props` — see
[`docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md`](docs/Checklists/ROADMAP.md) for what a
stable release will require. Dates are UTC.

## [Unreleased]

### Removed
- **`VideoRequest.VideoExtendModel`** — never consumed by any recipe on either side of the contract since the
  DTO's introduction (video extension was explicitly out of scope in the extension's own plan). Breaking
  record-shape change for transports that set it; the SwarmUI extension's mapping was removed in the same pass.

### Added
- **ComfyUI `int8_tensorwise` quantization, resident (± the `convrot` Hadamard rotation).** This is the format
  the *official* Lightricks LTX 2.5 and Comfy-Org MiniMax-H3 quantized releases ship in, and it was previously
  rejected by name at load. Weights now stay int8 on the device at 1 byte/param instead of expanding to BF16,
  so a 21 GB DiT fits a 24 GB card. Format notes:
  [`docs/Research/QUANTIZATION_COMFY_FORMATS.md`](docs/Research/QUANTIZATION_COMFY_FORMATS.md).
  - **`Tensor.QuantInfo`** (`QuantWeightInfo`) carries a packed weight's companions — per-output-row scale,
    ConvRot group size, `full_precision_matrix_mult` — the way `Fp8ScaleFactor` already carried fp8's scalar.
    Attaching them to the weight rather than to a per-model linear wrapper is what lets one backend branch
    serve every model that loads such a checkpoint, with no change to any model's code.
  - **`CudaBackend.Linear`** gained a resident-int8 branch reusing the existing W8A8 chain: activation ConvRot
    (`convrot.ptx`, a radix-4 butterfly — `H` is `kron(h4, …)`, so no matrix is materialized), per-row dynamic
    int8 quant, cuBLASLt IMMA, then the `rowScale·wScale + bias` epilogue. Chunked over rows against live free
    VRAM, because the int32 accumulator is 4 bytes per output element and the weights have already filled the
    card. Short sequences pad to the 32-row IMMA granularity rather than falling back, matching comfy-kitchen.
  - **`Int8ConvRotCodec`** provides the un-rotating dequant for the CPU/Vulkan backends and for layers tagged
    `full_precision_matrix_mult`. Verified against comfy-kitchen's eager reference at **relL2 5.1e-8–2.7e-7**
    with F32 activations; `H` itself matches bit-exactly.
  - **`ComfyQuantDescriptor`** replaces three divergent private copies of the `.comfy_quant` blob parser. The
    per-layer blob is now authoritative over the file-level `_quantization_metadata` mirror, which re-quants of
    the same model disagree with.
  - MiniMax-H3's `int8_convrot` rejection is deleted; `MiniMaxH3Assets` no longer sinks `convrot` filenames.
- **ComfyUI `nvfp4` weights stay resident too.** They were unpacked to BF16 at load, which turned the official
  18.72 GB LTX-2.5 distilled nvfp4 DiT into 42 GB. `Nvfp4Codec.TryAttachResident` now relabels the packed
  weight to `DType.F4E2M1 [N, K]` — the dtype already existed for exactly this — and `CudaBackend` dequantizes
  it in-kernel per GEMM under the existing `CacheWeightCasts` budget. Bit-exact against the host reference on
  real `qwen3vl_32b_minimax_h3_nvfp4_awq` layers. A **VRAM** win only: no consumer GPU here has FP4 tensor
  cores, so the GEMM still runs in F16. Opt-in per caller, since the eager unpack is what CPU/Vulkan need and
  AWQ layers with `pre_quant_scale` must take it regardless.
- **`Tensor.ReinterpretAs`** — a byte-count-validated, keep-alive-rooted dtype/shape view. `Reshape` could not
  serve: it holds element count fixed, and the whole point here is that one U8 byte becomes two F4E2M1 elements.
- **LTX-2.5 pipeline wiring — the Gemma 4 tower is now driven, not just built.** `Gemma4TextEncoder` and
  `Gemma4Tokenizer` existed and were parity-checked but had no consumers; `LtxVideo2Recipe` still constructed the
  Gemma-3 tower and refused a 2.5 bundle with a targeted error.
  - **`ILtx2TextTower` / `ILtx2PromptTokenizer`** name the contract the pipeline was already relying on
    structurally. Both encoders already exposed `EncodeMultiLayer`/`EnumerateWeights`/`NumLayers` with identical
    signatures, so neither implementation changed. `LtxVideo2Pipeline`'s 3840 caption channels and 49 harvested
    states hold for both families.
  - The recipe branches on `model.layers.0.layer_scalar` — **not** on a missing `v_proj`, which would misclassify
    every Gemma 4 checkpoint, since layer 0 is a sliding layer and has one. Gemma 4 conditions at 1024 tokens
    against Gemma 3's 256; that length is part of the conditioning, because the connector replaces learnable
    registers positionally.
  - The tokenizer is built from the `tokenizer_json` U8 tensor LTX-2.5 embeds **inside** its text encoder — there
    is no side file anywhere to fall back to.
  - **Converter routing**: a standalone Gemma 4 tower ships bare `model.layers.*` keys with no `text_encoder.`
    prefix and was falling through to the DiT mapper. Those now route to the text-encoder bucket, checked after
    the connector rule because `text_embedding_projection.*` lives in the same file but belongs to the connectors.
    Packager-embedded `hf_asset__*` side files (chat template, tokenizer/processor config) are dropped — they are
    not weights. Verified on the real 15.37 GB checkpoint: 328 packed int8 tower weights, zero leaking into the DiT.
  - Real-weight verified: the int8-convrot Gemma 4 encoder produces finite, deterministic, prompt-discriminating
    conditioning on a 4090 — identical tokens give bit-identical states, differing tokens diverge.
  - Unchanged on purpose: the diffusion-decoder guard (decode through the conv VAE — the diffusion decoder is
    managed-only), and both documented divergences from upstream (the 49th state's final norm, and padding side),
    which also affect the shipping 2.3 path.
- **LTX-2.5 support (components complete; end-to-end generation not yet wired).** Every piece is ported and
  checked against the reference, but the LTX-2 pipeline still constructs the Gemma-3 tower and the
  convolutional decoder, so a full 2.5 bundle is now refused with a targeted error instead of being
  mis-decoded. Details and the two open questions are in
  [`docs/Research/LTX_2_5.md`](docs/Research/LTX_2_5.md).
  - **Variant detection** replaces the hardcoded `LtxVideo2Config.V23`: `SafeTensorsLoader` exposes a file's
    `__metadata__`, and `LtxVideo2VariantDetector` resolves a config from it with tensor-key probes as the
    fallback — probing whenever the architecture config did not state a value, since repacks routinely ship
    metadata with no LTX config, and letting key presence win for the keyframe marker as ComfyUI does.
  - **Transformer**: `keyframes_abs_pos_embedding` applied to the first latent frame's tokens in all three
    forward paths including the captured step graph, plus a load-time cross-check that rejects a checkpoint
    contradicting the detected variant. The 2.3→2.5 architecture diff is only two config keys.
  - **`IBackend.Na3d`** — 3D neighborhood attention (NATTEN window semantics: the window slides inward at
    borders rather than truncating), verified against the reference at relL2 < 1e-6.
  - **`LtxVideo25DiffusionDecoder`** — the new diffusion video decoder, relL2 2.3e-7 on final pixels and a real
    decode of the shipped checkpoint. Managed-only, so it is a numerical reference rather than a fast path.
  - **`Gemma4TextEncoder` + `Gemma4Tokenizer`** — per-layer alternating geometry (global layers carry one KV
    head at a different head dim, no `v_proj`, and a 25% partial rotary), RMS weights stored directly rather
    than as Gemma 3's `1+w`, and a rank-merge BPE tokenizer that is bit-exact against the HuggingFace library
    on the real 262k vocab.
  - **Distilled sampling**: a checkpoint's baked-in sigma schedule replaces the dynamic flow-match shift, and
    the unconditional branch is skipped when both guidance scales are 1.
  - **Catalog**: `ltx-2.5` and `ltx-2.5-distilled` as separate ids, because the two checkpoints are
    byte-indistinguishable and only the id can carry which schedule was intended.
- **Device-resident CFG+Euler for the image denoise loops** (CUDA image bring-up): Lance, Lumina2, HiDream,
  F-Lite, Kandinsky5, SD3, and Z-Image's fast path now run guidance + the Euler update in-place on device via
  `CfgEulerStep` and the new fused `CfgRenormEulerStep` (Lance renorm), `CfgNormalizedEulerStep` (Lumina2
  `cfg_normalization`), and `AffineMix`/`MaskedAffineMixInPlace` (SD3 img2img noise + masked-inpaint blend/
  recomposite) — replacing per-step scalar host loops over `DataPointer`. New `MixContract`/`PatchTokenContract`/
  `SplitContract` validate the op geometry identically across backends. BF16 elementwise dispatch (Gelu, Clamp,
  GeGlu, RepeatKvHeads) no longer silently falls through to the F32 kernel — GeGlu gained a real BF16 kernel,
  RepeatKvHeads a 16-bit bit-copy launcher, and unsupported dtypes now throw.
- **Z-Image lifecycle hardening**: Base/Turbo checkpoint-variant detection from the filename, two independent
  prompt-cache layers, a Qwen3 tokenizer rewrite (byte-level encoding gap, `<think>` handling, tokenization-
  boundary fix) golden-tested against HF tokenizers 0.22.2 output, and a CUDA-graph-captured denoise step for
  the packed fast path. Scope note: the packed device-resident loop covers t2i, img2img, and CFG; masked
  inpaint and regional conditioning still run the host-stepped loop (per-step `ApplyZImageCfg`/`scheduler.Step`
  on the CPU) — porting those onto the SD3 `MaskedAffineMixInPlace` pattern is tracked follow-up work.
- **Kernel build reproducibility**: `conv/`, `vision/`, and `wan/` gain the same `build.sh` (nvcc, or the
  committed `nvrtc_compile` fallback) the other kernel domains already had — their shipped PTX previously had
  no scripted rebuild path at all; `dequant/build.sh` now covers `w8a8` (sm_75) and `fp8_quant` (sm_80, per its
  shipped target) and gains the nvrtc fallback; `dit/build.sh` covers `mg3_action`. 13 of 16 artifacts
  reproduce bit-identically from source; `stepcache`/`w8a8`/`wan_vae_norm` are regenerated with the current
  pinned toolchain (all kernel-family GPU tests pass against the regenerated artifacts).

- **Wan 2.2 A14B dual-expert swap through the native contract** (regression restore): `VideoRequest.VideoSwapModel`
  + `VideoSwapPercent` (fraction of steps for the low-noise expert; null = official 0.875/0.9 boundary) →
  `WanVideoRecipe` loads the second DiT and warps the fraction through the flow shift
  (`boundary = s·p/(1+(s−1)·p)`); swap-aware pipeline cache key; CLI `--swap-model`/`--swap-percent`.
- **FLUX.1 Redux through the native contract** (regression restore): `redux.stylemodel`/`redux.multiply`/
  `redux.merge` (number 0..1)/`redux.apply_start` Extra keys drive `ReduxResolver` from `Flux1RecipePipeline`;
  prompt images ride `IpAdapter.PromptImages`; Flux declares `ImageFeatures.IpAdapter` (Redux only — real
  IP-Adapter checkpoints are refused with a clear message). CLI `--style-model` + redux knobs.
- **Wan-Animate driving video**: `VideoRequest.DrivingVideo`/`DrivingPoseVideo`/`DrivingFaceVideo`/
  `DrivingAutoPreprocess` with in-engine YOLO11-pose skeleton render + face crop (ported from the extension's
  dead preprocessors); single-still tiling kept as fallback. CLI `--driving-video`/`--pose-video`/`--face-video`/
  `--no-auto-preprocess`.
- **`VideoFeatures.ReferenceImages/ReferenceVideos/ReferenceAudios/DrivingVideo`** gating bits — closes the
  silent drop of reference conditioning on families that never consumed it.
- **`ImageRequest.InstructPix2PixCfg` wired** to OmniGen2 (default 2.0) and Boogu (default 1.0) dual-CFG edit
  paths; CLI `--ip2p-cfg`.
- **Multi-GPU sharding, placement & parallelism — the full opt-in feature set** (`PlacementConfig` /
  `EngineOptions.Placement`; all-defaults is byte-identical to single-GPU). Works over plain PCIe with
  no P2P/NVLink (host-staged boundaries; P2P used when available). User guide: `docs/MULTI_GPU.md`.
  - **LLM layer split** (`ShardDevices`, or a `"cuda:0+cuda:1"` composite device key / `--device` in the
    CLI): N-way transformer layer split planned from live free VRAM (or explicit `ShardRatios`), per-stage
    asymmetric weight preload, KV cache per-layer on its stage's card, logits/sampler on the last stage.
    Verified: Llama-3.2-1B split = exact token parity vs single-GPU; **Qwen3-32B Q4_K_M (19.8 GB) OOMs a
    24 GB 4090 alone and runs at ~12.1 tok/s split across 4090+3060**. Exclusions: SSM, Gemma-4 PLE, VLM
    sidecars (warned + skipped); CUDA-graph/speculative decode disabled while staged.
  - **DiT block sharding** (`EnableDitSharding`, exactly 2 devices; CLI `--dit-shard-gpu`, extension
    `DitShardGpuId`): diffusion transformer block-range split with pooled (never replicated) weights.
    Verified on real weights: Krea2, Qwen-Image 20B, Flux.1 (plain generations; ControlNet/Kontext/
    inpaint/regional auto-fall-back), Chroma, HunyuanImage 2.1, MiniMax-H3 fp8. Disables step-graph/
    step-cache/block-streaming while sharded; mutually exclusive with CFG-parallel.
  - **Audio-LM layer split + precision policy** (CLI `--lm-shard-gpu`, extension `LmShardGpuId`): YuE's
    7B Stage-1 rides the same layer-split machinery; the load-time Q4_K quantization became a policy
    (`HARTSY_AUDIO_LM_QUANT=q4k|q8|off`) defaulting to **un-quantized bf16 when sharded** — pooled at
    8.7 + 4.3 GB across 4090+3060. `MusicLoadContext` carries shard backends + precision to loaders.
  - **TE/VAE component placement** (`TextEncoderDevice`/`VaeDevice`; CLI `--te-gpu`/`--vae-gpu`): Wan
    TI2V-5B 43.7 s → 32.7 s with umT5 on the second card; SDXL SSIM 0.9998; Flux/Qwen-Image/Chroma/
    HunyuanImage/LTX-1/LTX-2 wired. Composes with DiT sharding.
  - **CFG-branch parallelism** (`CfgParallelDevice`; CLI `--cfg-parallel-gpu`): negative branch runs
    concurrently on a second card with replicated weights (~1.8-1.9× per-step concurrency; Wan +
    Flux true-CFG), observably falling back to sequential when the replica doesn't fit.
  - **Same-GPU dual backends**: two engine instances share one physical GPU with isolated
    streams/caches/mempools; serialized per-ordinal by default (`HARTSY_SAME_GPU_CONCURRENT=1` opts into
    concurrent mode, which has a known allocator issue near VRAM capacity — left off).
  - Verification: `tests/run-multigpu-campaign.sh` (real-weight, fail-on-missing-checkpoint) covering
    every mode; measured tables in `benchmarks/results/2026-08-05_multigpu_speeds.md`.
- **YuE full-quality pipeline activated**: Stage-2 (m-a-p/YuE-s2-1B-general, cb0 → all 8 codebooks) and
  the per-stem 44.1 kHz Vocos vocoders are now weights-catalog entries (auto-download); the vocoder
  torch checkpoints auto-convert to safetensors on first load (`EnsureVocoders`, same pattern as
  x-codec). Without them YuE silently degraded to the vocal-cb0-only 16 kHz draft — the "garbled" mode
  (Whisper transcribes it as nothing; the full pipeline transcribes supplied lyrics near-verbatim).
  CLI `hartsy music` gained `-g|--genre` (YuE's prompt is the LYRICS — `[verse]`/`[chorus]` markers —
  and genre carries the style tags).
- **MiniMax-H3 ("Hailuo 03") — full port, real-weight video + audio verified on a 12 GB RTX 3060.** Single-stream
  packed-token DiT (`[text | cond | audio | video]`, hidden 5376, 50 blocks) denoising 24-channel video and 32-channel
  40 Hz stereo audio jointly, with a ViT3D video VAE, a DAC/BigVGAN audio VAE, and an NVFP4-AWQ Qwen3-VL text encoder.
  Verified end-to-end: 512×288 / 39 frames / 30 steps produces a tracking shot of a dog splashing through a stream with
  a 1.625 s stereo soundtrack at −12.5 dBFS peak. The 66 GB bf16 DiT is mmap-backed and loads at 943 MB RSS.
  Layout, dual-shift schedule, audio row packing and final-layer modulation rows are byte-identical to upstream
  ComfyUI master; the audio VAE decoder matches the reference module shipped inside the checkpoint at relL2 4.9e-6
  (CPU) / 1.3e-3 (CUDA), covered by the new `MiniMaxH3AudioVaeParityTests` + `tests/python-reference/
  minimax_h3_audio_vae_ref.py`.

- **SeedVR2-7B support (v1 NaDiT port).** The 7B checkpoint is the V1 architecture (`models/dit`, not the
  3B's `dit_v2`) — same windowing/attention/AdaSingle, but a different RoPE (pixel-basis freqs
  `linspace(1,128,10)·π` matching the checkpoint's `rope.rope.freqs`, 60 of 128 head dims rotated,
  positions normalized to `linspace(−1,1)` over each window axis, applied to video only — text is never
  rotated), plain GELU-tanh MLP with biases, all 36 blocks fully split (no `mm_layers`), and no tail
  norm/ada or last-layer text shortcut. `SeedVr2Config.Detect` now configures all of this from the
  plain-MLP+no-tail signature instead of throwing. Parity: a v1 tiny-config dump against ByteDance's own
  `models.dit.nadit` (`seedvr2_transformer_v1_parity_dump.py`) passes at blocks ≤9.1e-4 / output 8.96e-4
  (`Dit_TinyConfigV1_ForwardMatchesReference_PerBlock`).
- **SeedVR2 BF16 VAE activations** (CUDA default; `HARTSY_SEEDVR2_VAE_F32=1` reverts): the fp32
  whole-clip activation peak that OOM'd 24 GB at 720p-area is halved — 720p-area now restores a full
  25-frame clip at a measured 13.3 GB peak, 960×540-area at 9.0 GB, and **the 12 GB 3060 runs 960×540-area
  end-to-end (peak 7.8 GB)**. Pixel/latent boundaries stay F32; the mid-block attention runs F32 (the
  known F16-attention precision class). BF16 variants of the five `wan_vae` glue kernels; BF16 admitted
  through `SliceRows`/`Permute0213`. Output vs the f32 path: SSIM 0.9998. 1080p-area still exceeds 24 GB —
  tiled/sliced VAE remains open.

- **LTX-2.3 audio guidance rescale — the near-silent soundtrack is ~14 dB louder.** Root cause measured:
  CFG over-disperses the audio latent (σ 0.89 at guidance 1 → 2.22 at 3 → 2.69 at 7, against the
  checkpoint's own 1.17) and the decoded level falls with it; the audio VAE is not at fault (fed a
  training-distribution latent it decodes to a healthy level). `AudioGuidanceRescale` (default 1.0) applies
  diffusers' `rescale_noise_cfg` to the audio stream, restoring σ to 1.141. Real 20-step generation
  (512×320×25f, seed 42): **peak −43.9 → −28.2 dBFS, RMS −59.4 → −45.4 dBFS**, video unchanged.
  Implemented as an affine transform so only four scalars reach the host and the step stays on-device.
  Also adds `AudioGuidanceScale` (null = follow the video scale, the reference default) with
  `HARTSY_LTX2_AUDIO_CFG` / `HARTSY_LTX2_AUDIO_RESCALE` overrides. **Still ~20 dB below a healthy
  soundtrack** — the reference's STG + modality-isolation guidance remain unimplemented; see
  MODEL_STATUS_VIDEO. Note for anyone tempted: raising audio guidance to the authors' recommended 7.0
  *alone* makes it ~17 dB worse, because that recommendation assumes the rescale/STG stack.
- `HARTSY_LTX2_PROBE=1` now also dumps the audio stages (latent pre/post-denorm, VAE log-mel, vocoder
  waveform) via a shared `ProbeTensor` helper.

### Changed
- **`LlamaStyleEncoder` attention glue is device-resident** — the per-layer CPU reshape/RoPE/GQA-repeat/merge
  `float*` loops are replaced with the existing `Permute0213`/`ApplyRopeSingleHeadMajor`/`RepeatKvHeads`
  kernels (this encoder is shared by Qwen-Image, Z-Image, Krea2, Boogu, Flux.2, Ideogram 4, Lumina2, OmniGen2
  and others); tests assert the D2H sync count, not just numerics.
- **Lumina2 sampling schedule corrected — output images change.** The scheduler previously applied Flux-style
  dynamic shifting derived from the image token count (an experiment its own comment marked VALIDATION-PENDING,
  using Flux's base/max-shift constants); the official Alpha-VLLM/Lumina-Image-2.0 `scheduler_config.json` is
  `shift: 6.0` with `use_dynamic_shifting: false`, so the pipeline now uses the checkpoint's static shift. Same
  seed produces a (correctly) different image than prior releases.
- **SD3 patchify/final-layer/masked-mix run on device**; `flash_attn_v2_tf32` rejects partial query tiles
  (OOB read) and zero-fills its shared-memory K/V tail (stale-value poisoning); MaxPool distinguishes an empty
  window from a valid all-−Inf one; MSDA uses true −Inf softmax init and 64-bit index products; the cuDNN SDPA
  plan cache is keyed by attention scale; the step-cache treats a zero-denominator relative distance as
  Infinity (was a false cache HIT); Sage's F32→F16 V-narrowing is opt-in (`HARTSY_SAGE_UNSAFE_F32_V_NARROW`).
- **SeedVR2 DiT is device-resident** — the bring-up host-math forward (window gather/scatter, rope,
  qk-norm, AdaSingle on CPU spans; ~200 stream drains per forward) is replaced with backend-op
  composition: fused `QkvSplitNorm`, `RowGather`/`RowScatterAdd` window packing over cached per-geometry
  index tensors, `WanRopeInterleaved` with per-token identity-padded tables, and modulation vectors
  precombined per timestep (constant 1000) and cached across chunks (`SeedVr2DevicePlan`). No new
  kernels beyond GPU-resident `SeedVr2PixelShuffle`/`SeedVr2PadBottomRight` (the VAE upsampler/downsampler
  host loops, previously a multi-GB D2H+H2D round trip each). **Measured e2e: 960×540-area 14.5 → 2.7
  s/frame (362 → 68 s); 720p-area 25.7 → 8.4 s/frame (4090, BBB 25 f). The 3060 runs the same clip in
  169 s.** Existing tiny-config parity numbers are unchanged to the printed digit; per-chunk phase timing
  is logged at Debug level.

### Fixed
- **Z-Image Base checkpoints were silently corrupted when the filename carried no variant token.** The official
  Base release ships under the bare family name (`z_image_bf16.safetensors`); variant detection fell through to
  Turbo's policy, whose F16 attention narrowing overflows Base's >83k value-projection range into Inf — a
  garbage image with no error. Bare family naming now positively detects Base, and genuinely ambiguous
  filenames default to the numerically safe Base policy (F32 attention, shift 6) with a loud warning — a
  misfiled Turbo merely runs slower, instead of a misfiled Base corrupting. Verified with a real generation
  from the official Comfy-Org single-file on the exact previously-corrupted filename.
- **Z-Image `ReleaseDeviceCache` could leave the captured denoise step graph pointing at freed memory** — the
  graph bakes the caption-pin and RoPE-table device addresses it frees, so a later same-signature forward
  could replay against freed allocations (CUDA 700 context poison). The release path now invalidates the
  graph first, keeps an invalidation failure as the first error, and continues the rest of cleanup.
- **`CfgNormalizedEulerStep`/`ApplyCfgNormalized` produced NaN for `eps=0` with an all-zero guided row**
  (0/0 in the norm ratio; NaN then poisons `z` through `0·NaN`). A zero denominator now resolves the ratio to
  0 — exact, since an all-zero row contributes nothing — identically in the IBackend fallback, the CUDA
  kernel (`dit_f32.ptx` regenerated), and the host helper; new CPU+CUDA regression test.
- **cuDNN auto-fetch 404'd for CUDA < 12** (NVIDIA publishes no cuDNN 9.21 redist there) — now refused
  up-front with manual-install guidance instead of attempting the download.
- **`Qwen3Tokenizer` hardcoded its chat special-token ids** (`<think>`, `<|im_start|>`, pad/EOS), which only
  fit the embedded artifact — a caller-supplied `tokenizer.json` now has them resolved from its added-token
  table, with a logged fallback when absent.
- **Z-Image rejected legitimate solid-color output** — a uniformly black/white frame now only fails the
  generation when the decoded F32 tensor is actually non-finite; a finite solid frame (valid prompt outcome,
  inpaint over a solid source) is accepted with a log line.
- **CUDA BF16/F16 GroupNorm mis-read non-F32 affine weights.** `CastAffineDownIfF32` only converted an
  F32 affine down to the kernel dtype; an F16-checkpoint affine (e.g. the numz SeedVR2 VAE) was passed
  raw to the BF16 kernel — F16 bits reinterpreted as BF16 → garbage scale/shift and flat-gray output.
  Any affine dtype is now converted to the kernel's dtype. Caught by the SeedVR2 BF16-VAE bring-up: the
  isolated parity test passed against the f32 checkpoint while the pipeline (fp16 catalog checkpoint)
  produced uniform gray.

- **Borrowed views passed as an in-place op's OUTPUT silently discard the write on CUDA** — a hazard class, found via
  MiniMax-H3. The backend binds the result to the borrowed `View`/`RowView` and the dispose callback skips the D2H, so
  the store never lands; the CPU path is unaffected, which is why unit tests passed. In H3 this made RoPE, adaLN
  modulation and the gated residual no-ops across all 50 blocks — `h` never left the patch embedding and every frame
  decoded to a regular-grid mosaic. Fixed by forcing the read-back at the three sites; CPU-vs-CUDA parity went
  0.246 → 4.75e-4 and step-0 velocity rms 7.90 → 2.24. The rest of the repo was swept for the same pattern and is
  clean: the only other borrowed views outside `HartsyInference.Core` feed a host `float*` loop (video-VAE tile
  blending) or are read-only GEMM inputs (LLM stacked-weight slicers).
- **MiniMax-H3 now loads SwarmUI/Comfy's flat checkpoint layout, not just the vendor folder tree.** The vendor
  publishes `transformer/` + `video_vae/` + `audio_vae/` + `text_encoder/` folders; Comfy-Org repackages the same
  weights as one file per component under `diffusion_models/`, `vae/` and `text_encoders/`, and that is what SwarmUI's
  native H3 support downloads — so a Swarm-driven load previously failed looking for a `transformer/` subfolder that
  does not exist. New `MiniMaxH3Assets` resolves both layouts, walking up from the DiT to find components and ranking
  variants so an unloadable `int8_convrot` file never beats a loadable sibling. Nothing is downloaded: re-fetching
  under the engine's own model directory would duplicate the ~5.8 GB of VAEs Swarm already has. Falls back to the
  embedded Qwen BPE and to `MiniMaxH3VideoVaeConfig.Detect` since the flat repack ships no tokenizer or `config.json`.
- **MiniMax-H3 `pruned_fp8_scaled` checkpoints load, and are ~10x faster than bf16.** Two defects blocked them.
  (1) `ThrowIfInt8Convrot` rejected on the *presence* of Comfy's quantization companions, but Comfy tags every
  quantized build with the same `.weight_scale`/`.input_scale`/`.comfy_quant` suffixes — only the `.comfy_quant`
  descriptor distinguishes them, and the fp8 build says `{"format": "float8_e4m3fn"}`. The guard now reads the
  descriptor and rejects only genuine int8-convrot (`MiniMaxH3QuantGuardTests`; an absent/unreadable descriptor still
  rejects conservatively). (2) The converter never called the shared `CheckpointConvertUtils.ApplyFp8ScaledDequant`,
  so the scale companions were routed as unknown weights. **Verified on the real 21 GB
  `minimax_h3_fl2va_pruned_fp8_scaled` checkpoint:** 22 frames at 512x288 / 20 steps produces a coherent tracking shot
  with matched 0.9167 s audio at -21.5 dBFS peak, at **8.6 s/step and 22.5 GB VRAM, fully resident on a 24 GB 4090** —
  versus ~90 s/step for the 66 GB bf16 build, which cannot stay resident and re-reads most of itself from NVMe every
  step. This run is also the first exercise of the *pruned* checkpoint's `adaln_t_table` curve path (`curves=True`).

- **MiniMax-H3 is 25.9x faster: 50.2 -> 1.94 s/step** (512x288, 141 frames, RTX 4090; a full 30-step clip went
  1602 s -> 129 s). ComfyUI does the identical work at 1.67 s/step, so this closes a ~30x gap to ~1.16x.
  The cause was host round-trips, not weight residency or GEMM selection. `View`/`RowView` were built as
  `new Tensor((void*)t.DataPointer, ...)`, and `GpuTransferHelper`'s activation cache is keyed by Tensor object
  reference, so a view can never alias its parent's device buffer — and merely CONSTRUCTING one calls
  `DataPointer` -> `EnsureCpuData` -> cache-evict + `cuStreamSynchronize` + device-to-host copy. The worst
  offender was the QKV split, which host-copied a `[seq, 21504]` tensor three times per attention (~473 MB each
  way per block, ~47 GB/step). Restructured so views are never needed: `SliceLastDim` for the QKV and adaln
  splits, q/k allocated 4-D up front so `RmsNorm` runs in place and `ApplyRopeSingle` consumes them directly,
  the residual stream shaped `[seq, 1, hidden]` so `AffineBroadcastLastDim`/`GatedResidualLastDim` modulate the
  whole packed sequence in a single launch driven by a `RowGather`ed table, and `Concat` for segment assembly.
  **No new kernels.** Acceptance metric: D2H syncs per forward **74 -> 0** (`IBackend.GetD2hSyncCount`, whose own
  doc states a fully GPU-resident denoise loop must stay at ~0). Numerics unchanged — parity holds at video
  relL2 4.752E-004 / audio 6.555E-004.
- **MiniMax-H3 text encoder: ~2x less PCIe traffic per prompt.** The nvfp4 tower dequantized every layer into a
  full F32 weight and uploaded it per call (~97 GB per encode). It now narrows to BF16 inside the dequant loop
  and reuses one shared host scratch buffer instead of ~350 short-lived 200-500 MB allocations, and drops a
  redundant per-call `Sync()` (`FreeWeights` already syncs). BF16 rather than F16 because F16 overflows on the
  SwiGLU gated tensor. Qwen3-VL is BF16-trained, so this is closer to the reference than the old F32/TF32 path.

- **MiniMax-H3 geometry was wrong at its own declared defaults.** Three grids were mis-derived: frame counts must
  snap up onto `17k+5`, video latent frames are `(frames-5)/17*5 + 2` rather than `frames/4`, and pixel axes round to
  32 rather than 16 (a multiple of 16 that is not a multiple of 32 gives an odd latent axis, and the 2x2 patchifier
  silently drops its last row/column). At the shipped defaults `1360x768x121f` that meant 1344x768 output and 102
  delivered frames sized against ~5.0 s of audio — roughly 0.8 s of soundtrack generated and then trimmed away. The
  reference grids now live in `MiniMaxH3Geometry` and are pinned by `MiniMaxH3GeometryTests`, including a round-trip
  asserting the latent count re-expands to exactly the requested frames. Defaults corrected to 1344x768x124f.
- **`CudaBackend` now logs the device name at construction.** `CUDA_VISIBLE_DEVICES` defaults to fastest-first
  ordering, so it does not agree with `nvidia-smi` indices — every perf and VRAM figure from an H3 bring-up run was
  initially attributed to the wrong GPU because only the ordinal was logged.

## [2.0.0-alpha.8] — 2026-08-01

### Fixed
- **A short soundtrack silently dropped trailing video frames.** Muxers cut to the shorter stream
  (ffmpeg `-shortest`), and LTX-2.3's audio-latent count rounds down: a real 25-frame @24fps clip
  (1.0417s) came back with 1.010s of audio, so the muxed mp4 contained **24 frames, not 25**.
  `VideoAudioResolver` now fits the track to the clip in both directions — trim if long, silence-pad
  (`AudioBuffer.PadTo`) if short — so frame count is preserved; a shortfall over 0.25s still warns,
  since that indicates the wrong track rather than latent rounding. Verified on a real LTX-2.3
  generation: audio 1.0417s, muxed mp4 keeps all 25 frames, and the generated samples are
  bit-identical to the pre-fix run with the padding appended as pure silence.
  Found by the e2e run after alpha.7 was cut, hence the separate version.

### Note
- `2.0.0-alpha.7` was tagged but never appeared on nuget.org (both the flat-container and registration
  indexes still topped out at alpha.6 more than 30 minutes after publish). Consume alpha.8 instead.

## [2.0.0-alpha.7] — 2026-08-01

Video gets its sound back: generated audio now reaches the caller (closes `TODO(E-IMG-4/5)`), plus the
LTX-2 split-checkpoint decode fix.

### Added
- **`AudioBuffer`** (`Engine.Requests`) — engine-native raw planar-float PCM, the decoded counterpart to
  `AudioClip` (encoded in) and `AudioResult` (encoded out). Mono/stereo conversion + duration trim; the
  shared currency for moving a waveform between components in any modality.
- **`VideoGenerationResult`** — frames plus the soundtrack that belongs with them.
- **`VideoAudioResolver`** — one place that decides which track ships with a generation: what the pipeline
  attached beats `VideoRequest.VideoAudioInput` pass-through, then the track is trimmed to video length.
  `VideoAudioReference` is deliberately not a fallback (it is conditioning; a family that means it to be
  heard attaches it itself).
- `AudioClipCodec` is now public and gained `DecodeNative` (native rate/channels, no resample) and an
  `EncodeWav(AudioBuffer)` overload.
- REST `/v1/native/video/stream` emits an `audio` SSE event (base64 WAV + rate/channels).

### Fixed
- **LTX-2 split-checkpoint output was checkerboard garbage** (the known-broken `hartsy video -m ltx-2`
  path, which SwarmUI also hits). The split VAE file ships bare keys, and the converter's bare-key router
  only recognized `decoder.`/`encoder.`/`latents_` as VAE keys — `per_channel_statistics.{mean-of-means,
  std-of-means}` fell into the Transformer bucket, so latent denormalization silently became an identity
  no-op. With std-of-means as low as 0.074, the decoder received channels up to ~13× too hot; the up-stack
  amplified that to ±943 and the RGB clamp saturated to checkerboard. One added route in
  `LtxVideo2CheckpointConverter.RouteKey` fixes it: decode now lands in [-1,1] and the catalog path produces
  coherent frames (verified 512×320×25f, seed 42; transformer Sha256 pinned). Bundled single-file
  checkpoints were never affected.
- **LTX-2.3's generated soundtrack was dropped**, not muxed — `LtxVideo2RecipePipeline` logged a warning
  and discarded it because the pipeline contract carried frames only. It is now attached and muxed.
- **Wan2.2-S2V's driving speech was not muxed either.** The mux moved to the Engine when the extension was
  thinned to a wrapper, but was never implemented there; `VideoRequest.VideoAudioInput` was documented as a
  mux track with no consumer. S2V now attaches the speech it consumed, at source rate rather than the 16 kHz
  mono conditioning downmix.
- The SwarmUI extension's ffmpeg audio mux (`VideoOutputEncoder.AudioTrack`, `FormatSupportsAudio`) was
  unreachable dead code — never constructed, never passed. Reconnected, with a warning when the chosen
  container (gif/webp) cannot carry a track.

### Changed
- **Breaking:** `IVideoRecipePipeline.Generate` returns `VideoGenerationResult` instead of
  `IReadOnlyList<VideoFrame>`, and `IVideoService.GenerateAsync` returns `Task<VideoGenerationResult>`
  instead of `IAsyncEnumerable<VideoFrame>`. The enumerable never streamed — it awaited the full frame list
  before yielding — so no delivery behaviour is lost. Replaced rather than added alongside: a second
  frame-only overload would silently drop audio, which is the bug being fixed.
- `hartsy video` writes `audio.wav` beside the frame directory when a generation produces sound.

## [2.0.0-alpha.6] — 2026-08-01

SeedVR2 video/image restoration — a new modality, end to end.

### Added
- **SeedVR2 one-step video restoration** (`Modality.Restore`, catalog ids `seedvr2-3b`/`seedvr2-7b`):
  NaDiT windowed MM-DiT + s8c16t4 causal video VAE ported to pure C#, every stage parity-gated against
  the ByteDance reference — window partition **exact** (2,490 slices), preprocessing maxAbs **2.3e-6**,
  VAE relL2 **≤2.9e-6** vs real weights, full-model E2E **SSIM 0.99950 / 56.6 dB PSNR** vs the Python
  pipeline with injected reference noises. Surfaces: `hartsy restore <video|image>` (PNG frames + H.264
  MP4 out), `--restore` chain on `hartsy video`, REPL `/mode restore`, `POST /v1/native/restore[/stream]`,
  and the SwarmUI extension's "Video Restore" param group. 7-clip real-footage matrix verified on the
  4090 (USIA Reagan '87, NASA Apollo 11, JFK '61, Steamboat Willie, Prelinger '62, Big Buck Bunny
  ground-truth, still-image t==1 branch): 25-frame clips at 960×540-area, **~14 s/frame, 17.1 GB peak,
  zero OOM**. Ground-truth profile matches the paper: pixel metrics prefer bicubic (SSIM −0.05) but
  **LPIPS improves 26–28%** (0.735→0.541 extreme; 0.448→0.324 mild) — it repaints, it doesn't
  reconstruct; `--strength` guards oversharpening.
- **`FfmpegProcessDecoder`** (ffmpeg/ffprobe child processes) — first video-INPUT path in the engine;
  `VideoClip`/`RestoreRequest` DTOs; `TorchResize` (torchvision-exact antialiased bicubic, a=−0.5
  float32 weights — two silent-divergence bugs caught by parity, see PARITY_VERIFICATION).
- **Reference quirks ported deliberately** (SEEDVR2_ARCHITECTURE.md §2.5): the tail `vid_out_ada`
  cache-collision (uses the ATTN emb slice — the code as written is dimensionally impossible), last-layer
  `vid_only` semantics incl. the txt self-residual doubling, per-frame VAE GroupNorm stats, asymmetric
  (0,1,0,1) downsampler padding, MAGViT `(x y z c)` pixel-shuffle dropping output frame index 1.

### Known limitations
- fp32 whole-clip VAE activations cap restoration at ~960×540-area on 24 GB (5-frame chunks); 720p+
  needs bf16 activations or tiled VAE — tracked in MODEL_STATUS_VIDEO remaining work.
- DiT window gather/scatter and RoPE run host-side (bring-up shape): ~14 s/frame. Residency/CUDA-graph
  optimization is the follow-up perf pass.
- Catalog DiT + VAE download from the community safetensors mirror `numz/SeedVR2_comfyUI` (verbatim
  original state-dict keys, fp16; Sha256 pinned from a verified download → convert → restore run, and
  the fp16 output is visually equivalent to fp32 — remaining delta is generative high-frequency repaint).
  Only the 1.2 MB frozen pos/neg embeddings ship from `HartsyAI/SeedVR2-safetensors` (upstream has them
  as torch-pickle `.pt` only); until published, place `seedvr2_embeddings.safetensors` under
  `Models/Video/SeedVr2/`.
- **seedvr2-7b is catalog-registered but BLOCKED**: its smoke run revealed the 7B is the **v1 NaDiT**
  (`models/dit`, `qk_rope`/`shared_qkv`) whose state-dict keys coincide with v2 — it loaded and produced
  plausible-but-wrong mud (GT SSIM 0.71 vs 3B's 0.88). `SeedVr2Config.Detect` now throws on the v1
  signature instead of running it; the v1 port is tracked in MODEL_STATUS_VIDEO.

## [2.0.0-alpha.5] — 2026-07-27

Low-VRAM generation, a GPU-memory leak fix, and selectable devices.

### Added
- **Low-VRAM weight streaming across the image fleet** (`HARTSY_LOWVRAM`, three-state: `auto` default /
  `on` / `off`). The sliding-window machinery (`BlockStreamingController`) already existed but only one of
  ~25 image pipelines used it. **Four models that could not run on a 12 GB card now do**, all 1024²,
  quality-gate clean: **HunyuanImage-2.1** (19.7 s), **Ideogram 4** (205 s — a *pair* of 9.3 GB DiTs
  needing 19.7 GB against 9.2 GB available), **Qwen-Image** (231 s, 20B MMDiT), **Krea2** (71 s).
  `off` is a real escape hatch: the same request succeeds under `on` and raises `OutOfVramException`
  under `off`.
- **`VramPlanner`** — one place that decides resident-vs-streamed per generation phase, carries the
  `HARTSY_KEEP_MODELS` residency short-circuit, and logs the **weights-vs-activations split** (streaming
  can only move the weight term, so a phase dominated by activations needs a smaller working set, not a
  sliding window).
- **Selectable CUDA device**: `cuda:1`-style backend selectors, `InferenceEngine(selector, ordinal)`, and
  a real `GPU_ID` in the SwarmUI backend — previously logged and ignored. Verified by memory delta.
  Note the ordinal is CUDA's (fastest-first), which need not match `nvidia-smi`'s PCI order.
- **SD3.5 modular component loading** — CLIP-L / CLIP-G / T5-XXL / VAE each resolve independently when the
  checkpoint does not bundle them, which is the standard SD3.x distribution format. SD3.5-Medium now
  generates end-to-end; it previously threw before any sampling.

### Fixed
- **GPU memory leak on OOM.** `CudaBackend.PreloadWeights` had no exception path, so a mid-load OOM left
  already-uploaded weights registered against a model that would never finish — unreachable, therefore
  unfreeable. The process held ~11.5 GB with nothing running and **starved other processes on the same
  card**, including a separate ComfyUI. Now: typed `OutOfVramException`, per-batch rollback, and reclaim at
  both the generate and construct boundaries. An OOM'd process now holds **152 MiB instead of ~11.5 GB**,
  and a sequential multi-model sweep survives an OOM (3/3 models succeeded after one).
- **Streaming was inert for every GGUF model.** `DType.Q4_K.SizeInBytes` is 0 (a K-quant has no
  per-element size), so `ElementCount * SizeInBytes` totalled block weights to **zero bytes** and the
  "fits resident?" test was always trivially true. Fixed in four block implementations.
- **Lens rendered solid black** (16/16). SageAttention's INT8 path materializes V as F16; Lens does not
  RMS-norm V, and `max|V|` crossed F16's 65504 mid-generation. Verified against ComfyUI's own reference
  implementation on the same checkpoint — an engine bug, not a port bug.
- **Anima was 19-63× slower than ComfyUI**: 792 host round-trips per denoise step (14 per block × 28
  blocks × 2 CFG passes). Now **3**. Warm step 15,279 ms → 519 ms. Its documented "1024² hangs" was never
  a hang.
- **The VRAM planner under-reported free memory by ~4.6 GB**, because `cuMemGetInfo` counts the
  stream-ordered pool's reservations as used. The error is asymmetric — it biases toward streaming, which
  costs 5-8× — so a large card could silently take the slow path for a model that fits.
- `TextService.PrimaryDeviceKey()` hardcoded `"cuda:0"`, so a `cuda:1` engine would have rendered images on
  one GPU while its LLM landed on another.
- Lumina2's on-disk checkpoint was the wrong variant (`cap_embedder.*` naming vs the diffusers
  `time_caption_embed.*` the converter expects). Correct weights now load and generate — though this
  revealed a **separate, previously unreachable conditioning bug**: output is coherent but off-prompt.

### Changed
- `Chroma` checkpoint conversion is now streaming per tensor (removes a GC-timing dependence from the
  peak). **The documented "host RAM OOM" does not reproduce** — it peaks at 9.1 GB anon and completes;
  the reported 25 GB was total RSS including reclaimable file-backed page cache.

## [1.0.0-alpha.48]

Production-readiness push: closes the throughput gap toward python inference stacks (vLLM/TGI-class) and
adds the serving infrastructure a real deployment needs. Full technical detail in
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](docs/Checklists/ROADMAP.md)'s dated status
updates; this is the release-notes-level summary.

### Added
- **Fused GEMV kernels for Q4_0 and Q5_K** quantization formats — the last two of the six original
  quant types without a fused decode kernel; both previously fell to the ~10-20x-slower
  dequant-to-F16-then-cuBLAS path.
- **On-device repetition penalty for CUDA-graph decode.** Graph decode was previously greedy-only with a
  raw unpenalized argmax — a request with `RepetitionPenalty > 1.0` and graph decode enabled silently
  ignored the penalty. Fixed with two new device-resident kernels chained into the existing captured graph.
- **`/v1/chat/completions`** (OpenAI-compatible, streaming and non-streaming) on `HartsyInference.Server` —
  the server previously had no LLM chat endpoint at all (image generation only). Includes structured
  request logging (queue depth, prompt/completion tokens, latency, tokens/sec) and real cancellation that
  stops in-flight generation, not just the HTTP connection.
- **Paged KV cache** (`PagedKvPool`/`PagedKvCache`) — replaces the single-sequence `FixedKvCache` (hard
  `batch=1` restriction) with pages allocated on demand from a pool shared across sequences.
- **True continuous batching** (`DynamicBatchScheduler`/`IBatchScheduler`) — requests admit dynamically at
  any time and batch together into shared decode rounds; each sequence evicts the instant it
  finishes/stops/cancels. Replaces the old static-batch `ContinuousBatchScheduler` (fixed request list up
  front, zero production callers, removed). Backend-exclusivity is preserved via an injected gate so LLM
  batching never races with diffusion image generation on the shared GPU backend instance.
- **JSON-mode constrained decoding** (`response_format: {"type":"json_object"}`) — masks every candidate
  token so generation can only produce syntactically valid JSON. The richer `json_schema` mode is not
  implemented and is rejected with a clear 400 rather than silently ignored.
- Server integration test suite (`ChatCompletionsIntegrationTests`, in-process via `WebApplicationFactory`)
  covering chat-completions request validation — previously zero automated coverage on this HTTP surface.

### Changed
- `IBackend.SliceTimeRange` — new primitive (host default + CUDA kernel) extracting a contiguous
  time-range from a KV-shaped tensor; used by the paged KV cache.
- `GenericTransformer.ForwardBatchDecode`'s cache parameter widened from `FixedKvCache[]` to `IKvCache[]`.
- Chat-completions request validation now checks pure request-shape issues (empty messages, unsupported
  `response_format`) before consulting server state (is the model loaded) — fails fast on a malformed
  request regardless of what's currently loaded.

### Fixed
- Two real bugs in the new JSON-grammar state machine, both caught by unit tests before ever touching a
  live model: object keys didn't set the post-string parse transition (would have broken any JSON with a
  key — i.e. almost all real JSON); the state's `Clone()` was missing two fields added after it was first
  written (every candidate-token check clones the state, so this would have corrupted the container stack
  on every single trial in production).
- `ModelManager`'s diffusion-vs-LLM checkpoint routing no longer speculatively attempts the LLM loader on
  an unrecognized GGUF — a prior version of this logic (try-LLM-then-catch-fallback) fully materialized a
  multi-GB diffusion checkpoint's tensors before the fallback path could fire, causing a real OOM.
- Paged KV cache's VRAM footprint is now sized from a configurable byte budget
  (`HartsyInferenceServerOptions.KvPoolBytesBudget`, default 512MB) scaled to each loaded model's actual KV
  dimensions, replacing a fixed page count that comfortably fit a narrow-KV-dim model but eagerly
  pre-allocated several GB for a wider one — caught loading gemma-3 during a broader architecture sweep.

### Deferred (explicitly, not attempted)
- Prefix/prompt caching (share identical-content KV pages across sequences) — real additional scope (page
  reference-counting, prefix hashing, copy-on-write on divergence).
- Speculative decoding — a true stretch item, orthogonal to everything else in this release.
- `json_schema`-constrained decoding (schema-aware, not just syntax-valid JSON).
- Wider quant kernel coverage (Q2_K/Q3_K/IQx formats) — no template to adapt from, genuinely new kernel
  design (lookup-table dequant for IQx specifically).

## [1.0.0-alpha.47] and earlier

Not individually itemized here — see `git log` for the full history prior to this changelog's introduction.
