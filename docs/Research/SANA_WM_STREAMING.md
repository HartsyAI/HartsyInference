# SANA-WM streaming: design note

Source snapshot: NVlabs/Sana `b3bad956` (`inference_sana_wm_streaming.py`, `generate_streaming`, HF
`Efficient-Large-Model/SANA-WM_streaming`). This note records the upstream-to-engine map, the decisions and their reasons,
the carried-state layouts, the numerics, the memory plan and the test handoff. Status and parity evidence belong in
[MODEL_STATUS_WORLD.md](../Checklists/MODEL_STATUS_WORLD.md) and [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md)
once the weights have been run. Stage 1 streaming (Phase A) has been run on a 4090 against upstream's own 73-frame run
(see [Phase A on a GPU](#phase-a-on-a-gpu)). The refiner (Phase B) has only been compared on the CPU against upstream's
code on a tiny random model; see [its section](#phase-b-the-streaming-refiner). It has not run on a GPU or on real weights.

## What streaming is

Stage 1 (`sana-wm`) denoises the whole clip at once and is bidirectional in time, so its camera path must be known before
generation. Streaming (`sana-wm-streaming`) is a different checkpoint of the same DiT layout, distilled to be chunk-causal:

| Stage | Upstream | What it does per chunk |
|---|---|---|
| Student | `SanaMSVideoCamCtrlStreaming` + `SelfForcingFlowEulerCamCtrl` | 4 denoising forwards over the chunk's noise, then 1 forward over the clean chunk at t = 0 that writes the state the next chunk reads |
| Refiner (optional) | `RefinerChunkRunner` over the LTX-2 transformer | 3 distilled steps per 3-frame block against a sliding KV window, then 1 forward at σ = 0 that captures the block's keys and values |
| Decoder | `CausalVaeStreamingDecoder` over the causal LTX-2 VAE | decodes the chunk's latent frames, each causal conv keeping its input tail |

Cadence at the defaults: latent chunks of 3 frames, the first chunk 4 (the seed frame plus 3). A latent frame is 8 pixel
frames, so chunk 0 delivers 25 frames (the seed frame first) and every later chunk 24: 1.5 s of video at 16 fps per chunk.
A bounded stream therefore takes `24n + 1` frames. The camera pose of latent frame `i` is raw frame `8i`, and its Plücker
channels read raw frames `8i − 7 … 8i`, so a chunk needs the camera path only up to its own last raw frame. That is what
makes live steering possible: the path is extended 24 frames at a time, just before the chunk that uses them.

## Upstream to engine map

| Upstream | Engine | Package |
|---|---|---|
| `create_autoregressive_segments`, `chunk_index_from_chunk_size` | `ChunkPlanner`, `ChunkSplitStrategy`, `ChunkSpan` | Diffusion/Streaming |
| per-block `kv_cache` slot list, save pass | `IStreamingState`, `StreamingStateSet` (stage, then commit) | Diffusion/Streaming |
| GDN state slots 0, 1, 2 | `StreamingTensorSlot` | Diffusion/Streaming |
| short-conv and t-conv caches (slots 3, 4, 9), VAE `feat_cache` entries | `FrameTailCache` | Diffusion/Streaming |
| VAE `feat_cache` / `feat_idx` traversal | `CausalStreamCache` | Diffusion/Streaming |
| `_accumulate_softmax_kv_cache` (window + sink), refiner history trim | `RollingTokenWindow`, `TokenWindowPolicy` | Diffusion/Streaming |
| `FlowMatchEulerDiscreteScheduler` with `denoising_step_list` | `SelfForcingFlowSchedule` | Diffusion/Schedulers |
| `fused_gdn_chunkwise` phase B with `init_state` / `return_final_state` | `IBackend.FrameDeltaScan(initialState, finalState)` | Core, Cuda |
| `SanaMSVideoCamCtrlStreaming.forward_long` | `SanaWmDit.ForwardChunk` | Diffusion/…/SanaWm |
| `CachedChunkCausalGDN`, `_cached_cam_branch`, `_cached_temporal_short_conv` | stream parameters on `SanaWmGdnBidi`, `SanaWmCamBranch`, `SanaWmTemporalConv.ApplyWithContext` | Diffusion/…/SanaWm |
| `CachedChunkCausalSoftmaxAttn`, `_cached_cam_branch_softmax` | stream parameters on `SanaWmSoftmaxAttention`, `SanaWmSoftmaxCamBranch`, `SanaWmWindowedKeys` | Diffusion/…/SanaWm |
| `CachedGLUMBConvTemp` | `SanaWmGlumbTempFfn.Forward(…, tail, commit)` | Diffusion/…/SanaWm |
| per-chunk cache list | `SanaWmStreamCache`, `SanaWmBlockStream` | Diffusion/…/SanaWm |
| `LTX2VideoCausalConv3d` with `feat_cache`, `LTXVideoUpsampler3d` trim state | `CausalConv3d.ForwardStreaming`, `LtxVaeUpsampler3d.Forward(…, cache)`, `LtxVaeResnetBlock3d.Forward(…, cache)` | Diffusion/Models/Vae |
| `decode_per_frame_with_cache`, `CausalVaeStreamingDecoder` | `LtxVideo2VaeDecoder.DecodeStreaming`, `IStreamingVideoDecoder`, `LtxVideo2StreamingDecoder` | Diffusion/Models/Vae |
| `CameraPoseIntegrator` + `VelocityState` (live demo) | `ICameraIntegrator<T>`, `SanaWmCameraIntegrator`, `LiveCameraPath<T>` | World/Camera |
| `prepare_camera` sliced per chunk (`_inject_sliced_extras`) | `SanaWmCameraConditioning.PackRange`, `SanaWmStreamCamera` | World |
| `sample_chunks` + `run_streaming_inference` (stage 1 and decode) | `SanaWmStreamingPipeline`, `SanaWmStream` | World/Pipelines |
| the chunk generator contract | `IChunkedWorldStream` | World/Sessions |
| CUDA-stream orchestration, progressive MP4 | `IChunkedWorldPipeline`, `ChunkedWorldSession` | Engine |
| CLI defaults | `WorldStreamingCapabilities`, `WorldPlanResolver`, catalog `sana-wm-streaming` | Engine |
| `RefinerChunkRunner` (noise, 3-step Euler, capture, window trim) | `IStreamingLatentRefiner`, `LtxVideo2ChunkRefiner`, `FlowRefineSchedule` | Diffusion/Streaming, Schedulers |
| `_forward_video_only_with_rope`, `_forward_video_block` | `LtxVideo2ChunkTransformer`, `LtxVideo2VideoBlock` | Diffusion/Models/Denoisers |
| `_streaming_self_attention` (`rf_shifted_sink` prefix, capture hooks) | `LtxVideo2PrefixSelfAttention` | Diffusion/…/DiTBlocks |
| `_sink_kv_pre`, `_history_kv_post`, `_history_frames` | `LtxVideo2ChunkState` (`StreamingTensorSlot` pair + `RollingTokenWindow` per block) | Diffusion/Models/Denoisers |
| `_build_rotary_emb_for_absolute_positions` | `LtxVideo2Rope.BuildVideo(…, firstFrame)`, `LtxVideo2ChunkScope` | Diffusion/…/DiTBlocks |
| `_pack_text_embeds` + `LTX2TextConnectors` (video branch) | `LtxVideo2PackedTextConnector` | Diffusion/…/DiTBlocks |
| `_encode_prompt` (Gemma-3-12B) | `SanaWmGemma3RefinerTextEncoder` | World |
| `streaming_pipeline.py` stage 1 → refiner → decode | `SanaWmStream` with `ISanaWmRefinerComponents` | World |

## The reusable contracts

A different chunk-causal world model plugs in at four levels, lowest first.

1. **Carried state** (`HartsyInference.Diffusion.Streaming`). Every piece of state is an `IStreamingState`: writes are
   staged while a chunk closes and `Commit` makes them current together. Denoising passes only read. `StreamingStateSet`
   fans `Commit`, `Discard` and `Reset` out over a model's states. The building blocks are `StreamingTensorSlot` (one
   tensor), `FrameTailCache` (the newest N frames of a stream along any axis, the left context of a causal temporal
   operator), `CausalStreamCache` (one tail per causal operator, addressed by traversal order, for convolutional
   networks) and `RollingTokenWindow` (parallel K/V streams with a permanent sink and a history bounded by chunks, by
   frames, or both). All of them pin their tensors on the backend, so an activation sweep cannot free them.
2. **Sampling.** `ChunkPlanner` gives the chunk spans for a bounded or unbounded sequence; `SelfForcingFlowSchedule` gives
   the per-frame timesteps and the Euler step, with clean (conditioning) frames held fixed.
3. **The chunk generator** (`IChunkedWorldStream`). A model exposes `GenerateChunk`, `SendAction`, `NextChunkFrames` and
   `Completed`. It owns its temporal state and is called from one thread.
4. **The session** (`IChunkedWorldPipeline` + `ChunkedWorldSession`). The engine runs any `IChunkedWorldStream` on one
   dedicated compute thread under the world execution lease, delivers chunks through `IWorldSession.StreamAsync` with
   back-pressure, forwards `SendAction` at any time, and releases the stream before the lease. A model declares
   `WorldCapabilities.Streaming`; the planner resolves `WorldRequest.Streaming` and `Unbounded` against it.

5. **The refiner** (`IStreamingLatentRefiner`). A second stage takes the stream's sink frames once (`SetSink`) and then
   one block of clean latents at a time (`Refine`), staging what the block adds until `Commit`. It knows nothing about the
   first stage. `LtxVideo2ChunkRefiner` is the LTX-2 implementation; it serves any first stage that produces LTX-2 VAE
   latents.

The streaming VAE decode is model-agnostic as well: `IStreamingVideoDecoder` over any causal LTX-2 VAE, usable by an
LTX-2 video pipeline that loads a causal decoder. `LiveCameraPath<TControls>` and `ICameraIntegrator<TControls>` carry no
SANA-WM types.

## Decisions

**Stream state on the existing SANA-WM layers, not parallel classes.** Chunk-causal attention differs from the
bidirectional layer in three places only: the forward scans start from a carried state, the key convolution's causal half
reads the previous chunk's tail, and the softmax blocks see earlier keys and values. Each layer takes an optional
`SanaWmBlockStream`; with none the stage-1 path is unchanged, statement for statement. A stream's first chunk has no state,
so `ForwardChunk` on chunk 0 must equal `Forward` on the same inputs, which is the cheapest regression check there is.

**`FrameDeltaScan` extended, not forked.** The chunk-causal scan is the bidirectional scan with the forward recurrence
seeded: upstream runs the same phase B kernel with `direction=1` plus `init_state` and `direction=2` without. The op gained
optional `initialState` and `finalState` (`[heads, keyDim, valueDim]`). The reverse state of the last frame is exactly
zero, so the CUDA build reads the final forward state out of the per-frame states it already has; the recurrence kernel
gained one pointer argument and nothing else changed.

**The key convolution with context is the same symmetric filter.** Upstream computes
`causal(cache ⊕ x) + reverse_within_chunk(x) − center`. For an output frame inside the chunk that is the stage-1 symmetric
7-tap filter over `[cache ⊕ x]` with zeros after the chunk, so `ApplyWithContext` concatenates, runs the existing
`DepthwiseConvFrames`, and drops the context frames. The FFN's temporal conv is handled the same way with one frame of
context.

**Tails carry across chunks shorter than the tail.** Upstream's key-conv cache keeps the last `K − 1` input frames of the
current chunk only, and its slicing assumes a full cache, so upstream does not run chunks shorter than 3 frames.
`FrameTailCache` keeps the newest frames across chunk boundaries instead, which is the causal continuation and is what
upstream's VAE `feat_cache` does. The two agree for every chunk of 3 frames or more, which includes the default plan.

**Staged writes, one commit per chunk.** Upstream mutates its cache during the save pass. Staging costs nothing extra and
means a chunk cancelled during denoising or during the save pass leaves the stream exactly at the previous chunk
(`SanaWmStream.GenerateChunk` can simply be called again). After the commit the stream decodes; a failure there marks the
stream faulted, because the decoder would be behind the DiT.

**One stream state per guidance branch.** Upstream batches the unconditional and conditional items and its cache holds
both. Here each branch has its own `SanaWmStreamCache` and runs as its own forward, which is the same arithmetic and keeps
every stream forward at batch 1. The student is trained for guidance 1, where only the conditional branch exists.

**Per-chunk noise is seeded by chunk index.** Upstream draws the whole latent volume up front from a CUDA generator, which
neither matches any host generator nor suits an unbounded stream. Chunk `k` uses seed `seed + k·0x61C88647`. Parity runs
inject upstream's noise through `SanaWmStreamRequest.ChunkNoise` and `FirstLatent`.

**The save pass uses per-frame modulation with all-zero timesteps.** Upstream passes a 1-D timestep there and takes the
non-frame-aware block path; with every frame at t = 0 the two paths compute the same modulation, so the engine keeps one
path.

**Decode mirrors upstream's call structure by default.** `LtxVideo2StreamingDecoder` decodes one latent frame per decoder
pass, as `decode_per_frame_with_cache` does. More frames per pass are the same convolutions over the same inputs (the CPU
build gives bit-identical output for 1, 2 and 4 frames per pass on a synthetic decoder); it stays a constructor argument
until a GPU run shows the cuDNN algorithm choice does not move the result.

**The decoded seed frame is delivered.** Upstream drops the first pixel frame; the engine's contract is that a world
session's frames start with the seed frame, as the batch model's do, so a bounded stream delivers exactly `plan.Frames`.

**A separate catalog id.** `sana-wm-streaming` has its own DiT and VAE files and only streams
(`WorldStreamingCapabilities.Only`), so it is its own entry rather than a mode of `sana-wm`.

**Live actions are held keys or queued scripts.** `SendAction("wd")` holds a key set from the next unscripted frame on;
`SendAction("w-48,none-24")` queues scripted frames that play first. Planning keeps a streaming request's `--actions`
script as written (the opening move) instead of expanding it into a fixed pose list, so the script's length does not have
to match the frame count. A request that carries poses streams along that fixed path and refuses live actions.

**A new Engine session rather than `BackgroundComputeSession`.** The World package's `BackgroundComputeSession` is a
lossy real-time transport: it drops frames when the consumer falls behind and has no notion of the execution lease, world
phases or a plan's frame count. `ChunkedWorldSession` is lossless, back-pressured and lease-aware, which is what the CLI
and the HTTP route need. An `IFrameStepper` adapter over `IChunkedWorldStream` is a small follow-up if a real-time UI wants
the lossy transport.

**The producer waits for room before it generates.** With a buffer of one chunk, the next chunk starts only when the
consumer has taken the previous one, so it reads the newest actions. Action-to-photon latency is one chunk's generation
time plus up to one chunk of playback; that is inherent to 1.5 s chunks.

## Carried state

Per DiT block and guidance branch (`SanaWmBlockStream`), at 720p (`S = 22·40 = 880` tokens per frame, `C = 2240`,
`H = 20`, `D = 112`):

| State | Blocks | Shape | Upstream slot | Notes |
|---|---|---|---|---|
| numerator scan state | GDN (15) | F32 `[H, D, D]` | 0 `S_kv` | key-major `[Dk, Dv]`: the transpose of upstream's `(B, H, D, D)` |
| denominator scan state | GDN | F32 `[H, D, 1]` | 1 `S_z` | from the unrotated q and k |
| camera scan state | GDN | F32 `[H, D, D]` | 2 `cam_S_kv` | key-major, equal to upstream's layout (no transpose) |
| key conv tail | GDN | F32 or BF16 `[3·S, C]` | 4 | raw key projection of the newest 3 frames |
| camera key conv tail | GDN | F32 or BF16 `[3·S, C]` | 3 | raw camera key projection |
| main K, V window | softmax (5) | F32 or BF16 `[1, H, n·S, D]` ×2 | 0, 1 | K after RoPE at its own absolute frames, V raw |
| camera K, V window | softmax | F32 or BF16 `[1, H, n·S, D]` ×2 | 2, 3 | after the ray transform and camera RoPE |
| FFN tail | all (20) | FFN dtype `[S, C]` | 9 | point-conv output of the newest frame |

Softmax blocks are 3, 7, 11, 15 and 19 (`softmax_every_n = 4`). The default window is
`TokenWindowPolicy.RecentChunks(2, sink: true)`: chunk `c` attends to chunk 0 (4 frames, permanent) and chunk `c − 1`
(3 frames), matching `num_cached_blocks = 2` with `sink_token`. RoPE for a chunk is built at its absolute frame positions
(`casual_wan_rope` over `[start_f, end_f)`), and cached keys keep the positions they were rotated at.

The VAE decoder's state (`CausalStreamCache`) is one `FrameTailCache` of 2 frames per causal conv, in traversal order,
plus one first-use flag per temporal upsampler (the first call drops the leading shuffle frame, later calls keep it).

## Numerics

- Sigmas are `timestep / 1000` in float32 and timesteps `sigma · 1000` in float32, as diffusers computes them; the DiT then
  truncates timesteps to integers. `1000, 960, 889, 727` survive that round trip exactly.
- The Euler step is `x + (σ − σ_next)·(−v)` in float32 on BF16-rounded `x` and `v`, rounded to BF16: the stage-1 sampler's
  arithmetic. Clean frames are skipped, which equals upstream's restore from `init_latents`.
- Softmax K and V are held between chunks as computed (F32) or, with `vram.sanaWmStreamState` BF16 or `auto` on a small card
  (see [Memory on a 12 GB card](#memory-on-a-12-gb-card)), in BF16 as upstream's cache holds them
  (`SanaWmBlockStream.StorageDType`). The rest of this bullet and the next describe the BF16 mode. A
  chunk's own keys and values are used in F32 for its passes and rounded to nearest even only when they are staged. Each
  pass widens the window's parts and joins them with the chunk, so the attention call (`AttentionNumerics.Bf16Storage`,
  which narrows its F32 operands to BF16 inside) receives exactly the operands it did when the window was F32: the window's
  values are BF16 numbers either way, and narrowing a BF16 number is the identity. The first chunk has no window and runs
  as before.
- The two key tails are held in BF16 too (upstream's slots are `[S, 3, C]` BF16) and widened where the convolution reads
  them; only the 3 frames kept are rounded. Scan states stay F32 (they are small and recur through the whole stream), the
  FFN tail keeps its own dtype. The engine's stage-1 layers keep q, k and v in F32 between ops and are at parity that way.
- CPU runs differ in one respect: `IBackend.CastToBf16`'s default narrows by truncation. The streaming state avoids it
  (`StreamingTensors.CastTo` rounds to nearest even on the CPU backend), so CPU and CUDA hold the same stored values.
- Text cross-attention uses the additive `(1 − mask)·−10000` bias, which is upstream's own non-xformers path (the streaming
  script sets `DISABLE_XFORMERS=1`).
- Streaming pixels are `clamp(127.5·x + 127.5, 0, 255)` truncated, in float32 (`SanaWmStream.ToRgb24`); upstream calls
  `.float()` first. The batch pipeline's BF16-product conversion is not used here.

## Memory plan

Estimates from tensor shapes, not measurements; the first GPU run should replace them.

Phase A (student + causal VAE decoder resident together, text and image encoders loaded and released first):

| Item | Estimate |
|---|---|
| DiT weights (BF16 matrices, F32 vectors), as stage 1 | about 5.4 GB |
| causal VAE decoder weights (BF16) | to be measured; the stage-1 decoder is the same layout |
| softmax K/V windows, 5 blocks × 4 streams × 7 frames, F32 | about 1.1 GB |
| GDN key tails, 15 blocks × 2 × 3 frames, F32 | about 0.7 GB |
| FFN tails and scan states | under 0.2 GB |
| chunk activations (3 to 4 frames of 880 tokens) | small next to stage 1's 21 frames |

That was the estimate; the measurement did not fit 12 GB, see [Memory on a 12 GB card](#memory-on-a-12-gb-card). The two
levers named here, BF16 softmax windows and BF16 key tails, are applied. A second guidance branch doubles the carried state.

The refiner's memory is planned in [its own section](#memory-plan-with-the-refiner).

## Phase A on a GPU

Reported by the parent session, 2026-11-09, on a 4090: demo_0 pose, seed 42, guidance 1.0, chunks of 4, 3 and 3 latent
frames, stage 1 only, upstream's noise and first latent injected; a real run, not a CPU check.

| Check | Result |
|---|---|
| all 15 upstream DiT calls replayed through one `SanaWmStreamCache`, committing after each save pass | worst cosine 0.99961, relative L2 1.8 to 2.8 %, chunks 1 and 2 as close as chunk 0 |
| CUDA `FrameDeltaScan` with carried state against the host reference | relative L2 1.8e-4 (112-wide heads), 1.5e-5 for the denominator |
| CLI path end to end (planner, `WorldService.Open`, `ChunkedWorldSession`, streaming decode), 73 frames | PSNR min 44.9 / mean 47.7 dB, SSIM min 0.9973 / mean 0.9981 against upstream's `decode_chunk` pixels |
| time | chunk 0 at 12.9 s including model loads, then 2.11 s and 1.74 s per 24-frame chunk (0.86× real time); upstream 2.13 s |
| device memory | 10.98 GiB in use, about 1 GiB of it other processes |

**Live action script on the 4090** (parent session, 2026-11-09, `d55f74893`): `hartsy world -m sana-wm-streaming … --intrinsics
fx,fy,cx,cy --frames 73 --actions "w-24,d-24,s-24"` (forward, turn right, reverse) runs end to end through the live camera
path (no poses; the script is queued into `LiveCameraPath`) and writes 73 frames and an MP4 in about 39 s of wall time; the
view follows the script. Against an upstream streaming run with the same script (`--camera-source action`) and upstream's
noise injected: per-frame PSNR 49 dB for the forward chunk, 46 to 47 dB for the turn, 34 to 42 dB for the reverse; SSIM
min 0.986. The frames are the same picture; the differences sit on the edges of invented texture, and the sky and horizon
differ by about 0, so the camera is right. The reverse chunk's lower PSNR is the model, not the port: reversing makes stage 1
invent dark cracked rock at the bottom right, upstream does the same from the same input, and small BF16 differences grow on
invented detail. The refiner may clean it up (not tested). The CLI writes each frame to disk as it arrives and keeps none
of the pixels, so memory stays flat however long an `--unbounded` run goes (a 720p frame is 1.2 MB of PNG on disk and
2.7 MB of RGB in memory), and Ctrl+C or `q` on that run saves what was generated and muxes the MP4. Still open: there is no
live viewer, so the video cannot be watched while it generates.

Open from the same report: `SanaWmWindowedKeys.Join` concatenates window and chunk on every pass, a copy per pass.
Harmless at a window of two chunks; worth caching the joined window across the five passes of a chunk if the window grows.
(The report's other item, BF16 softmax storage, is applied; see the next section.)

### Memory on a 12 GB card

Reported by the parent session from `vram.trace`, 3060 (11.9 GiB, about 0.5 GiB of it the CUDA context). The per-call DiT
replay passes there (worst cosine 0.99963, about 5.1 s per warm 3-frame chunk) but the stream, DiT plus decoder, runs out
of memory decoding chunk 1: the decoder asked for 110 MB with 71 MB free. Upstream does not fit on the card at all.

| Item (MiB) | After |
|---|---|
| weights: DiT 5157, causal decoder +1975 once it first runs | 7132 |
| activations: 49 | chunk 0's denoise passes |
| 1431 | chunk 0's save pass: the committed state of one 4-frame chunk, F32 |
| 1989 | chunk 0 decoded: the decoder's carried cache is about 560 |
| 3221 | chunk 1's save pass: committed and staged state coexist |
| 3672 | chunk 2's save pass on the 4090, the steady-state peak |

Pool reserved against pool used on the 4090 was 4800 against 3672 at steady state, and on the 3060 3968 against 3221 at the
failing save pass: the allocator held 0.7 to 1.9 GiB it was not using, as free blocks the driver could not give to the
decoder, whose allocations go to the driver. That is the direct cause of the failure; the state itself fit. Three changes:

1. **The pool is trimmed at every DiT commit** (`SanaWmDitChunkDenoiser.Commit` calls `IBackend.TrimMemoryPool`). The commit
   has just released the replaced state, and the decoder runs next. Cost: the next chunk's passes re-map what they need
   again, small against a chunk's seconds. A second trim after the decode, or at a chunk's first pass, is the next step if
   the save pass itself runs short (it ended at 11904 of 11912 MiB on the 3060).
2. **Softmax windows can be held in BF16** (`RollingTokenWindow` takes a storage dtype; chunks are converted as they are
   staged, so the F32 chunk is released at once). Steady state 7 frames × 880 tokens × 2240 channels × 4 tensors × 5 blocks:
   1.1 GB F32, 0.55 GB BF16, plus about 0.24 GB saved on the chunk staged during a save pass.
3. **The two key tails can be held in BF16** (`FrameTailCache` takes a storage dtype; only the 3 frames kept are converted):
   15 blocks × 2 tails × 3 frames × 880 × 2240 is 0.71 GB in F32, 0.36 GB in BF16, and about 0.18 GB less when staged.

Expected: about 0.9 GB less state at steady state and about 1.3 GB less at a save pass, plus the pool slack. Nothing is
measured; the parent reruns the 3060 and 4090 lanes. Not done: alternating residency per chunk (DiT weights on the host
while decoding, decoder weights on the host while sampling), about 5 GB plus 2 GB of PCIe traffic per chunk; use it only if
these are not enough.

**Second 3060 run** (after the three changes above; trace `streaming-3060-vram-trace-v2.tsv`, reported by the parent). Replay
parity unchanged (worst cosine 0.999625). The DiT's committed state fell from 1431 to 792 MiB after chunk 0's save pass
and from 3221 to 2018 MiB after chunk 1's; device memory after chunks 0, 1 and 2 fell from 8.08, 9.36 and 10.33 GiB to
7.55, 8.36 and 9.36. The stream still failed decoding chunk 1 (123 MB requested, 109 MB free, inside `Conv2D` under
`CausalConv3d.Forward`'s per-tap path), with the pool at 3520 MiB reserved against 2018 used before the decode: the
decode's own transients fill the card. The largest item left was the decoder's weights, which went from 5159 to 7157 MiB
when it first ran: 1998 MiB for a decoder of 528.4M parameters, 1008 MiB in BF16.

4. **`CausalConv3d` holds one weight form.** It built both the 5-D weight (what a native 3-D conv reads) and the kt per-tap
   2-D slices (what the per-tap path reads) at construction, and a device backend promotes the weights a conv keeps using, so
   a conv the native op declines held both on the device, and every conv held both on the host. The slices are now cut the
   first time a call needs them; a conv the backend's native 3-D op has declined twice and never served releases its 5-D form
   (host and device); a conv the native op serves never cuts slices. Two declines, not one, because the native call also
   declines when it runs out of workspace for one geometry. A decoder therefore holds both forms for its first two calls
   per conv (the first chunk's first two passes in a stream) and one form from then on. `EnumerateWeights` yields whichever forms exist,
   so enumerate it after the first forward. This is shared code (the Wan, LTX-2, HunyuanVideo, Kandinsky 5 and SeedVR2 VAEs
   build `CausalConv3d`): outputs are unchanged on the CPU, and the GPU paths each need a decode to confirm. A conv the
   native op declined twice only because it ran out of workspace stays per-tap for the rest of the run. Expected saving about
   1 GB of device memory at the 3060's geometry.

After that the decode's own peak, about 2.5 GB above the carried state on the 3060, is the next item, then alternating
residency.

**4090 run of the BF16 state** (parent session): replay worst cosine 0.999613, end to end PSNR min 44.6 / mean 47.6 dB
(was 44.9 / 47.7), SSIM min 0.9972: within the BF16 floor. Device peak 13.3 GB (was 14.2), committed state at the save passes
792, 2018 and 2244 MB (was 1431, 3221, 3672). Speed: 1.89 s per chunk against 1.74 s (+9%); a pass that reads the window took
212 to 231 ms against 170 to 185 (about +41 ms, the same in chunks 1 and 2 although chunk 2's window is larger, so it is not
bandwidth; chunk 0, which has no state, is unchanged at about 205 ms).

**The slowdown was not the widening.** An A/B of the replay showed the same +43 ms per windowed pass with the state in F32,
and the replay's opt-in stage profile (`SANAWM_STREAM_PROFILE=1`) put all of it in `mlp.temporal`: the FFN temporal conv
joins its BF16 tail with the BF16 point-conv output through `StreamingTensors.Concat`, and the 16-bit scatter added for the
CPU backend (below) ran on CUDA too, one strided copy per part, about five times slower than `Concat` on that op. The fix
(`d55f74893`, by the parent session) takes `backend.Concat` on every backend but the CPU. After it: 4090 with F32 state
1.75 to 1.76 s per chunk (the original 1.74), replay passes 170 to 187 ms, device peak 13.1 GB; 3060 with BF16 state 7.6 to
7.8 s per chunk, device peak 10.79 GB. What widening a BF16 state costs on its own was not separately measured after
the fix, because the 4090 runs F32 under `auto`.

**`vram.sanaWmStreamState`** (`auto`, `f32`, `bf16`; `SanaWmStreamState`) picks the storage: `auto` holds the state in BF16 on
a device of 16 GiB or less, which is where F32 runs out of memory, and as computed (F32) on a larger one, which has the room
and skips the conversions. The log line `[SANA-WM] streaming state held as …` says which was chosen. The threshold is a
memory argument; if a later A/B shows BF16 costs nothing on a large card, `auto` could take BF16 everywhere.

Considered and not done:
- *Widening the window once per chunk and reusing it for the chunk's passes.* The window cannot change during a chunk, so
  this was the obvious fix for what looked like a widening cost (it was the `Concat` scatter, above). The cached F32 copy (1.1 GB for the windows, 0.7 GB for the tails) would sit
  beside the committed BF16 state and the staged chunk at the save pass, 0.5 GB over the F32 design at the one point the
  3060 had no memory (11904 of 11912 MiB). Releasing it before the save pass leaves a widen per chunk for that pass.
- *Attention over BF16 operands without widening* (Q, the joined K and V and the output in BF16, the output widened after):
  numerically the same cuDNN BF16 graph the F32 call runs after its internal casts, with half the join traffic, so it could
  be faster than either. It is not verified that an attention row serves the call (the planner refuses mixed dtypes, so Q
  and the output would have to be BF16 too), and the CPU attention does not read BF16, so it would be a second code path.
  Worth trying if the 3060's chunk time matters; `AttentionLayer.PlansBatch` shows how to ask the planner first.

Two properties worth knowing: the BF16 window values are the ones the attention call would have narrowed to anyway, so
GPU numerics should not move beyond a rounding tie; and the 16-bit `Concat` in `StreamingTensors` is a strided scatter,
because the CPU backend's `Concat` writes four bytes per element and corrupts the heap on a BF16 tensor.

## Checkpoint conversion (parent session)

`sana_dit/model.pt` is a pickle. Upstream's loader (`_build_model`) takes `state["generator"]` when that key exists, then
`["state_dict"]` when present, otherwise strips a leading `model.` from every key, and drops `pos_embed`. Save the
resulting state dict to safetensors with the keys unchanged (F32 or BF16; the engine's converter rounds to its own dtype
policy). The expected layout is the stage-1 layout exactly (`SanaWmCheckpointConverter.ExpectedTensors`), with `pos_embed`
optional. `SanaWmCheckpointConverter.ValidateKeys` lists every mismatch, so run it on the header first.

File layout the resolver expects, as on the RAID (override any path with `--model-path`, `--vae-path`,
`--text-encoder-path`). Like stage 1, every companion is named relative to the DiT file:

```
World/SanaWm/streaming/sana_dit/model.safetensors                  --model-path (the converted DiT)
World/SanaWm/streaming/ltx2_causal_vae/diffusion_pytorch_model.safetensors   --vae-path
World/SanaWm/text_encoder/gemma-2-2b-it/{gemma-2-2b-it.safetensors, tokenizer.model}   --text-encoder-path (shared with stage 1)
```

A streaming release that sits beside the stage-1 release (`World/SanaWmStreaming/…`) works the same way: the shared
Gemma-2 is looked for at `../../SanaWm/text_encoder/…` and `../../text_encoder/…`, whichever exists.

**The DiT is a pickle upstream.** Passing `sana_dit/model.pt` resolves to the converted `model.safetensors` beside it when
that exists; otherwise the resolver refuses with the conversion rule above. There is no automatic repack: the pickle is
32 GB with optimizer and critic state in it, and the conversion is a one-off.

**The causal VAE decoder is retrained, not shared.** The release's decoder differs from stage 1's in 90 tensors (the
encoder is bit-identical after BF16 rounding), so `OpenStreamingDecoder` loads the decoder from the streaming file
(`_files.Vae`) and never from the stage-1 VAE; the file is F32 on disk and is cast to the engine's VAE dtype at load.
`OpenImageEncoder` reads the encoder from the same file, which is equivalent.

**Why the catalog entry has no `Assets`.** `ModelResolver` takes a catalog entry's primary asset (its `transformer`
role, else its first asset) as the model path, and `ModelAcquisition` offers to download every missing asset. The
streaming DiT has no hosted safetensors, so an asset for it would be a 32 GB pickle the engine cannot load, and any other
asset would become the primary file. The entry gains `Assets` when the converted DiT is hosted; until then the model
path is explicit (`--model-path`). The refiner's files are never catalog assets, so a stage-1-only run never offers a 40 GB
download.

## Test handoff

CPU checks the author ran while building, as scratch tests now in the shared notes folder
(`streaming-scratch-tests/`): the scan's carried state equals one whole scan; `ForwardChunk` on chunk 0 equals `Forward`;
five chunks stream through GDN and softmax blocks with identical repeat passes; streaming VAE decode equals the causal
whole decode bit for bit; the window trims by chunks and by units; `SanaWmStream` and `ChunkedWorldSession` behave with
scripted components. They are a starting point, not the committed suite.

Unit lane (no weights, no GPU):

| Component | Verify |
|---|---|
| `ChunkPlanner` | boundaries against upstream `create_autoregressive_segments` and `chunk_index_from_chunk_size` for totals 4 to 64, base 3 and 4, every strategy; unbounded spans; `FirstAbsorbsRemainder` refuses no total |
| `SelfForcingFlowSchedule` | sigmas and timesteps for `1000,960,889,727,0` against diffusers `set_timesteps(sigmas=…)`; `Step` against `scheduler.step(per_token_timesteps=…)` on a small tensor; clean frames untouched; validation errors |
| `FrameDeltaScanReference` | two calls with a carried state give the same final state and last-frame readout as one call; `finalState` aliasing `initialState` is refused; null state equals zeros |
| `FrameTailCache` | chunk longer, equal and shorter than the tail; carry across short chunks; `ReplicatePadded`; staged versus committed |
| `RollingTokenWindow` | `RecentChunks(N, sink)` retains what `_accumulate_softmax_kv_cache` retains for chunks 0 to 6 and N in {−1, 1, 2, 3}; `RecentUnits` cuts inside the oldest chunk; ownership on a failed `Stage` |
| `CausalStreamCache` | traversal order mismatch throws; first-use flags flip only on commit |
| `SanaWmTemporalConv.ApplyWithContext` | equals upstream `_cached_temporal_short_conv` on random data (PyTorch fixture) |
| `SanaWmDit.ForwardChunk` | chunk 0 equals `Forward`; repeat denoising passes are identical; `Discard` restores; a later chunk differs from a stateless run |
| `SanaWmGlumbTempFfn` with a tail | equals upstream `CachedGLUMBConvTemp` over two chunks (PyTorch fixture) |
| `LtxVideo2StreamingDecoder` | equals `Decode` of a causal decoder over the whole latent, for 1 and N frames per pass, on a synthetic decoder |
| `SanaWmCameraIntegrator`, `LiveCameraPath` | `Integrate` unchanged (existing parity tests); held and scripted controls; thread safety of `Hold` during `ExtendTo` |
| `SanaWmCameraConditioning.PackRange` | every range of a clip equals the same slice of `Pack` |
| `SanaWmStream` | pass count and timesteps per chunk; frame counts 25 then 24; guided runs both branches; cancel before commit leaves the stream retryable; a fixed path refuses actions |
| `ChunkedWorldSession` | ordered delivery, single enumeration, back-pressure bound, actions before and during the stream, cancel and dispose release the lease and the ledger, a failing open reaches the consumer |
| `WorldPlanResolver` | streaming on a batch model refused; batch on a streaming-only model refused; unbounded needs the capability and no frame count; live camera keeps the script; intrinsics still required; re-resolving a resolved request is idempotent |

Real-weight lane (needs the converted checkpoint and upstream dumps):

| Component | Compare against |
|---|---|
| CUDA `FrameDeltaScan` with state | the host reference, at the real shapes (20 heads, 112 dims, 3 and 4 frames, 880 tokens) |
| one GDN block, chunk 1 | upstream block output and `kv_cache` slots 0, 1, 2 after the save pass (slot 0 transposed) |
| one softmax block, chunk 2 | upstream output with the `[sink, previous]` window |
| DiT `ForwardChunk` | upstream `forward_long` output per pass of chunks 0, 1 and 2 (5 passes each), with the same latents, timesteps, camera slices and cache |
| sampler | the chunk latents after each chunk of a 3-chunk upstream run, with upstream's `z` injected (`ChunkNoise`, `FirstLatent`) |
| causal VAE | `CausalVaeStreamingDecoder.decode_chunk` pixels for chunks 0 and 1 |
| end to end | a bounded 73-frame run against upstream stage-1-only output (upstream with the refiner bypassed), PSNR per frame |

## What the parent session needs to provide

1. ~~The streaming checkpoint header (key, shape, dtype) after conversion, to confirm the layout assumption.~~ Received
   2026-11-09: the converted header is the stage-1 layout exactly, without `pos_embed`, every tensor BF16.
2. ~~`ltx2_causal_vae/config.json`.~~ Received 2026-11-09: stage-1's VAE config plus `decoder_causal: true`,
   `decoder_upsample_type: null`; reflect decoder padding, residual upsamplers, `scaling_factor` 1.0. Matches the
   assumption.
3. Upstream dumps for the real-weight lane above. The least to dump from one 3-chunk run at `--num_frames 73`: the
   initial `z`, the per-chunk `kv_cache` after each save pass, the model output of each of the 15 forwards, the latents
   after each chunk, and `decode_chunk` pixels per chunk.
4. A GPU measurement of resident memory and seconds per chunk on the 3060 and the 4090.

## Phase B: the streaming refiner

Upstream's refiner is the LTX-2.0 (19B) audio-video transformer run video-only, block by block
(`diffusion/refiner/diffusers_ltx2_refiner.py`, `RefinerChunkRunner`). It is optional here (`--refiner`,
`WorldRequest.Refiner`) and off by default.

| Step | When | What |
|---|---|---|
| Sink capture | once, when the stream opens | the seed frame's latent alone at σ = 0, frame position 0; every block stages its raw key and value projections |
| Re-noise | per block | `x = (1 − σ₀)·clean + σ₀·ε` with σ₀ = 0.909375 |
| 3 distilled steps | per block | σ = 0.909375, 0.725, 0.421875 → 0; each forward attends the block's queries to `[sink ⊕ history ⊕ block]` |
| History capture | per block | one forward over the refined block at σ = 0 under the same prefix; its rotated keys and values join the history |

A block is the stage-1 chunk without the stream's first frame: frames `[1 + 3b, 4 + 3b)`. The seed frame's latent is the
refiner's sink and is decoded unrefined; the DiT's own carried state keeps the unrefined latents. With the refiner a
chunk costs 5 DiT forwards, 4 refiner forwards and the decode.

### Decisions

**A video-only LTX-2.0 transformer beside the dual-stream one.** `LtxVideo2Transformer` runs LTX-2.3 and 2.5: nine
modulation rows per block, modulated and gated text cross-attention, per-modality connectors, Lightricks key names. The
refiner checkpoint is the LTX-2.0 diffusers layout: six rows, an unmodulated ungated cross-attention, the caption
projection inside the transformer, `norm_q`/`norm_k`. `LtxVideo2ChunkTransformer` reads that layout and only its video
keys (1215 of 3510 tensors, 13.04B of 18.88B parameters), so the audio stream is never paged in. What the two share was
extracted rather than copied: `LtxVideo2BlockOps` (modulation, norm-and-modulate, gated residual, FFN, output layer),
`LtxVideo2AdaLnSingle`, the connector blocks, and `LtxVideo2Attention` now accepts either QK-norm key name. Without a
stream state the class is the plain bidirectional video-only forward, so it is not SANA-WM specific.

**One modulation per pass.** Every token of a block carries the same sigma, so the AdaLN-Single modulation is computed
once per forward and broadcast. Upstream evaluates it per token (`[tokens, 6·4096]`); the values are the same.

**Token-major keys and values.** The carried tensors are `[tokens, 4096]` rows, as the projections produce them. Joining
the prefix is then one contiguous copy per part along axis 0, and the attention call is the token-major entry the
dual-stream path already uses (`AttentionLayer.AttendFlat`), so no head permute runs on CUDA. Every op in the block is
one the dual-stream path runs in production: `LinearMulti`, `RmsNormRotary` with split rows, token-major attention,
`NormModulate`, `GatedResidualLastDim`, `LinearFused`.

**The sink key is kept before the norm.** Upstream caches the sink's key after the QK norm and before the rotation, and
re-rotates it for every block at `block_start − history_frames − sink_frames`. Keeping the raw `to_k` projection
instead and running the fused norm-and-rotate with the sink's tables at each use gives the same key (the norm is
deterministic) and needs no stand-alone rotation of an F16 tensor. History keys are kept after the rotation, at the
absolute frames they were produced at, as upstream keeps them.

**Refine before either model commits.** The refiner reads only the clean latents, so `SanaWmStream` runs it after the
DiT's state-writing pass and before the commit: a chunk cancelled during refinement discards both staged states and can
be generated again. After the commit a failure still faults the stream, as in Phase A.

**The capture pass stops at the last block's projections.** Nothing reads the last block's output in a capture pass, so
it stages its keys and values and returns. The values staged are identical; one block's attention and FFN are saved per
block. (Upstream has this behind `SANA_WM_REFINER_CAPTURE_KV_ONLY_LAST`.)

**Noise is seeded per block.** Upstream draws from a CUDA generator seeded with 42. Block `b` uses
`seed + b·0x61C88647` here, so a block refined again after a discard reproduces. Parity runs inject upstream's noise
(`SanaWmRefinerSettings.Noise`, `SanaWmStreamInjection.RefinerNoise`).

**The refiner's prompt is encoded first and alone.** It is the raw prompt (no CHI prefix, no negative), through
Gemma-3-12B and the LTX-2.0 connector, once, before the DiT loads; the encoder is released before anything else is
placed. The tower is given only the prompt's own tokens: upstream left-pads to 1024 and masks the padding, and since
the tower's rotation is relative and padding is never attended to, the hidden states are the same. The connector then
fills the sequence to 1024 with its registers, as upstream's does.

**The refiner's Gemma-3-12B is the engine's FP8-scaled LTX-2 side model by default.** `SideModels.GemmaLtx2`
(`text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`, 13.2 GB, Comfy layout, per-tensor `weight_scale`) is a
quantization of the same checkpoint as the release's `gemma3_12b/` (five BF16 shards, 24.4 GB, whose sizes equal
google/gemma-3-12b-it's), so within FP8 rounding it gives the same hidden states, and it is the one that fits a 24 GB
card. Pass `--refiner-text-encoder-path` for the release's BF16 shard folder (parity runs). Both layouts load: the
transformers export nests the decoder under `language_model`, which `Gemma3DecoderKey` maps onto the `model.layers.*`
layout the encoder reads, dropping the vision tower, projector and head. **Do not point it at
`text_encoders/gemma_3_12B_it.safetensors`**: that file is a mixed FP8/U8 quantization with `weight_scale_2`, an
NVFP4-style format the encoder does not read.

**The weight-cast cache is off while an FP8 Gemma-3 is loaded.** With the cache on, each FP8 matrix is widened once and
the F16 copy kept beside the packed weight, about twice the footprint (`SideModels.GemmaLtx2` carries the same advice for
24 GB cards). The stage turns `IBackend.CacheWeightCasts` off when the tower's weights are FP8 and restores it when the
stage releases. The tokenizer is the Gemma-3 `tokenizer.model` found by the LTX-2 recipe's locator
(`LtxVideo2Recipe.LocateGemmaTokenizer`, shared) beside the encoder; on the RAID it is `text_encoders/tokenizer.model`,
which is the 4.69 MB Gemma-3 vocabulary, not the Gemma-2 one.

**Weights are placed by `BlockStreamingScope`.** The refiner opens last. Its resident prefix is sized from the device
memory the DiT and the decoder left free, minus the carried state at a full window (reserved up front, since it arrives
over the first blocks) and `vram.sanaWmRefinerHeadroomMb`. Blocks that do not fit are streamed from the host on every
forward. A window the device cannot hold at all is refused when the refiner opens, not several chunks in.

**Sequential, not three CUDA streams.** Upstream overlaps stage 1 of chunk `k+1`, the refiner of chunk `k` and the
decode of chunk `k−1` on three CUDA streams. The session runs them one after the other on its compute thread. The
output is the same; overlap is a later optimization and needs backend support for concurrent streams.

**Not ported.** Upstream's approximations and speed modes, all off by default there: reusing the last denoising
forward's keys and values instead of the capture pass (`FAST_KV_CAPTURE`), the cross-attention K/V cache, FP8 storage
of the history, NVFP4/FP8 block scaling through Transformer Engine, QKV fusion and `torch.compile`. The legacy
single-shot refiner (`block_size=None`) is not ported either.

### Carried state of the refiner

Per transformer block (48), at 720p (880 tokens per frame, inner width 4096), in the block stream's activation dtype
(F16 on CUDA with `numerics.ditF16`, the default; F32 otherwise):

| State | Shape | Kept | F16 size |
|---|---|---|---|
| sink key, raw `to_k` projection | `[880, 4096]` | whole stream | 6.9 MiB |
| sink value | `[880, 4096]` | whole stream | 6.9 MiB |
| history keys, rotated at their own frames | up to `[10·880, 4096]` | newest `window − 1` frames | 68.8 MiB |
| history values | up to `[10·880, 4096]` | newest `window − 1` frames | 68.8 MiB |

Over 48 blocks: 0.64 GiB for the sink and 0.645 GiB per history frame, 7.1 GiB at the default window of 11 frames.
Per chunk the scope also holds F32 rotation tables (`[tokens, 2048]` cos and sin: 43 MiB for a 3-frame block, 14 MiB
per sink placement), and the stream holds the projected caption (`[1024, 4096]`).

The window is `RollingTokenWindow` with `TokenWindowPolicy.RecentUnits(window − 1)`: a block that only partly fits is
cut to its newest frames, which is upstream's token-level trim. `LtxVideo2ChunkWindow.HistoryLayerStride` keeps the
history on every N-th block only (and the last); the others read the sink at `block_start − 1`.

### Numerics of the refiner

- The block stream runs in the engine's LTX-2 activation dtype (F16); upstream runs BF16. `numerics.ditF16=false` runs
  F32 for parity work.
- The sampler is float32 with the sigma ratio taken in double from the float32 sigmas, as upstream's Python floats are.
  With `RoundToBf16` (the default) the noise, each velocity, each clean estimate and each stepped latent are rounded to
  BF16, where upstream casts back to its BF16 dtype.
- The model timestep is `σ·1000` in float32; capture passes run at 0.
- Rotation phases are computed in double (`LtxVideo2Rope`, as for LTX-2.3); upstream computes them in float32, which is
  up to about 1e-3 rad off in the highest-frequency lanes.
- The text packing statistics (mean, min, max per hidden state) are computed in float32 and double; upstream computes
  them on BF16 tensors.
- CPU parity against upstream's own `RefinerChunkRunner`, `_forward_video_only_with_rope`, `_pack_text_embeds` and
  diffusers 0.37.0 modules, on a tiny random model in float32 without BF16 rounding (2 heads of 16, 3 blocks, 2×3
  latent grid, 4 blocks of 3 frames): refined latents differ by at most 3.0e-4 on values up to 11.6 for windows of 1, 5
  and 11 frames; the text connector by 7.3e-5 on values up to 2.5; a plain forward by 2.5e-4 on values up to 9.5. The
  fixture script and file are in the shared notes folder (`streaming-scratch-tests/refiner-parity-fixture/`).

### Memory plan with the refiner

Sizes from the checkpoint header and tensor shapes, in GiB; nothing is measured yet.

| Item | Size |
|---|---|
| refiner transformer, video keys, BF16 as shipped | 24.3 (0.51 per block, 0.29 shared) |
| the same in FP8 | about 12.2 (0.26 per block) |
| refiner carried state, window 11, F16 | 7.1 (0.64 + 0.645 per history frame) |
| Gemma-3-12B, the engine's FP8-scaled single file (default) | about 12.3, cast cache off |
| Gemma-3-12B, the release's BF16 shards (`--refiner-text-encoder-path`) | about 22 (language model only) |
| text connector (`text_proj_in` 2.7 in F32, plus two blocks) | about 2.9 |
| stage 1 (DiT and its carried state, Phase A estimate) | about 7.5 |

*48 GB and up.* Everything is resident in BF16: 7.5 + 24.3 + 7.1 + 4 of headroom is about 43.

*24 GB (the 4090).*
1. Text: the default, the FP8-scaled single file, fits (12.3 GiB, cast cache off) and leaves before the DiT loads. The
   BF16 shards (22 GiB) do not fit beside the tower's own activations.
2. Transformer: with stage 1 resident about 15.5 is free. At the defaults (window 11, 4 of headroom) about 4.4 is left
   for blocks: 8 BF16 blocks resident and 40 streamed, 20 GiB over PCIe per forward and 4 forwards per block. That
   runs but is slow.
3. Levers, cheapest first: an FP8 transformer file (17 blocks resident, 7.9 GiB streamed per forward);
   `vram.sanaWmRefinerWindowFrames=7` (state 4.5); `vram.sanaWmRefinerLayerStride=2` (state about 3.9, an
   approximation); `vram.sanaWmRefinerHeadroomMb` lowered once the DiT's and the decoder's real transients are
   measured. FP8 weights with a window of 7 leave about 27 blocks resident.

*12 GB (the 3060).* Stage 1 only. The refiner's carried state alone does not fit beside the DiT, and the refiner
refuses to open with that message.

Compute, for scale: about 0.95 TFLOP per transformer block per forward at a full window, 182 TFLOP per 3-frame block
(1.5 s of video). A 4090 will not run the refiner in real time even fully resident.

### Files and options

Release layout beside the DiT folder; each path can be overridden:

```
World/SanaWm/streaming/refiner_diffusers/transformer/diffusion_pytorch_model.safetensors   --refiner-path
World/SanaWm/streaming/refiner_diffusers/connectors/diffusion_pytorch_model.safetensors    --refiner-connectors-path
text_encoders/gemma_3_12B_it_fp8_scaled.safetensors   (SideModels.GemmaLtx2, found by the catalog's side-model path)
                                                                                           --refiner-text-encoder-path (file or shard folder)
text_encoders/tokenizer.model   (beside a single-file encoder; inside a shard folder)       --refiner-tokenizer-path
```

The files are named when the model resolves and checked only when a run asks for the refiner, so a stage-1 install
needs none of them. Settings: `vram.sanaWmRefinerWindowFrames` (11), `vram.sanaWmRefinerLayerStride` (1),
`vram.sanaWmRefinerHeadroomMb` (4096). Run: `hartsy world -m sana-wm-streaming … --refiner`.

### Test handoff for the refiner

Scratch tests the author ran on the CPU, in the shared notes folder (`streaming-scratch-tests/`):
`Diffusion_ZzScratchRefinerParity.cs` (set `REFINER_PARITY_FIXTURE` to the fixture file),
`World_ZzScratchStreamRefinerSmoke.cs`, `Engine_ZzScratchRefinerWiring.cs`.

Unit lane (no weights, no GPU):

| Component | Verify |
|---|---|
| `FlowRefineSchedule` | `Renoise` and `Step` against upstream's arithmetic on a small tensor, with and without BF16 rounding; the last step lands on the clean estimate; validation errors |
| `LtxVideo2Rope.BuildVideo(firstFrame)` | equals rows `[firstFrame·H·W, …)` of a whole-clip build; frame 0's clamped coordinate |
| `LtxVideo2ChunkWindow`, `LtxVideo2ChunkState` | history layers for strides 1, 2, 3 and offsets; sink offsets beside and without history for blocks 0 to 5 against upstream's formula; `EstimateBytes`; `CommitSink` refuses a partial capture; `Discard` and `Reset` |
| `LtxVideo2PrefixSelfAttention` | against a host reference (RMS norm, split rotation, softmax) with and without a prefix; a capture-only pass stages what a full pass stages; a failed pass leaves the stream untouched |
| `LtxVideo2ChunkTransformer` | the tiny-model fixture: plain forward, three window sizes; `IsVideoKey` on the real header (1215 keys); a nine-row table is refused with the LTX-2.3 message; scope and state mismatches are refused |
| `LtxVideo2ChunkRefiner` | the fixture end to end; a repeated `Refine` before `Commit` is identical; `Commit` without a staged block throws; cancel between passes discards; seeded noise differs per block and repeats per seed |
| `LtxVideo2PackedTextConnector` | the fixture; a prompt longer than the sequence and a length that is not a register multiple are refused |
| `SanaWmStream` with a refiner | open order and sink at open; chunk 0 refines 3 of 4 frames and decodes the sink unrefined; commit order; cancel during refinement discards both; progress counts 9 passes; components without the refiner refuse `Refine` |
| engine wiring | `Gemma3DecoderKey` for both exports; refiner file naming and overrides; a plan with the refiner reaches `SanaWmStreamRequest`; `sana-wm` refuses the refiner; missing refiner files fail with the aux key named |

Real-weight lane:

| Component | Compare against |
|---|---|
| text path | `DiffusersLTX2Refiner._encode_prompt` embeddings for one prompt, with the BF16 tower and with the FP8 file |
| one transformer block, block 1 | upstream block output with the prefix of block 0 |
| `Predict` | upstream `_forward_video_only_with_rope` velocity for each of the 3 steps of blocks 0, 1 and 4 |
| carried state | upstream `_sink_kv_pre` (post-norm; compare after norming ours) and `_history_kv_post` after blocks 0 and 3 |
| refiner end to end | `refine_block` outputs for a 73-frame run with upstream's noise injected |
| stream end to end | upstream's refined video, PSNR per frame |

CUDA-specific things the CPU run cannot show, in the order worth checking: F16 activations through the token-major
attention with more keys than queries (12 320 against 2 640); `RmsNormRotary` on the sink's 880 rows with its own
tables; `Concat` along axis 0 of F16 parts; `PinActivation` keeping the carried tensors across the DiT's and the
decoder's work; block streaming with the carried state growing after placement; the FP8 and INT8 attention rows, if
those knobs are on.

### What the parent session needs to provide for the refiner

1. ~~The streaming repo's refiner header and configs.~~ Received 2026-11-09: the same 3510 keys and shapes as the
   bidirectional repo's, the same transformer and connector configs. One difference: all 194 modulation tables
   (`scale_shift_table`, `video_a2v_cross_attn_scale_shift_table`, the audio ones and the two top-level tables) are BF16
   here, F32 there; the loader widens them to F32 (`TensorCasts.LoadF32`), so the forward is unchanged.
2. ~~Whether `gemma3_12b` is the same as the FP8 file.~~ Answered 2026-11-09: the release shards are BF16 and
   google/gemma-3-12b-it's sizes byte for byte; the FP8-scaled side model quantizes the same weights. See the decision
   above for which file to use.
3. Upstream dumps from one 73-frame run with the refiner: the refiner prompt embeddings, the per-block noise, the
   velocity of each forward, `_sink_kv_pre`, `_history_kv_post` after each block, and each `refine_block` output.
   Note that upstream's `_streaming_self_attention` reads `attn.to_gate_logits`, which diffusers 0.37.0's
   `LTX2Attention` does not define; the fixture script sets it to `None`.
4. Measurements on the 4090: resident blocks, seconds per block, and the DiT's and the decoder's peak transients (to
   set the headroom default).
5. An FP8 conversion of the refiner transformer's video keys, if the 4090 is to run it at a usable speed.
