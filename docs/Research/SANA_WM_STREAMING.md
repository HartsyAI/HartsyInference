# SANA-WM streaming: design note

Source snapshot: NVlabs/Sana `b3bad956` (`inference_sana_wm_streaming.py`, `generate_streaming`, HF
`Efficient-Large-Model/SANA-WM_streaming`). This note records the upstream-to-engine map, the decisions and their reasons,
the carried-state layouts, the numerics, the memory plan and the test handoff. Status and parity evidence belong in
[MODEL_STATUS_WORLD.md](../Checklists/MODEL_STATUS_WORLD.md) and [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md)
once the weights have been run; nothing here has run on real weights or on a GPU yet.

## What streaming is

Stage 1 (`sana-wm`) denoises the whole clip at once and is bidirectional in time, so its camera path must be known before
generation. Streaming (`sana-wm-streaming`) is a different checkpoint of the same DiT layout, distilled to be chunk-causal:

| Stage | Upstream | What it does per chunk |
|---|---|---|
| Student | `SanaMSVideoCamCtrlStreaming` + `SelfForcingFlowEulerCamCtrl` | 4 denoising forwards over the chunk's noise, then 1 forward over the clean chunk at t = 0 that writes the state the next chunk reads |
| Refiner (Phase B) | `RefinerChunkRunner` over the LTX-2 transformer | 3 distilled steps per 3-frame block against a sliding KV window |
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
| key conv tail | GDN | F32 `[3·S, C]` | 4 | raw key projection of the newest 3 frames |
| camera key conv tail | GDN | F32 `[3·S, C]` | 3 | raw camera key projection |
| main K, V window | softmax (5) | F32 `[1, H, n·S, D]` ×2 | 0, 1 | K after RoPE at its own absolute frames, V raw |
| camera K, V window | softmax | F32 `[1, H, n·S, D]` ×2 | 2, 3 | after the ray transform and camera RoPE |
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
- Softmax K and V are kept in F32 and the attention runs with `AttentionNumerics.Bf16Storage`, which rounds operands to
  BF16 on use. Upstream stores them as BF16; the values attended are the same.
- Scan states and conv tails are F32. Upstream's are whatever its BF16 autocast leaves them; the engine's stage-1 layers
  already keep q, k and v in F32 between ops and are at parity that way.
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

That is tight but plausible on 12 GB. Two levers if it does not fit: keep the softmax windows in BF16 (halves the
largest item; needs the attention call to accept BF16 K/V operands) and keep the key tails in BF16 (the conv input is then
rounded, a numerics change to measure). A second guidance branch doubles the carried state.

Phase B adds the 18.9B-parameter LTX-2 refiner; see the Phase B section.

## Checkpoint conversion (parent session)

`sana_dit/model.pt` is a pickle. Upstream's loader (`_build_model`) takes `state["generator"]` when that key exists, then
`["state_dict"]` when present, otherwise strips a leading `model.` from every key, and drops `pos_embed`. Save the
resulting state dict to safetensors with the keys unchanged (F32 or BF16; the engine's converter rounds to its own dtype
policy). The expected layout is the stage-1 layout exactly (`SanaWmCheckpointConverter.ExpectedTensors`), with `pos_embed`
optional. `SanaWmCheckpointConverter.ValidateKeys` lists every mismatch, so run it on the header first.

File layout the resolver expects (override any path with `--model-path`, `--vae-path`, `--text-encoder-path`):

```
World/SanaWmStreaming/sana_dit/<dit>.safetensors
World/SanaWmStreaming/ltx2_causal_vae/diffusion_pytorch_model.safetensors
World/SanaWm/text_encoder/gemma-2-2b-it/{gemma-2-2b-it.safetensors, tokenizer.model}   (shared with stage 1)
```

The catalog entry has no `Assets` yet: the DiT has no safetensors upstream, so the model path is explicit until a converted
file is hosted.

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

1. The streaming checkpoint header (key, shape, dtype) after conversion, to confirm the layout assumption.
2. `ltx2_causal_vae/config.json`: the decoder is assumed to be the bidirectional VAE's decoder with
   `decoder_causal: true` (`upsample_residual` all true, reflect padding, `scaling_factor` 1.0). If any of that differs,
   `SanaWmEngineComponents.OpenStreamingDecoder` is the one place to change.
3. Upstream dumps for the real-weight lane above. The least to dump from one 3-chunk run at `--num_frames 73`: the
   initial `z`, the per-chunk `kv_cache` after each save pass, the model output of each of the 15 forwards, the latents
   after each chunk, and `decode_chunk` pixels per chunk.
4. A GPU measurement of resident memory and seconds per chunk on the 3060 and the 4090.

## Phase B: the streaming refiner

Not built yet. Upstream's refiner is the LTX-2 19B video transformer run block by block (`RefinerChunkRunner`): per
3-latent-frame block it re-noises the stage-1 latent to `σ_max`, runs 3 distilled Euler steps where the block attends to
a KV prefix (a pre-RoPE sink re-rotated to sit just before the window, plus up to 10 frames of post-RoPE history), then
one more forward captures the block's post-RoPE K/V for the history. The shared pieces above cover it: `RollingTokenWindow`
with `TokenWindowPolicy.RecentUnits(10)` for the history, a `StreamingTensorSlot` per layer for the sink, and the same
stage-then-commit discipline. The plan and its memory budget are recorded here when the phase lands.
