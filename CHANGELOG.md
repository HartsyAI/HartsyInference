# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

## Unreleased

- **Added: UniRig skeleton generation (stage 1 of the auto-rigger), validation-pending.** `HartsyInference.ThreeD.Models.UniRig`
  ports the `VAST-AI/UniRig` articulation-xl skeleton stage: the Michelangelo-style point encoder (65,536 points with normals, 1,024
  farthest-point queries), the OPT decoder hosted on `GenericTransformer` with the checkpoint's keys remapped, the 256-bin
  skeleton-tree tokenizer and its grammar as a sampler step, and mesh preparation (welding, normalization, upstream's `mix` point
  sampler) in pure C#. The `.ckpt` is read with the safe-subset pickle loader. Not yet wired to `hartsy 3d` or `MeshService`; skinning
  and rigged-GLB export follow in later PRs. Deviations from upstream: one sampled sequence instead of 15-beam sampling, a seeded
  Fisher-Yates point subset instead of numpy's stream, no decimation, retry with the next seed when a draw does not decode.
  `HartsyInference.ThreeD` now references `HartsyInference.LLM`. On the real checkpoint, an Xbot humanoid yields a 52-joint body with
  fingers and a horse a 32-joint quadruped (CUDA, RTX 3060, about 2 s per generation after load); no Python reference was run.

## alpha.415

- **Fixed: multi-image and video prefill no longer grows with frames squared.** The gated cross-attention layer (MOSS-VL)
  ran every prompt row's attention once per image group, projected each, and masked away all but one group's rows. Groups whose
  rows are one contiguous range (every MOSS-VL group) now attend only their own rows over their key prefix, the results are
  stitched in row order and projected once, and their `[1, T, hidden]` row masks are no longer built. Same results (segmented and
  single forwards agree within 1.2e-3; the video reply is unchanged). MOSS-VL GGUF prefill on Vulkan (RTX 3060):

  | frames | before | after |
  |---|---|---|
  | 16 | 5.2 s | 0.9 s |
  | 32 | 35.0 s | 2.3–3.4 s |
  | 64 | 378 s | 10.2 s |

  Decode with 64 frames held went from 12.1 to 20.4 tok/s; up to 32 frames it stays at 30.5 to 31.8 tok/s (28.4 before at 32
  frames). Groups without a range keep the masked path, with the output projection now applied once to the summed groups.
- **Added: `MossVlProfileTests`** (`Category=GpuBenchmark`, opt-in `HARTSY_MOSS_VL_PROFILE=1`): vision tower per frame (about
  325 ms for a 256×256 frame on Vulkan), prefill and decode by frame count.

## alpha.414

- **Added: the MOSS-VL NF4 and FP8 releases load from their folders.** `hartsy text --model-path <MOSS-VL-Realtime-NF4> --precision
  q8_0 -i photo.png "…"` names a red circle and a blue square on Vulkan at 28.8 tok/s; the FP8 release with `--precision q4_k` does
  too at 9.7 tok/s. The first load converts once into `_gguf-cache` inside the folder (about 3 minutes); later loads map it.
- **Added: quantized Hugging Face checkpoints load through one shared path.** `HfQuantWeightCache` (ModelAssets) reads any
  checkpoint `HfQuantCheckpoint` indexes (NF4, FP8 block or per-channel, GPTQ, MXFP4, NVFP4, MLX), borrows the weights used as
  stored, and caches the converted ones as GGUF; FishAudio Drama 3 uses it too. `DenseTextWeightPolicy` (LLM) picks each weight's
  load type: 4-bit sources as Q4_K (else Q4_0), 8-bit ones as Q8_0, where the backend keeps them packed.
- **Added: compressed-tensors FP8 with per-row or per-tensor scales** (llm-compressor `FP8_DYNAMIC`), the layout of the MOSS-VL FP8
  release. Decoded against BF16: 2.7% relative RMS.
- **Added: `--precision` / `TextRequest.WeightPrecision`.** Re-quantizes a checkpoint folder's plain BF16 projections (q8_0, q6_k,
  q5_k, q4_k, q4_0) so a model too large in BF16 fits a smaller card.
- **Changed: the GGUF quant cache converts several tensors at once** (`GgufWriter.ProduceAhead`), still writing them in order.
- **Fixed: the MOSS-VL vision tower of a quantized folder reads only its own weights**, instead of folding the text model's FP8
  scales and failing.

## alpha.413

- **Added: live MOSS-VL sessions over HTTP.** `POST /v1/native/realtime/sessions` opens one (the model loads through the
  long-running queue). `POST …/{id}/frame` takes a base64 PNG and its timestamp, `POST …/{id}/prompt` a prompt, and
  `GET …/{id}/stream` is server-sent events (`round_start`, `text`, `silence`, `control`, `frames_dropped`, `error`) with a
  keep-alive comment while the model is silent. `GET …/{id}` returns the counters and `DELETE …/{id}` closes the session and frees
  its device slot. Sessions idle longer than `RealtimeSessionIdleTimeoutMinutes` (10) are closed. Checked end to end on Vulkan
  against the GGUF: open, three frames, a prompt, the stream, stats, close, then 404.
- **Added: `video_url` content parts on `/v1/chat/completions`.** A `data:video/…;base64,` part becomes the message's video for
  video-capable models (128 MiB at most, one per message); a remote URL or a non-`video/` type is refused. Chat completions now
  takes bodies up to `MaxVideoRequestBodyBytes` (256 MiB), as native video does, instead of Kestrel's 30 MB default.
- **Changed: live-session streams.** One stream per session (a second gets 409); a session whose loop failed reports its `error`
  event and is closed, freeing its device slot, and later input to it is a 404; closing a session never reports a failure for one
  already removed.
- **Changed: world and realtime sessions share `SessionRegistry<TSession>`.**
- **Added: `tests/HartsyInference.API.Tests`.** It also runs `SeedVr2ModelLayoutTests`, which had no project before; its one
  Windows path-separator bug is fixed.

## alpha.412

- **Added: live MOSS-VL-Realtime sessions.** `hartsy text --realtime -m moss-vl-realtime --video clip.mp4 "…"` (or `--frames-dir`)
  replays the frames at their own pace into a running session, takes further prompts from stdin, and prints the reply as it
  streams. The engine side is `ITextService.OpenRealtimeAsync`, which returns an `IRealtimeVlmSession` (push frames and prompts,
  stream round-start, text, silence, control and dropped-frame events). A session holds its device's text slot until disposed and
  takes the device gate per forward. On Vulkan, frames of a red disc with "What color is the circle?" give "The circle is red."
- **Added: `RealtimeVlmSession`, a model-free live loop.** It follows the reference realtime sampler: after each token it splices
  pending prompts and frames into the stream, idles on the silence token with a 20 ms poll, inserts the protocol's silence when a
  cycle brings both prompts and frames, drops the oldest frame from a full queue, flushes text on whole characters, and paces the
  token rate. A model plugs in through `IRealtimeVlmProtocol`, `IStreamingVlmDecoder` and `IStreamingFrameEncoder`;
  `MossVlStreamingDecoder` adapts the MOSS decode session. Control tokens other than silence (MOSS: `<|response|>`) are reported
  apart from the reply text.

## alpha.411

- **Added: video input for MOSS-VL-Realtime.** `hartsy text -m moss-vl-realtime --video clip.mp4 "…"` samples the clip with ffmpeg
  per the model's video settings (1 fps, up to 256 frames, `--video-fps`/`--video-max-frames` to change). `--frames-dir` takes a
  folder of PNG frames instead and needs no ffmpeg. The API takes `TextMessage.Video` or `TextMessage.VideoFrames`. The frames share
  one size (the reference's video smart-resize, now `SmartResize.Qwen2VlVideo`) and go into one timed `<|vision_start|>…<|vision_end|>`
  block. On Vulkan a five-frame clip of a red disc crossing the frame is described as "A red circle … moves from the left side of the
  frame to the right side". Other models refuse a request with video by name.
- **Changed: the MOSS-VL decode session appends.** `MossVlDecodeSession.Forward` adds a segment of text and images to a running
  stream. Positions continue from the last segment, each row sees every earlier image, and each image's vision keys and values are
  projected once into a growing `CrossKvCache`. A stream built in two segments gives the same logits as one forward (max diff 0 on
  Vulkan). This is the base for the realtime session. A cancelled segment leaves the session usable.
- **Fixed: MOSS-VL still images use the checkpoint's prompt layout.** The chat template renders an image as a bare `<|image_pad|>`;
  the engine had wrapped it in `<|vision_start|>`/`<|vision_end|>`. Asked whether two copies of one image are the same, the model
  now answers "Yes, they are the same."
- **Added: `IRealtimeVlmProtocol`** (prompt layouts and control tokens of a streaming VLM) with MOSS-VL's implementation, and a shared
  `VlmDecodeOps.DecodeLoop` (stop tokens, streaming deltas, cancellation) for VLM generators.

## alpha.410

- **Added: MOSS-VL-Realtime runs from `hartsy text` and the text API.** `hartsy text -m moss-vl-realtime -i photo.png "…"` (or
  `--model-path` to the GGUF or the checkpoint folder) loads the decoder and vision tower on one backend, preprocesses every
  image of the last user message, and streams the reply. Catalog id `moss-vl-realtime`: the community Q4_K GGUF (vision inside)
  plus the tokenizer and JSON configs from the official repo, all hash-pinned; variant `bf16` is the official checkpoint.
  A request without an image is refused by name. On Vulkan (RTX 3060) a synthetic red disc and blue square are named correctly at
  about 29 tok/s decode.
- **Added: a MOSS-VL GGUF needs no `config.json`.** `MossVlConfig.FromGguf` reads the `moss_vl.*` metadata and matches the
  checkpoint's config. The released GGUF embeds no tokenizer, so a GGUF without `tokenizer.json` beside it is refused, with the
  file and repo named.
- **Changed: `MossVlGenerator` streams.** A new overload reports each decoded piece and honours cancellation, and returns the token
  counts and whether the limit was hit.

## alpha.409

- **Added: shared VLM image preprocessing in Core (`HartsyInference.Core.Imaging`).** `SmartResize.Qwen2Vl` is an exact port of
  `transformers`' `smart_resize`. `SmartResize.ShareBudget` splits a batch pixel budget by area. `VisionPatchPreprocessor` turns
  8-bit RGB into merge-block patch rows for any patch, merge and temporal geometry. `ResampleOptions.PilBicubic` matches Hugging
  Face's fast Qwen-VL processors. `Resample` moved here from `HartsyInference.Vision.Codec`, so LLM and Diffusion code can use
  it. Qwen3-VL's processor now delegates its smart-resize and patchify to these types, with unchanged output.
- **Added: MOSS-VL reads raw pixels.** `MossVlImageProcessor` applies the checkpoint's `preprocessor_config.json`, including
  `multi_image_max_pixels` shared across the images of one request. Against `transformers` on four committed cases the grids
  are identical, and 0.2–0.3% of values differ, each by one 8-bit level. A synthetic red-disc and blue-square image run
  through the processor, the GGUF tower and the 48-layer decoder on Vulkan is described correctly.
- **Added: video frame sampling.** `VideoFrameSampling.Plan` and `FrameSize` implement the Qwen/MOSS video rule (fps, frame
  bounds, numpy `linspace` indices, a shared pixel budget). `VideoTimestampText.Seconds1` formats like Python's `f"{t:.1f}"`.
  `FfmpegFrameSource` keeps only the sampled frames while ffmpeg streams a file, and can replay it at its timestamps.
  `FfmpegProcessDecoder` gains `ProbeStreamAsync` (frame count) and `StreamFramesAsync`.

## alpha.408

- **Added: MOSS-VL loads in stored precision for the backend that runs it.** `MossVlTextDecoder.Open`/`OpenGguf` and
  `MossVlVisionTower.Open` take the backend (`MossVlLoadOptions.Backend`). BF16 projections stay BF16 on CUDA (Ampere or newer)
  and are cast once to F16 on Vulkan, which refuses by name any weight F16 cannot hold. GGUF quants stay packed where the backend
  has a kernel and widen to F16 elsewhere. CPU, and a load with no backend, keep the old F32 path. The rule lives in the shared
  `DenseWeightCasts`, so any model can use it.
- **Added: the MOSS-VL vision tower loads from the GGUF.** `MossVlVisionTower.OpenGguf` reads the `v.*`/`mm.*` blocks the MOSS-VL
  GGUF carries, with the shape from its `moss_vl.vision.*` metadata, so a GGUF user no longer needs the 22.7 GB safetensors
  checkpoint for vision. The converter pads the fused qkv (q and k per rotary half, 36 to 40 rows; v per head, 72 to 80);
  `UnpadQkv` restores the checkpoint layout. On CPU the GGUF tower matches the reference at 1.9e-6 relative.
- **Added: `GgufMetadata.GetInteger`/`GetIntegerArray`** read an integer stored as any GGUF integer type; converters differ.

## alpha.407

- **Added: cuBLAS auto-provisioning.** A machine with only the NVIDIA driver can now run the CUDA backend. When no cuBLAS is
  installed, the first CUDA backend construction, an explicit `cuda` device, or `auto` on a card the kernels support
  downloads NVIDIA's cuBLAS redist for the driver's CUDA version once (13.10.0.20 for CUDA 13, 12.9.2.10 for CUDA 12), checks
  its sha256 and unpacks it into `~/.cache/hartsyinference/cublas/cudaN/lib`. A cuBLAS built for a newer CUDA than the driver is
  never loaded. Probes such as `CudaContext.IsAvailable` stay download-free. Settings: `paths.cublasAutofetch` (default on),
  `paths.cublasDir`, `paths.cublasUrl` (a mirror URL or a local archive), `paths.cublasVersion`.
- **Changed: the cuDNN download shares the new provisioner.** It now logs progress, and on Linux it keeps one copy of each
  library and recreates the soname links instead of copying the file once per name.
- **Fixed: a CUDA request on a machine without cuBLAS said only that the device was not found.** The message now names the
  folders searched and the setting that enables the download.

## alpha.406

- **Added: CoLa3D scene decomposition.** `hartsy 3d <scene.obj|glb> -m cola3d --model-path <Models>` and `POST /v1/native/mesh`
  (model `cola3d`, `sceneMesh = {data, format}`) split a triangle scene into one mesh node per object and return one GLB. The
  chain is dual-grid voxelization, the shape VAE encoder, Utonia point features, the CoLa3D decomposer, then per-object shape VAE
  decoding and dual-grid meshing. The grid defaults to 256 and is capped at 1024 (`--grid`, dual-grid voxels per axis, a
  multiple of 16).
- **Added: OBJ and GLB scene readers.** `ObjReader` splits fields on any whitespace and warns once per record type it does not
  read. `GlbReader` places each primitive by its node transform, so objects positioned by their nodes come out in world space.
  It refuses malformed, sparse and buffer-less accessors with a domain error. `GlbWriter` writes several meshes into one scene.
- **Fixed: a CoLa3D run that is cancelled or fails releases every object latent it has not decoded yet.**

## alpha.405

- Kimi K3.1 Pro (`kimi_k31`): CPU reference for the KDA forget gate, the KDA recurrence and output, the latent MoE
  order, the routing rule, and attention residuals, with the SiTU activation shared with the GLU reference.
- Expert-parallel dispatch host reference (`ExpertParallelDispatch`): contiguous expert ownership, route planning,
  send counts, and a combine that matches a single-device evaluation.
- MXFP4 checkpoint branch in `HfQuantCheckpoint` for the `mxfp4-pack-quantized` routed-expert tensors.
