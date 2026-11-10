# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

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
