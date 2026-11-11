# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

## Unreleased

- **Added: TRELLIS.2 texture path, validation-pending.** `Trellis2ImageTo3DPipeline.GenerateTextured` samples the texture latent with
  the texture flow (64 input channels: noise plus the normalized shape latent), decodes it with a texture decoder that follows the
  shape decoder's subdivisions (`ShapeVaeSubdivisions`, upstream's `guide_subs`), bakes the 6-channel PBR volume through
  `PbrMeshPostProcessor` and writes a double-sided Y-up PBR GLB (`Trellis2ImageTo3DPipeline.ToGlb`). `ShapeVaeDecoder` now also builds
  the subdivision-free texture decoder (`ShapeVaeDecoderConfig.Texture`); `pipeline.json`'s `tex_slat_*` entries and the three
  texture files (`Trellis2ModelFiles`) are optional, so shape-only installs are unchanged. The trilinear sampler's `floor` convention
  was checked against the FlexGEMM CUDA kernel. `hartsy 3d -m trellis2 --texture [--texture-size N]`, `MeshRequest.Texture` and
  `MeshRequest.TextureSize` select it; the existing request and the shape-only output are unchanged, and the catalog still downloads
  the shape files only. Tested on tiny synthetic networks only; no texture checkpoint has run.

- **Added: projection mode for the TRELLIS.2 flows (groundwork for Pixal3D, not yet runnable end to end).** `Trellis2FlowConfig` now
  reads `image_attn_mode` and `proj_in_channels` and accepts the `ElasticSLatFlowModel` sidecar name; a projection flow's blocks add a
  per-block `proj_linear` of per-token features to the cross-attention output and read the cross-attention tensors under
  `cross_attn.cross_attn_block`. `Trellis2SparseStructureFlow` and `Trellis2SlatFlow` gain `Forward` overloads that take the features;
  the existing overloads and TRELLIS.2 results are unchanged. Safetensors headers may now declare `C64`, which can be listed but not
  materialised, so the Pixal3D structure flow (which saves its RoPE table) opens; `Trellis2FlowWeights.Load` leaves that buffer unread.
  Header-checked against the real Pixal3D flows (700 F32 tensors each); nothing has been run with them yet.

## alpha.422

- **Fixed: Windows processes exited at CUDA start when cuDNN was in the engine cache.** NVIDIA's `cudnn64_9.dll` loads
  `cudnn_graph64_9.dll` and its other sub-libraries by bare name through the normal Windows search, which skips its own folder,
  and calls `exit()` when one is missing ("Invalid handle. Cannot load symbol cudnnGetCudartVersion", exit code 9). The engine
  loaded the dispatcher by full path from `~/.cache/hartsyinference/cudnn/cuda<N>/lib`, so every CUDA backend start died in
  `CudnnRuntime.LogStatus`. Machines with no system CUDA toolkit hit it from alpha.407, when the cached cuBLAS made CUDA usable.
  The resolver now puts the cuDNN folder first on the process `PATH` before loading it, and skips a folder that lacks any
  sub-library the dispatcher needs.
- **Added: Hunyuan3D-2.1 shape (validation-pending).** `Hunyuan3DShapePipeline.LoadFromPath` runs a repacked `tencent/Hunyuan3D-2.1` checkpoint: a new 21-block DiT with U-skips, a time token and a mixture of experts (`Hunyuan3DPlainDit`), DINOv2-large conditioning and the existing VAE and mesher; the config reader understands the 2.1 yaml. The DiT matches the Python reference at fp32 rounding level for one forward (block outputs and final velocity correlation 1.0, relative L2 at most 6e-5; VAE occupancy 0.99999) and a 10-step mesh renders on an RTX 3060 in 34 s. There is no catalog id yet because the upstream checkpoint is a pickle; see the status doc.
- **Added: Hunyuan3D-2 Fast, Turbo, mini and multiview builds.** `hartsy 3d -m hunyuan3d-fast|hunyuan3d-turbo|hunyuan3d-mini` (each its own catalog id and download set; `hunyuan3d-mv` downloads but has no CLI path yet because a request carries one image). The checkpoint's own `config.yaml` sets the architecture: Fast and Turbo embed the guidance scale and run one forward per step, Turbo samples sigmas from its consistency table (5 steps by default), mini has 8+16 blocks and 512 latents, and Turbo decodes with FlashVDM (turbo VAE, hierarchical grid with adaptive KV selection). Against the upstream Python the DiT velocity correlates 1.00000000 for every build (relative L2 4e-6 to 4e-5) and the FlashVDM field 0.99999979. `Hunyuan3DShapePipeline.GenerateMultiView` takes one to four views for 2mv checkpoints. mini-turbo (DINO at 1022) is refused. See the status doc.
- **Fixed: Hunyuan3D-2 meshes were melted blobs facing inward.** The sampler passed the DiT `sigma*1000` where it expects `sigma`, so the DiT saw a time 1000 times too large at every step (velocity correlation with upstream 0.53 to 0.73, now 1.00000000), and the marching-cubes triangles faced inward. On the same image and seed a 20-step, grid-128 mesh goes from 228k triangles in 101 pieces to 77k triangles in one watertight piece, matching the upstream Python mesh. The schedule now starts at sigma 0 as upstream does and skips the final zero-width step. Hunyuan3D-2 stays in the validation-pending table: one image checked, see the status doc.
- **Added: Hunyuan3D architecture dossier** with the Tencent Hunyuan 3D 2.0 Community License terms (territory excludes the EU, UK and South Korea; no training other models on outputs) and a real-weight reference dump script.
- **Added: UniRig skeleton generation (stage 1 of the auto-rigger), validation-pending.** `HartsyInference.ThreeD.Models.UniRig`
  ports the `VAST-AI/UniRig` articulation-xl skeleton stage: the Michelangelo-style point encoder (65,536 points with normals, 1,024
  farthest-point queries), the OPT decoder hosted on `GenericTransformer` with the checkpoint's keys remapped, the 256-bin
  skeleton-tree tokenizer and its grammar as a sampler step, and mesh preparation (welding, normalization, upstream's `mix` point
  sampler) in pure C#. The `.ckpt` is read with the safe-subset pickle loader. Not yet wired to `hartsy 3d` or `MeshService`; skinning
  and rigged-GLB export follow in later PRs. Deviations from upstream: one sampled sequence instead of 15-beam sampling, a seeded
  Fisher-Yates point subset instead of numpy's stream, no decimation, retry with the next seed when a draw does not decode.
  `HartsyInference.ThreeD` now references `HartsyInference.LLM`. On the real checkpoint, an Xbot humanoid yields a 52-joint body with
  fingers and a horse a 32-joint quadruped (CUDA, RTX 3060, about 2 s per generation after load); no Python reference was run.
- **Added: UniRig skinning (stage 2), validation-pending.** The skin network of `VAST-AI/UniRig` in pure C#: UniRig's point
  transformer (k=5 sparse-conv stem with BatchNorm, QK-norm patch attention, no pooling; `UtoniaEncoder` was checked and does not
  fit), the global point encoder shared with stage 1, bone encoder, mesh-to-bone cross-attention and the per-bone scoring head, the
  geometric voxel prior (grid 196, geodesic distances over mesh edges and interior voxels, no pyrender), and upstream's `reskin`
  (nearest-sample median, one diffusion step, small-weight cut-off). Weights are non-negative and each row sums to 1. Runs on the CPU
  backend in minutes (Horse, 494 vertices, 32 joints: 400 s); GPU timing is not measured.
- **Added: `hartsy 3d -m unirig`, mesh in, rigged GLB out.** `MeshService` routes the new `unirig` id (`MeshModelIds.RigMesh`) through
  the existing `MeshRequest.SceneMesh` input and the GLB `MeshResult`; no request or result field changed. The pipeline reads one stage's
  checkpoint at a time (1.4 GB skeleton, 4.4 GB skin, F32), writes bones named `bone_0…`, tails in node `extras`, inverse bind matrices
  and the top four influences per vertex (`SkinnedGlbWriter`, separate from the hair-card writer). Pass the folder holding
  `skeleton/articulation-xl_quantization_256/model.ckpt` and `skin/articulation-xl/model.ckpt` with `--model-path`; `--steps` and
  `--grid` are rejected. Validation-pending: no Python reference was run, so parity is not claimed.
- **Added: TRELLIS.2 shape-only image to mesh.** `hartsy 3d <image.png> -m trellis2` and `POST /v1/native/mesh` (model `trellis2`) turn one
  image into a GLB with positions and computed normals (no texture, no UVs). The chain is foreground preprocessing, DINOv3 conditioning,
  the 1.3B dense structure flow and decoder, the sparse shape flow (once at 512, or the 512 to 1024 cascade), the shape VAE decode and
  dual-grid meshing, all in C#. `--grid` is the final dual-grid resolution for this model: 512, 1024 (default, `1024_cascade`) or 1536
  (`1536_cascade`); other values are rejected before any weight loads. `--grid` keeps its meaning for the other models. An image with an alpha
  channel is used as the cutout; one without needs the BiRefNet-matting checkpoint (a stand-in for upstream's RMBG-2.0, which is not used).
- **Fixed: TRELLIS.2 and HairDeck now find their Vision-folder checkpoints (DINOv3, BiRefNet-matting) through the configured Vision folder, and a matting checkpoint installed after the first TRELLIS.2 load is used without a restart.**
- **Added: config-driven TRELLIS.2 flow classes** (`Trellis2SparseStructureFlow`, `Trellis2SlatFlow`: shared adaLN with a per-block
  modulation, QK RMS norm, 3D RoPE), the guidance-interval FlowEuler sampler with CFG rescale, `pipeline.json` parsing, cascade
  token-cap logic, and `ShapeVaeDecoder.Upsample` (the decoder's first levels, returning coordinates). The existing full decode is unchanged.
- **Fixed: TRELLIS.2 meshes shaded with dark vertical streaks and speckle.** A dual-grid quad is wound the same way whichever side of the surface
  is inside, so about half of a raw mesh's triangles faced inward and the vertex normals averaged across them were wrong. The pipeline now
  unifies the winding (`FaceOrientation.Unify`, as upstream's `unify_face_orientations` does) before computing normals. CoLa3D shares the
  dual-grid mesher and is unchanged.
- Parity: the DINOv3 conditioning, the 512 shape flow forward and the guided sampler match the upstream Python on the CPU (correlation 0.99993 or better); the structure stage is not checked. The catalog entry is Structural and the status row is 🔧.
- **Added: a model-independent mesh post-processing and PBR baking toolkit in `HartsyInference.ThreeD`.** `MeshDecimator`
  (quadric edge collapse with the upstream 3x-then-target schedule), `MeshCleaner`, `TriangleBvh`, `UvUnwrapper` (chart
  segmentation, planar projection, atlas packing), `Rasterizer`, `SparseAttributeVolume` with `SparseAttributeSampler`,
  `TextureBaker`, `Inpainter` (Telea) and `PbrGlbWriter` (embedded PNG textures, glTF metallic-roughness packing, optional Z-up
  to Y-up). `PbrMeshPostProcessor` chains them for a mesh plus a sparse attribute volume. Pure C#, no model dependency, and
  `GlbWriter` and `HairCardGlbWriter` are unchanged. Differences from upstream `to_glb` are listed in
  [MESH_POSTPROCESS.md](docs/Research/MESH_POSTPROCESS.md).

## alpha.421

- **Added: Vulkan `LeakyRelu` and `AdaInstanceNorm1d`.** Both threw "not yet implemented" on the Vulkan backend, which stopped
  Kokoro, StyleTTS 2 and CosyVoice 2 from running there. The same stubs also blocked the CAM++ speaker encoder, the AuK VAE
  encoder, Silero VAD and other callers on Vulkan. `LeakyRelu` is an elementwise op; `AdaInstanceNorm1d` is a new two-pass kernel
  that matches the CPU and CUDA semantics.

## alpha.420

- **Added: host APIs for MOSS-VL-Realtime**, so an app that lists and drives local models (the SwarmUI LLMAssistant extension)
  needs no model-family knowledge. `TextService.InputCapabilities(path)` answers whether a GGUF or checkpoint folder takes images,
  video and live sessions from its header or `config.json` alone (MOSS-VL has no `mmproj` sidecar, so the old sidecar test called it
  text-only); it loads nothing and never throws. `TextService.LoadedPlacements` now lists a loaded MOSS-VL model with a
  single-device placement, so "loaded, and where" works for it.
- **Added: the NF4 and FP8 releases, and the BF16 folder, are catalog variants you can download.** `ModelDownloader.DownloadVariantAsync`,
  `hartsy pull moss-vl-realtime --variant nf4|fp8|bf16` and `POST /admin/models/pull` with `variant` fetch a folder at its pinned
  revision into `LLM/moss-vl-realtime-<variant>/`, verify every file's SHA-256, and check that every shard the index names is on
  disk. The BF16 variant moved from the GGUF's folder to its own, because a folder holding both a GGUF and a shard index is ambiguous.
- **Added: `paths.ffmpegDir`.** Video decode and encode now look for `ffmpeg` and `ffprobe` in that folder before `PATH`, and the
  error for a missing ffmpeg names the setting.

## alpha.419

- **Added: a live window for world models, `hartsy world --view`.** It plays frames at the plan's frame rate as the session generates
  them, and steers the run from held keys: hold a key to move, release it to stop. Esc or Q, or closing the window, ends the run the
  way `q` does, so the PNGs and the MP4 are still written. The window needs a display and cannot be combined with `--interactive`. It is
  built on Avalonia 11.3.22, the version already in the local NuGet cache; the rest of the engine has no window dependency.
- **Added: control schemes for world models.** `IWorldControlScheme` maps held keys to the action a session takes, and a model declares
  one with `WorldCapabilities.LiveControls`. Built in: the SANA-WM camera (`wasd`, `ijkl`), Oasis (WASD, space, shift, ctrl, arrows
  for the camera) and DIAMOND (fire and left or right). A model with no live control (SANA-WM stage 1) plays its clip when the run
  completes. A new world model adds a scheme and nothing else; `docs/Research/WORLD_LIVE_VIEWER.md` has the steps.
- **Changed: streamed frames are written by background PNG writers.** PNG encoding took about 110 ms per 1280x704 frame on the consumer
  thread, so a 24-frame chunk spent 2.7 s writing and the model waited on it. Four writers now encode in parallel behind a queue of
  32 frames, and the stream waits only when that queue is full. The frames and the MP4 are byte-identical to before.
- **Fixed: a live window closes on every exit, and a failed run says so.** A failed stream is reported to the window, which stays up
  with the error until it is closed. Ctrl+C on an unbounded run keeps its files and reports the run, not "Cancelled.". Oasis and
  DIAMOND take one action per step, so their window buffers two frames rather than two seconds, and they stop after `--frames`.
- **Added: a where-the-time-went breakdown** printed at the end of every `--view` run, and under `diagnostics.stageProfile`.
>
## alpha.418

- **Added: ERNIE-5.5-Flash-Base multi-token-prediction layer on the host reference.** With `Ernie5LoadOptions.Mtp`,
  `Ernie5Model` loads layer 43 and drafts the token after next: `ForwardStream` returns the target's pre-head stream,
  `MtpForward` runs the MTP layer on it (chained steps feed its output back), and `MtpHidden` plus `Logits` read the draft through
  the shared LM head. Verified on a synthetic checkpoint only; the Text path does not draft yet. A checkpoint is no longer refused
  for having MTP layers.

## alpha.417

- **Fixed: a SANA-WM stream over HTTP ran its whole job outside the long-running queue.** Opening a batch or chunked world session
  reserves nothing, so its stream loads the model and runs it; it now takes a long-running slot like any other generation (or gets
  a 429 when the queue is full), where before it could share the GPU with an image or video job and run out of memory. An
  interactive session is unchanged. `IWorldSession` gains `DefersDeviceWork` (default false) for this.
- **Fixed: a world stream that failed left nothing in the server log, and a client that disconnected made the error handler throw a
  second time.** Failures are logged and sent to the client as an error event; a disconnect ends the stream quietly. An action sent
  to a session closed a moment earlier now returns 404, not 500, and `hartsy world --actions ","` is refused with a message.
- **Fixed: a SANA-WM DiT whose weight upload ran out of memory partway kept the weights it had already put on the GPU.** On the
  12 GB card a failed load now frees them.
- **Fixed: `Conv2dDepthwise` accepted an output with another batch than its input and wrote past the end of it.** The shared shape
  contract now refuses it.
- **Fixed: `DepthwiseConvGlu` failed to launch on an input with more than 65,535 frame rows.** The vectorized kernel puts them on the
  grid's y axis, and the general kernel now takes over past the cap. The frame scan refuses more than 65,535 heads with a clear
  message.
- **Changed: the frame scan's GEMMs ignore `numerics.gemmF16`.** Its Gram errors compound over the recurrence, so it keeps TF32
  (or plain F32) whatever that setting says. Default settings are unchanged.
- **Changed: the model stage timer has its own setting, `diagnostics.stageProfile`.** `diagnostics.profile` no longer adds the
  device syncs around every stage, and the SANA-WM softmax blocks report one row each, as the GDN blocks do.
- **Fixed: `HartsyInference.Engine.Tests` did not compile** against the Cuda project, which now grants it access to
  `CudaRedistProvisioner`'s internal override.

## alpha.416

- **Added: SANA-WM stage 1 (`sana-wm`).** `hartsy world -m sana-wm --image <png> --prompt-file <txt> --intrinsics fx,fy,cx,cy
  (--actions "w-80,dw-40,aw-40" | --camera poses.npy)` renders up to 161 frames at 704x1280 and 16 fps from one image, a prompt
  and a camera path (per-frame poses, or an action script over `wasdijkl`). It matches NVIDIA's stage-1 reference on an RTX 4090
  and on a 12 GB RTX 3060, where the upstream code does not fit: a 161-frame clip takes about 3 minutes on the 4090 and 14.5
  minutes on the 3060. The refiner, FP8/FP4 and Pi3X intrinsics are refused when the request is planned. The weights are
  non-commercial (the LTX-2 VAE is under the LTX-2 community licence).
- **Added: SANA-WM streaming (`sana-wm-streaming`).** The same model distilled to be chunk-causal. It generates 24 video frames
  at a time and decodes each chunk as soon as it is done, and the camera can change while it runs: `--actions "w-24,d-24,s-24"`
  drives it, `--interactive` steers it from the keyboard (`wasdijkl`, space to release, `q` to stop), and `--unbounded` runs
  until stopped. Each frame is written to disk as it arrives, and `q` or Ctrl+C saves what was generated and muxes the MP4. A
  chunk takes 1.75 s on the 4090 and 7.7 s on the 12 GB 3060; every frame matches upstream's streaming sampler at PSNR 44.9 dB
  or better on both. The refiner (`--refiner`) is written and off by default; it has not run on a GPU.
- **Added: new backend ops for SANA-WM.** `FrameDeltaScan` (with an optional carried state), `DepthwiseConvGlu`,
  `UcpeTransform`, `DepthwiseConvFrames` and `RmsNormRelu`, each with a host reference and a CUDA kernel, plus `Exp` and
  `Reciprocal`. A SANA-WM step takes 2.59 s on the 4090 against 3.05 s for upstream PyTorch. Existing ops are unchanged.
- **Added: streaming building blocks.** `ChunkPlanner`, `RollingTokenWindow`, `FrameTailCache`, `CausalStreamCache` and
  `SelfForcingFlowSchedule` (Diffusion), `LiveCameraPath` and `ICameraIntegrator` (World), `ChunkedWorldSession` and
  `StreamingFrameWriter` (Engine). Another chunk-causal world model can reuse them. `vram.sanaWmStreamState` (default `auto`)
  holds the carried state in BF16 on cards of 16 GiB or less, as upstream does, and in F32 on larger ones.
- **Added: shared pieces for SANA-WM.** `VideoTemporalTiling` for decoding a long video latent in overlapping tiles, `CameraRayFrame`,
  `CoverCropMapping` and a `PluckerLayout` option on `PluckerEmbedding` (the existing overload is byte-identical), and a
  `World.Tests` project.
- **Fixed: CUDA `BatchedMatMul` wrote F32 into a BF16 output buffer.** The cuBLAS output type now follows the output tensor's
  dtype. `Conv2dDepthwise` also takes a BF16 output on CUDA.
- **Changed: Gemma-2 text encoders apply the attention-logit softcap (50).** Lumina-2 uses Gemma-2-2B, so its images change; this
  was a deliberate fix. Gemma-2 also encodes right-padded batches and can return the post-final-norm hidden state.
- **Changed: `CausalConv3d` holds one weight form** instead of both: the per-tap slices are cut on first use, and a convolution
  the native 3-D op keeps declining releases its 5-D form. SANA-WM's decoder drops from 1762 to 999 MB of weights after its first
  decode with byte-identical frames, and an LTX-2.5 clip is byte-identical too. The first decode in a process pays the slicing,
  about 1.7 s.
- **Changed: world planning.** `hartsy world --frames` below 2 is refused, random seeds are drawn and recorded, and an unknown
  world model id throws instead of loading Oasis.

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
