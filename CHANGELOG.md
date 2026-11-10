# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

## alpha.411

- **Added: a model-independent mesh post-processing and PBR baking toolkit in `HartsyInference.ThreeD`.** `MeshDecimator`
  (quadric edge collapse with the upstream 3x-then-target schedule), `MeshCleaner`, `TriangleBvh`, `UvUnwrapper` (chart
  segmentation, planar projection, atlas packing), `Rasterizer`, `SparseAttributeVolume` with `SparseAttributeSampler`,
  `TextureBaker`, `Inpainter` (Telea) and `PbrGlbWriter` (embedded PNG textures, glTF metallic-roughness packing, optional Z-up
  to Y-up). `PbrMeshPostProcessor` chains them for a mesh plus a sparse attribute volume. Pure C#, no model dependency, and
  `GlbWriter` and `HairCardGlbWriter` are unchanged. Differences from upstream `to_glb` are listed in
  [MESH_POSTPROCESS.md](docs/Research/MESH_POSTPROCESS.md).

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
