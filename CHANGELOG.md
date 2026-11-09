# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

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
