# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N`. The single source of truth is
`<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props`. Dates are UTC.

## alpha.412

- **Added: TRELLIS.2 shape-only image to mesh.** `hartsy 3d <image.png> -m trellis2` and `POST /v1/native/mesh` (model `trellis2`) turn one
  image into a GLB with positions and computed normals (no texture, no UVs). The chain is foreground preprocessing, DINOv3 conditioning,
  the 1.3B dense structure flow and decoder, the sparse shape flow (once at 512, or the 512 to 1024 cascade), the shape VAE decode and
  dual-grid meshing, all in C#. `--grid` is the final dual-grid resolution for this model: 512, 1024 (default, `1024_cascade`) or 1536
  (`1536_cascade`); other values are rejected before any weight loads. `--grid` keeps its meaning for the other models. An image with an alpha
  channel is used as the cutout; one without needs the BiRefNet-matting checkpoint (a stand-in for upstream's RMBG-2.0, which is not used).
- **Added: config-driven TRELLIS.2 flow classes** (`Trellis2SparseStructureFlow`, `Trellis2SlatFlow`: shared adaLN with a per-block
  modulation, QK RMS norm, 3D RoPE), the guidance-interval FlowEuler sampler with CFG rescale, `pipeline.json` parsing, cascade
  token-cap logic, and `ShapeVaeDecoder.Upsample` (the decoder's first levels, returning coordinates). The existing full decode is unchanged.
- Parity against the Python reference is not verified; the catalog entry is Structural and the status row is 🔧.

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
