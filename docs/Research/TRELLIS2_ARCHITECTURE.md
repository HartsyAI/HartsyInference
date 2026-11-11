# TRELLIS.2 architecture (image to mesh, with optional PBR texture)

Research dossier for `microsoft/TRELLIS.2`, as implemented by `Trellis2ImageTo3DPipeline`: the shape path (`Generate`) and the texture
path (`GenerateTextured`, [below](#texture-path)). Status and what has been verified are in
[MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md).

## Pins

| Item | Revision |
|---|---|
| Upstream code `microsoft/TRELLIS.2` | commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd` (2026-06-05); every file below was read at this commit |
| Weights `microsoft/TRELLIS.2-4B` | `af44b45f2e35a493886929c6d786e563ec68364d` (MIT, not gated) |
| Structure decoder `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` (MIT, not gated) |
| DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` (DINOv3 License, gated) |
| RMBG-2.0 `briaai/RMBG-2.0` | `2fd7b4b0f2393ddb39a267fb62df6c010928a6c6` (CC BY-NC 4.0, gated; not used here) |

Source files read: `trellis2/models/{sparse_structure_flow,structured_latent_flow}.py`, `models/sc_vaes/{fdg_vae,sparse_unet_vae}.py`,
`modules/transformer/{blocks,modulated}.py`, `modules/attention/{modules,rope,full_attn}.py`, `modules/sparse/attention/{modules,rope}.py`,
`modules/sparse/transformer/modulated.py`, `modules/sparse/basic.py`, `modules/image_feature_extractor.py`,
`pipelines/trellis2_image_to_3d.py`, `pipelines/samplers/*`, `pipelines/rembg/BiRefNet.py`, `o-voxel/o_voxel/postprocess.py`.
Texture sampling and baking also read `JeffreyXiang/FlexGEMM` commit `6dd94a859c26ee8246888502eada3dd8ad85532e`
(`flex_gemm/kernels/cuda/grid_sample/grid_sample.cu`, `flex_gemm/ops/grid_sample/*`).

## Stage map (pipeline types `512`, `1024_cascade`, `1536_cascade`)

| Stage | Upstream | This port |
|---|---|---|
| Preprocess | `preprocess_image`: long edge to at most 1024 (LANCZOS), RMBG-2.0 matte unless the image has non-opaque alpha, crop the square around alpha above 0.8, RGB times alpha | `Trellis2ImagePreprocessor`; the matte comes from `IForegroundMatter` (BiRefNet-matting stand-in) |
| Conditioning | DINOv3 ViT-L/16 at 512 (and 1024), all tokens after the last block, `F.layer_norm`, no DINOv3 final norm; negative cond is zeros | `Trellis2ImageConditioner` over `Dinov3VisionEncoder.EncodeBlocks`; 1029 / 4101 tokens |
| Structure flow | `SparseStructureFlowModel`, dense 16^3 x 8, width 1536, 30 blocks, 12 heads, RoPE, shared adaLN | `Trellis2SparseStructureFlow` |
| Structure decode | `ss_dec_conv3d_16l8` to a 64^3 logit grid, `> 0`, max-pool to 32^3 for the 512 type and the cascades, `argwhere` | v1 `SparseStructureDecoder` (same checkpoint, same key map), `Trellis2Cascade.OccupiedVoxels` |
| Shape flow | `SLatFlowModel` 512 (grid 32) and 1024 (grid 64), one token per voxel, RoPE on integer coordinates | `Trellis2SlatFlow` |
| Cascade | LR flow, denormalize, `decoder.upsample(4)`, `unique((c + 0.5) / 512 * (res / 16))`, token cap 49152 retreating by 128 to a floor of 1024, HR flow | `ShapeVaeDecoder.Upsample`, `Trellis2Cascade.FitTokenCap` |
| Shape decode | `FlexiDualGridVaeDecoder` with `set_resolution(res)`, vertices `(1 + 2m) sigmoid - m`, intersected `> 0`, quad lerp `softplus` | `ShapeVaeDecoder` + `ShapeVaeHead.Split` (shared with CoLa3D) |
| Mesh | `flexible_dual_grid_to_mesh`, aabb `[-0.5, 0.5]^3` | `DualGridMesherStage` (origin -0.5, voxel size 1/res), then `Trellis2ImageTo3DPipeline.ToYUp` |
| GLB | `to_glb` swaps `y, z = z, -y` | the same rotation is applied to positions and normals |
| Texture flow | `sample_tex_slat`: shape latent re-normalized, 32-channel noise, `concat_cond` | `Trellis2TextureFlow`, `Trellis2TextureFlowStage` |
| Texture decode | `tex_slat_decoder(slat, guide_subs=subs) * 0.5 + 0.5` | `ShapeVaeDecoder` with `ShapeVaeDecoderConfig.Texture`, `Trellis2TextureDecoderStage` |
| Bake | `o_voxel.postprocess.to_glb` | `PbrMeshPostProcessor` then `PbrGlbWriter` ([MESH_POSTPROCESS.md](MESH_POSTPROCESS.md)) |

`pipeline.json` is parsed at load: samplers (steps 12; structure guidance 7.5, rescale 0.7, interval [0.6, 1.0], `rescale_t` 5.0;
shape 7.5, 0.5, [0.6, 1.0], 3.0; `sigma_min` 1e-5) and the 32-channel shape normalization.

## Flow block, from source

`ModulatedTransformerCrossBlock` (dense) and `ModulatedSparseTransformerCrossBlock` (sparse) are the same computation:

- Shared modulation: the model computes `adaLN_modulation(t_emb) = Linear(SiLU(.))`, 6C wide. Each block adds its own `modulation`
  parameter `[6C]` and chunks the sum into shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
- `norm1` and `norm3` have no affine parameters (eps 1e-6); `norm2` (before the cross-attention) is affine, eps 1e-6.
- Self-attention input `norm1(x) * (1 + scale_msa) + shift_msa`; output gated by `gate_msa`. Cross-attention has no gate and no
  modulation. The MLP input is `norm3(x) * (1 + scale_mlp) + shift_mlp`, output gated by `gate_mlp`.
- QK RMS norm is `F.normalize(x) * gamma * sqrt(head_dim)` with `gamma [heads, head_dim]`, on self- and cross-attention (cross-attention
  has no RoPE). The MLP is `Linear, GELU(tanh), Linear`, hidden `int(1536 * 5.3334) = 8192`.
- Timestep: sinusoidal embedding of `1000 t`, 256 wide, cosines then sines, max period 10000, `Linear, SiLU, Linear`.
- Final: `F.layer_norm` (eps 1e-5, no affine) then `out_layer`.
- RoPE: `freq_dim = head_dim // 2 // 3 = 21`; `phases = outer(coord, 1 / 10000^(i / 21))` for x, y, z concatenated (63 phases) and padded to 64
  with a zero phase; complex multiplication over adjacent channel pairs `(2j, 2j + 1)`. The 2 trailing channels pass through.
  The dense flow's tokens are the `[C, 16, 16, 16]` grid flattened x-major; the sparse flow reads each voxel's integer coordinates, not
  scaled by the grid, so the 1024 flow serves a 96 grid unchanged.
- The checkpoint holds 640 tensors per flow (10 model-level, 21 per block), all BF16; `Trellis2FlowKeyMap` lists them and the loader
  rejects missing, extra and mis-shaped tensors.

## Sampler, from source

`FlowEulerGuidanceIntervalSampler`: `t = linspace(1, 0, steps + 1)`, `t = rescale_t t / (1 + (rescale_t - 1) t)`, the model gets `1000 t`,
`x -= (t - t_prev) v`. Inside `[low, high]` the prediction is `s pos + (1 - s) neg` (strength exactly 1 skips the second call, exactly 0
uses the negative only). With `guidance_rescale r > 0`: `x0 = (1 - sigma_min) x - (sigma_min + (1 - sigma_min) t) pred` for the conditional and
the guided prediction, the guided `x0` is scaled by `std(x0_pos) / std(x0_cfg)` over all elements, blended with weight `r`, and converted back.
Upstream's sparse path computes a population std and its dense path an unbiased one; the ratio is the same, and the port uses one.

## Texture path

Read from source at the pins above. Each of the 512 and 1024 flows has a texture twin; the shape flows are unchanged.

| Item | Value |
|---|---|
| Texture flows | `slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16`: `SLatFlowModel`, grid 32 / 64, 64 input channels, 32 output, otherwise the shape flow's architecture (width 1536, 30 blocks, 12 heads, `mlp_ratio` 5.3334) |
| Input | `sparse_cat([x_t, concat_cond], dim=-1)` at every call: 32 noise channels then the 32 shape-latent channels |
| `concat_cond` | the sampled shape latent re-normalized: `(shape_slat - shape_mean) / shape_std`; the noise is `randn` over the same coordinates |
| Conditioning | the 512 type uses the 512-token DINOv3 conditioning with the 512 texture flow; both cascades use the 1024-token conditioning with the 1024 texture flow, over the fine shape latent's voxels. There is no 1536 texture flow |
| Sampler | `FlowEulerGuidanceIntervalSampler`, 12 steps, strength 1.0 (one conditional pass; the interval is never used), rescale 0.0, interval [0.6, 0.9], `rescale_t` 3.0, `sigma_min` 1e-5 |
| Output | `slat * tex_std + tex_mean` with `tex_slat_normalization` from `pipeline.json` (32 values each) |
| Decoder | `tex_dec_next_dc_f16c32_fp16`: `SparseUnetVaeDecoder`, the shape decoder's body (channels 1024-512-256-128-64, blocks 4-16-8-4-0), `out_channels` 6, `pred_subdiv` false, so no `to_subdiv` weights |
| Guidance | each `SparseResBlockC2S3d` takes `subdiv = guide_subs[i]` from the shape decode's `return_subs` and keeps the children whose logit is `> 0`; `ShapeVaeSubdivisions` records those logits per level, and a guide from other coordinates is rejected |
| Head | `F.layer_norm` (eps 1e-5), `output_layer`, then `* 0.5 + 0.5`; channels base color 0..2, metallic 3, roughness 4, alpha 5 |
| Volume | `MeshWithVoxel(origin = -0.5, voxel_size = 1 / resolution, coords = voxel coords, attrs = decoded features)` |

`to_glb` then runs `fill_holes`, two decimation passes, `unify_face_orientations`, UV unwrap, a rasterized bake with
`flex_gemm.grid_sample_3d(mode='trilinear')` at snapped surface points, `cv2.inpaint`, and a double-sided material (non-remesh path).

- **Sampler convention (checked).** The CUDA kernel behind `grid_sample_3d(trilinear)` takes `base = floor(q - 0.5)` per axis, weighs
  neighbor `x` by `1 - |q - x - 0.5|`, drops unoccupied neighbors and divides by `max(w_sum, 1e-12)`. `SparseAttributeSampler` matches that
  (it uses `floor`). The torch fallback `grid_sample_torch.py` truncates with `.int()`, which differs only for samples within half a voxel
  of the grid's minimum faces; the pipeline uses the CUDA behavior.
- **Frame.** Texture voxels, the mesh and the bake all live in TRELLIS.2's Z-up cube `[-0.5, 0.5]^3`. `GenerateTextured` returns that frame and
  `Trellis2ImageTo3DPipeline.ToGlb` sets `PbrGlbOptions.ConvertZUpToYUp = true` and `DoubleSided = true`; shape-only `Generate` rotates its mesh itself.
- **Request.** `Trellis2Options.Texture` selects the path for `GenerateTextured` (`Generate` ignores it); `TextureSize` (default 2048) and `DecimationTarget`
  (default 1,000,000 faces) are upstream's `to_glb` defaults. The texture steps use the same step override as the other samplers.
- **Contract.** The existing request is unchanged; texture is an opt-in on top of it. `hartsy 3d <image.png> -m trellis2 --texture [--texture-size N]`,
  and `MeshRequest.Texture` / `MeshRequest.TextureSize` for the engine and the native HTTP mesh request, return a textured GLB instead of the
  shape-only one (`Format` is `glb` either way). `TextureSize` without `Texture`, or either on another model, is rejected, and a texture size
  outside 16 to 8192 fails before the checkpoints load. The catalog entry still downloads the shape files only: the 6.1 GB of texture files are
  not part of `hartsy download trellis2` until they are verified on real weights.
- **Files** (`microsoft/TRELLIS.2-4B` at the pin; sizes in bytes, sha256 from the Hugging Face tree API):

  | File | Size | sha256 |
  |---|---|---|
  | `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors` | 2,584,672,728 | `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d` |
  | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` | 2,584,672,728 | `580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686` |
  | `ckpts/tex_dec_next_dc_f16c32_fp16.safetensors` | 948,458,812 | `97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c` |

  Each flow has a small `.json` sidecar next to it (the 1024 flow is required for the cascades, the 512 flow only for `--grid 512`).
  `Trellis2ModelFiles` finds them beside the shape flows; without them `GenerateTextured` fails before any stage runs.

### Real-weight run (RTX 3060 12 GB, CUDA, seed 1, 2026-11-11)

Header check: the 512 and 1024 texture flows (640 BF16 tensors each, in 64, out 32) and the texture decoder (no `to_subdiv` keys) match
`Trellis2FlowKeyMap` and `ShapeVaeKeyMap.Decoder(ShapeVaeDecoderConfig.Texture)` exactly (`Trellis2RealWeightsTests`). Files were fetched from
`microsoft/TRELLIS.2-4B` at the pin, sizes and sha256 as in the table above, sidecars by size (458, 458, 705 bytes).

A real example image (a marble bust, 1052 px RGBA) and a flat synthetic mug, full 12-step samplers, `--texture`, default 2048 texels and
1,000,000 faces. Stage times come from the progress stamps of one process per run; peak VRAM is the card's `memory.used`, which includes about
1.3 to 1.8 GB held by other processes.

| Stage | `--grid 512` bust | `--grid 1024` bust (cascade) |
|---|---|---|
| conditioning and structure flow | 190 s (cold file cache; 60 s warm) | 71 s |
| coarse shape flow (12 steps) | 28 s | 29 s |
| upsample, fine conditioning and fine shape flow (12 steps, 10,955 tokens) | | 227 s |
| shape decode and meshing | | 23 s |
| texture flow (12 steps, one pass each) | 23 s | 133 s |
| texture decode (guided) | 7 s | 27 s |
| decimate, unwrap, bake, inpaint (CPU) | 21 to 29 s | 46 to 54 s |
| total | 269 s (140 s warm) | 564 s |
| peak VRAM (whole card) | 7.7 to 11.8 GB (see note) | 9.3 GB |

The 512 peak of 11.8 GB belongs to the run that also had other processes resident; the whole-card figure includes about 1.3 to 1.8 GB that is
not ours, and the two 512 runs (7.7 GB and 11.8 GB) differ by what else held the card.

Findings:

- **Device memory left behind by one stage made the next flow 6 to 35 times slower, with no error and no log line.** Before the fix a
  `--grid 1024` run spent 5 fine shape steps at 650 to 775 s each and every texture step at 360 to 460 s (a 1024 run was stopped at step 50 of
  53 after 2 hours); the same steps take 15 to 20 s when the pool is trimmed. The per-op profile with a stream sync after every op showed
  the same code at full speed, the SDPA kernels were fast in isolation, and the process was CPU-bound in the driver (half of its CPU time in
  the kernel). What fixed it: returning the stream-ordered pool, cached activations and attention plans to the driver before each flow
  (`Trellis2DeviceMemory.ReleaseBetweenStages`, shape flows), and dropping every cached device weight before the texture flow
  (`ReleaseAll`; the shape decoder never preloads, so the backend promotes its weights on its own and keeps them, 1.5 to 1.8 GB). The cause
  inside the CUDA backend (which route the slow path takes when the pool is large) was not isolated; the shape-only pipeline had the same
  slow first high-resolution steps (see the status row's "first high-resolution step took 12 minutes").
- **The CPU bake is not the bottleneck.** Decimating a 11M-triangle 1024 mesh to 1M faces, unwrapping, baking and inpainting took 54 to 68 s;
  the 512 mesh took 19 to 22 s.
- **Sweeping the device is not a read-back, and it zeroed the conditioning.** The first version of the release above came out copper and
  teal (mean metallic 0.2 at 1024 and 0.36 at 512, mesh 937,485 vertices instead of 938,050 at 512) for the marble bust. `FreeActivations` and
  `FreeAllDeviceMemory` drop the device copy of every activation without reading it back, and a tensor an op produced that nobody read on
  the host, the DINOv3 conditioning among them, has only a zero-filled host buffer (its host sum reads 0.0, 461,844 after
  `OffloadActivation`). The next flow then ran on zero conditioning. Every sweep now offloads the tensors a later stage still reads
  (`Trellis2DeviceMemory`: conditioning, the shape latent, the subdivision guides) first; the 512 bust is then byte-identical (vertices and
  base color) to the run made before any sweep existed (cream and tan, roughness 0.91, metallic 0.0). The sweep stays in the TRELLIS.2 stages,
  not in the `SampleFlow` that Pixal3D shares, because Pixal3D's own device-only tensors would need the same offload first. The texture
  stage's `ReleaseAll` also evicts the DINOv3 and decoder weights that a warm second request would otherwise find cached; the timings above
  include that re-upload and it pays for itself, so it should not be optimised away without a measurement. The sweeps assume one generation at
  a time on the engine's backend, as the other pipelines' `FreeActivations` calls do. A flat
  synthetic mug gave a saturated red (R 0.94, G 0.03, B 0.03; roughness 0.87) at 512 from a copper-brown image, before any sweep existed;
  it is unexplained. No Python reference exists for any texture tensor.
- **The atlas is the weak part of the textured GLB.** On both meshes `UvUnwrapper` ended with roughly one chart per few triangles (vertices
  per triangle 0.94 to 1.5, against about 0.5 for a connected mesh), so only 5 to 11% of the 2048x2048 texels are written by the bake and
  the rest is inpainted. The render shows dark specks and thin seams. This is the limitation `MESH_POSTPROCESS.md` already lists
  ("a rough or noisy surface can end in many small charts"), a property of the shared post-processor and not of this path.

## Differences that remain from upstream

- Noise comes from the repository's `SeedGenerator`, not `torch.manual_seed`/`randn`; the same seed gives a different latent than Python.
- Compute is F32 on CPU (and the CUDA backend); upstream runs the flows in BF16 and the VAE in FP16.
- `m.fill_holes()` (CuMesh, in `decode_latent`) is not applied to the shape-only mesh: the raw dual-grid mesh may have small holes. The texture path fills holes inside `PbrMeshPostProcessor`.
- The winding is unified by `FaceOrientation.Unify` (a breadth-first flip over shared edges, then each patch turned outward by its signed volume) where upstream calls CuMesh's `unify_face_orientations` in `to_glb`.
- RMBG-2.0 is replaced by the BiRefNet-matting checkpoint the repository already supports, and only when the image has no alpha.
- The direct `1024` pipeline type (64^3 structure grid) is not exposed.

## Memory and time observations

Measured on an RTX 3060 (12 GB): the 512 type took about 95 s of step time, the 1024 cascade about 23 minutes (its first high-resolution step 12 minutes, the rest 25 to 52 s), with GPU memory at 11.7 GB (2.8 GB of it other processes) and 13.7 GB resident. The structure and shape flows are 1.3B parameters each; at F32 a flow is 5.2 GB, so each is read from disk
for its stage and released. Attention runs through `IBackend.ScaledDotProductAttention` on head-major tensors, never a materialized N x N matrix.

## Licences (from the sources read; not legal advice)

- TRELLIS.2 code and weights, TRELLIS-image-large: MIT.
- DINOv3: Meta DINOv3 License (custom; gated, manual approval). Non-exclusive, royalty-free; trade-control and end-use limits; no
  non-commercial clause was found in the text read.
- RMBG-2.0: CC BY-NC 4.0 (gated). Not used; upstream uses it for background removal.
- BiRefNet-matting (the stand-in matter): MIT per Hugging Face metadata.
- nvdiffrast and nvdiffrec (upstream texture baking): NVIDIA Source Code License, research and non-commercial. Not used in this port.
- CuMesh, flex_gemm, o-voxel: MIT.

## Open items

- No Python reference dumps exist, so no stage is parity-verified; see the status row. The one exception is the DINOv3 conditioning, checked against HF `DINOv3ViTModel` on the CPU on the marble bust: correlation 0.999995 at 512 (1029 tokens) and 0.999996 at 1024 (4101 tokens), max abs 0.10 and 0.17.
- The shape VAE's parity in CoLa3D used a naive convolution shim and an assumed kernel-axis convention; the TRELLIS.2 shape path inherits that caveat.
- The texture path has run on real weights (headers match, 512 and 1024 chains complete, GLBs render), but nothing is compared with Python (see the run above).
- The mesh post-processing is CPU and approximates cumesh, nvdiffrast and OpenCV (see [MESH_POSTPROCESS.md](MESH_POSTPROCESS.md)); its UV atlas fragments on these meshes.
- The texture latent and the texture decode use this repository's noise generator and F32 compute, like the shape stages.
