# TRELLIS.2 architecture (shape-only image to mesh)

Research dossier for the shape path of `microsoft/TRELLIS.2`, as implemented by `Trellis2ImageTo3DPipeline`. Status and what has
been verified are in [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md). Texture and PBR are not part of this port.

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

## Differences that remain from upstream

- Noise comes from the repository's `SeedGenerator`, not `torch.manual_seed`/`randn`; the same seed gives a different latent than Python.
- Compute is F32 on CPU (and the CUDA backend); upstream runs the flows in BF16 and the VAE in FP16.
- `m.fill_holes()` (CuMesh, in `decode_latent`) is not applied: the raw dual-grid mesh may have small holes.
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

- No Python reference dumps exist, so no stage is parity-verified; see the status row.
- The shape VAE's parity in CoLa3D used a naive convolution shim and an assumed kernel-axis convention; the TRELLIS.2 shape path inherits that caveat.
- Texture and PBR (texture flow with 64 input channels, texture decoder with guide subdivisions, GLB baking) are a later PR.
