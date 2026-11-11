# SAM 3D Objects architecture (image + mask to Gaussian splat or mesh, with layout)

Research dossier for `facebookresearch/sam-3d-objects` (Meta), written from the public GitHub source and the public Hugging Face
file listing only. **Nothing gated was read and nothing was downloaded**: the pipeline and per-model YAML configs and every
checkpoint sit in the gated `facebook/sam-3d-objects` repository, so every value that lives there is marked *unverified*. Nothing
here is built yet; the status row is in [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md).

## Pins (source snapshot, 2026-11-11)

| Item | Revision |
|---|---|
| Upstream code `facebookresearch/sam-3d-objects` | commit `f91db411c50efee93d8db7aeb323885650f6f722` (2026-06-02, "SA-3DAO Release - README Update (#190)"); every source file below was read at this commit |
| Weights and configs `facebook/sam-3d-objects` | revision `2e73555018d2741ccd486e56c24fac41155a1dc6` (last modified 2026-06-12), `gated: manual`, tag `license:other`. Only the unauthenticated file listing (names, sizes, LFS sha256) was read; no file was fetched |
| MoGe (pointmap model) | code pinned by `requirements.txt` at `microsoft/MoGe@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`; which MoGe checkpoint the pipeline loads is in the gated `pipeline.yaml`: unverified |

Source files read: `README.md`, `doc/setup.md`, `LICENSE`, `demo.py`, `requirements*.txt`, `pyproject.toml`, `notebook/inference.py`,
`sam3d_objects/pipeline/{inference_pipeline,inference_pipeline_pointmap,inference_utils,preprocess_utils,layout_post_optimization_utils}.py`,
`pipeline/depth_models/*`, `pipeline/utils/pointmap.py`, `model/io.py`, `model/layers/llama3/ff.py`,
`model/backbone/generator/{classifier_free_guidance,flow_matching/model,flow_matching/solver,shortcut/model}.py`,
`model/backbone/dit/embedder/{dino,embedder_fuser,pointmap,point_remapper}.py`,
`model/backbone/tdfy_dit/models/{mot_sparse_structure_flow,mm_latent,sparse_structure_flow,sparse_structure_vae,structured_latent_flow,timestep_embedder}.py`,
`models/structured_latent_vae/{base,decoder_gs,decoder_mesh}.py`, `modules/transformer/{blocks,modulated}.py`,
`modules/attention/modules.py`, `modules/sparse/transformer/modulated.py`, `modules/sparse/spatial.py`,
`representations/gaussian/gaussian_model.py`, `representations/mesh/cube2mesh.py`, `utils/postprocessing_utils.py`,
`data/dataset/tdfy/{preprocessor,img_processing,img_and_mask_transforms,pose_target}.py`. Not read: `tables.py` (FlexiCubes lookup tables),
sparse attention kernels, the training and visualisation code, the notebooks.

## Licence (from the text of `LICENSE`, "SAM License", last updated 2025-11-19; not legal advice)

- **Grant (1.a):** non-exclusive, worldwide, non-transferable, royalty-free, limited licence to use, reproduce, distribute, copy, create
  derivative works of and modify the SAM Materials (documentation, model code, weights, inference, training and fine-tuning code). There is
  **no non-commercial clause and no field-of-use limit other than 1.b.v**. The text carries no territory carve-out beyond "worldwide".
- **Patent grant (1.a.i):** terminates if you sue anyone alleging the materials infringe a patent. The paragraph refers to "the Work" and to
  "SAM 3 Materials", wording carried over from the SAM 3 licence; it reads as a template artefact and the text is taken as written.
- **Redistribution (1.b.i):** any distribution of the materials *or derivative works* must be under this Agreement and ship a copy of it.
  A repacked checkpoint (for example safetensors) therefore needs the `LICENSE` file beside it.
- **Research publication (1.b.ii):** work submitted for publication that uses the materials must acknowledge them.
- **Compliance (1.b.iii):** applicable law, including trade-control, privacy and data-protection law.
- **No reverse engineering (1.b.iv):** "will not involve or encourage others to reverse engineer, decompile or discover the underlying
  components". The public source is not covered by that wording in any obvious way, but a clean reimplementation should be built from the
  public Python only and must not probe the weights for undocumented structure beyond loading them.
- **Trade controls (1.b.v):** the licensee must not be the target of Trade Controls (sanctions, export and import controls of the US, UN, EU
  and UK); no use "for any activities subject to" ITAR; no end uses "prohibited by Trade Controls, including those related to military or
  warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons". The end-use list
  binds every downstream user of an app that ships this model, so it belongs in any user-facing model card or download prompt.
- **Warranty and liability (2, 3, 4):** as is; the disclaimer covers "any output and results".
- **Ownership (5.a):** you own your derivative works and modifications, subject to Meta's ownership of the materials. The text says nothing
  explicit about who owns generated outputs: unresolved, do not state it either way.
- **Litigation and indemnity (5.b):** suing Meta over the materials or outputs ends your licence; you indemnify Meta against third-party
  claims arising from your use or distribution.
- **Term (6):** ends on breach; then delete and stop using the materials (3, 4 and 7 survive).
- **Law (7):** California law, California courts. **Changes (8):** Meta may amend the Agreement and continued use is acceptance.

**Access terms outside the licence text.** The Hugging Face repository is `gated: manual`: the access form asks for first and last name,
date of birth, country, affiliation, job title, IP-derived geolocation, and acceptance of the licence and Meta's privacy policy.
`doc/setup.md` adds that the weights are "available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions", and
that requests from there are rejected. Approval is per Hugging Face account and is Meta's decision, not ours.

**Licences of other code in the upstream tree** (headers read):

| File or dependency | Licence found | Consequence for this port |
|---|---|---|
| `sam3d_objects/**` (most files) | "Copyright Meta Platforms", under the SAM License | the architecture is implemented from this source |
| `representations/gaussian/gaussian_model.py` | Meta header only | the activation formulas (see below) may be ported |
| `renderers/gaussian_render.py`, `representations/gaussian/general_utils.py` | Inria GRAPHDECO header: "free for non-commercial, research and evaluation use" | **do not port these files**; the repository has no splat rasteriser (see [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md)) and any future one must come from the published method, not this code |
| `representations/mesh/flexicubes/flexicubes.py` | Apache-2.0, NVIDIA | portable with attribution; `tables.py` header not read: unverified |
| MoGe, DINOv2 (via `torch.hub`), spconv, kaolin, gsplat, pytorch3d, nvdiffrast | not read | check each before port; unverified |

## Stage map (public pipeline `InferencePipelinePointMap.run`, defaults of `notebook/inference.py::Inference`)

The public `Inference.__call__` runs with `with_mesh_postprocess=False`, `with_texture_baking=False`, `with_layout_postprocess=False` and
`use_vertex_color=True`; `demo.py` writes only the Gaussian splat (`output["gs"].save_ply`). Output keys are the splat `gs`, the mesh `glb`,
and the pose (`rotation` quaternion w-first, `translation`, `scale`).

| # | Stage | Upstream (from source) | Existing C# | Gap |
|---|---|---|---|---|
| 0 | Inputs | RGB image, one binary mask per object; the mask is written into the alpha channel (`mask * 255`), image `uint8` | `Sam3Pipeline`, `Sam2MaskRefiner` produce masks; `SceneMeshPrompt.Masks` exists for CoLa3D | a request field for image + mask for this model; no mask is derived inside the model |
| 1 | Pointmap | `MoGe.infer(image, force_projection=False)` gives `points` (H, W, 3) and, if present, `intrinsics`; points are rotated into the PyTorch3D camera convention by `look_at_view_transform(eye (0,0,-1), up (0,-1,0))` rotation; otherwise intrinsics are recovered by `infer_intrinsics_from_pointmap` (MoGe `recover_focal_shift`) | none | **a MoGe port** (not specified here; needs its own dossier) or a stand-in; see "Layout decision" |
| 2 | Preprocess (shape stage) | `crop_around_mask_with_padding(box_size_factor=1.0, padding_factor=0.1)`, then `rembg` (image times mask), then `pad_to_square_centered` and bicubic resize to 518; mask padded and resized to 518 nearest. The pointmap is normalised (below), cropped and padded identically with NaN fill, then goes through the same crop | `Trellis2ImagePreprocessor` is similar but not the same (square crop around alpha above 0.8, matte, RGB times alpha) | a SAM 3D preprocessor, pure tensor code, specified by the public source |
| 3 | Conditioning (shape) | `EmbedderFuser` over several embedders keyed by kwargs (`image`, `mask`, `rgb_image`, `rgb_image_mask`, `pointmap`, `rgb_pointmap`; which embedder takes which is in the gated YAML: unverified); per embedder a `LayerNorm` + Llama-3 SwiGLU projection, a learned per-group index embedding added, tokens concatenated on the token axis | `Dinov2VisionEncoder` (ViT with registers, LayerScale), `Dinov2ImagePreprocessor` | `PointPatchEmbed`, `PointRemapper`, the fuser, the SwiGLU projection; the DINO weights come from the checkpoint with `torch.hub` key names (HF-style names are what the encoder reads): unverified |
| 4 | Structure generator | flow matching over a dict of latents with a mixture-of-transformers (MOT) block (below); `shape` latent is 4096 tokens of 8 channels (a 16^3 grid) | `Trellis/SparseStructureFlow` (single latent, patchify) and `TrellisV2/Trellis2SparseStructureFlow` | a MOT flow; the pose-latent names and sizes are in the gated YAML: unverified |
| 5 | Sampler | Euler, 25 steps, `rescale_t` 3, CFG strength 7, interval [0, 500], `no_shortcut` true (below) | `Trellis/TrellisSparseStructureSampler`, `TrellisV2/Trellis2FlowSampler` | the time convention differs (t runs 0 to 1 here); new sampler class |
| 6 | Structure decode | `ss_decoder(latent.permute(0,2,1).view(B,8,16,16,16))`, `coords = argwhere(ss > 0)`, then `downsample_sparse_structure` if more than 42000 voxels | `Trellis/SparseStructureDecoder` (same class lineage, 8 to 64^3 occupancy) | the downsample rule; config values: unverified |
| 7 | Pose decode | closed-form from the pose latents and the pointmap statistics (below) | none | new, pure math |
| 8 | SLAT flow | sparse DiT over the voxels, 25 steps, `rescale_t` 3, CFG strength 5, interval [0, 500]; conditioned on the image (no pointmap) with a second preprocessor instance | `Trellis/SlatFlowModel` (the TRELLIS v1 flow: sparse res-blocks, APE, shared structure), `TrellisV2/Trellis2SlatFlow` (RoPE, different) | the optional `d` embedder; config values: unverified |
| 9 | Denormalise | `slat * SLAT_STD + SLAT_MEAN` (constants below) | `TrellisSlatSampler` does the v1 constants | new constants |
| 10 | Gaussian decode | `SLatGaussianDecoder` (swin-window sparse transformer, 8 channels to per-voxel gaussians) | `Trellis/SlatGaussianDecoder`, `TrellisGaussianRepresentation` | the representation constants differ or may differ (below); `scaling_activation` is configurable upstream and hard-coded softplus here |
| 11 | Mesh decode | `SLatMeshDecoder`: transformer, two `SparseSubdivideBlock3d` (each `GroupNorm`, subdivide, two sparse 3x3x3 convolutions), FlexiCubes at resolution `4 * resolution` | none (`TrellisV2/ShapeVae*` is a different decoder; `Geometry/Ops/MarchingCubes` is not FlexiCubes) | sparse convolution + subdivide, FlexiCubes extraction |
| 12 | Export | splat to PLY; mesh to a trimesh with vertex colours `vertex_attrs[:, :3]` and a z-up to y-up rotation | `PlyWriter.WriteSplats`, `GlbWriter`, `MeshService` wiring | wiring only |
| 13 | Optional mesh post-process and texture bake | `simplify` 0.95 through PyVista decimate, `fill_holes` by rasterised visibility (1000 views at 1024), xatlas UVs, texture bake with `nvdiffrast` or PyTorch3D, 100 views at 1024, `texture_size` 1024 | `docs/Research/MESH_POSTPROCESS.md` toolkit (decimation, atlas, bake) | not needed for the default path |
| 14 | Optional layout optimisation | see "Layout decision" | none | dropped |

## Constants and rules read from the source

### Time, sampler and guidance (`flow_matching/model.py`, `classifier_free_guidance.py`, `shortcut/model.py`, `inference_pipeline.py`)

- Interpolant `x_t = (1 - (1 - sigma_min) t) x0 + t x1` with `x0` noise and `t = 1` the data; `sigma_min` default 0 (rectified flow). The
  velocity target is `x1 - (1 - sigma_min) x0`. Inference integrates `t` from 0 to 1, the opposite of TRELLIS.2's `1 to 0`.
- Schedule: `t = linspace(0, 1, steps + 1)`, then if `rescale_t`: `t = t / (1 + (rescale_t - 1)(1 - t))`. The model receives `1000 t`
  (`time_scale` 1000). Solver `euler` (`x += v dt`); `midpoint`, `rk4` and an `sde` solver exist and are not used by the pipeline defaults
  (the `sde` code references an undefined `variance` and is unusable as published).
- Pipeline defaults (`InferencePipeline.__init__`): `ss_inference_steps` 25, `ss_rescale_t` 3, `ss_cfg_strength` 7, `ss_cfg_interval` [0, 500],
  `ss_cfg_strength_pm` 0.0; `slat_inference_steps` 25, `slat_rescale_t` 3, `slat_cfg_strength` 5, `slat_cfg_interval` [0, 500]; `dtype`
  `bfloat16`; `pad_size` 1.0; `downsample_ss_dist` 0. A constructor argument can be overridden by the gated `pipeline.yaml`: the values
  actually used are unverified.
- CFG: `(1 + s) y_cond - s y_uncond` per tensor of the latent dict. The strength applies only while `interval[0] <= t <= interval[1]` with `t`
  the **scaled** time (0 to 1000); outside it the unconditional pass is skipped. With `rescale_t` 3 the value `t = 500` is reached at
  `u = 0.75` of the uniform grid, so guidance covers the first 19 of 25 steps (`u = k / 25`, `k = 0 .. 18`).
- The unconditional branch is `unconditional_handling = "add_flag"`: the backbone receives `cfg=True` and, if `force_zeros_cond` is set in
  its config (unverified), multiplies the condition tokens by zero.
- Pointmap guidance (`PointmapCFG`; the pipeline sets `reverse_fn.strength_pm` on the structure generator, which suggests the YAML selects it, but that is
  unverified): `y_cond + s_pm (y_cond - y_nopm) + s (y_nopm - y_uncond)`, where `y_nopm` is the pass with the `pointmap` and `rgb_pointmap` condition
  tokens multiplied by zero, and `y_uncond` the `cfg=True` pass. It costs three backbone calls, not two, while guidance is active. With `strength_pm` 0 it
  is **not** the ordinary formula: it reduces to `y_cond + s (y_nopm - y_uncond)`, so the two forms must not be conflated. The structure sampler must
  therefore support both (`strength_pm` default 0.0 in the pipeline constructor; the configured value is unverified).
- Shortcut model (`ShortCut`, arXiv 2410.12557): the backbone takes an extra step-size `d` embedded by a second `TimestepEmbedder` and added
  to the time embedding. With `no_shortcut = True` (the pipeline default when `use_stage*_distillation` is false) `d = 0`, **but the `d`
  embedder is still evaluated at 0 and added**, so a checkpoint with `is_shortcut_model: true` needs that term even for plain sampling. With
  the distillation flag, `d = 1 / steps`, `strength = 0` (no CFG) and fewer steps are intended. Whether the released checkpoints set
  `is_shortcut_model` is unverified.
- Noise: `torch.randn` after `torch.manual_seed(seed)`; the C# generator cannot reproduce it (same limitation as TRELLIS.2).

### Timestep embedder, norms and blocks (`timestep_embedder.py`, `modules/transformer/*`)

- `TimestepEmbedder`: sinusoid of width 256, `freqs = exp(-ln(10000) i / 128)`, cosines then sines, `Linear(256, C) SiLU Linear(C, C)`.
- Dense block (`ModulatedTransformerCrossBlock`): `norm1` and `norm3` without affine, `norm2` affine, all eps 1e-6; self-attention input
  `norm1(x)(1 + scale) + shift`, gate on its output; cross-attention (no gate, no modulation) after `norm2`; MLP `Linear, GELU(tanh), Linear` with
  `mlp_ratio` default 4, modulation from `adaLN_modulation = SiLU, Linear(C, 6C)` per block or one shared (`share_mod`). This is the TRELLIS
  v1 block that `Trellis/SsFlowBlock` already implements.
- QK RMS norm: `F.normalize(x) * gamma * sqrt(head_dim)` with `gamma [heads, head_dim]`. `AbsolutePositionEmbedder`: `freq_dim = C / 3 / 2`,
  `sin` then `cos` per axis, zero padded to `C`. RoPE (when `pe_mode` is `rope`) uses `freq_dim = C / 3 / 2` and adjacent-pair complex
  rotation; it is **not implemented** in the MOT attention (`raise NotImplementedError`), so the structure generator uses learned or fixed
  per-latent position embeddings instead.
- Final layer of every latent: `F.layer_norm(h)` (no affine) then `out_layer` (`Latent.to_output`); input is `input_layer(x) + pos_emb`.

### Mixture-of-transformers structure generator (`mot_sparse_structure_flow.py`, `mm_latent.py`, `attention/modules.py`)

- The latent is a dict. Each entry has its own `Latent` module (`input_layer`, `out_layer`, `pos_emb` as a learned parameter, a fixed random
  buffer or a fixed 3-D sin-cos table, chosen by `ShapePositionEmbedder`, `RandomPositionEmbedder` or `LearntPositionEmbedder` in the YAML).
  Latents named in `latent_share_transformer` are concatenated on the token axis and share one set of per-latent modules; other latents keep their
  own. Timestep embedding and `adaLN_modulation` are shared by all latents; the block has **per-latent** `norm1/2/3`, QKV and output projections,
  QK RMS norms, cross-attention and MLP.
- `MOTMultiHeadSelfAttention`: the protected modality `shape` attends only to itself. All other latents concatenated attend to each other
  **and** to the shape keys and values (detached in training; no difference at inference). RoPE is not supported. Cross-attention is
  per-latent to the same condition tokens.
- Pose latent names that the decoder maps: `shape`, `quaternion`, `6drotation`, `6drotation_normalized`, `translation`, `scale`,
  `translation_scale`. Which of them the released checkpoint uses, their token counts and widths: unverified (gated YAML).

### Condition embedders (`embedder_fuser.py`, `pointmap.py`, `point_remapper.py`, `dino.py`)

- `Dino`: `torch.hub.load("facebookresearch/dinov2", dino_model)` with `input_size` default 224 (the configured value is unverified; the
  pipeline feeds 518 px crops), ImageNet mean and std, bilinear resize, output either `[x_norm_clstoken, x_norm_patchtokens]` or, with
  `prenorm_features`, `F.layer_norm(x_prenorm)`. `DinoForMasks` embeds the mask through the same backbone (a 1-channel mask is repeated to
  three channels).
- `PointPatchEmbed` (`input_size` 256, `patch_size` 8, `embed_dim` 768 defaults; configured values unverified): pointmap resized by **nearest**
  to `input_size`; invalid (non-finite) points set to 0, remapped, projected by `Linear(3, D)` and replaced by the learned `invalid_xyz_token`; windows
  of `patch_size^2` points get a prepended CLS token plus `pos_embed_window`, pass one timm `Block` (16 heads, mlp 2.0, `qkv_bias`, LayerNorm eps 1e-6),
  and the window's CLS output plus a bilinear-interpolated `pos_embed` gives one token per window (`(input_size / patch_size)^2` = 1024 tokens).
  `dropout_prob` and `force_dropout_always` swap tokens for a learned `dropped_xyz_token`.
- `PointRemapper` (default `exp`): `xy / (1 + z)`, `log1p(z)`. Others: `linear`, `sinh`, `sinh_exp`, `exp_disparity`. Which one is configured: unverified.
- `EmbedderFuser`: per embedder `LayerNorm` then a Llama-3 `FeedForward` (`hidden = int(2/3 * multiplier * D)` rounded up to a multiple of 256,
  `silu(w1 x) * w3 x` then `w2`, no biases; `projection_net_hidden_dim_multiplier` default 4.0); a learned `idx_emb` per positional group is added;
  tokens are concatenated on the token axis (or channel-concatenated and projected if `compression_projection_multiplier > 0`). `force_drop_modalities`
  multiplies the named groups' tokens by 0. Weights `idx_emb`, `projection_nets.*`, the embedders' own keys: key names are known, shapes are
  unverified.

### Pointmap normalisation and pose decode (`img_and_mask_transforms.py`, `pose_target.py`, `inference_utils.py`)

- Default normaliser `SSIPointmapNormalizer` (the YAML may select `ObjectCentricSSI` or others: unverified): `shift = (0, 0, nanmedian(z))`,
  `scale = nanmean(|pointmap - shift|)` over all elements, same value on the three axes; `normalized = (pointmap - shift) / scale`. The
  shape stage gets the normalised, masked, cropped pointmap; `rgb_pointmap` is the full-frame normalised pointmap with the same scale and shift.
  The unnormalised full pointmap is kept for the layout step.
- Rotation: `6drotation_normalized * ROTATION_6D_STD + ROTATION_6D_MEAN` (constants below), then Gram-Schmidt:
  `b1 = normalize(a1)`, `b2 = normalize(a2 - (b1 . a2) b1)`, `b3 = b1 x b2`, matrix columns `[b1 b2 b3]`, converted to a quaternion (w first).
  `scale` and `translation_scale` latents pass through `exp`. Then the convention class (`ScaleShiftInvariant`, `ApparentSize`,
  `DisparitySpace`, `ZeroPredictionScaleShiftInvariant`; `default` returns an empty pose; the configured one is unverified) turns the
  targets and the pointmap `scale`/`shift` into the object-to-camera `scale` (mean over axes, expanded to three), `translation` and `rotation`.
  `ScaleShiftInvariant`: compose scale, rotation, translation, then post-compose with `ssi_to_metric(scene_scale, scene_shift)` (scale then translate).
  `DisparitySpace`: `translation_xy = (xy + shift_xy) * z * exp(shift_z_log)`, `scale = s * z * exp(shift_z_log)`.
  After decode the pipeline multiplies `scale` by `downsample_factor` (1 or 2, from the voxel count rule).
- `downsample_sparse_structure(coords, max_coords=42000, factor=2)`: if at most 42000 voxels, unchanged; otherwise rescale the coordinates
  about the bounding box centre to half extent, round, clamp, deduplicate, and if still above the cap keep a random subset. The comment ties 42000
  to `max(int32) / (64 * 768)` and to the mesh decoder only.
- `SLAT_MEAN` = [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133, -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826];
  `SLAT_STD` = [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695, 2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338].
- `ROTATION_6D_MEAN` = [-0.06366084883674913, 0.008438224692279752, 0.00017084786438302483, 0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688];
  `ROTATION_6D_STD` = [0.6656971967514863, 0.6787012271867754, 0.30345010594844524, 0.4394504420678794, 0.39817973931717104, 0.6176286868761914].
- `ScaleShiftInvariant` training normalisation (applied only when `normalize=True`, which the pipeline never passes): scale mean/std 1.0232692956924438 / 1.3773751258850098;
  translation mean (0.003191213821992278, 0.017236359417438507, 0.9401122331619263), std (1.341888666152954, 0.7665449380874634, 3.175130605697632).

### Decoders (`decoder_gs.py`, `decoder_mesh.py`, `base.py`, `gaussian_model.py`, `cube2mesh.py`, `sparse_structure_vae.py`)

- `SparseStructureDecoder`: `Conv3d(latent, channels[0], 3)`, `num_res_blocks_middle` ResBlocks, per stage `num_res_blocks` ResBlocks and a
  conv-then-pixel-shuffle x2 upsample, `ChannelLayerNorm32 / SiLU / Conv3d(channels[-1], out, 3)`; ResBlock `norm, SiLU, Conv3d, norm, SiLU, zero-init Conv3d`
  plus skip. The default class arguments (`layer` norm, 2 middle blocks) match what `Trellis/SparseStructureDecoder` reads; the configured channels are
  unverified.
- `SparseTransformerBase` (decoder trunk): `SparseLinear(in, C)`, APE added, blocks with `attn_mode` `swin` (windowed, `window_size` 8 default,
  odd blocks shifted by `window_size / 2`) or the serialised modes; this is the TRELLIS v1 decoder trunk.
- `SLatGaussianDecoder`: per voxel `num_gaussians` gaussians; output layout `_xyz [G,3]`, `_features_dc [G,1,3]`, `_scaling [G,3]`, `_rotation [G,4]`, `_opacity [G,1]`,
  each scaled by `lr[name]`; position `(coord + 0.5) / resolution + tanh(offset * lr + perturbation) / resolution * 0.5 * voxel_size`, where the
  perturbation is `atanh((hammersley(3, i, G) * 2 - 1) / voxel_size)` when `perturb_offset`. Gaussian activations: scale `sqrt(act(raw + bias)^2 + kernel^2)`
  with `act` `exp` or `softplus` and `bias = inverse_act(scaling_bias)`; rotation `normalize(raw + [1,0,0,0])`; opacity `sigmoid(raw + logit(opacity_bias))`;
  xyz in the aabb `[-0.5, 0.5]^3`. `num_gaussians`, `voxel_size`, `3d_filter_kernel_size`, `scaling_bias`, `opacity_bias`, `scaling_activation`, `lr`,
  `perturb_offset` and `resolution` come from the gated `slat_decoder_gs.yaml`: **unverified**; `Trellis/TrellisGaussianRepresentation` hard-codes the
  TRELLIS v1 values (32, 1.5, 0.0009, softplus, 0.004, 0.1).
- `SLatMeshDecoder`: trunk, then `SparseSubdivideBlock3d(C -> C/4, resolution)` and `(C/4 -> C/8, 2 * resolution)`, `SparseLinear(C/8, feats_channels)`;
  `SparseFeatures2Mesh(res = 4 * resolution)`: per-voxel channels `sdf [8]`, `deform [8,3]`, `weights [21]`, and with `use_color` `color [8,6]`; `sdf += -1/res`;
  corners are merged to a dense grid, FlexiCubes extracts the mesh with `beta = weights[:12]`, `alpha = weights[12:20]`, `gamma_f = weights[20]`. At `resolution` 64 the
  extraction grid is 256^3. Whether `use_color` is set and the resolution: unverified.
- Memory limit stated in source: more than about 42000 occupied voxels breaks mesh decoding by int32 overflow; the pipeline halves the structure instead.

### Layout optimisation (`layout_post_optimization_utils.py`, `inference_utils.py`)

Reads the mesh (or splat), the quaternion, translation and scale, the mask, the unnormalised pointmap and intrinsics (focal set to `min(fx, fy)`), and:
(1) skips the whole step if `check_occlusion` flags the mask (helpers for border contact, occlusion by other masks and internal holes), (2) an alignment step (`run_alignment`, not read in detail) against the pointmap, (3) optional ICP (disabled by the pipeline: `Enable_shape_ICP=False`),
(4) render-and-compare: 5 Adam steps (lr 1e-2, translation and scale) then 25 (lr 5e-3, quaternion, translation, scale), loss weights mask 200 (mesh) or mask 10 + rgb 10 (splat), regularisers
0.1, 0.05, 0.05, the render being a PyTorch3D silhouette rasteriser (mesh) or a gsplat render (splat), (5) accepts the result only if IoU is at least 0.5 and improved. It needs autograd through a
differentiable rasteriser. The pipeline's own default is `with_layout_postprocess=False` and the public `Inference` wrapper never turns it on.

## What maps to the existing C# and what is missing

Reusable as is or with small changes (all unverified against SAM 3D weights until a checkpoint is available):

- `Dinov2VisionEncoder` (CLS, registers, LayerScale), `Dinov2ImagePreprocessor`; it reads HF-style key names while the upstream `torch.hub` model uses its own, so a key map is needed.
- `Trellis/SsFlowBlock`, `Trellis/SlatFlowModel`, `Trellis/SlatResBlock3d`, `Trellis/SparseStructureDecoder`, `Trellis/SlatGaussianDecoder` (swin trunk): the blocks implement the same TRELLIS v1
  computation (`SsFlowBlock` is `internal` today, so reuse needs it widened or moved). The SLAT flow needs the optional shortcut `d` embedder.
- `PlyWriter.WriteSplats`, `GlbWriter`, `GaussianSplatCloud`, `MeshService` wiring pattern, `Sam3Pipeline` / `Sam2MaskRefiner` for masks, `PickleCheckpointRepacker` / `PytorchPickleLoader` for the `.ckpt` files.

Missing (new code, all of it specified by the public source and testable on tiny synthetic networks without weights):

1. Flow-matching sampler with the 0-to-1 time convention, `rescale_t` map, scaled-time CFG interval, optional `d`, and the pointmap CFG (a new `SamFlowSampler`; extract to a shared base only if the TRELLIS.2 sampler can take the same contract).
2. MOT block and multi-latent flow (per-latent modules, protected-modality attention).
3. `PointPatchEmbed`, `PointRemapper`, `EmbedderFuser` with the SwiGLU projection.
4. SAM 3D preprocessor (mask-box crop, pad, resize, NaN-filled pointmap crop) and SSI normaliser.
5. Pose decoder (6D rotation, conventions) and `downsample_sparse_structure`.
6. Gaussian representation parameterised by the YAML values (scaling activation, kernel, lr, perturbation switch).
7. Mesh branch: sparse convolution, `SparseSubdivide`, FlexiCubes extraction (needs `tables.py`, whose licence is not read).
8. MoGe (pointmap and intrinsics), if the pose output is wanted; its own dossier.
9. A safe `.ckpt` path: every released weight is a Lightning `.ckpt` pickle (see below).

## Layout decision

**Drop the layout optimiser.** It is off by default in the public pipeline and in the public `Inference` wrapper, it needs a differentiable
silhouette rasteriser and a splat rasteriser with gradients (the repository has neither, and the splat one may not be ported from the
Inria-licensed upstream file), and it only ever adjusts the final pose by a bounded amount (it keeps the original pose if IoU does not improve).
What stays is the **feed-forward pose**, which is closed-form and specified above. Two modes follow:

- *Object only:* no MoGe. The pointmap condition tokens are multiplied by zero (the trained-in `force_drop_modalities` path) and the pose is not
  produced; the result is the shape and appearance in object space. Whether the released structure generator gives sensible shapes with the pointmap
  group dropped is **unverified** and can only be checked with the weights.
- *With pose:* needs a MoGe stand-in. Until MoGe is ported, a caller may supply a pointmap (`run(pointmap=...)` accepts one; the intrinsics are then inferred from it by `infer_intrinsics_from_pointmap`, which needs MoGe's focal-shift solver ported or reimplemented).

## Memory and time (reported, not measured here)

`doc/setup.md` requires "a NVIDIA GPU with at least 32 Gb of VRAM", Linux 64-bit, Python environment with CUDA 12.1, PyTorch 2.5.1, spconv,
xformers 0.0.28, kaolin 0.17, gsplat, pytorch3d and flash-attn (flash-attn is selected automatically on A100, H100, H200 only). The checkpoint
sizes in the next section give 6.69 GB for the structure generator and 4.91 GB for the SLAT generator on disk; the stored dtype is unverified
(the pipeline default compute dtype is bfloat16, which would put the generators near 3.3 B and 2.5 B parameters if the files are bf16, or about half that if F32).
The 12 GB RTX 3060 is a later concern; the existing pattern (each flow read from disk for its stage and released, as for TRELLIS.2) is the likely starting point.

## What the user must provide

Everything below is the user's action; this repository cannot do any of it.

1. **Hugging Face access** to `facebook/sam-3d-objects`: submit the gated form (name, date of birth, country, affiliation, job title, licence
   acceptance) with the user's own account and wait for Meta's manual approval. Approval is refused for comprehensively sanctioned jurisdictions.
2. **A token** with read access, supplied by the user for the download step (never committed, never logged).
3. **The files** (public listing at revision `2e73555018d2741ccd486e56c24fac41155a1dc6`; sizes in bytes, sha256 is the Hugging Face LFS oid):

| File | Size | sha256 | Needed for |
|---|---|---|---|
| `checkpoints/pipeline.yaml` | 3,548 | (not LFS) | every config value; first file to read |
| `checkpoints/ss_generator.yaml` | 5,076 | (not LFS) | structure generator and condition-embedder config |
| `checkpoints/ss_generator.ckpt` | 6,690,136,964 | `225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146` | structure generator + condition embedders |
| `checkpoints/slat_generator.yaml` | 1,986 | (not LFS) | SLAT flow config |
| `checkpoints/slat_generator.ckpt` | 4,906,537,684 | `91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb` | SLAT flow + its condition embedder |
| `checkpoints/ss_decoder.yaml` / `.ckpt` | 244 / 147,609,242 | `6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340` | occupancy decoder |
| `checkpoints/slat_decoder_gs.yaml` / `.ckpt` | 576 / 171,476,155 | `f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111` | Gaussian decoder (the default output) |
| `checkpoints/slat_decoder_mesh.yaml` / `.ckpt` | 300 / 363,726,862 | `85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33` | mesh decoder (second pass; `slat_decoder_mesh.pt` is a 363,728,714-byte duplicate, skip it) |
| `LICENSE` | 8,204 | (not LFS) | ship with any repack |

   Minimum for a first splat: the first seven rows through `slat_decoder_gs`, **about 11.92 GB** (11,915,771,475 bytes with the five YAMLs); with the mesh decoder about 12.28 GB.
   Not needed for inference: `slat_decoder_gs_4` (170,269,801), `ss_encoder.*` (119,085,402 plus a `.safetensors` the listing shows with size 0), `slat_encoder.ckpt` (173,263,986); the encoders are for
   latent-space work (`notebook/demo_ss_encoder.ipynb`) and are out of scope.
4. **The MoGe checkpoint** named in `pipeline.yaml` (repository, size, licence unknown until the YAML is read), only for the pose path.
5. **A decision on destination and approval to download**, relayed by the parent session. The expected destination is `Models/3D/sam3d-objects/` (the folder is never touched by agents otherwise), with the
   `LICENSE` beside the files. Nothing is downloaded until the user says yes with the repository, files, sizes, sha256, licence and purpose in front of them.

**Pickle handling.** All released weights are Lightning `.ckpt` files and the upstream loader calls `torch.load(..., weights_only=False)`. This repository must never execute a pickle:
the files go through the existing safe `PytorchPickleLoader` (a pickle machine that builds tensors without running arbitrary callables) and are repacked to safetensors by
`PickleCheckpointRepacker`. A `.ckpt` also carries non-tensor state (hyper-parameters, optimiser state); the repack keeps `state_dict` entries under the prefixes used by the pipeline
(`_base_models.generator.`, `_base_models.condition_embedder.`) and drops the rest. Whether the loader copes with these particular files is unverified. The `LICENSE`'s redistribution rule applies to any repacked copy the user shares.

## Validation plan (when weights exist)

1. Read `pipeline.yaml` and the five YAMLs, then fill every *unverified* item above; update this file before writing the loaders.
2. Key map per checkpoint, rejecting missing, extra and mis-shaped tensors (the TRELLIS.2 pattern), then a header-only test (`Sam3dRealWeightsTests`).
3. Reference dumps from the upstream Python on the CPU or a rented GPU (the same noise, voxel set and condition tokens saved as inputs): condition tokens for each embedder, one structure-flow velocity at the schedule's
   first non-trivial `t` (cond and uncond), the SS decode occupancy, one SLAT-flow velocity, the Gaussian decode, the pose. Compare by correlation and relative L2 as in the Hunyuan3D-2.1 rows; share the saved inputs, not seeds.
4. A rendered look of the splat from several angles (needs a splat viewer; there is no rasteriser in the engine) and a mesh render for the FlexiCubes branch.
5. Only then a status row above 🔧; ✅ needs real-weight parity and a rendered look.

## Open items (all unverified until the gated files are read)

- Every value in `pipeline.yaml`, `ss_generator.yaml`, `slat_generator.yaml`, `ss_decoder.yaml`, `slat_decoder_gs.yaml`, `slat_decoder_mesh.yaml`: model widths, block and head counts, latent names and sizes, `share_mod`,
  `pe_mode`, `is_shortcut_model`, `force_zeros_cond`, embedder list and kwargs routing, DINO variant and input size, pointmap embedder sizes and remap type, pose convention, normaliser, representation constants, mesh `use_color`.
- The stored dtype of each checkpoint, the key prefixes and whether a plain `state_dict` can be extracted without executing anything.
- Whether `ss_generator.ckpt` carries the DINO and pointmap embedder weights (the pipeline loads the condition embedder from the same file through the `_base_models.condition_embedder.` prefix) or fetches DINO from `torch.hub` at run time.
- The MoGe version and checkpoint, and the licences of MoGe and `tables.py`.
- How the structure generator behaves with the pointmap tokens zeroed (object-only mode).
- Whether the release's quality needs the 25-step CFG schedule at all on the 12 GB target, and the cost of the shape stage when the 1024-token pointmap group and the image groups are all present.
