# SAM 3D Objects architecture (image + mask to Gaussian splat or mesh, with layout)

Research dossier for `facebookresearch/sam-3d-objects` (Meta), written from the public GitHub source, the public Hugging Face file
listing and the seven small text files (`LICENSE` and the six pipeline and model YAMLs, 19,934 bytes in all). The text files were read
from the ungated mirror `jetjodh/sam-3d-objects` at revision `f29e43c1` after every git blob id matched Meta's gated listing, so
they are Meta's files. **No checkpoint was downloaded or read.** Config values below are verified against those YAMLs; whatever
needs a checkpoint (key names, shapes, stored dtype, numerical behaviour) stays marked *unverified*. Nothing here is built yet; the
status row is in [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md).

## Pins (source snapshot, 2026-11-11)

| Item | Revision |
|---|---|
| Upstream code `facebookresearch/sam-3d-objects` | commit `f91db411c50efee93d8db7aeb323885650f6f722` (2026-06-02, "SA-3DAO Release - README Update (#190)"); every source file below was read at this commit |
| Weights and configs `facebook/sam-3d-objects` | revision `2e73555018d2741ccd486e56c24fac41155a1dc6` (last modified 2026-06-12), `gated: manual`, tag `license:other`. The unauthenticated file listing (names, sizes, LFS sha256 and git blob ids) was read directly; the seven text files were read from the mirror below |
| Config mirror `jetjodh/sam-3d-objects` | revision `f29e43c1`, ungated. `LICENSE` and the six YAMLs have the same git blob ids as Meta's listing (so identical bytes); its six `.ckpt` files also list the same sizes and LFS sha256 as Meta's. No checkpoint was fetched |
| MoGe (pointmap model) | code pinned by `requirements.txt` at `microsoft/MoGe@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`; `pipeline.yaml` loads `Ruicheng/moge-vitl` through `moge.model.v1.MoGeModel.from_pretrained`; that repository's licence and files were not read |

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

## Configs read (`pipeline.yaml`, `ss_generator.yaml`, `slat_generator.yaml`, `ss_decoder.yaml`, `slat_decoder_gs.yaml`, `slat_decoder_mesh.yaml`)

`pipeline.yaml` instantiates `InferencePipelinePointMap` and its keys are constructor arguments, so it overrides the code defaults. The
generator YAMLs also carry sampler fields (`inference_steps` 2 for the structure generator, 12 for the SLAT generator, `strength`,
`interval`, `rescale_t`), but the constructor overwrites them on load (`override_ss_generator_cfg_config`, `override_slat_generator_cfg_config`);
the pipeline values below are the ones that run.

| Setting | Value that runs | Code default it replaces |
|---|---|---|
| compute dtype (`dtype`) | float16 | bfloat16 |
| structure sampler | 25 Euler steps, `rescale_t` 3, CFG strength 7, interval [0, 500], `strength_pm` 0 (all code defaults; the YAML does not set them) | same |
| SLAT sampler | 25 Euler steps, `rescale_t` **1**, CFG strength **1**, interval [0, 500] | `rescale_t` 3, strength 5 |
| `downsample_ss_dist` | **1** | 0 |
| `pose_decoder_name` | `ScaleShiftInvariant` | `default` |
| `slat_mean`, `slat_std` | values in "Constants" below | `SLAT_MEAN`, `SLAT_STD` constants in `inference_utils.py` |
| `pad_size` | 1.0 | same |
| `ss_condition_input_mapping`, `slat_condition_input_mapping` | `[]` | `["image"]`; empty means every preprocessor output key is passed as a keyword to the condition fuser |
| `version`, `compile_model` | `3dfy_v9`, true | `v0`, false; not behaviour that a port needs |
| `depth_model` | `MoGe` wrapper, checkpoint `Ruicheng/moge-vitl` | none |
| preprocessors | explicit, see stage 2 | `get_default_preprocessor()` (box factor 1.0, padding 0.1, `rembg`): **not what runs** |

Per-model files:

| File | Class and values |
|---|---|
| `ss_generator.yaml` | `ShortCut` (flow matching) over `ClassifierFreeGuidanceWithExternalUnconditionalProbability` (plain two-pass CFG, `unconditional_handling add_flag`, `p_unconditional` 0.1) over `SparseStructureFlowTdfyWrapper`: `model_channels` 1024, `num_blocks` 24, `num_heads` 16, `mlp_ratio` 4, `patch_size` 1, `resolution` 16, `pe_mode ape`, `qk_rms_norm` true, `cond_channels` 1024, `in_channels` and `out_channels` 8, `force_zeros_cond` true, **`is_shortcut_model` true**, `use_fp16` false, `condition_embedder` null (the fuser is a separate module). Latents: `shape` (8 channels, `ShapePositionEmbedder`, resolution 16, patch 1: 4096 tokens), `6drotation_normalized` (6), `translation` (3), `scale` (3), `translation_scale` (1), each of the four pose latents one token with a `LearntPositionEmbedder`. `latent_share_transformer`: the four pose latents form one group that shares per-latent modules; `shape` stays alone |
| `ss_generator.yaml` embedders | `EmbedderFuser` (`use_pos_embedding learned`, `projection_net_hidden_dim_multiplier` 4.0, `freeze` true, `dropout_prob` 0.1 with the `[pointmap, rgb_pointmap]` group dropped at weight 1.0, training only). Three embedders: `Dino` `dinov2_vitl14_reg` (`input_size` 518, `normalize_images` true) for kwargs `image` (group `cropped`) and `rgb_image` (group `full`); a **second** `Dino`, same settings, for `mask` (`cropped`) and `rgb_image_mask` (`full`); `PointPatchEmbed` (`embed_dim` 512, `input_size` 256, `patch_size` 8, `remap_output linear`) for `pointmap` (`cropped`) and `rgb_pointmap` (`full`) |
| `slat_generator.yaml` | `FlowMatching` (no shortcut) over plain `ClassifierFreeGuidance` over `SLatFlowModelTdfyWrapper`: `model_channels` 1024, `num_blocks` 24, `num_heads` 16, `mlp_ratio` 4, `patch_size` 2, `resolution` 64, `pe_mode ape`, `qk_rms_norm` true, `io_block_channels` [128], `num_io_res_blocks` 2, `cond_channels` 1024, in and out 8, `force_zeros_cond` true, `use_fp16` true. Embedders: the two `Dino` instances above with `prenorm_features` true and no pointmap; no `d` embedder |
| `ss_decoder.yaml` | `SparseStructureDecoderTdfyWrapper`: `latent_channels` 8, `out_channels` 1, `channels` [512, 128, 32], `num_res_blocks` 2, `num_res_blocks_middle` 2, `reshape_input_to_cube` false (the pipeline reshapes the 4096x8 latent to 8x16x16x16 itself) |
| `slat_decoder_gs.yaml` | `SLatGaussianDecoderTdfyWrapper`: `resolution` 64, `model_channels` 768, `latent_channels` 8, `num_blocks` 12, `num_heads` 12, `mlp_ratio` 4, `attn_mode swin`, `window_size` 8, `use_fp16` true; representation: `num_gaussians` 32, `voxel_size` 1.5, `perturb_offset` true, `3d_filter_kernel_size` 0.0009, `2d_filter_kernel_size` 0.1, `scaling_bias` 0.004, `opacity_bias` 0.1, `scaling_activation softplus`, `lr` 1.0 for `_xyz`, `_features_dc`, `_opacity`, `_scaling` and 0.1 for `_rotation` |
| `slat_decoder_mesh.yaml` | `SLatMeshDecoderTdfyWrapper`: `resolution` 64, `model_channels` 768, `latent_channels` 8, `num_blocks` 12, `num_heads` 12, `mlp_ratio` 4, `attn_mode swin`, `window_size` 8, `use_fp16` true, `representation_config.use_color` true |

The Gaussian values match the constants `Trellis/TrellisGaussianRepresentation` already hard-codes (32, 1.5, 0.0009, softplus, 0.004, 0.1), so
the splat decoder needs no representation change. Not yet checked: the `lr` value 0.1 for `_rotation` against how the C# code scales rotation.

## Stage map (public pipeline `InferencePipelinePointMap.run`, defaults of `notebook/inference.py::Inference`)

The public `Inference.__call__` runs with `with_mesh_postprocess=False`, `with_texture_baking=False`, `with_layout_postprocess=False` and
`use_vertex_color=True`; `demo.py` writes only the Gaussian splat (`output["gs"].save_ply`). Output keys are the splat `gs`, the mesh `glb`,
and the pose (`rotation` quaternion w-first, `translation`, `scale`).

| # | Stage | Upstream (from source) | Existing C# | Gap |
|---|---|---|---|---|
| 0 | Inputs | RGB image, one binary mask per object; the mask is written into the alpha channel (`mask * 255`), image `uint8` | `Sam3Pipeline`, `Sam2MaskRefiner` produce masks; `SceneMeshPrompt.Masks` exists for CoLa3D | a request field for image + mask for this model; no mask is derived inside the model |
| 1 | Pointmap | `MoGe.infer(image, force_projection=False)` gives `points` (H, W, 3) and, if present, `intrinsics`; points are rotated into the PyTorch3D camera convention by `look_at_view_transform(eye (0,0,-1), up (0,-1,0))` rotation; otherwise intrinsics are recovered by `infer_intrinsics_from_pointmap` (MoGe `recover_focal_shift`) | none | **a MoGe port** (not specified here; needs its own dossier) or a stand-in; see "Layout decision" |
| 2 | Preprocess (shape stage) | explicit YAML preprocessor, **no `rembg`**. The pointmap is normalised first by `ObjectCentricSSI` (below) using the full mask. Cropped branch: `resize_all_to_same_size` (image, mask and pointmap to the image size), then `crop_around_mask_with_padding(box_size_factor 1.2, padding_factor 0.0)` (RGBA crop of the mask bbox scaled by 1.2, padded to a square with zeros, pointmap padded with NaN), then `pad_to_square_centered` and bilinear `Resize(518)` for the image, nearest for mask (`interpolation 0`) and pointmap. Full branch (`rgb_image`, `rgb_image_mask`, `rgb_pointmap`): no crop, only pad-to-square and the same resizes; `rgb_pointmap` uses the scale and shift of the cropped branch (the `rgb_pointmap_normalizer` is not set, so it falls back to the same normaliser) | `Trellis2ImagePreprocessor` is similar but not the same (square crop around alpha above 0.8, matte, RGB times alpha) | a SAM 3D preprocessor, pure tensor code, specified by the public source |
| 3 | Conditioning (shape) | `EmbedderFuser` with three embedders (YAML values in "Configs read"): `Dino` for `image` and `rgb_image`, a second `Dino` for `mask` and `rgb_image_mask`, `PointPatchEmbed` for `pointmap` and `rgb_pointmap`; each kwarg is one call, a `LayerNorm` + Llama-3 SwiGLU projection to 1024, a learned per-group (`cropped`, `full`) index embedding added, all tokens concatenated on the token axis. Expected token count (derived, not run): 4 DINO calls of 1370 (class token plus 37x37 patches at 518 px) plus 2 pointmap calls of 1024 = 7528 | `Dinov2VisionEncoder` (ViT with registers, LayerScale), `Dinov2ImagePreprocessor` | `PointPatchEmbed`, `PointRemapper` (`linear`), the fuser, the SwiGLU projection; the DINO weights come from the checkpoint with `torch.hub` key names (HF-style names are what the encoder reads): unverified |
| 4 | Structure generator | flow matching over a dict of five latents with a mixture-of-transformers (MOT) block (below): `shape` 4096 tokens of 8 channels (a 16^3 grid) plus four one-token pose latents; 24 blocks, 1024 wide, 16 heads | `Trellis/SparseStructureFlow` (single latent, patchify) and `TrellisV2/Trellis2SparseStructureFlow` | a MOT flow with the shortcut `d` embedder |
| 5 | Sampler | Euler, 25 steps, `rescale_t` 3, CFG strength 7 on every latent of the dict, interval [0, 500] (19 of 25 steps guided), `no_shortcut` true so `d = 0` is still fed (below) | `Trellis/TrellisSparseStructureSampler`, `TrellisV2/Trellis2FlowSampler` | the time convention differs (t runs 0 to 1 here); new sampler class |
| 6 | Structure decode | `ss_decoder(latent.permute(0,2,1).view(B,8,16,16,16))`, `coords = argwhere(ss > 0)`, then **`prune_sparse_structure(max_neighbor_axes_dist 1)`** (`downsample_ss_dist` is 1), then `downsample_sparse_structure` if more than 42000 voxels | `Trellis/SparseStructureDecoder` (same class lineage, 8 to 64^3 occupancy) | the prune and downsample rules |
| 7 | Pose decode | closed-form from the pose latents and the pointmap statistics (below) | none | new, pure math |
| 8 | SLAT flow | sparse DiT over the pruned voxels, 25 Euler steps, `rescale_t` **1**, CFG strength **1** (`2 cond - uncond`), interval [0, 500] (13 of 25 steps guided: scaled `t = 40 k`, `k = 0 .. 12`); conditioned on the image and mask (no pointmap) with two `prenorm_features` DINO calls per kwarg pair; no `d` embedder (derived token count 4 x 1374, since `x_prenorm` keeps the four register tokens) | `Trellis/SlatFlowModel` (the TRELLIS v1 flow: sparse res-blocks, APE, shared structure), `TrellisV2/Trellis2SlatFlow` (RoPE, different) | a DINO path that returns the pre-norm tokens including registers, then a plain `layer_norm`; the key and shape match are unverified |
| 9 | Denormalise | `slat * slat_std + slat_mean` with the **`pipeline.yaml`** values (constants below), not the code constants | `TrellisSlatSampler` does the v1 constants | new constants |
| 10 | Gaussian decode | `SLatGaussianDecoder` (swin-window sparse transformer, 8 channels to per-voxel gaussians) | `Trellis/SlatGaussianDecoder`, `TrellisGaussianRepresentation` | none expected: the YAML representation values equal the C# hard-coded ones (verify against the checkpoint) |
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
- Values that run: see "Configs read" (structure: 25 steps, `rescale_t` 3, strength 7; SLAT: 25 steps, `rescale_t` 1, strength 1; interval [0, 500]
  for both; `dtype` float16). The public `Inference` wrapper passes `stage1_inference_steps=None`, so the constructor values stand.
- CFG: `(1 + s) y_cond - s y_uncond`, applied by a pytree map to **every tensor of the latent dict** with the same scalar strength, so the four pose
  latents are guided as well as `shape`. The strength applies only while `interval[0] <= t <= interval[1]` with `t` the **scaled** time (0 to 1000);
  outside it only the conditional pass runs. Structure, `rescale_t` 3: `t = 500` is reached at `u = 0.75` of the uniform grid, so guidance covers the
  first 19 of 25 steps (`u = k / 25`, `k = 0 .. 18`). SLAT, `rescale_t` 1: scaled `t = 40 k`, so `k = 0 .. 12`, 13 steps.
- The unconditional branch is `unconditional_handling = "add_flag"`: the backbone receives `cfg=True`. Both generator YAMLs set `force_zeros_cond: true`, and both wrappers
  (`mot_sparse_structure_flow.py`, `structured_latent_flow.py`) then compute the fused condition tokens as usual and multiply the whole tensor by
  zero, so the unconditional pass is "all condition tokens zero", cross-attention keys and values included; the embedders still run.
- Pointmap guidance (`PointmapCFG`: `y_cond + s_pm (y_cond - y_nopm) + s (y_nopm - y_uncond)`, three passes) exists in the source but the released
  `ss_generator.yaml` selects `ClassifierFreeGuidanceWithExternalUnconditionalProbability`, a plain two-pass CFG, and `ss_cfg_strength_pm` is 0 and
  unused by that class. **The released configuration does not need pointmap guidance**; implement it only if a later checkpoint selects it.
- Shortcut model (`ShortCut`, arXiv 2410.12557): the structure generator is a `ShortCut` with `is_shortcut_model: true`, so the backbone takes an extra
  step-size `d` embedded by a second `TimestepEmbedder` (`freeze_d_time_embedder` true) and added to the time embedding. With `no_shortcut = True` (the
  pipeline default when `use_stage*_distillation` is false) `d = 0`, **but the `d` embedder is still evaluated at 0 and added** (`_generate_dynamics`
  always passes `d`, scaled by 1000), so plain sampling needs that term. With the distillation flag, `d = 1 / steps`, `strength = 0` (no CFG) and fewer
  steps are intended; the public wrapper never turns that on. The SLAT flow is a plain `FlowMatching` and has no `d`.
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
  `translation_scale`. The released YAML uses `shape` (4096 x 8), `6drotation_normalized` (1 x 6), `translation` (1 x 3), `scale` (1 x 3) and
  `translation_scale` (1 x 1), all model width 1024. The pose latents are the `latent_share_transformer` group. `ScaleShiftInvariant` does not read
  `translation_scale` (its `x_translation_scale` is a constant one); only the `ScaleShiftInvariantWTranslationScale` variant does.

### Condition embedders (`embedder_fuser.py`, `pointmap.py`, `point_remapper.py`, `dino.py`)

- `Dino`: `torch.hub.load("facebookresearch/dinov2", dino_model)`, configured `dinov2_vitl14_reg` with `input_size` 518 and `normalize_images` true,
  ImageNet mean and std, bilinear resize, output either `[x_norm_clstoken, x_norm_patchtokens]` (structure stage) or, with
  `prenorm_features` (SLAT stage), `F.layer_norm(x_prenorm)` without affine over all pre-norm tokens, which in a registers model include the registers. `DinoForMasks` embeds the mask through the same backbone (a 1-channel mask is repeated to
  three channels).
- `PointPatchEmbed` (`input_size` 256, `patch_size` 8, `embed_dim` 512 and `remap_output linear` as configured; the class default `embed_dim` is 768): pointmap resized by **nearest**
  to `input_size`; invalid (non-finite) points set to 0, remapped, projected by `Linear(3, D)` and replaced by the learned `invalid_xyz_token`; windows
  of `patch_size^2` points get a prepended CLS token plus `pos_embed_window`, pass one timm `Block` (16 heads, mlp 2.0, `qkv_bias`, LayerNorm eps 1e-6),
  and the window's CLS output plus a bilinear-interpolated `pos_embed` gives one token per window (`(input_size / patch_size)^2` = 1024 tokens).
  `dropout_prob` and `force_dropout_always` swap tokens for a learned `dropped_xyz_token`.
- `PointRemapper` (default `exp`): `xy / (1 + z)`, `log1p(z)`. Others: `linear`, `sinh`, `sinh_exp`, `exp_disparity`. Configured: `linear`.
- `EmbedderFuser`: per embedder `LayerNorm` then a Llama-3 `FeedForward` (`hidden = int(2/3 * multiplier * D)` rounded up to a multiple of 256,
  `silu(w1 x) * w3 x` then `w2`, no biases; `projection_net_hidden_dim_multiplier` default 4.0); a learned `idx_emb` per positional group is added;
  tokens are concatenated on the token axis (or channel-concatenated and projected if `compression_projection_multiplier > 0`, not set in the YAML). `force_drop_modalities`
  multiplies the named groups' tokens by 0. The fuser width `embed_dims` is the largest embedder width (1024, DINO ViT-L); the 512-wide pointmap tokens are
  lifted by their own projection net. `idx_emb` has one row per positional group plus one: groups `cropped` and `full` give 3 x 1024. Weights `idx_emb`,
  `projection_nets.*`, the embedders' own keys: key names are known, shapes are derived here and unverified against a checkpoint.

### Pointmap normalisation and pose decode (`img_and_mask_transforms.py`, `pose_target.py`, `inference_utils.py`)

- Configured normaliser: `ObjectCentricSSI` (`use_scene_scale` true, `allow_scale_and_shift_override` true, `scale_factor` 1.0, `normalize_pointmap` true).
  The mask is resized to the pointmap by nearest; `shift` is the per-axis `nanmedian` of the **masked** points (all three axes, not only z); with
  `use_scene_scale` true `scale = nanmedian` over **all pixels** of `max_axis |pointmap - shift|` (the scene, not the object), the same value on the
  three axes; `normalized = (pointmap - shift) / scale`. If the mask holds no finite point, scale is `scale_factor` and shift 0. The cropped branch is
  normalised with the full mask first, then cropped; `rgb_pointmap` reuses that scale and shift through the override flag. The unnormalised full
  pointmap is kept for the layout step. (The class default `SSIPointmapNormalizer`, `shift = (0, 0, nanmedian(z))` and `scale = nanmean(|p - shift|)`, is
  not what the released config selects.) `_apply_metric_to_ssi` applies the inverse of `ScaleShiftInvariant.ssi_to_metric(scale, shift)` (scale, then translate),
  which is the `(p - shift) / scale` above.
- Rotation: `6drotation_normalized * ROTATION_6D_STD + ROTATION_6D_MEAN` (constants below), then Gram-Schmidt:
  `b1 = normalize(a1)`, `b2 = normalize(a2 - (b1 . a2) b1)`, `b3 = b1 x b2`, matrix columns `[b1 b2 b3]`, converted to a quaternion (w first).
  `scale` and `translation_scale` latents pass through `exp`. Then the convention class (`ScaleShiftInvariant`, `ApparentSize`,
  `DisparitySpace`, `ZeroPredictionScaleShiftInvariant`; `default` returns an empty pose; configured: `ScaleShiftInvariant`) turns the
  targets and the pointmap `scale`/`shift` into the object-to-camera `scale` (mean over axes, expanded to three), `translation` and `rotation`.
  `ScaleShiftInvariant`: compose scale, rotation, translation, then post-compose with `ssi_to_metric(scene_scale, scene_shift)` (scale then translate).
  `DisparitySpace`: `translation_xy = (xy + shift_xy) * z * exp(shift_z_log)`, `scale = s * z * exp(shift_z_log)`.
  After decode the pipeline multiplies `scale` by `downsample_factor` (1 or 2, from the voxel count rule).
- `prune_sparse_structure(coords, max_neighbor_axes_dist=1)` runs first because `downsample_ss_dist` is 1: build a dense occupancy grid over the
  coordinate bounding box, count occupied voxels in each voxel's `(2d + 1)^3` neighbourhood (box filter, zero padding) and keep only voxels whose
  count is below `(2d + 1)^3 = 27`. Interior voxels with a full neighbourhood are dropped, so the SLAT flow and decoders see a **surface shell**,
  not the filled volume (`coords_original` keeps the full set). A voxel on the bounding-box face counts as surface because of the zero padding.
- `downsample_sparse_structure(coords, max_coords=42000, factor=2)`: if at most 42000 voxels, unchanged; otherwise rescale the coordinates
  about the bounding box centre to half extent, round, clamp, deduplicate, and if still above the cap keep a random subset. The comment ties 42000
  to `max(int32) / (64 * 768)` and to the mesh decoder only.
- SLAT denormalisation, **from `pipeline.yaml`** (these override the constructor defaults): `slat_mean` = [0.12211431, 0.37204156, -1.26521907, -2.05276058, -3.10432536, -0.11294304, -0.85146744, 0.45506954];
  `slat_std` = [2.37326008, 2.13174402, 2.2413953, 2.30589401, 2.1191894, 1.8969511, 2.41684989, 2.08374642]. The code constants in `inference_utils.py` are different
  (`SLAT_MEAN[0]` -2.1687545776367188, `SLAT_STD[0]` 2.377650737762451, ...) and are not what the released pipeline uses; do not copy them.
- `ROTATION_6D_MEAN` = [-0.06366084883674913, 0.008438224692279752, 0.00017084786438302483, 0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688];
  `ROTATION_6D_STD` = [0.6656971967514863, 0.6787012271867754, 0.30345010594844524, 0.4394504420678794, 0.39817973931717104, 0.6176286868761914].
- `ScaleShiftInvariant` training normalisation (applied only when `normalize=True`, which the pipeline never passes): scale mean/std 1.0232692956924438 / 1.3773751258850098;
  translation mean (0.003191213821992278, 0.017236359417438507, 0.9401122331619263), std (1.341888666152954, 0.7665449380874634, 3.175130605697632).

### Decoders (`decoder_gs.py`, `decoder_mesh.py`, `base.py`, `gaussian_model.py`, `cube2mesh.py`, `sparse_structure_vae.py`)

- `SparseStructureDecoder`: `Conv3d(latent, channels[0], 3)`, `num_res_blocks_middle` ResBlocks, per stage `num_res_blocks` ResBlocks and a
  conv-then-pixel-shuffle x2 upsample, `ChannelLayerNorm32 / SiLU / Conv3d(channels[-1], out, 3)`; ResBlock `norm, SiLU, Conv3d, norm, SiLU, zero-init Conv3d`
  plus skip. The default class arguments (`layer` norm, 2 middle blocks) match what `Trellis/SparseStructureDecoder` reads; configured: 8 latent channels,
  channels [512, 128, 32], 2 blocks per stage and 2 in the middle, 1 output channel (the same shape family as the TRELLIS v1 decoder; key names unverified).
- `SparseTransformerBase` (decoder trunk): `SparseLinear(in, C)`, APE added, blocks with `attn_mode` `swin` (windowed, `window_size` 8 default,
  odd blocks shifted by `window_size / 2`) or the serialised modes; this is the TRELLIS v1 decoder trunk.
- `SLatGaussianDecoder`: per voxel `num_gaussians` gaussians; output layout `_xyz [G,3]`, `_features_dc [G,1,3]`, `_scaling [G,3]`, `_rotation [G,4]`, `_opacity [G,1]`,
  each scaled by `lr[name]`; position `(coord + 0.5) / resolution + tanh(offset * lr + perturbation) / resolution * 0.5 * voxel_size`, where the
  perturbation is `atanh((hammersley(3, i, G) * 2 - 1) / voxel_size)` when `perturb_offset`. Gaussian activations: scale `sqrt(act(raw + bias)^2 + kernel^2)`
  with `act` `exp` or `softplus` and `bias = inverse_act(scaling_bias)`; rotation `normalize(raw + [1,0,0,0])`; opacity `sigmoid(raw + logit(opacity_bias))`;
  xyz in the aabb `[-0.5, 0.5]^3`. `num_gaussians`, `voxel_size`, `3d_filter_kernel_size`, `scaling_bias`, `opacity_bias`, `scaling_activation`, `lr`,
  `perturb_offset` and `resolution` come from `slat_decoder_gs.yaml` (values in "Configs read": 32, 1.5, 0.0009, 0.004, 0.1, softplus, perturb on, resolution 64);
  `Trellis/TrellisGaussianRepresentation` hard-codes the same TRELLIS v1 values (32, 1.5, 0.0009, softplus, 0.004, 0.1).
- `SLatMeshDecoder`: trunk, then `SparseSubdivideBlock3d(C -> C/4, resolution)` and `(C/4 -> C/8, 2 * resolution)`, `SparseLinear(C/8, feats_channels)`;
  `SparseFeatures2Mesh(res = 4 * resolution)`: per-voxel channels `sdf [8]`, `deform [8,3]`, `weights [21]`, and with `use_color` `color [8,6]`; `sdf += -1/res`;
  corners are merged to a dense grid, FlexiCubes extracts the mesh with `beta = weights[:12]`, `alpha = weights[12:20]`, `gamma_f = weights[20]`. At `resolution` 64 the
  extraction grid is 256^3. Configured: resolution 64, `use_color` true.
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

1. Flow-matching sampler with the 0-to-1 time convention, `rescale_t` map, scaled-time CFG interval applied to every latent in the dict, and the `d` input fed as 0 (a new `SamFlowSampler`; extract to a shared base only if the TRELLIS.2 sampler can take the same contract). Pointmap CFG is not needed by the released config.
2. MOT block and multi-latent flow (per-latent modules, protected-modality attention, the `d` timestep embedder).
3. `PointPatchEmbed`, `PointRemapper` (`linear` only), `EmbedderFuser` with the SwiGLU projection; a DINO path that can return the pre-norm tokens with registers.
4. SAM 3D preprocessor (mask-box crop with factor 1.2, pad, resize, NaN-filled pointmap crop) and the `ObjectCentricSSI` normaliser.
5. Pose decoder (6D rotation, `ScaleShiftInvariant`), `prune_sparse_structure` and `downsample_sparse_structure`.
6. A Gaussian representation needs no new constants (the YAML values equal the existing hard-coded ones); only the `lr` 0.1 for rotation needs checking against the C# code.
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
(the configured compute dtype is float16, which would put the generators near 3.3 B and 2.5 B parameters if the files are 16-bit weights only, or about half that if F32;
the generator files also hold the embedders, DINO included, and a Lightning `.ckpt` can carry optimiser moments or EMA copies as well, in which case file size overstates the
parameter count, so treat these figures as upper bounds until the tensor list is read).
The 12 GB RTX 3060 is a later concern; the existing pattern (each flow read from disk for its stage and released, as for TRELLIS.2) is the likely starting point.

## What the user must provide

The configs are read (see "Configs read"). What remains is the checkpoints, which are the user's decision and action; this repository cannot do any of it.

1. **A source for the weights.** Either (a) Hugging Face access to `facebook/sam-3d-objects`: submit the gated form (name, date of birth, country,
   affiliation, job title, licence acceptance) with the user's own account and wait for Meta's manual approval (refused for comprehensively sanctioned
   jurisdictions), plus a read token supplied for the download step (never committed, never logged); or (b) an explicit yes to fetch the same files
   from an ungated mirror. `jetjodh/sam-3d-objects` (`f29e43c1`), `tuandao-zenai/sam-3d-objects` and `asdafdfafffg/sam-3d-objects` list every
   `.ckpt` with the same size and LFS sha256 as Meta's listing, so a download can be verified byte for byte against the table below. A mirror
   sidesteps Meta's form, not the SAM License: its terms bind on use and on any redistribution. Third-party safetensors conversions
   (`apozz/sam-3d-objects-safetensors`) exist but have not been compared with Meta's files and are not a substitute for the sha256 check.
2. **The files** (public listing at revision `2e73555018d2741ccd486e56c24fac41155a1dc6`; sizes in bytes, sha256 is the Hugging Face LFS oid):

| File | Size | sha256 | Needed for |
|---|---|---|---|
| `checkpoints/pipeline.yaml` | 3,548 | (not LFS) | read; every pipeline-level value |
| `checkpoints/ss_generator.yaml` | 5,076 | (not LFS) | read; structure generator and condition-embedder config |
| `checkpoints/ss_generator.ckpt` | 6,690,136,964 | `225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146` | structure generator + condition embedders |
| `checkpoints/slat_generator.yaml` | 1,986 | (not LFS) | read; SLAT flow config |
| `checkpoints/slat_generator.ckpt` | 4,906,537,684 | `91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb` | SLAT flow + its condition embedder |
| `checkpoints/ss_decoder.yaml` / `.ckpt` | 244 / 147,609,242 | `6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340` | occupancy decoder |
| `checkpoints/slat_decoder_gs.yaml` / `.ckpt` | 576 / 171,476,155 | `f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111` | Gaussian decoder (the default output) |
| `checkpoints/slat_decoder_mesh.yaml` / `.ckpt` | 300 / 363,726,862 | `85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33` | mesh decoder (second pass; `slat_decoder_mesh.pt` is a 363,728,714-byte duplicate, skip it) |
| `LICENSE` | 8,204 | (not LFS) | ship with any repack |

   Minimum for a first splat: the first seven rows through `slat_decoder_gs`, **about 11.92 GB** (11,915,771,475 bytes with the five YAMLs); with the mesh decoder about 12.28 GB.
   Not needed for inference: `slat_decoder_gs_4` (170,269,801), `ss_encoder.*` (119,085,402 plus a `.safetensors` the listing shows with size 0), `slat_encoder.ckpt` (173,263,986); the encoders are for
   latent-space work (`notebook/demo_ss_encoder.ipynb`) and are out of scope.
3. **The MoGe checkpoint**, `Ruicheng/moge-vitl` per `pipeline.yaml` (size and licence not read), only for the pose path.
4. **A destination and an explicit approval to download.** The expected destination is `Models/3D/sam3d-objects/` with the `LICENSE` beside the files. Nothing further is
   downloaded until the user approves with the repository, files, sizes, sha256, licence and purpose in front of them.

**Pickle handling.** All released weights are Lightning `.ckpt` files and the upstream loader calls `torch.load(..., weights_only=False)`. This repository must never execute a pickle:
the files go through the existing safe `PytorchPickleLoader` (a pickle machine that builds tensors without running arbitrary callables) and are repacked to safetensors by
`PickleCheckpointRepacker` (its `recursiveFlatten` and a key map that returns `null` to drop a key already cover a Lightning top-level `state_dict` and a prefix filter). A `.ckpt` also carries non-tensor state (hyper-parameters, optimiser state); the repack keeps `state_dict` entries under the prefixes used by the pipeline
(`_base_models.generator.`, `_base_models.condition_embedder.`) and drops the rest. Whether the loader copes with these particular files is unverified. The `LICENSE`'s redistribution rule applies to any repacked copy the user shares.

## Validation plan (when weights exist)

1. Done: the YAMLs are read and the config items above are filled. Next, with the checkpoints: fill the remaining *unverified* items (key names, shapes, stored dtype) and update this file before writing the loaders.
2. Key map per checkpoint, rejecting missing, extra and mis-shaped tensors (the TRELLIS.2 pattern), then a header-only test (`Sam3dRealWeightsTests`).
3. Reference dumps from the upstream Python on the CPU or a rented GPU (the same noise, voxel set and condition tokens saved as inputs): condition tokens for each embedder, one structure-flow velocity at the schedule's
   first non-trivial `t` (cond and uncond), the SS decode occupancy, one SLAT-flow velocity, the Gaussian decode, the pose. Compare by correlation and relative L2 as in the Hunyuan3D-2.1 rows; share the saved inputs, not seeds.
4. A rendered look of the splat from several angles (needs a splat viewer; there is no rasteriser in the engine) and a mesh render for the FlexiCubes branch.
5. Only then a status row above 🔧; ✅ needs real-weight parity and a rendered look.

## Checkpoints read (2026-11-11, from the sha256-verified ungated mirror)

Read through `SamCheckpoint` (the safe-subset pickle reader; nothing from the files is executed). The four `.ckpt` files below match the sizes and hashes in the table above; `slat_decoder_mesh.ckpt` was not fetched (its safetensors conversion was, see below).

| File | Tensors | dtype | Key layout |
|---|---|---|---|
| `ss_generator.ckpt` | 1741 | all F32 | envelope `state_dict.`, then `_base_models.generator.reverse_fn.backbone.` (the flow: 945 tensors), `_base_models.condition_embedder.` (722) and `_base_models.decoder.` (the occupancy decoder, bundled; `ss_decoder.ckpt` has the same layout; the two were not compared tensor by tensor) |
| `slat_generator.ckpt` | 1225 | all F32 | same envelope: `generator.reverse_fn.backbone.` (526) and `condition_embedder.` (699: two DINO towers plus `idx_emb` and two `projection_nets`; no pointmap embedder) |
| `ss_decoder.ckpt` | 74 | 38 F32 (norms), 36 F16 (convolutions) | no envelope, no prefix: `input_layer`, `middle_block.i`, `blocks.i`, `out_layer.{0,2}` |
| `slat_decoder_gs.ckpt` | 101 | 5 F32, 96 F16 | no envelope, no prefix: `blocks.i.{attn,mlp,...}` |

Confirmed against the files:

- Structure flow keys equal what `SamMotBlock`, `SamLatentProjection` and `SamTimestepEmbedder` read (streams `shape` and `6drotation_normalized`, `latent_mapping.{name}.pos_emb` of `[4096, 1024]` for the shape grid and `[1, 1024]` per pose latent, `d_embedder`, per-head `gamma` `[16, 64]`, `adaLN_modulation` per block so `share_mod` is false). `SamMotFlow`, `SamEmbedderFuser` (hidden 2816, `idx_emb` `[3, 1024]`) and `SamPointPatchEmbed` (under `condition_embedder.module_list.2.`) construct from the real tensors with the released configs, which checks every key and shape they read.
- The condition embedders sit under `condition_embedder.module_list.{0,1}.backbone.` (the two DINOv2-L towers, F32, `torch.hub` names with `ls1`/`ls2` layer scale, registers, `pos_embed` `[1, 1370, 1024]`) and `module_list.2.` (the pointmap embedder). The DINO weights are therefore in the checkpoint and need no `torch.hub` fetch.
- The SLAT flow keys (`input_layer`, `input_blocks.i`, `blocks.i`, `out_blocks.i`, `out_layer`, `t_embedder`; spconv convolutions `[out, 3, 3, 3, in]`; no `d_embedder`, no position-embedder weights) load into the existing `Trellis/SlatFlowModel`; `Trellis/SparseStructureDecoder` and `Trellis/SlatGaussianDecoder` load `ss_decoder.ckpt` and `slat_decoder_gs.ckpt` without a key map. This shows the keys are present; it does not show the outputs agree.
- A conversion to safetensors should keep or drop the envelope and prefixes (`SamCheckpointKeys` accepts both); the generators are F32 and the decoders mixed, so only the decoders need widening before an F32-only module reads them.

Still unverified: every number (no reference dumps exist), the MoGe checkpoint's use, and behaviour with the pointmap group dropped.

### The safetensors set (`apozz/sam-3d-objects-safetensors`, revision `cbb995af`, fetched 2026-11-11)

Ungated third-party conversion, placed in `Models/3D/sam3d/` with `SAM3D_LICENSE` beside it (the SAM License binds on use and on redistribution, including through this mirror). Every file's size and sha256 was checked against the Hugging Face tree listing.

| File | Bytes | sha256 | Tensors, dtypes |
|---|---|---|---|
| `ss_generator.safetensors` | 6,689,521,984 | `9f0fd0a7e288eadf7e51b6ead30f70aea0cd2ac4fe4160ee1deb412916bcfa0d` | 1741, F32 |
| `slat_generator.safetensors` | 4,906,129,160 | `ee0c1601f3259f61cb22030bad839a3fe30a95b5f78bc9451babdbd8836fd06d` | 1225, F32 |
| `ss_decoder.safetensors` | 147,591,972 | `f192156b047dc467f82f8efa851157c32ad9d7a008f01bc92c846f5ef0a4293f` | 74, 38 F32 + 36 F16 |
| `slat_decoder_gs.safetensors` | 171,450,952 | `48a3dc3271f56efe9c8b1e601d0350d9ab7914fec0e635048682aa8f7c9e86be` | 101, 5 F32 + 96 F16 |
| `slat_decoder_mesh.safetensors` | 363,718,540 | `51f27a1c7a7892eff9576cf73bb888a255d4a184a76a9bfd9675f0d9ba464f5a` | 120, F32 |
| `dinov2_vitl14_reg.safetensors` | 1,217,523,408 | `30e20dce587ad621a8dfc20e4ed66198d2998974928d44f06a6baf7732503dcc` | 344, F32 |
| `moge_vitl.safetensors` | 1,256,740,032 | `dbdf89d1f7651b9213a507923b4f100fed87489bb7425fda7f7d430bcb31d8a9` | 421, F32 |

Compared against Meta's `.ckpt` files (the four above that exist locally, loaded through `SamCheckpoint`, nothing executed): every tensor of `ss_generator`, `slat_generator`, `ss_decoder` and `slat_decoder_gs` is present with the same shape and dtype and **byte-identical contents** (3141 tensors). The generators drop the Lightning envelope (`state_dict._base_models.`), the decoders keep their unprefixed names. `dinov2_vitl14_reg.safetensors` is byte-identical to **both** DINO towers bundled in `ss_generator.ckpt` (`condition_embedder.module_list.{0,1}.backbone.`, 344 tensors each), so the two towers hold the same weights. This shows the conversion is lossless for those files; it says nothing about whether the C# forward matches upstream.

Measured on the RTX 3060 (12 GB, CUDA backend, 2026-11-11, one process): weights open from `ss_generator.safetensors` 11 s (1741 tensors, F32), the point-patch embedder on a synthetic 256 x 256 pointmap with invalid corner pixels 0.56 s (output `[1024, 512]`, finite), and one structure-flow forward (24 blocks, 4096 shape tokens, 4764 synthetic condition tokens in place of the unwired DINO towers, `d` 0) 2.5 s cold and 2.4 s warm with finite velocities for all five latents. VRAM read with `nvidia-smi` after each stage: about 1.2 GB before loading, 3.0 GB after the first forward and 6.7 GB after the second (device total, the desktop's 1 GB included; not a peak trace). Roughly 125 s for 25 CFG steps at two forwards each, an estimate, not a run. These numbers show the real tensors run through the flow without error; the velocities are not compared with anything.

Conditions on the real weights (2026-11-11, synthetic image, mask and pointmap; `SamStageCondition`): the structure condition (two DINO towers on the `forward_features` path, the point-patch embedder, the fuser) gives `[1, 7528, 1024]` (4 x 1370 DINO rows plus 2 x 1024 pointmap rows, the derived count) and the SLAT condition (`prenorm_features` path with the registers kept) `[1, 5496, 1024]` (4 x 1374), all finite. The structure flow also ran on the real condition, finite. The GPU was shared with other sessions' jobs during this run (99 % utilisation, 10.6 GB used in total at the peak), so the times are inflated and are not a benchmark: structure condition 47.8 s cold and 14.9 s warm, one flow forward on 7528 condition tokens 36 s (2.5 s on an idle GPU with 4764 tokens), SLAT condition 3.7 s cold and 1.7 s warm. Still no reference dump, so the tokens are not compared with upstream; the DINO wrapper itself is checked against a naive double-precision ViT on a tiny configuration.

Not compared with a Meta file: `slat_decoder_mesh.safetensors` (the `.ckpt` was not fetched; header names match the `blocks.i.attn`, `blocks.i.mlp`, `input_layer`, `out_layer` and `upsample.i` layout of the YAML's 12 swin blocks at width 768) and `moge_vitl.safetensors` (backbone plus a multi-scale head, with `image_mean` and `image_std`; its licence is not recorded yet).

## Open items

- Settled by reading the checkpoints (see "Checkpoints read"): per-block `adaLN_modulation`, stored dtypes, key prefixes, safe extraction of the `state_dict`, and that the DINO and pointmap embedder weights are in the checkpoints.
- The licences of `Ruicheng/moge-vitl`, MoGe and `tables.py`, and the MoGe checkpoint's files.
- How the structure generator behaves with the pointmap tokens zeroed (object-only mode).
- Whether the release's quality needs the 25-step CFG schedule at all on the 12 GB target, and the cost of the shape stage when the 1024-token pointmap group and the image groups are all present.
