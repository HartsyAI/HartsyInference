# Pixal3D architecture (source spike)

Research dossier for `TencentARC/Pixal3D` (SIGGRAPH 2026, arXiv 2605.10922): single image to mesh with PBR texture, built on the TRELLIS.2
backbone. Written before any C# exists, to decide what the existing TRELLIS.2 code (see [TRELLIS2_ARCHITECTURE.md](TRELLIS2_ARCHITECTURE.md))
covers. Everything below was read from the pinned sources and the safetensors headers (JSON only, no tensor data, no pickle executed).

## Pins

| Item | Revision |
|---|---|
| Code `TencentARC/Pixal3D` (GitHub, default branch `master`) | commit `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (2026-09-01, "support multi-view image input") |
| Weights `TencentARC/Pixal3D` (Hugging Face) | `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f` (2026-08-31, not gated) |
| TRELLIS.2 code used for the diff | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` (the pin of the TRELLIS.2 dossier) |
| NAF upsampler `valeoai/NAF` | commit `37f2dfc180f2de53d98bd601109c0da0dd6b0f43` (2026-04-10); weights release asset `naf_release.pth` |
| DINOv3 mirror `camenduru/dinov3-vitl16-pretrain-lvd1689m` | `3c276edd87d6f6e569ff0c4400e086807d0f3881` |

The Hugging Face `main` branch is the TRELLIS.2-based version (this document). A `paper` branch, built on Direct3D-S2, reproduces the paper
and is not covered.

## Licence

- Pixal3D code and weights: MIT, "Copyright (c) 2026 Tencent" (`LICENSE`, and `license: mit` in the Hugging Face front matter). The `NOTICE`
  adds a non-binding responsible-use paragraph (no non-consensual or sexualized content, no minors, no impersonation) that "does not modify
  the license terms".
- Third-party components named in `NOTICE`: dinov2 (Apache-2.0), TRELLIS.2, Direct3D-S2 and MoGe (MIT).
- DINOv3 weights keep Meta's DINOv3 License (custom, gated upstream); see the TRELLIS.2 dossier. The `camenduru` mirror used by the Pixal3D
  scripts is marked `gated: false` but still carries `license: other / dinov3-license` and `LICENSE.md`.
- NAF: code Apache-2.0 per the GitHub licence file; `src/layers/rope.py` carries a Meta DINOv3 License header (copied from DINOv3). The
  licence of the `naf_release.pth` weights is not stated separately; treat it as Apache-2.0 with that caveat until clarified.
- MoGe-2 (`Ruicheng/moge-2-vitl`, `model.pt`): `license: mit` in the Hugging Face metadata. Not needed (see [MoGe](#moge)).

## What Pixal3D changes

Pixal3D keeps TRELLIS.2's three-stage cascade (structure flow, 512 shape flow, 1024 shape flow, texture flow, same decoders) and changes how
the image reaches the flows. TRELLIS.2 cross-attends every flow block to all DINOv3 tokens. Pixal3D instead:

1. Estimates a pinhole camera (field of view and distance) for the input image.
2. Projects every cell of the flow's 3D grid into the image and samples DINOv3 features there (view-aligned "back-projection"), so each
   voxel token receives the feature of the pixel it lands on.
3. Adds that per-voxel feature to the block's cross-attention output, `cross_attn(norm2(x), global) + proj_linear(proj)`; the cross-attention
   itself only sees the 5 global tokens (class + 4 registers).

No flow block changes anywhere else. All the inference-relevant TRELLIS.2 modules (sparse and dense blocks, RoPE, sparse conv, samplers,
both VAE decoders, preprocessing, `postprocess.to_glb`) are byte-identical to the TRELLIS.2 pin; the only inference edits are additive:
the `proj` image-attention mode, `flash_attn_4` and `sdpa` attention backends, and the dict-shaped conditioning in the CFG mixin.

## Stage map and reuse

| Stage | Upstream (Pixal3D) | Existing C# | Verdict |
|---|---|---|---|
| Preprocess | `preprocess_image`: long edge at most 1024, RMBG-2.0 unless alpha, crop around alpha > 0.8 with 10 % margin, RGB over **black** | `Trellis2ImagePreprocessor` + `IForegroundMatter` / `Trellis2LazyMatter` | 1:1 (same function as TRELLIS.2; black background is its default) |
| Camera | `distance_from_fov`, FOV from MoGe-2 or `--fov` | none | new, small (formulas below); MoGe not ported |
| Global tokens | DINOv3 ViT-L/16, all tokens after the last block, `F.layer_norm`, no final norm; keeps class + 4 register tokens | `Trellis2ImageConditioner.Encode` (same call returns every token) | 1:1; take the first 5 tokens |
| Patch grid | the remaining tokens as `[h, w, 1024]`, `h = w = size / 16` | the same tokens, row-major | 1:1, needs a view |
| NAF upsample | `valeoai/NAF`, `[1024, 32 or 64]^2` to `[1024, 512 or 1024]^2` guided by the RGB image | none | new (cross-scale neighbourhood attention; see [NAF](#naf)) |
| Projection | `ProjGrid`: grid points, pinhole projection, `grid_sample` (bilinear, border padding, `align_corners=False`) | none (`TriplaneGridSample` is a different op) | new: 2D bilinear gather op plus the camera math |
| Structure flow | `SparseStructureFlowModel`, `image_attn_mode="proj"`, `proj_in_channels` defaults to 1024 | `Trellis2SparseStructureFlow`, `Trellis2FlowBlock`, `Trellis2FlowConfig`, `Trellis2FlowKeyMap` | config change: proj branch in block + key map; checkpoint also holds a `rope_phases` buffer |
| Structure decode | `ss_dec_conv3d_16l8_fp16` | `SparseStructureDecoder` | 1:1, same file (sha256 `1c76d4a4...`) |
| Shape flows (512, 1024) | `ElasticSLatFlowModel` (inference identical to `SLatFlowModel`), proj mode, `proj_in_channels` 2048 | `Trellis2SlatFlow` | config change (as above) |
| Cascade | LR flow, `upsample(4)`, quantize, token cap 49152, HR flow | `ShapeVaeDecoder.Upsample`, `Trellis2Cascade` | config change: Pixal3D's quantizer differs (formulas below) |
| Shape decode, mesh | `FlexiDualGridVaeDecoder`, `fill_holes`, dual-grid mesh | `ShapeVaeDecoder`, `DualGridMesherStage` | 1:1, same file (sha256 `e3b718d3...`) |
| Texture flow | `imgshape2tex` flow, 64 input channels (shape latent concatenated), 1024 | none (the TRELLIS.2 texture path is being added by another session) | out of scope for shape-only; same block as the shape flow once proj is supported |
| Texture decode, bake | `SparseUnetVaeDecoder`, `to_glb` with remesh and texture baking | none yet; [MESH_POSTPROCESS.md](MESH_POSTPROCESS.md) covers the toolkit | later |
| Samplers | `FlowEulerGuidanceIntervalSampler` | `Trellis2FlowSampler` | 1:1 (file identical to TRELLIS.2; `pipeline.json` sampler and normalization blocks equal the local TRELLIS.2 ones) |

### Shared with the local files

| Pixal3D file | Local file | Same bytes |
|---|---|---|
| `shape_dec_next_dc_f16c32_fp16.safetensors` | `Models/TRELLIS2-ShapeVAE/` | yes, sha256 `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581` |
| `ss_dec_conv3d_16l8_fp16.safetensors` | `Models/3D/trellis2/` | yes, sha256 `1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a` |
| DINOv3 `model.safetensors` (camenduru mirror) | `Models/DINOv3/` | yes, sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |
| `pipeline.json` | `Models/3D/trellis2/pipeline.json` | no (name, rembg and `default_pipeline_type` differ), but the two samplers and both normalizations are equal |

## Checkpoint layouts (safetensors headers)

All Pixal3D flow files are **F32** on disk despite the `bf16` in the name (`"dtype": "bfloat16"` in the sidecar is the runtime dtype). That is
why they are 5.5 GB where the TRELLIS.2 files are 2.6 GB: the parameter count is only about 7 % higher.

| Property | TRELLIS.2 flow | Pixal3D flow |
|---|---|---|
| Storage dtype | BF16 | F32 (`Trellis2FlowWeights` already widens through `TensorCasts.EnsureF32`, F32 is borrowed) |
| Tensors | 640 (10 model-level, 21 per block) | 700 (10 model-level, 23 per block) |
| Cross-attention keys | `blocks.N.cross_attn.{to_q,to_kv,to_out,q_rms_norm,k_rms_norm}` | `blocks.N.cross_attn.cross_attn_block.<same>` plus `blocks.N.cross_attn.proj_linear.{weight,bias}` |
| `proj_linear` | none | `[1536, 2048]` + `[1536]` in the three shape/texture flows; `[1536, 1024]` + `[1536]` in the structure flow |
| Sidecar `name` | `SLatFlowModel`, `SparseStructureFlowModel` | `ElasticSLatFlowModel`, `SparseStructureFlowModel`; extra args `image_attn_mode: "proj"`, `proj_in_channels` (2048; absent in the structure flow, so it defaults to `cond_channels` 1024) |
| Other tensors | none | structure flow only: `rope_phases`, dtype **C64** (complex64) `[4096, 64]`; `SafeTensorDTypes` has no C64 |
| Config | width 1536, 30 blocks, 12 heads, `mlp_ratio` 5.3334, RoPE, shared modulation, QK RMS norm | identical; `cond_channels` 1024 |

Texture flow: `in_channels` 64, `out_channels` 32, otherwise identical. `_mv` flows have exactly the same keys, shapes and sidecar as the
single-view ones; only the weights differ.

Files (revision `b0cb2e1b`, sizes in bytes, LFS sha256):

| File | Size | sha256 |
|---|---|---|
| `ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 5359822584 | `6d0147bb347eb61fa81c304cc68c7fd69d447e91870357a549b47e1d0fd77242` |
| `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | 5546764048 | `63569838c1da93f010e2fa805b202ecf802cc758830230abac202a81b70e5b45` |
| `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors` | 5546764048 | `3e6d1df0e62bc5bc863320d1d63d6b996235eef4abb8b851e878c88f316369f1` |
| `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors` | 5546960656 | `65df179ad02f9e92bc18335d7861fadd95f268e6249217352468dd83cb79d1d0` |
| `ckpts/tex_dec_next_dc_f16c32_fp16.safetensors` | 948458812 | `97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c` |
| `*_mv` variants of the four flows | same sizes | `ss` `e7ce6305...`, `512` `043a0d7f...`, `1024` `37f75c8f...`, `tex` `887fed61...` |
| `naf_release.pth` (GitHub `valeoai/NAF`, release `model`) | 2664431 | `c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f` (GitHub asset digest) |

The non-`_mv` Pixal3D set is 24.04 GB (the reported ~24 GB). The shape-only set that is not already on disk is the three flows, 16.45 GB, plus
NAF (2.7 MB); the texture flow and decoder add 6.5 GB.

## Camera and projection (from `inference.py` and `image_conditioned_proj.py`)

World frame is TRELLIS.2's Z-up cube `[-0.5, 0.5]^3`. For a camera field of view `fov` (horizontal) and a stage image size `S`
(512 or 1024):

- `f_pixels = S / (2 tan(fov / 2))` (upstream writes it through a 32 mm sensor, which cancels).
- `distance = f_pixels * x_world / x_ndc` with `x_world = -0.5 / mesh_scale` (the grid's `(-1, 0, 0)` corner) and `x_ndc = -S / 2`, which reduces
  to `distance = 1 / (2 mesh_scale tan(fov / 2))` for the default `extend_pixel = 0`: the cube's width exactly fills the image width at its centre
  plane. It does not depend on `S`. `mesh_scale` defaults to 1.
- The camera sits at `(0, -distance, 0)` looking along +y with +z up: camera x is world x, camera y is world z, depth is `y + distance`. The fixed
  front pose upstream calls `front_view_transform_matrix` is `[[1,0,0,0],[0,0,-1,-distance],[0,1,0,0],[0,0,0,1]]` (camera to world).
- Projection grid: `linspace(-1, 1, R)^3` (**corner-inclusive**, x-major flattening, so index `i` sits at `-1 + 2i/(R-1)`), rotated by
  `(x, y, z) -> (x, -z, y)`, divided by `2 mesh_scale`, then pinhole projected: `x_pix = f x_cam / depth + S/2`, `y_pix = -f y_cam / depth + S/2`.
- Sampling: normalized coordinate `(pix + 0.5) / S * 2 - 1`, `F.grid_sample(bilinear, align_corners=False, padding_mode="border")`. Because the
  coordinate is normalized, the same call samples the 32x32 patch map, the 64x64 patch map and the 512 or 1024 NAF map. Points outside the
  image are clamped to the border, not zeroed (`valid_mask` is computed but unused).
- Grid sizes `R`: structure 16 (dense, 4096 tokens, x-major, the dense flow's own order); shape 512 stage 32; shape 1024 stage `hr / 16`
  (64 for 1024, **96 for 1536**, overriding the extractor's default 64); texture the same as the HR shape stage. For the sparse flows the dense
  `[R, R, R, C]` volume is indexed at each token's `(x, y, z)`.
- Stage inputs: structure and shape-512 run DINOv3 at 512 (32x32 patches); shape-1024 and texture at 1024 (64x64 patches). NAF target size is 512
  for both shape stages and 1024 for the texture stage. The projected feature is `[lowres(1024) ; NAF(1024)]` = 2048 channels for
  shape and texture, 1024 (no NAF) for the structure stage.
- The negative (unconditional) condition is zeros for both parts, so the negative branch still evaluates `proj_linear(0) = bias`; a port must
  run the linear, not skip it.

The multi-view pipeline projects the same grid through each view's pose `F @ inv(C_0) @ C_i` and averages the per-view projected features and
the per-view global tokens (`multiview_fusion: "average"`, the only mode `inference_mv.py` configures), so it reuses the single-view flows
unchanged apart from the `_mv` weights. Cameras come from a `transforms.json` (Blender convention, `camera_angle_x`, `transform_matrix`).

## Cascade

Pipeline types are `1024_cascade` and `1536_cascade` only (no direct 512 or 1024 type; `default_pipeline_type` is `1536_cascade`, `inference.py`
uses 1536, or 1024 with `--low_vram`). The shape LR stage is always 512 (grid 32). Upsampling, the token cap (49152) and the retreat by 128 to a
floor of 1024 are TRELLIS.2's. The quantizer differs:

| | TRELLIS.2 | Pixal3D |
|---|---|---|
| HR voxel from an upsampled voxel `c` | `int((c + 0.5) / 512 * (res / 16))` | `round((c + 0.5) / 512 * (res / 16 - 1)).int()` |

`Trellis2Cascade.QuantizeUnique` implements the first in exact integer arithmetic; the second needs a parameterised form (and its tie
behaviour: `torch.round` rounds half to even). The corner-inclusive projection grid is why Pixal3D uses `res/16 - 1`.

## NAF (the `natten` dependency) <a id="naf"></a>

`natten` is **not** imported anywhere in Pixal3D's own code (the only mention is a wheel in `requirements-hfdemo.txt`). It comes in through
NAF, which Pixal3D's `DinoV3ProjFeatureExtractor` loads with `torch.hub.load("valeoai/NAF", "naf", pretrained=True)` when `use_naf_upsample` is
set, i.e. in the shape and texture stages (hence `proj_in_channels` 2048). The structure stage does not use NAF.

NAF (dim 256, 4 attention heads, 4 RoPE heads, kernel 9) is small:

- `image_encoder`: two conv stacks on the RGB image in `[0, 1]` (1x1 conv to 128 then 2 `EncBlock`s without residual connections with GroupNorm(8) and SiLU,
  and a 3x3 variant), concatenated to 256 channels, adaptive-average-pooled to the **output** size, then an axial RoPE
  (`periods` buffer, base 100; coordinates in `[-1, 1]`, `rescale_coords` is training-only).
- Queries are those encoded features; keys are the same map pooled to the low-resolution feature size and nearest-exact resized back; values are
  the DINOv3 patch features nearest-exact resized to the output size.
- `na2d(q, k, v, kernel_size=(9, 9), dilation=output / lowres)` with scale `head_dim^-0.5`: 2D neighbourhood attention over the output grid with
  a dilation of 16 (512 / 32 or 1024 / 64) or 8 (512 / 64). Output is the 1024-channel feature map, which `ProjGrid` samples.

The engine already has `IBackend.NeighborhoodAttention` (window slides inward at borders, NATTEN's rule, window depth/height/width over a
`[batch, T, H, W, heads, headDim]` tensor). Dilation `d` splits the map into `d * d` interleaved sub-grids and applies the same rule inside each;
that is a reshape into the batch axis (with `T = 1`) and back, valid because the maps are multiples of `d`. No new attention kernel is needed.
Cost to plan for: a 512 or 1024 output with 1024 channels is 1 GiB or 4 GiB in F32, so a port should evaluate NAF only where the projection
samples it, or in tiles.

`naf_release.pth` is a pickle. Never execute it; the repository has a safe-subset reader (`PytorchPickleLoader`, used for the UniRig checkpoint)
that is the intended way in, after the file is listed for approval. Its tensor names and shapes are therefore **not** verified here; the
module layout above comes from the source.

## MoGe <a id="moge"></a>

MoGe-2 (`Ruicheng/moge-2-vitl`, 1.3 GB `model.pt`, a pickle) is used only to estimate the horizontal field of view in `inference.py`: `fx_norm`
from its intrinsics gives `fov = 2 atan(1 / (2 fx_norm))`, run on the already preprocessed image. Passing `--fov` (radians) skips it entirely;
the README suggests 0.2 rad when distortion appears, and `get_proj_cond_*` defaults to 0.8575560450553894. It is therefore **not required at
inference** if the user supplies a field of view. Porting MoGe-2 would be a separate model and is not planned; the port takes a field of view
with a documented default.

## Findings that change the plan

1. The conditioning is no longer one token tensor: each flow takes a pair (5 global tokens `[1, 5, 1024]`, per-voxel features
   `[N, 1024 or 2048]`). `ITrellis2Conditioner`, the stage interfaces and the flow `Forward` signatures carry a single `Tensor cond`.
2. Weights are F32 and 2x the TRELLIS.2 files; each is read from disk per stage, as the TRELLIS.2 stages already do.
3. The structure flow's `rope_phases` (C64) makes `SafeTensorsLoader` reject the file today; the table is rebuilt in C# anyway
   (`Trellis2RopeTable`), so the tensor should be skipped by name and its shape checked.
4. `Trellis2FlowConfig.FromJson` rejects `ElasticSLatFlowModel` and ignores `image_attn_mode` and `proj_in_channels`; `Trellis2FlowKeyMap` and
   `Trellis2FlowBlock` use `cross_attn.*` names.
5. The cascade quantizer differs (above), and `mesh` scale conventions are TRELLIS.2's, so `ToYUp` is reused.
6. No parity reference exists locally; none of this is numerically verified.

## Plan for the implementation

Proposed split (each a draft PR; the first two are generalisations with TRELLIS.2 numerics pinned by tests):

1. **Flow proj mode.** `Trellis2FlowConfig` reads `image_attn_mode` / `proj_in_channels` and accepts `ElasticSLatFlowModel`; the key map and
   block take an optional per-block `proj_linear` and the `cross_attn_block` prefix; `rope_phases` is skipped. The flows' `Forward` takes a
   condition pair. TRELLIS.2 forward outputs are asserted bit-equal before and after on tiny networks.
2. **Condition type and cascade quantizer.** A small record for (global, per-voxel) conditioning through `ITrellis2StructureStage` /
   `ITrellis2ShapeFlowStage`, and a parameterised `QuantizeUnique` that keeps TRELLIS.2's result.
3. **Projection ops.** Camera math, 2D bilinear gather (CPU reference and an `IBackend` hook with the generic fallback), DINOv3 patch-grid view.
   Structure stage end to end first: it needs no NAF and only the 5.4 GB structure flow, so it can be checked with real weights early.
4. **NAF.** Encoder, RoPE, dilated neighbourhood attention through the existing op, `.pth` through the safe-subset reader.
5. **Pipeline and service.** `Pixal3DImageTo3DPipeline` (shape-only), `MeshService` id `pixal3d`, catalog entry, CLI help, status row.
6. Later, separate: texture flow and decoder, multi-view, MoGe.

`--grid` for `pixal3d` would mean what it means for `trellis2`: the final dual-grid resolution, `1024` or `1536` (both run the 512-to-HR cascade;
`512` is not a Pixal3D pipeline type and is rejected). Proposed default 1024, matching the TRELLIS.2 row and upstream's own low-VRAM default
(upstream's standard default is 1536). Vision folders and matting follow the `trellis2` pattern: DINOv3 from `Vision/Dinov3`, `DINOv3` or the
sibling folders, matting through `Trellis2LazyMatter` / `Trellis2ModelFiles.FindMatterFile` and `ModelFolders`.

## Not verified

- No weight was downloaded: the flow and NAF layouts come from headers and source, not from loading them.
- NAF's tensor names and shapes (pickle not read), and its weights licence.
- Whether `Trellis2PipelineConfig.Load` accepts Pixal3D's `pipeline.json` as is (the fields it reads are equal to the local TRELLIS.2 file).
- Any numerical agreement with the Python reference; no reference dumps exist.
