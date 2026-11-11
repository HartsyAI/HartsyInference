# Hunyuan3D Paint (texturing): architecture, licence and reuse map

Source spike read 2026-11-11. Research, not implementation status: the status row is
[MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md#hunyuan3d-paint). Shape models are in
[HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md) and [HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md](HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md);
the bake toolkit is in [MESH_POSTPROCESS.md](MESH_POSTPROCESS.md).

## Pins

| Item | Pin |
|---|---|
| Paint 2.1 code | `Tencent-Hunyuan/Hunyuan3D-2.1` `main` at `82920d64` (2025-10-17), package `hy3dpaint` |
| Paint 2.0 code | `Tencent-Hunyuan/Hunyuan3D-2` `main` at `f8db6309` (2025-10-28), package `hy3dgen/texgen` |
| Paint 2.1 weights | `tencent/Hunyuan3D-2.1` revision `0b946776`, folder `hunyuan3d-paintpbr-v2-1/`, not gated |
| Paint 2.0 weights | `tencent/Hunyuan3D-2` revision `9cd649ba`, folders `hunyuan3d-paint-v2-0`, `-turbo`, `hunyuan3d-delight-v2-0`, not gated |

| File | Bytes | sha256 |
|---|---|---|
| 2.1 `unet/diffusion_pytorch_model.bin` (pickle) | 3,925,293,863 | `675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8` |
| 2.1 `vae/diffusion_pytorch_model.bin` (pickle) | 334,707,217 | `1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc` |
| 2.0 `hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors` (F16) | 3,662,636,472 | `1c5ce434ba976b30bbb51a080917ecd39f8d8761691887b5df3be765ef4bd9e9` |
| 2.0 `hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors` (F16) | 167,335,310 | `abcec86e499e1ce9f05d1630725d386dc533b61fe0947ab034f07f89042e7a61` |
| 2.0 `hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors` | 3,722,161,032 | `d6acffa4a22f4da61d87f446bfa83e7ac245481c1535fbf25b200fe4462d0b22` |
| 2.0 `hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors` (F16, 8-channel input) | 3,463,772,592 | `0ce61d15a43d11ba19079ab8f24dfce78b876d3f5291470079ef64b17e08ca58` |
| `facebook/dinov2-giant` `model.safetensors` (Apache-2.0, `611a9d42`) | 4,546,005,432 | `917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f` |

The 2.1 UNet and VAE are pickles, so they go through the safe-subset parser and a repack. The 2.0 files are safetensors. Both Paint
repositories ship a CLIP text encoder, but the released UNets carry learned text tokens (`learned_text_clip_*`), so no text encoder
or tokenizer runs.

## Licence

`Tencent-Hunyuan/Hunyuan3D-2.1` `LICENSE` at `82920d64` (the Hugging Face copy differs in whitespace only): Tencent Hunyuan 3D 2.1
Community License, release date June 13, 2025. Paint falls under it ("inference-enabling code, parameters, and weights", `Notice.txt`).

- **Territory.** Worldwide excluding the European Union, United Kingdom and South Korea; the agreement does not apply there. Use,
  reproduction, modification, distribution or display of the Works **or of Output** outside the Territory is unlicensed (5c).
- **Scale.** More than 1 million monthly active users in the month before the 2.1 release date needs a separate licence from Tencent
  (4); the request must also describe plans to modify the Works or create Model Derivatives.
- **Outputs.** Tencent claims no rights in Outputs; the user is responsible for them (6d).
- **No training other models** on the Works, Output or results (5b).
- **Services.** Section 3e: a service built on the Works must name its real provider and state that Tencent is not affiliated; no
  Tencent marks. This clause is in the root licence and absent from the copy in `hy3dpaint/LICENSE` (which also names "THL Q Limited"
  as Tencent instead of the entity wording of the root copy). The root text is the stricter and is the one to follow.
- **Acceptable Use Policy** applies (military use, high-stakes decisions, unlabelled machine-generated public content); it must be
  passed on in downstream terms (5a).
- **Third-party components** (`Notice.txt`): Stable Diffusion under MIT and CreativeML Open RAIL++-M (the paint UNet is
  `ByteDance/sd2.1-base-zsnr-laionaes6`, finetuned from `stabilityai/stable-diffusion-2-1-base`; use-based restrictions apply and must be
  passed on), DINOv2 Apache-2.0, Real-ESRGAN BSD-3-Clause. The 2.0 paint files fall under the 2.0 agreement, whose terms
  ([HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md#licence-tencent-hunyuan-3d-20-community-license-agreement)) are the same on territory, scale,
  outputs and training.

Personal use here is as for the shape models. A host that redistributes or serves Paint carries the territory, notice and 3e terms.

## What the model is

Both generations are one network family: **Stable Diffusion 2.1-base (zero terminal SNR, v-prediction) with a "2.5D" extension**
(`UNet2p5DConditionModel`), run once per request on all views at once.

| Item | 2.0 (`hunyuan3d-paint-v2-0`) | 2.1 (`hunyuan3d-paintpbr-v2-1`) |
|---|---|---|
| Output | one albedo per view | albedo and metallic-roughness per view |
| Views | six fixed: azim 0, 90, 180, 270 at elev 0; azim 0 at 90; azim 0 at -90 | same six first, up to `max_num_view` (6 to 12) chosen greedily |
| UNet | 12-channel input: 4 noisy latent + 4 normal latent + 4 position latent; `class_embedding` of 49 camera classes added to the time embedding; `learned_text_clip_gen/ref` (1,77,1024) | same 12-channel input (`noise_in_channels: 12`); per-material `learned_text_clip_{albedo,mr,ref}` (77,1024); no camera class embedding |
| Per transformer block | self attention `attn1`, reference attention `attn_refview`, multiview attention `attn_multiview`, text cross attention `attn2`, GEGLU feed-forward | the same plus material attention (separate `to_q/k/v/out` for `mr`), DINO attention `attn_dino` after `attn2`, 3D rotary encoding on the multiview attention |
| Reference | dual-stream: a second UNet (`unet_dual`, 4-channel input, 686 tensors) runs the reference latent at timestep 0 and records `norm1(hidden)` per transformer block | same |
| Sampler | Euler-ancestral, 30 steps (turbo: LCM, 30 steps in the call, guidance as a timestep-conditioning embedding, no guidance batch); scheduler config DDIM v-pred, trailing spacing, `rescale_betas_zero_snr` | UniPC multistep, trailing spacing, 15 steps, guidance 3.0 |
| Guidance | two-way: batch `[uncond, cond]` with `ref_scale = [0, 1]`, `uncond + 2.0 * (cond - uncond)` | three-way: `[uncond, ref, full]` with `uncond + g*s*(ref - uncond) + g*s*(full - ref)` and a view scale `s` from azimuth |
| Conditioning image | delight pass (InstructPix2Pix SD, 50 steps, image cfg 1.5) then recentre and white composite | white composite only; DINOv2-giant at 224 px |
| VRAM stated upstream | not stated | 21 GB for 6 views at 512 |

### 2.5D transformer block (2.1 order, `Basic2p5DTransformerBlock.forward`)

Input layout is `(b, n_pbr, n_view)` flattened; `b = 3` under guidance. With `x` the block input and `h1 = norm1(x)`:

1. **Material attention (MDA).** Per material `m` in `[albedo, mr]`, self attention over the tokens of each view with that material's
   own `to_q/to_k/to_v/to_out` (albedo uses the base `attn1`; `mr` uses `attn1.processor.to_*_mr`). `x += attn`.
2. **Write mode** (reference UNet only): store `h1` as `(b, n_ref * L, C)` under the layer name.
3. **Reference attention (RA)** (`ref_scale` is 0 for the unconditional branch, 1 otherwise): queries are the **albedo** rows of `h1`
   for all views, keys and values come from the stored reference `h1`; one Q and K for both materials and **one V per material**.
   The fused value is the channel concatenation `[v_albedo | v_mr]` viewed as `(heads, 2 * head_dim)`, the result split into two
   `head_dim` chunks: albedo output is the first half of each head, mr the second. Because the views interleave the two projections,
   a head sees consecutive channels of the concatenation, not its own slices of each. It is the same kind of layout trap as the
   2.1 shape DiT and must be reproduced exactly. `x += ref_scale * attn_m` per material.
4. **Multiview attention (MA)** (only when more than one view): all `n_view * L` tokens of one material attend to each other, with
   3D rotary encoding applied to Q and K; `x += mva_scale * attn`. 2.0 has no encoding (turbo adds it).
5. **Text cross attention** with the learned token block, then **DINO attention** (`attn_dino`, same `norm2` output, keys and values
   from the projected DINO tokens), both residual.
6. **Feed-forward** `norm3`, GEGLU (diffusers `FeedForward`, 4x), residual.

Zero-initialised output projections (`attn_multiview`, `attn_refview`, `attn_dino`) are a training detail and are irrelevant at inference.

### 3D rotary encoding (`RotaryEmbedding`, `compute_discrete_voxel_indice`)

Position maps are the white-background renderings described under Cameras. For each attention resolution `H`
(latent side 64 at 512 px: grids `H, H/2, H/4, H/8`, voxel resolutions `8H, 4H, 2H, H`), the position map (pixels) is pooled into an
`H x H` grid: pixels with value 1 in all three channels are background and zeroed, the grid cell is the mean of valid pixels, a cell
with fewer than `(pix_h * pix_w) // 16` valid pixels is zeroed, then `round(clamp(cell, 0, 1) * (voxel_resolution - 1))` gives integer
`(x, y, z)` per token. The table is `cos, sin` of `index * freq` with axis dimensions `dim_xy = hd // 8 * 3` for x and y and
`dim_z = hd // 8 * 2` for z (24, 24, 16 for head dimension 64), `freq_k = 1 / 10000^(2k / dim_axis)`, each value repeated for the
interleaved pair, concatenated x, y, z. Rotation is the interleaved complex form: `out = x * cos + rot(x) * sin` with
`rot(x)_{2i} = -x_{2i+1}`, `rot(x)_{2i+1} = x_{2i}`. The tables are cached per `(head_dim, sequence length)` and shared across the
materials; the table keyed by sequence length selects the resolution, so the number of tokens per UNet level must be distinct.

### Pipeline arithmetic

- Latents: `(b, n_pbr, n_view, 4, h, w)` with `h = w = view_size / 8` (512 or 768). Condition images are encoded by the VAE (posterior
  **sample**, times 0.18215), normal and position maps as `images * 2 - 1` after white-compositing; the reference image likewise.
- Under guidance the batch is `[uncond, ref, full]`: DINO tokens are zero for the first two, `ref_scale = [0, 1, 1]`, normal and
  position embeddings repeat in all three, prompts are the learned tokens for every branch.
- View scale from azimuth `a` (degrees): `a / 90 + 1` for `0 <= a < 90`, `2` for `90 <= a < 330`, `5 - a / 90` for `a >= 330`.
- `scheduler.step` is given `latents[:, :4]`; the decode is `vae.decode(latents / 0.18215)`.
- DINO: `AutoImageProcessor` defaults (shortest edge 256, centre crop 224, ImageNet mean and std) on the 512-px reference, `last_hidden_state`
  (257 tokens x 1536 after the final layer norm), then `ImageProjModel`: `Linear(1536, 4 * 1024)` per token, reshape to 4 tokens, `LayerNorm(1024)`,
  giving 1028 tokens of width 1024.

## Cameras, renderer and bake

- **Mesh frame** (`MeshRender.set_mesh`): `(x, y, z) -> (-x, z, -y)`, centre = bounding-box centre, scale = `1.15 / (2 * max |v - c|)`.
- **Camera** (`get_mv_matrix`, orthographic): distance 1.45, `elev -> -elev`, `azim -> azim + 90`, position
  `(d cos e cos a, d cos e sin a, d sin e)`, up `(0, 0, 1)`, look at the origin; projection scale 1.2 (`left = -0.6 ... right = 0.6`),
  near 0.1, far 100.
- **Maps.** Normal: face normal in the normalised mesh frame, `(n + 1) / 2`, background white. Position: `0.5 - v / 1.15` interpolated,
  background white. Alpha: triangle id plus one. Both render at 2048 (texture 2048 for 2.0, 4096 for 2.1); the diffusion model sees them at the view
  size (512 or 768); `L`-mode maps are thresholded at 2.
- **View selection** (2.1): render the 6 fixed views and 24 extra at elevation +-20 (azim every 30 deg, weight 0.01) at 1024 px;
  keep the six fixed, then add up to `max_num_view - 6` candidates greedily by new visible face area (area ratio), stopping when the
  best gain is below 0.01. Weights: `[1, 0.1, 0.5, 0.1, 0.05, 0.05]` for the fixed six.
- **Back projection** (`back_project`, mode `back_sample`): rasterize the view, face normals in camera space, `cos = <(0,0,-1), n>`
  zeroed below `cos(75 deg)`; the unreliable border is the depth image's Canny edges (30, 80) dilated by `2 / 512 * resolution`
  texels, plus the eroded silhouette. For every texel of the UV rasterization (`extract_textiles`: world position per texel), project
  to the view, require `|z - depth| < 3e-3`, a visible pixel and positive weight, sample the view bilinearly, and write colour and
  the cos weight.
- **Merge** (`fast_bake_texture`): weight = `view_weight * cos^4`; views already 99 percent painted are skipped; texture is the weighted
  mean; mask is the summed weight above 1e-8.
- **Inpaint:** upstream first fills from mesh-vertex neighbours (`meshVerticeInpaint`, C++) and then `cv2.inpaint` Navier-Stokes radius 3.
- **Super-resolution:** Real-ESRGAN x4plus (`RRDBNet`, 23 blocks) on each generated view, then resize to the render size.
- **Output:** the mr texture's R is metallic and G roughness (the writer packs glTF G = roughness, B = metallic); xatlas UV wrap; pymeshlab
  and trimesh decimation to 40,000 faces before painting.

## Reuse map

| Piece | Verdict |
|---|---|
| SD VAE (`VaeEncoder`, `VaeDecoder`, `VaeConfig.Sd15`, scale 0.18215) | Exists. The encoder returns the posterior mean (upstream samples); sampling needs the posterior std, which the encoder does not expose, so a mean-only run differs by the posterior noise (small). |
| Real-ESRGAN x4plus (`RrdbNet`, `UpscalePipeline`, catalog `real-esrgan-x4plus`) | Exists; the weights are a pickle the safe loader reads. Not on disk here. |
| DINOv2-giant (`Dinov2Preset.Giant`, `Dinov2VisionEncoder`) | Exists. Native image size is 518; 224 px goes through its bicubic position-embedding interpolation (`EncodeIntermediates`; `Encode` refuses a non-native size), to be checked against `Dinov2Model`. |
| Scheduler | `DdimVPredScheduler` is the sigmoid schedule, not this one. The scaled-linear zero-SNR beta table, v-prediction and UniPC multistep (and Euler-ancestral for 2.0) are new for a discrete-time model; `UniPcSampler` is the flow-matching variant. |
| SD UNet blocks (`UNet`, `CrossAttentionBlock`) | Do not fit: the transformer layer is sealed inside `CrossAttentionBlock`, and the 2.5D block needs three extra attentions, per-material weights and a capture mode. A hook would touch the SD1.5 and SDXL hot path, so `HunyuanPaintUnet` composes the public pieces (`UNetResNetBlock`, `TimestepEmbedding`, `AttentionLayer`) in its own block. |
| Rasterizer, `TextureBaker`, `Inpainter` (Telea), `PbrGlbWriter`, `UvUnwrapper`, `MeshDecimator` | Exist (UV-space raster, post-process). The Paint bake needs a **screen-space depth-buffered rasterizer**, an alpha/normal/position renderer, view selection, the back-projection weights and the merge: new. |
| Canny edges, Navier-Stokes inpaint, vertex inpaint | Not in the repo; replaced by a depth-gradient edge test and the Telea `Inpainter`, a documented difference. |
| xatlas, pymeshlab | Replaced by `UvUnwrapper`, `MeshDecimator`, as for TRELLIS.2. |

## Not yet checked

The 2.1 `.bin` key layout (read the pickle after approval; expected `unet.*`, `unet_dual.*`, `attn1.processor.to_*_mr`, `attn_dino`,
`image_proj_model_dino`, `learned_text_clip_{albedo,mr,ref}`); the 2.0 layout was read from the safetensors header: 849 `unet.*` and 686 `unet_dual.*`
tensors, `unet.conv_in.weight` `[320, 12, 3, 3]`, `unet_dual.conv_in.weight` `[320, 4, 3, 3]`, `unet.class_embedding.weight` `[49, 1280]`,
`unet.learned_text_clip_{gen,ref}` `[1, 77, 1024]`, attention keys under `transformer_blocks.N.{transformer.attn1,transformer.attn2,transformer.ff,
transformer.norm1..3,attn_multiview,attn_refview}`, all F16. The attention inner dimensions are heads x 64 (`attention_head_dim` `[5, 10, 20, 20]` is a head count).
