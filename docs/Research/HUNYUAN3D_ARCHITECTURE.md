# Hunyuan3D shape models: architecture, license and findings

Source snapshot, read 2026-11-10. This is research, not implementation status; the status doc is
[MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md#hunyuan3d-2).

## Sources pinned

| Item | Pin |
|---|---|
| Upstream code | `Tencent-Hunyuan/Hunyuan3D-2` `main` at `f8db6309` (2025-10-28), package `hy3dgen/shapegen` |
| Weights | `tencent/Hunyuan3D-2` revision `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf`, not gated |
| Checkpoint | `hunyuan3d-dit-v2-0/model.fp16.safetensors`, 4,928,151,562 bytes, sha256 `360bc281fc956d4acac0c3d36d5ec0ebf8cdddbf4b8892e894d12419388d479b` |
| Header | 1645 tensors, all F16: `model.*` 652 (DiT), `vae.*` 266 (decoder only, no encoder or `pre_kl`), `conditioner.*` 727 (DINOv2-giant) |

## Licence: Tencent Hunyuan 3D 2.0 Community License Agreement

Read from the repository `LICENSE` (release date January 21, 2025). The Hugging Face copy at the pinned revision
differs only in whitespace.

- **Territory.** Worldwide excluding the European Union, United Kingdom and South Korea. The agreement "does not
  apply" there. Use, reproduction, modification, distribution or display of the Works, **or of Output**, outside the
  Territory is unlicensed (section 5c; Acceptable Use Policy item 1).
- **Scale.** If the licensee's products had more than 1 million monthly active users in the calendar month before the
  version release date, a separate licence from Tencent is required (section 4).
- **Outputs.** Tencent claims no rights in Outputs; the user is solely responsible for them (6d). Outputs are not Model
  Derivatives by themselves (1g).
- **No training other models.** The Works, Output or results must not be used to improve any other AI model (5b).
  Distillation or synthetic-data training that makes another model imitate the Works creates a Model Derivative (1g).
- **Distribution.** Only inside the Territory; recipients get a copy of the agreement; modified files carry change
  notices; a "Notice" text file is required for distribution other than through a Hosted Service (3a-3d).
  Section 3e requires disclosing the real provider of any service built on the Works and forbids implying Tencent
  endorsement.
- **Use restrictions.** The Acceptable Use Policy (last modified November 5, 2024) forbids, among others, military
  use (19), high-stakes automated decisions (14), unlabelled machine-generated content in a public context (12) and
  use outside the Territory (1). It must be included as an enforceable provision in downstream terms (5a).
- **Termination.** Tencent may terminate on breach; the user must then delete and stop using the Works (8b).
  Governing law and courts are Hong Kong SAR.

The engine is used personally here. A host that redistributes this model must carry the notice and territory terms.
Other Hunyuan3D checkpoints (2.1, mini, mv, Fast, Turbo) carry their own licence files, recorded when each is
verified.

## Hunyuan3D-2 stage map (upstream to C#)

| Stage | Upstream | C# | Checked |
|---|---|---|---|
| Image prep | `ImageProcessorV2`: recentre with 15% border, composite on white, cubic resize to 512, scale to [-1, 1] | none: the caller supplies the 512x512 RGB bytes | not ported |
| Condition | `DinoImageEncoder`: bilinear (antialias) to 518, ImageNet mean and std, `Dinov2Model` giant (40 layers, 1536, 24 heads, SwiGLU, layerscale 1.0), all 1370 tokens | `Dinov2VisionEncoder` + `Dinov2ImagePreprocessor` | corr 1.0 earlier; not re-run on the pinned input |
| Unconditional | zeros of the same shape | zeros | by reading |
| Sampler | `np.linspace(0, 1, steps)` plus a trailing 1.0, `timestep = sigma`, `prev = x + (sigma_next - sigma) * v`, CFG `u + 5 * (c - u)` | `Hunyuan3DFlowSchedule`, `GuidedEulerStep` | schedule unit-tested; velocity corr 1.0 |
| DiT | `Hunyuan3DDiT`: `latent_in`, `cond_in`, `time_in(timestep_embedding(t, 256, max_period=1000))` where `t = 1000 * sigma`, 16 double + 32 single blocks, no RoPE, `final_layer` | `Hunyuan3DDit` | corr 1.00000000 |
| Latent scale | `latents / scale_factor` (0.9990943) | same | by reading |
| ShapeVAE | `post_kl` then 16 residual self-attention blocks (QK LayerNorm, GELU MLP) | `Hunyuan3DShapeVae` | processed latents corr 0.99999586 |
| Occupancy | Fourier features (8 octaves, no pi, input first, then sin, then cos), `query_proj`, one cross-attention block, `ln_post`, `output_proj` | `Hunyuan3DGeoDecoder` | corr 0.99999999 |
| Grid | `linspace(-1.01, 1.01, res + 1)` on three axes, `indexing="ij"` | `res` samples, same spacing formula | same coordinates when called with `res + 1` |
| Surface | skimage Lewiner at level 0.0, vertices `v / (res + 1) * size + min`, faces flipped | `Hunyuan3DSurface` (negated field, `MarchingCubes`) | topology identical on the reference field: 81,024 triangles, 1 component |
| Post-process | none inside the pipeline | none | |

## Findings

- The recorded per-component parity (conditioner, DiT, VAE) never exercised the sampler, so a wrong timestep and an inward
  winding both survived it. The DiT contract is `Forward(latent, cond, sigma)` with `sigma` in `[0, 1]`.
- Upstream's last sampler step has zero width (`sigma_next == sigma == 1`), so its velocity cannot change the latent; the
  C# loop skips that forward.
- The same repository holds the Fast and Turbo builds as separate directories (`hunyuan3d-dit-v2-0-fast`, `-turbo`, each about
  4.93 GB, `model.fp16.safetensors` and `config.yaml`) and the VAE builds `hunyuan3d-vae-v2-0`, `-turbo` and `-withencoder`.
  Their configs and sampler differences are read in the variants step (3b), not here.

## Variants (checked 2026-11-10)

Pins: `tencent/Hunyuan3D-2` `9cd649ba` (base, Fast, Turbo, turbo VAE), `tencent/Hunyuan3D-2mini` `f90a0f7d`, `tencent/Hunyuan3D-2mv`
`3a761b53`. None is gated. All three repositories carry the same Tencent Hunyuan 3D 2.0 Community License text (whitespace
differences only), so the terms above apply to every build below.

Each build is one `model.fp16.safetensors` (all F16, `model.*` DiT, `vae.*` decoder only, `conditioner.*` DINOv2-giant) plus a
`config.yaml`. What differs from the base config, read from the configs and the safetensors headers:

| Build | Size | Differences from base |
|---|---|---|
| Fast (`hunyuan3d-dit-v2-0-fast`) | 4,930,777,530 B | `guidance_embed: true`: adds `guidance_in.{in,out}_layer.{weight,bias}` (4 tensors, 1649 total). One forward per step, no CFG batch. |
| Turbo (`hunyuan3d-dit-v2-0-turbo`) | 4,930,777,530 B | Fast's DiT, plus `ConsistencyFlowMatchEulerDiscreteScheduler` with `pcm_timesteps: 100`. |
| mini (`hunyuan3d-dit-v2-mini`) | 3,819,958,234 B | 8 double + 16 single blocks, 512 latents, `scale_factor` 1.0188137. Same width, heads, DINO. |
| mini-Fast, mini-Turbo | 3,822,584,202 B | mini plus guidance embedding; mini-Turbo also feeds DINO at 1022 px and a 1022 preprocessor, which needs position-embedding interpolation: refused. |
| 2mv (`hunyuan3d-dit-v2-mv`) | 4,928,151,562 B | `DinoImageEncoderMV`: per-view DINO tokens plus a fixed sin/cos view embedding, views front, left, back, right in that order, any subset. DiT weights are the same shape as base. |
| Turbo VAE (`hunyuan3d-vae-v2-0-turbo`) | 407,410,402 B | Geo decoder narrowed to 512 with a `latents_proj` (1024 to 512), MLP ratio 1, no `ln_post` and no QK norm. Used by FlashVDM for every DiT. |

Sampler details (upstream `Hunyuan3DDiTFlowMatchingPipeline`):

- With `guidance_embed` the pipeline skips classifier-free guidance and passes `guidance_scale` to the DiT; the DiT adds
  `guidance_in(timestep_embedding(guidance))` to the timestep vector, with the same sinusoid and `time_factor` as the time.
  Upstream's gradio defaults are 5 steps for Turbo, 10 for Fast and 30 for the standard builds; the library default is 50.
- The consistency scheduler builds a 100-entry table `linspace(0, 1, 1000)[[0] + (arange(1, 100) * 10 - 1)]` and takes entries
  `floor(linspace(0, 100, steps, endpoint=False))`, so five steps run at sigma 0, 0.1992, 0.3994, 0.5996, 0.7998.
- FlashVDM (`FlashVDMVolumeDecoding`, mean mode): the coarse grid has `round(r0 / 4) * 4 - 1` cells per axis in 4x4x4 blocks, each
  block attending to the top 1024 of 3072 latents (256 of 512) chosen per head by the mean of every 100th query; finer levels
  evaluate a dilated band around sign changes, grouped into 6x6x6 spatial cells with the same selection from every 50th query.
  The finest grid is `r0 * 2^(levels - 1) + 1` points (253 per axis for octree 256: levels 63, 126, 252), and unevaluated points are NaN.

## Variant selection

`ModelResolver.Resolve` parses a `family:variant` selector only for Image and Video; for Mesh `hunyuan3d:fast` is not a catalog id and
falls through to a folder guess. 3D catalog rows had no `Assets`, `ModelDownloader` reads only `CatalogEntry.Assets`, and nothing
reads `CatalogEntry.Variants` (the recipe variant catalog serves image and video). The mechanism chosen is **one catalog id per
build** (`hunyuan3d`, `-fast`, `-turbo`, `-mini`, and `-mv` for download only), each with its own `Assets` in `3D/<id>/`:

- it needs no resolver or downloader change, so `hartsy pull` and path lookup work as for every other 3D model;
- the builds are separate 3.8 to 4.9 GB checkpoints with different sampling contracts, so a user wants one without fetching all;
- the id only chooses files: the checkpoint's `config.yaml` sets the architecture, and a build is never recognised by name;
- it touches the shared engine files in one row each (`ModelCatalog`, `MeshModelIds`, one branch in `MeshService`).

Extending the resolver would need Mesh selector parsing, a variant-to-asset-subset rule in `ModelDownloader`, and a folder
convention, all new shared surface for one family.
