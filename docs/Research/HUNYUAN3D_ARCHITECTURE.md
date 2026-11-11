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
