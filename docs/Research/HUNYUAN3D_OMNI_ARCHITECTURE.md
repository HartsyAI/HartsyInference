# Hunyuan3D-Omni: source spike

Source read, nothing run, nothing implemented. Pins: `Tencent-Hunyuan/Hunyuan3D-Omni` `4d47c0cc2bd0c4281963a7314ab330a5af36bfa8` (code, 2025-10-17), weights `tencent/Hunyuan3D-Omni` `70e803bfb4e127d534049d8ab8c8cb511780d485` (not gated). Read 2026-11-11. The configs under `model/`, `cond_encoder/`, `vae/`, `scheduler/` and the root `config.json` were read as text (a few hundred bytes each); no weight file was opened.

## What it is

Image-to-shape like Hunyuan3D-2.1, plus one extra control signal: a point cloud, a voxel set, a 3D bounding box or a skeleton (bone endpoints). It "inherits the structure of Hunyuan3D 2.1" (README), and the configs confirm it: the delta from the 2.1 shape model is a small control encoder and fine-tuned DiT weights.

## Licence

Tencent Hunyuan 3D Omni Community License Agreement (release 2025-09-26, root `License.txt` and `Notice.txt`): the same family as the 3D 2.1 and 3D-Part licences. The grant covers the Territory only (worldwide excluding the EU, UK and South Korea), more than 1M monthly active users needs a licence from Tencent, the Works and Output may not be used to improve another AI model, distribution carries the licence text and a Notice, and a product built on the Works must disclose Tencent's name to end users. Hong Kong law. The code files carry a stale "Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE" header in `hy3dshape/models/denoisers/__init__.py`; the root licence governs the release.

## Components (all from the configs)

| Part | Config | Same as in this repository |
|---|---|---|
| DiT | `HunYuanDiTPlain`: input 4096 x 64, width 2048, depth 21, 16 heads, `context_dim` 1024, `text_len` 1370, RMS q/k norm, no biases, no position embedding, MoE in 6 layers (8 experts, top 2) | The Hunyuan3D-2.1 shape DiT, which `Models/Hunyuan3D` already runs (`HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md`); the weights are fine-tuned |
| Shape VAE | `ShapeVAE`: 4096 latents, `embed_dim` 64, width 1024, 8 encoder and 16 decoder layers, `scale_factor` 1.0079887561230991, `point_feats` 3, `sharpedge_label` | The 2.1 VAE family (`Hunyuan3DShapeVae`); whether the Omni VAE equals the 2.1 one is unknown |
| Image encoder | `facebook/dinov2-large`, 518 px, CLS token kept (1 + 37 x 37 = 1370 tokens), frozen | `Dinov2Preset.Large`; the 2.1 shape checkpoints carry it under `conditioner.main_image_encoder.model.` |
| Sampler | flow matching, velocity prediction, Euler, 50 steps, logit-normal training, `shift_scale` 0.125; guidance 7.5 | The Hunyuan3D flow sampler |
| Surface | marching cubes, octree resolution 384, `box_v` 1.01, optional FlashVDM | Hunyuan3D extraction (MC and FlashVDM exist) |

## The control encoder (`hy3dshape/models/conditioners/omni_encoder.py`, `OmniEncoder`)

All of it is small. `cond_encoder/pytorch_model.bin` is 464,828 bytes.

- `pe`: a Fourier embedder (8 frequencies, `include_pi` false); `linear`: `Linear(pe.get_dims(6), 1024)`, `RMSNorm(1024)`, GELU.
- `cond_signal_embedding`: `Embedding(4, 8)` then `cond_signal_linear`: `Linear(8, 1024)`. The signal id is pose 0, bbox 1, voxel 2, point 3, and its embedding is repeated for 10 tokens.
- The context is `cat([dino tokens (1370), control tokens, 10 signal tokens], dim=1)`:
  - pose: the bone endpoints `[x1 y1 z1 x2 y2 z2]` go straight through `pe` (6 values per bone, one token per bone);
  - bbox: the three extents (length, height, width, each 0 to 1) are repeated to 6 values, one token;
  - voxel: the input points are quantised to a 16 x 16 x 16 grid, the unique voxel centres in [-1, 1] are repeated to 6 values, one token per occupied voxel;
  - point: the points repeated to 6 values, one token per point.
- Classifier-free guidance: the unconditional branch replaces only the DINO tokens (`unconditional_embedding`) and keeps the control and signal tokens, so the conditional and unconditional batches differ in the image part only.
- `sampled_point` (pose bone starts, bbox corners, voxel centres, points) is returned for visualisation; the denoiser does not read it.
- Training-only paths (`drop_point`, `random_noise`, `noise_ratio`) are off at inference (`disable_drop = True`).

## Weights (`tencent/Hunyuan3D-Omni` at the pin)

| File | Bytes | sha256 (HF tree API) |
|---|---|---|
| `model/pytorch_model.bin` | 12,203,119,470 | `01c95e5f2588e324ebc05268231c92b3b1d9e83e5163723100ac958226c12a24` |
| `model/pytorch_model_ema.bin` | 12,203,277,342 | `1071255f12a0dfa5c2a2785bb7114dee280dc32f3844119445bf234bdb6a48d2` |
| `vae/pytorch_model.bin` | 1,311,145,138 | `379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340` |
| `cond_encoder/pytorch_model.bin` | 464,828 | `f3f308e73153000ed566e985e2b1e38e9b40c12e23b7e6ecf89ce3ea1699c432` |
| the `config.json` files, `image_processor/config.json` | 78 to 1,217 B | not LFS |

All three weight files are pickles (read only through the safe pickle subset, never executed). By size the DiT is F32: 12.2 GB over about 3.05 B parameters equals the 2.1 DiT without its conditioner or VAE (the 2.1 shape file is 7.37 GB in F16 with DINO and VAE inside), and the VAE (1.3 GB, about 330 M parameters) likewise; dtypes unread. The `_ema` file is the same size; the upstream `--use_ema` flag selects it. The DiT needs 6.1 GB in F16, so a 12 GB card is plausible and tight, as for 2.1.

## Reuse map

| Need | Existing | Gap |
|---|---|---|
| DiT, MoE, flow sampler, MC / FlashVDM, VAE, DINOv2-large | `Models/Hunyuan3D` | loading from the pickles (the 12.2 GB F32 file is read tensor by tensor through the safe parser and widened or narrowed once); the context length is no longer fixed at 1370 |
| Fourier embedder | the ShapeVae point embedder (8 frequencies, `include_pi` false) | check that it accepts 6-wide input |
| OmniEncoder | none | `Linear`, `RMSNorm`, GELU, an embedding table and the concatenation: a few dozen lines |
| Control preprocessing | `Geometry/Ops` (surface sampler), PLY and mesh readers | PLY point reader, the 16-cube voxeliser, the bbox and bone-file formats (`demos/pose/*.txt`: one `x1 y1 z1 x2 y2 z2` line per bone); coordinates in [-1, 1] |
| Image preprocessing | the existing Hunyuan3D image path (`ImageProcessorV2`) | none expected |

## Plan and what is needed

1. Port `OmniEncoder` and the four control preprocessors (CPU-testable against the demo `data.json` inputs without any weights), add the control signal and the longer context to the Hunyuan3D shape pipeline, with a synthetic smoke test (tiny random weights).
2. Real-weight run: `cond_encoder/pytorch_model.bin` (465 KB) and `vae/pytorch_model.bin` (1.3 GB) first (the VAE may equal the 2.1 one, which is already on disk: compare before downloading), then `model/pytorch_model.bin` (12.2 GB, the non-EMA file first). That is a download of about 13.5 GB for the full set and needs an explicit yes; nothing is downloaded.
3. The 12.2 GB file is read through the safe pickle parser and held as F16 (6.1 GB): the F32 widen path used for Paint would need 12 GB on the host and is not an option, so the loader has to narrow per tensor.

## Not checked

No weight was opened, no code was run, the dtypes and key names are unread, the Omni VAE is not compared with the 2.1 VAE, the Fourier embedder is not compared with the ShapeVae one, the pipeline's unread parts (`prepare_image`, the guidance schedule `prev_guidance_scale`, the post-processors `FloaterRemover` and `DegenerateFaceRemover`) are not read in full.
