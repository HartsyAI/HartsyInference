# Hunyuan3D-Omni: source spike

Source read, nothing run, nothing implemented. Pins: `Tencent-Hunyuan/Hunyuan3D-Omni` `4d47c0cc2bd0c4281963a7314ab330a5af36bfa8` (code, 2025-10-17), weights `tencent/Hunyuan3D-Omni` `70e803bfb4e127d534049d8ab8c8cb511780d485` (not gated). Read 2026-11-11. The configs under `model/`, `cond_encoder/`, `vae/`, `scheduler/` and the root `config.json` were read as text (a few hundred bytes each); no weight file was opened on that day; the weights were downloaded and read on 2026-11-12, see "Weights downloaded and compared" below, which supersedes the statements here that they were unread.

## What it is

Image-to-shape like Hunyuan3D-2.1, plus one extra control signal: a point cloud, a voxel set, a 3D bounding box or a skeleton (bone endpoints). It "inherits the structure of Hunyuan3D 2.1" (README), and the configs confirm it: the delta from the 2.1 shape model is a small control encoder and fine-tuned DiT weights.

## Licence

Tencent Hunyuan 3D Omni Community License Agreement (release 2025-09-26, root `License.txt` and `Notice.txt`): the same family as the 3D 2.1 and 3D-Part licences. The grant covers the Territory only (worldwide excluding the EU, UK and South Korea), more than 1M monthly active users needs a licence from Tencent, the Works and Output may not be used to improve another AI model, distribution carries the licence text and a Notice, and a product or service built on the Works must disclose to all end users the full legal name and entity of its actual provider and state conspicuously that Tencent is not affiliated with, sponsoring or endorsing it, and must not show any Tencent name, logo or mark in a way that suggests otherwise (3e). Hong Kong law. The code files carry a stale "Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE" header in `hy3dshape/models/denoisers/__init__.py`; the root licence governs the release.

## Components (all from the configs)

| Part | Config | Expected match in this repository (inferred from the configs, not checked) |
|---|---|---|
| DiT | `HunYuanDiTPlain`: input 4096 x 64, width 2048, depth 21, 16 heads, `context_dim` 1024, `text_len` 1370, RMS q/k norm, no biases, no position embedding, MoE in 6 layers (8 experts, top 2) | The Hunyuan3D-2.1 shape DiT, which `Models/Hunyuan3D` already runs (`HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md`); the weights are fine-tuned |
| Shape VAE | `ShapeVAE`: 4096 latents, `embed_dim` 64, width 1024, 8 encoder and 16 decoder layers, `scale_factor` 1.0079887561230991, `point_feats` 3, `sharpedge_label` | The 2.1 VAE family (`Hunyuan3DShapeVae`); it does not equal the 2.1 one: same keys, different values (see the 2026-11-12 section) |
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
- Classifier-free guidance: the unconditional branch replaces only the DINO tokens (`unconditional_embedding`) and keeps the control and signal tokens, so the conditional and unconditional batches differ in the image part only. The replacement is zeros: `DinoImageEncoder.unconditional_embedding` returns `torch.zeros(batch, patch_nums, hidden_size)` with `patch_nums` 37 x 37 + 1 = 1370 and `hidden_size` 1024 (read 2026-11-12), so the 2.1 zero-conditioning CFG path applies with the control and signal tokens appended to both branches.
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

All three weight files are pickles (read only through the safe pickle subset, never executed). By size the DiT is F32: 12.2 GB over about 3.05 B parameters equals the 2.1 DiT without its conditioner or VAE (the 2.1 shape file is 7.37 GB in F16 with DINO and VAE inside), and the VAE (1.3 GB, about 330 M parameters) likewise; dtypes F32 (read 2026-11-12). The `_ema` file is the same size; the upstream `--use_ema` flag selects it. The DiT needs 6.1 GB in F16, so a 12 GB card is plausible and tight, as for 2.1.

## Reuse map

| Need | Existing | Gap |
|---|---|---|
| DiT, MoE, flow sampler, MC / FlashVDM, VAE, DINOv2-large | `Models/Hunyuan3D` | loading from the pickles: the 12.2 GB F32 file has to be narrowed to F16 per tensor, and `HunyuanPaintLoader` today calls `PytorchPickleLoader.GetAllTensors()` for a whole file, so a per-tensor streaming read is a gap to check; the context length varies with the control signal (no 1370 constant was found under `Models/Hunyuan3D`, but the cross-attention and KV paths must be checked for a size fixed once from the config) |
| Fourier embedder | `FourierEmbedSpec(InputDims, Bands, IncludeInput)` in `Core/Backends` (already generic: the VAE uses `InputDims: 3`, UniRig uses it too); Omni is `InputDims: 6`, 8 bands, input included, 102 wide | none expected; not compared numerically |
| OmniEncoder | none | `Linear`, `RMSNorm`, GELU, an embedding table and the concatenation: a few dozen lines |
| Control preprocessing | `Geometry/Ops` (surface sampler), PLY and mesh readers | PLY point reader, the 16-cube voxeliser, the bbox and bone-file formats (`demos/pose/*.txt`: one `x1 y1 z1 x2 y2 z2` line per bone); coordinates in [-1, 1] |
| Image preprocessing | the existing Hunyuan3D image path (`ImageProcessorV2`) | none expected |

## Plan and what is needed

1. Port `OmniEncoder` and the four control preprocessors (CPU-testable against the demo `data.json` inputs without any weights), add the control signal and the longer context to the Hunyuan3D shape pipeline, with a synthetic smoke test (tiny random weights).
2. Real-weight run: `cond_encoder/pytorch_model.bin` (465 KB) and `vae/pytorch_model.bin` (1.3 GB) first (the VAE does not equal the 2.1 one, compared after the download), then `model/pytorch_model.bin` (12.2 GB, the non-EMA file first). That download (13.5 GB, without the EMA file) was approved and done on 2026-11-12.
3. The 12.2 GB file is read through the safe pickle parser and held as F16 (6.1 GB): the F32 widen path used for Paint would need 12 GB on the host and is not an option, so the loader has to narrow per tensor.

## Not checked

No Omni code was run (the pickles were read only as names and shapes, and the VAE values compared, with a Python unpickler restricted to tensors; nothing from the pickles was executed), the Fourier embedder is not compared with the ShapeVae one, the pipeline's unread parts (`prepare_image`, the guidance schedule `prev_guidance_scale`, the post-processors `FloaterRemover` and `DegenerateFaceRemover`) are not read in full.

## Port and CPU parity (2026-11-12)

Code: `Models/Hunyuan3D/Omni` (`OmniControlEncoder`, `OmniConfigFile`, `OmniCheckpoint`, `OmniContext`) and `Pipelines/Hunyuan3DOmniPipeline`; the DiT, the VAE and the sampler are the Hunyuan3D-2.1 ones (`Hunyuan3DPlainDit`, `Hunyuan3DShapeVae`, `Hunyuan3DShapePipeline`).

- **Config:** `OmniConfigFile` reads `config.json`, `model/config.json` and `vae/config.json` into the 2.1 `Hunyuan3DConfig` (4,096 latents, width 2,048, MoE in the last 6 of 21 blocks, VAE scale 1.0079887561230991, 50 steps, guidance 7.5, grid 384, box 1.01, mc level 0) and refuses what is not implemented.
- **Loading:** `PytorchPickleLoader.OpenLazy` and `ReadTensor` read a ZIP pickle one tensor at a time and convert each to F16 or F32 as it goes. The DiT file keeps all 752 tensors as views of a single 12 GB storage entry (a ZIP entry cannot be seeked), so `OmniCheckpoint` reads them in storage order, one pass over the file: 14 s on this machine, and the host holds 6.1 GB of F16 matrices (12.2 GB F32 on the CPU backend). The VAE keeps only the decoder half (the `sal.` keys `post_kl`, `transformer`, `geo_decoder`).
- **DINOv2-large:** the Omni repository does not contain it (upstream fetches `facebook/dinov2-large`). `facebook/dinov2-large` @ `47b73eef`, `model.safetensors` (1,217,522,888 B, sha256 `399fba97...c92e23`, apache-2.0, not gated) was downloaded on 2026-11-12 with the owner's approval into `3D/hunyuan3d-omni/dinov2-large`; all 439 of its tensors equal the Hunyuan3D-2.1 checkpoint's `conditioner.main_image_encoder.model.*` after F16 rounding (no difference), so either file works as the source.
- **Context:** `OmniContext.Build` gives the conditional context `[DINO | control | 10 signal]` and the unconditional one with the DINO part zeroed; the pipeline's sampler takes both, 2-way guidance as in 2.1 (the schedule is the same: 50 time points on `linspace(0, 1)`, 49 Euler steps; the SiT transport samples the ODE on the same grid, `shift_scale` is used only in training).
- **Parity:** the full DiT forward against upstream's `HunYuanDiTPlain` in float32 on the CPU, on the real weights: velocity relative L2 2.6e-5, blocks 2.9e-5 to 1.1e-4, see `PARITY_VERIFICATION.md`. Python takes 71 s per forward, the C# CPU backend 46 minutes.
- **Not done:** the voxel control on a GPU, the Omni VAE decode against upstream, the zero-DINO guidance branch against the Python value in the fixture (the GPU run used guidance 7.5, so the branch ran, but it was not compared).

### Engine and CLI

`hunyuan3d-omni` is a catalog entry (the three pickles and three configs of `tencent/Hunyuan3D-Omni` and `facebook/dinov2-large`, pinned sha256, folder `3D/hunyuan3d-omni`, loads from a folder) and a mesh model: `MeshModelIds.Omni`, input kind `ImageAndControl`, a profile (steps 0 to 1000, default 50; grid 0 or 32 to 512, default 384; seed; Preprocess Auto frames the image like the other Hunyuan3D models), the validator and planning, `MeshService.GenerateOmniAsync` (the pipeline is cached per folder, the mesh goes through the shared repair with floating pieces under 0.5 % of the area dropped, the stand-in for upstream's `FloaterRemover`) and the asset status, which follows the folder the loader is given.

The control is `MeshRequest.Control` (`MeshControl`: kind, geometry or values), built as upstream's `inference.py` builds it (`OmniControlInputs`): **point**: the vertices of an obj, glb or ply point cloud, centred and scaled to a longest side of 1.96, one token per point; **voxel**: an obj or glb mesh with faces, turned by -90 degrees about x, scaled to 2 x 0.9999, 81,920 area-weighted samples with the request seed, quantised to 16 per axis; **bbox**: length, height and width in (0, 1]; **pose**: six numbers per bone. CLI: `hartsy 3d <image.png> -m hunyuan3d-omni --control point --control-file points.ply`, `--control bbox --control-values 0.8,0.64,1`, `--control pose --control-file bones.txt`. Only the request checks, the control mapping and the catalog are covered by tests; no request has been run through the engine on a GPU.

**Guidance follows upstream's published script, not the pipeline signature:** 4.5 (`inference.py` passes it for every control; `Hunyuan3DOmniSiTFlowMatchingPipeline.__call__` defaults to 7.5). The other settings differ from the script: its octree is 512 and its seed 1234, the engine's default grid is the config's 384 and the seed is random (1234 only seeds a voxel control's surface samples when no seed is given), and the floaters are removed with the shared repair. The first GPU run below used guidance 7.5 and a 2,048-point control sampled from the same bust, so it shows that the pipeline runs and obeys a control, not that it reproduces upstream's demos.

### Runs through the CLI (2026-11-12, RTX 3060, origin/main `3c38d3cd7`)

`hartsy 3d <image> -m hunyuan3d-omni --control ...` with the upstream demo inputs and the engine defaults (guidance 4.5, grid 384, mc_level 0, shared repair with small floating pieces dropped), one process at a time, exit 0 each:

| Control | Input | Total | Peak (card total, desktop 0.96 GB) | Mesh |
|---|---|---|---|---|
| bbox 0.8, 0.64, 1 | `demos/bbox/furniture0.png` | 307 s (load 13 s, DiT 2.04 s per step) | 7.8 GB | 1,088,924 faces, watertight, 1 component, extent ratios 0.816 / 0.660 / 1 |
| point (2,048 points) | `demos/point/imgs/008.png`, `plys/008.ply` | 315 s | 7.8 GB | 970,268 faces, watertight, 1 component; control points on average 0.016 from the surface (p90 0.030, longest side 2) |
| pose (51 bones) | `demos/pose/263.png`, `a_pose_bone.txt` | 310 s | 8.2 GB | 216,408 faces, watertight, 1 component; joint x-extent 1.252 against 1.256 for the mesh, 96 of 102 joint points inside the mesh box |

Rendered looks, judged by eye: a sofa group with the requested proportions, a stone cottage with brick and roof detail, a standing figure in the A-pose with its arms out; surfaces are clean. The CLI is 22 % faster than the pipeline run below at the same grid (394 s). The DINOv2-large encoder matches `transformers` to 2.3e-6 relative L2 and its preprocessing matches torchvision to 1.6e-7. Not run: the voxel control.

### First real run (2026-11-12, RTX 3060)

`Hunyuan3DOmniPipeline` (scratch runner `omnirun`, not in the repository), the DiT matrices as F16 on the device (6.1 GB), `facebook/dinov2-large` for the image, the TRELLIS.2 bust photo as the image and **2,048 points sampled from the bust's own mesh as the point control** (normalised to [-1, 1], 2,048 rows plus 10 signal tokens, so a context of 1,370 + 2,058 tokens), seed 42, 50 steps (49 Euler steps, 2 forwards each for guidance 7.5), one process, nothing else on the GPU (baseline 1.2 to 2.2 GB of other sessions):

| Octree grid | Load | DiT, 50 steps | Decode and marching cubes | Total | Peak device memory (delta) | Mesh |
|---|---|---|---|---|---|---|
| 256 | 26 s (pickles read in 25 s) | 137 s (2.8 s per step, cuDNN fused attention F16) | 121 s | 285 s | 8.9 GB (6.7 GB) | 174,964 vertices, 349,924 faces |
| 384 (upstream's default) | 27 s | 132 s | 235 s | 394 s | 9.1 GB (7.0 GB) | 395,972 vertices, 791,940 faces |

Mesh checks (trimesh, both grids): watertight, consistent winding, no boundary or non-manifold edges; the main component holds 99.96 to 99.98 % of the faces and two to three closed floaters of 8 to 136 faces (spheres, Euler number 2 each) remain (the pipeline does not remove them). The main component is not a sphere: its Euler number is -4 at 256 and -2 at 384 (genus 3 and 2, a few small handles), so the whole-mesh Euler number of 2 is a coincidence of the floaters and the handles. The mesh follows the control: after normalising both to a longest side of 2, the extents are 1.48 x 2.00 x 0.97 against 1.39 x 2.00 x 0.98 for the bust, and the Chamfer distance of 30,000 surface samples each is 0.018 (0.9 % of the extent) in the identity frame; no axis permutation does better. By eye (`omni256_render.png`, `omni384_render.png` in the session notes, two flat-shaded views): a recognisable bust with the face, moustache, hair strands, jacket lapels and buttons, a clean back; the 384 grid is smoother and sharper than 256. This is one subject, one seed and one control kind; nothing here is compared with a Python run of the whole pipeline, and the control comes from the same mesh as the photo, so it shows that the control is obeyed, not how the model behaves with a conflicting one.

## Weights downloaded and compared (2026-11-12)

User-approved download of `tencent/Hunyuan3D-Omni` @ `70e803bfb4e127d534049d8ab8c8cb511780d485` into `Models/3D/hunyuan3d-omni/` (`.part` then rename, size and sha256 checked against the HF tree API): `model/pytorch_model.bin` (the non-EMA file), `vae/pytorch_model.bin`, `cond_encoder/pytorch_model.bin` and the six small configs. The EMA file was not downloaded. The pickles were read only through the restricted unpickler of `tests/python-reference/dump_paint21_unet.py` (nothing executed); the DiT and cond_encoder were read as names and shapes only, without loading the tensors.

| File | Tensors | Finding |
|---|---|---|
| `model/pytorch_model.bin` | 752, all F32 | The keys are bare (no `model.` prefix, so `Hunyuan3DPlainDit.LoadWeights` takes prefix `""`); the key set and every shape equal the `model.*` group of the 2.1 shape file (`hunyuan3d-21/dit/model.fp16.safetensors`, also 752): the Omni DiT is the 2.1 DiT architecture unchanged, so `Hunyuan3DPlainDit` loads it as it is; the values were not compared |
| `vae/pytorch_model.bin` | 410, all F32, keys prefixed `sal.` | The same 410 keys and shapes as the 2.1 VAE (`vae.*` in the 2.1 shape file) apart from the prefix, but the values differ: none of the 410 tensors equals the 2.1 one after rounding to F16 (largest absolute difference 0.50, in `geo_decoder.cross_attn_decoder.mlp.c_proj.weight`, 21 % relative to that tensor; the median tensor differs by 9 %, so the whole VAE was retrained, which `scale_factor` 1.0079887561230991 here against 1.0039506 for 2.1 agrees with: the Omni path takes the scale from its own config). Omni needs its own VAE weights, loaded with prefix `sal` by the existing `Hunyuan3DShapeVae`, which reads only the decoder side (the 8 encoder layers in the file go unused unless a later control path encodes point clouds); the 2.1 VAE cannot stand in |
| `cond_encoder/pytorch_model.bin` | 6, all F32 | `cond_signal_embedding.weight` [4, 8], `cond_signal_linear.weight` [1024, 8] and `.bias` [1024], `linear.0.weight` [1024, 102] and `.bias` [1024], and `linear.1.weight` [1024] (the RMSNorm, no bias; the `Sequential(Linear(102, 1024), RMSNorm(1024), GELU)` above): the point-cloud, voxel, bounding-box and skeleton control signals share one small encoder |

So the engine work for Omni is the control encoder with its four preprocessors, the longer context (DINO tokens, control tokens, 10 signal tokens), reading the 12.2 GB F32 pickle per tensor straight into F16 (the host would otherwise hold 12 GB), and the pipeline wiring; no DiT or VAE architecture changes are expected, but two things from earlier in this document are still open: the cross-attention and KV paths of the DiT must take the variable context length (1370 DINO tokens + control tokens + 10 signal tokens) and not a constant fixed once from the config, and whether the unconditional DINO replacement is zeros (the CFG question above). No Omni model code was run; the only tensor values compared are the VAE ones.

