# Hunyuan3D Part (P3-SAM and X-Part): source spike

Source read, nothing run, nothing implemented. Pins: `Tencent-Hunyuan/Hunyuan3D-Part` `e96be065375438962375b55326416291342958a7` (code, 2025-12-05), weights `tencent/Hunyuan3D-Part` `677174466c53571e8bacd5050dff5948734a1a4d` (not gated). Date of reading: 2026-11-11.

## Licence

Tencent Hunyuan 3D-Part Community License Agreement (release 2025-09-23), root `LICENSE` and `NOTICE` of the repository and the weights repository. Same shape as the 3D 2.1 licence: the grant covers the Territory only (worldwide excluding the EU, UK and South Korea), and using or distributing the Works or the Output outside it is unlicensed (5c); more than 1M monthly active users needs a licence from Tencent (4); the Works and Output must not be used to improve any other AI model (5b); distribution needs the licence text and a Notice file (3); a product or service built on the Works must disclose to all end users the full legal name and entity of its actual provider and state conspicuously that Tencent is not affiliated with, sponsoring or endorsing it, and must not show any Tencent name, logo or mark in a way that suggests otherwise (3e). Tencent claims no rights in Output (6d). Hong Kong law. Behaviour that follows: a hosted service may not offer these models in the EU, UK or South Korea, and nothing generated here may be used as training data.

Dependencies: Sonata (`facebook/sonata`, a Pointcept PTv3 pretraining; the model card carries no licence text, so the licence of its weights has to be read from the Sonata repository before any use) and the Hunyuan3D shape VAE family (already in this repository).

## Pipeline

```
mesh -> P3-SAM (point-prompted 3D segmentation, 400 FPS prompts, NMS, face labels, AABBs)
     -> X-Part (conditioned flow-matching DiT: one latent per part, decoded by a shape VAE)
```

## P3-SAM (`P3-SAM/model.py`, `P3-SAM/demo/auto_mask.py`)

- Weights: `p3sam/p3sam.safetensors` 450,968,044 B, sha256 `eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c`, plus `p3sam/config.json` (134 B). Header read 2026-11-11 (ranged read of the first 53,448 bytes, JSON only, the file was not saved): 513 F32 tensors, no metadata. The Sonata backbone is inside: `sonata.embedding` (5 tensors) and `sonata.enc` (448 tensors, 108.46 M parameters), so `facebook/sonata` is not needed. The heads are `mlp` (1.16 M), `seg_mlp_1..3` (0.53 M each), `seg_s2_mlp_g` and `seg_s2_mlp_1..3` (0.27 M each), `iou_mlp` (0.33 M) and `iou_mlp_out` (0.13 M), each 6 tensors (three Linear weights and biases), which matches the module definitions in `model.py`. The full key list is in the session scratchpad, not in the repository.
- Feature extractor: Sonata (PTv3 encoder, `in_channels` 9 = coord, color (ones), normal; stages 48/96/192/384/512 with depths 3/3/3/12/3, heads 3/6/12/24/32, patch 1024, serialisation orders z, z-trans, hilbert, hilbert-trans with shuffling, stride 2 pooling, `enc_mode`). `get_feat`: sample 100,000 surface points, normalise to the unit cube, run Sonata, unpool by concatenating each parent's features with its children's (`pooling_parent`/`pooling_inverse`) down to the finest level, giving 1232 channels, then an MLP 1232 -> 512 -> 512 -> 512, then gather per point.
- Heads (plain MLPs, GELU): stage 1, three mask heads over `[feat 512, point xyz, prompt xyz]` (518 -> 512 -> 512 -> 1); stage 2, a global MLP over `[.., stage-1 logits]` (521 -> 256 x3), max over points, concatenated back, then three mask heads (777 -> 256 -> 256 -> 1); an IoU head (max-pooled, 256 -> 3). `iter=1` stage-2 passes. Outputs: three masks per prompt (sigmoid > 0.5) and three predicted IoUs.
- Auto mask (`mesh_sam`): clean the mesh, build face adjacency, sample 100k points and their face normals, 400 farthest-point-sampling prompts (`fpsample`), prompts in batches of 32, sort by predicted IoU, mask NMS by IoU, assign faces from points, fix labels by connected regions over face adjacency (with AABB checks), optional post-process (merge parts whose integral area exceeds 0.95 and that are under 1 % of the area), then per-part AABBs.
- Not portable as is: `fpsample` (FPS, a small C++ extension; the toolkit equivalent is `SurfaceSampler.FarthestPointIndices`), trimesh sampling and face adjacency, numba adjacency, `utils/chamfer3D` (CUDA, only in demos). All of it is plain geometry that the toolkit can host (the toolkit already has surface sampling, BVH, adjacency).

## X-Part (`XPart/partgen`, `config/infer.yaml`)

- Weights: `model/model.safetensors` 6,630,571,584 B (the DiT; dtype unread, size implies 2-byte), `shapevae/shapevae.safetensors` 655,539,474 B, `conditioner/conditioner.safetensors` 1,755,920,004 B, `p3sam/p3sam.safetensors` above; configs `model/config.json`, `shapevae/config.json`, `conditioner/config.json`, `scheduler/config.json`. The release is "a light version of X-Part" (README); the full model is only on Hunyuan3D-Studio. Recommended inputs: scanned or AI-generated meshes.
- DiT (`PartFormerDITPlain`): width 2048, depth 21, 16 heads, RMS q/k norm, no biases, learned part embedding (`valid_num` 50), two cross-attentions (object tokens and part-in-bbox tokens, both 1024-wide), MoE in 6 layers (8 experts, top-2), 1024 latent tokens of 64 channels. This is the same family as the Hunyuan3D-2.1 shape DiT, which this repository already runs (MoE routing included). Structure checked in `partformer_dit.py` and `moe_layers.py`: the MoE block has a shared expert (as in 2.1); U-skips (`skip_linear` over a concatenation, then a norm) on layers 11 to 20 (`layer > depth // 2`); the time embedding is prepended as one token (`torch.cat([c, x])`). Not yet compared: the fused-QKV head regroup (`Hunyuan3DPlainBlock.Regroup`). A parity trap: with `use_part_embed` the forward adds `part_embed[torch.randperm(valid_num)[:num_parts]]` (random part ids at inference), so a reference run needs the permutation pinned.
- Scheduler: `FlowMatchEulerDiscreteScheduler`, 1000 training steps, flow matching with an Euler integrator (the repository already has the Hunyuan3D flow sampler).
- Shape VAE (`VolumeDecoderShapeVAE`): 1024 latents x 64, width 1024, 8 encoder and 16 decoder layers, 8 Fourier frequencies, `qk_norm`, 81,920 surface points, scale factor 1.0039506. Hunyuan3D shape VAE family, with a different latent count; the conditioner uses two copies (4096 latents for the object, bbox-local for the part).
- Conditioner: ShapeVAE object and local encoders plus a Sonata feature extractor (`use_seg_feat`), the P3-SAM output giving the part boxes.

## Reuse map

| Need | Existing code | Gap |
|---|---|---|
| Sonata (PTv3) | `Models/Utonia` (`UtoniaEncoder`, serialisation z / hilbert, pooling stages), `Models/UniRig/UniRigPtv3*` | Sonata is the predecessor: different stem and attention details (to be read in `sonata/model.py`, 874 lines); `enc_mode` unpooling by concatenation; upstream uses flash-attn kernels; the engine's attention op replaces them |
| P3-SAM heads and prompt batching | none | small MLPs; the prompt batch is K x N x 777 F32 at stage 2: 32 prompts x 100k points is about 9.9 GB, so chunking is needed even at the upstream batch size of 32 |
| FPS, adjacency, NMS, connected regions | `Geometry/Ops` (`SurfaceSampler.FarthestPointIndices`, BVH, cleaner), `MeshOps` | the label fix-up and NMS; parity needs the exact call: `auto_mask.py` uses `fpsample.fps_sampling(_points, prompt_num)` with its default variant and start index (not yet read from `fpsample`), which decides the 400 prompts |
| X-Part DiT, MoE, flow sampler, shape VAE | `Models/Hunyuan3D` (DiT, MoE, ShapeVae, flow sampler) | latent count, part embedding, second cross-attention, `use_attention_pooling` off |
| Conditioner | ShapeVae encoder, Sonata above | the 1.76 GB conditioner file layout |

## Plan and what is needed

1. P3-SAM first (small: 451 MB, one backbone, MLP heads, geometry post-processing, no sampler): port Sonata (check `Models/Utonia` against `sonata/model.py`), the heads and `mesh_sam`. Verification needs the Python reference dumps (features, masks, labels) on one mesh, so a Python environment with the repository's requirements is needed or a CPU-only transcription run.
2. X-Part second: reuse the Hunyuan3D DiT, add the part conditioning, a 6.6 GB DiT and a 1.7 GB conditioner. dtype unknown (headers not read); 6,630,571,584 B over about 3.3 B parameters (a 2048 x 21 MoE DiT like the 2.1 one, which has about 3.0 B, plus the second cross-attention) implies F16 or BF16 already, so there is no F16 saving: 6.6 GB of DiT, 1.7 GB of conditioner and the VAE plus activations is tight on 12 GB, as for 2.1.
3. Omni follows its own spike.
4. Once a catalog id exists, the licence terms (the 3e provider disclosure and non-affiliation statement, the territory exclusion) go in the model catalog or licence notes.

Download needed for step 1: `tencent/Hunyuan3D-Part` @ `677174466c…` `p3sam/p3sam.safetensors` (450,968,044 B, sha256 above) and `p3sam/config.json` (134 B), to `Models/3D/hunyuan3d-part/p3sam/` (not downloaded yet; only the header has been read). `facebook/sonata` is not needed. X-Part files (about 9 GB) are a separate later request.

## Not checked

The tensor values (only the P3-SAM header was read), no code was run, the Sonata internals are not yet compared with `Models/Utonia`, the Sonata key names inside the P3-SAM file were grouped but not mapped to `sonata/model.py`, the X-Part file dtypes (headers not read; sizes imply 2-byte), the `fpsample` default variant and start index (which decide the 400 prompts), the post-processing thresholds beyond those quoted were not read in full (`mesh_sam` is 480 lines), and the X-Part pipeline (`partformer_pipeline.py`, 736 lines, and the conditioner files) was read only through its configuration.

## Implemented (2026-11-12, branch `feat/hunyuan-part-p3sam`)

`ThreeD/Models/Hunyuan3D/Part`: `P3SamCheckpoint` (the `sonata.` tensors to the Utonia encoder, the rest to the heads), `P3SamGridSample` (Sonata's `CenterShift` and `GridSample` at 0.005 with the FNV hash), `P3SamFeatures` (encoder, stage gathering through the pooling maps, the 1232 to 512 MLP, per-point expansion), `P3SamHeads` and `P3SamMlp` (the mask and IoU MLPs, with the prompt-independent first-layer terms computed once per mesh and held as backend tensors; a prompt runs on the backend with F32 GEMMs and only the per-point logits and the two max-pool inputs return to the host, so no prompt batch is needed: the work buffers are three `[points, 512]` tensors, about 2.3 GB on the device with the eight first-layer terms at 100k points; the max-pool reads two `[points, 512]` tensors back per prompt until a device row-max exists; `Own` sets `ForceF32Gemm`, so the 1232 to 512 feature MLP also runs F32 GEMMs), `P3SamPartLabeler` (the step-for-step port of the auto-mask heuristics), `P3SamAutoMask` (surface sampling, 400 farthest-point prompts, best-IoU mask per prompt, per-stage log). `UtoniaConfig.Sonata` and the `UseRope` switch make the existing Utonia encoder the Sonata backbone (same tensor names, no `attn.rope.inv_freq`).

Tested on random weights first (superseded by the real-weight comparison below): the heads against a transcription of upstream's concatenate-then-MLP (4 decimals; it shares this repository's reading of the column order, which the real-weight comparison proves), the grid sample, the feature gather, the labeler on synthetic cubes, and a tiny whole run. Not done: engine and CLI wiring, GPU runs. Differences from upstream: the encoder uses the fixed order sequence, while upstream permutes the serialisation orders at random on every pooling (see the real-weight section), the surface sampler is this repository's generator (the Python seed does not reproduce its points), `fpsample` is replaced by `SurfaceSampler.FarthestPointIndices` (the prompts can differ), the voxel pick is random as in upstream's train-mode `GridSample` (a deterministic mode exists), and the mesh cleaning (`merge_vertices`, `process`) is left to the caller.

## Real weights and parity (2026-11-12)

`p3sam.safetensors` (450,968,044 B, sha256 `eb76550cfbe06f154c6e9b17167ccfc28222bb4a216ec7b12ac2bf7d762de38c`) and `p3sam/config.json` (134 B, sha256 `fe06e1ef…`) are in `Models/3D/hunyuan3d-part/p3sam`, fetched at revision `67717446` with a size and hash check. Every tensor loads into the C# modules with no leftover (only `sonata.embedding.mask_token`, a training token, is unused).

Oracle: `tests/python-reference/dump_p3sam_features.py` runs upstream's Sonata modules (`XPart/partgen/models/sonata`) on the CPU with stand-ins for `spconv` (a brute-force `SubMConv3d`, kernel index k meaning offset k - 1), `torch_scatter`, `timm` and `addict`, and transcribes `get_feat` and the head forward; the input is 6000 points on an ellipsoid (`HARTSY_P3SAM`, `HARTSY_P3SAM_REF`, `P3SamRealWeightsTests.FeaturesAndHeads_MatchUpstreamOnTheCpu`). Results against this repository (CPU, F32; tolerance relRMS 5e-3 and correlation 0.9999 enforced):

| Stage | max abs diff | relative RMS | correlation |
|---|---|---|---|
| grid sample: selected voxel points, inverse map, grid coordinates (5945 voxels) | 0 values differ | | |
| Sonata stage 0 to 4 (5945, 5777, 5216, 3625, 1448 voxels; pooled grid coordinates identical) | 2.4e-5, 1.2e-5, 2.0e-5, 2.7e-3, 1.4e-3 | 2.2e-6, 1.9e-6, 1.9e-6, 7.6e-5, 1.1e-4 | 1.000000 |
| gathered 1232 channels | 2.7e-3 | 7.6e-5 | 1.000000 |
| features, 512 per point | 1.9e-4 | 4.2e-5 | 1.000000 |
| mask probabilities and predicted IoU, two prompts | 2.8e-6 / 8.9e-7 | 5.7e-7 / 8.2e-7 | 1.000000 |

Times on the CPU (i7-9700K): this repository, features for 6000 points 12.5 s (5945 voxels), one prompt 0.56 s; the Python oracle, Sonata forward 10 s, one prompt 0.1 to 0.2 s.

Findings: (1) upstream's `PointTransformerV3.__init__` builds its `GridPooling` layers without `shuffle_orders`, so they keep the default `True` and permute the serialisation orders at random on every pooling, at inference too: upstream's features from stage 1 on differ from run to run (about 10 % RMS between two runs of the unmodified code). The oracle pins it off; this repository always uses the fixed order sequence, which is one of the outputs upstream can produce. (2) `GridSample` in the `train` mode picks a random point per voxel; the oracle and this repository's `First` pick agree on the voxel selection, the random pick is not reproducible against Python.

Not compared: the auto-mask geometry (`P3SamPartLabeler`), the surface sampler, the farthest-point prompts (`fpsample` versus `SurfaceSampler`), a whole run on a real mesh, the GPU. The Sonata weights licence is not stated by the model card (the pickle `sonata.pth` and the Sonata repository were not read); this file is distributed under the Tencent licence above, but Sonata's own terms are unverified.

