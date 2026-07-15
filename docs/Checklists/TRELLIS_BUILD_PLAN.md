# TRELLIS (image → 3D) — architecture map + phased build plan (2026-07-15)

Microsoft **TRELLIS-image-large** (MIT). Image → **Structured 3D Latents (SLAT)** → Gaussian splat / mesh / radiance
field. This is a large from-scratch build in `HartsyInference.ThreeD` — a **two-stage flow-matching** system over a
new **sparse-voxel** representation, needing backend ops we don't have yet (3D conv, sparse tensor + sparse conv +
sparse attention, flexicubes). Reference: `/tmp/TRELLIS` (github microsoft/TRELLIS); weights
`/tmp/TRELLIS-weights/ckpts/*` (HF `microsoft/TRELLIS-image-large`).

**Discipline (hard-won — both TripoSR and Hunyuan3D started as wrong-architecture scaffolds):** get each sub-model's
architecture right against the reference BEFORE optimizing; every component gets a Python reference dump + a parity
test (corr ~1.0) before it counts as done. Validate on CUDA (GPU-only cache/reshape bugs hide on CPU).

## Pipeline (`TrellisImageTo3DPipeline.run`, exact)

1. **Preprocess** — rembg bg-removal → crop to alpha bbox (×1.2) → resize 518² → premultiply alpha. (Same
   Python-side bg-removal gap as TripoSR/Hunyuan3D; C# expects a premultiplied 518² RGB for now.)
2. **Conditioner** — DINOv2 **ViT-L/14 with registers** (`dinov2_vitl14_reg`), 518² input, ImageNet normalize →
   `x_prenorm` (features BEFORE the final norm) → `F.layer_norm(x, [C])` → patchtokens `[1, ~1374, 1024]`.
   `neg_cond = zeros`. cond_channels = 1024.
3. **Stage 1 — sparse structure (dense 16³):**
   - `SparseStructureFlowModel` (DiT-L): noise `[1,8,16,16,16]` → **FlowEuler CFG** sampler → `z_s [1,8,16,16,16]`.
     resolution 16, in/out 8ch, model 1024, cond 1024, **24 blocks**, 16 heads, mlp 4, patch 1, APE 3D pos,
     qk_rms_norm. Modulated self-attn + cross-attn to cond.
   - `SparseStructureDecoder` (conv3d VAE): `z_s [1,8,16,16,16]` → occupancy `[1,1,64,64,64]`; channels [512,128,32],
     2 res-blocks + 2 middle, upsample ×2 per stage. `coords = argwhere(occ > 0)[:, [0,2,3,4]]` → active voxels `[N,4]`.
4. **Stage 2 — structured latent (sparse over active voxels @ 64³):**
   - `SLatFlowModel` (sparse DiT-L): `SparseTensor(feats randn[N,8], coords)` → **FlowEuler CFG** → `slat [N,8]`.
     resolution 64, in/out 8ch, model 1024, **24 blocks**, patch 2, io_res_blocks 2 (io_block_channels [128]), APE,
     qk_rms_norm. Sparse self-attn (serialized/windowed) + cross-attn to cond. Then denormalize `slat = slat*std+mean`.
5. **Decode SLAT** (sparse VAE decoders, swin window-8, DiT-B model 768, 12 blocks, 12 heads):
   - `slat_decoder_gs` → **Gaussian splats** (32 gaussians/voxel: xyz/features_dc/opacity/scaling/rotation;
     softplus scaling, voxel_size 1.5). ← **first e2e target** (we already have `GaussianSplatCloud` + PLY export).
   - `slat_decoder_mesh` → **flexicubes** → mesh. `slat_decoder_rf` → radiance field.

## New backend infrastructure (what we don't have)

| Need | For | Notes |
|---|---|---|
| **Conv3D** (dense, + transposed/upsample) | SS VAE decoder | New `IBackend.Conv3d`; im2col-style or direct. Only 16³→64³, small. |
| **SparseTensor** type (feats `[N,C]` + coords `[N,4]`) | all of stage 2 | The core new type. Hash/lookup for neighbor gather. |
| **Sparse conv** (submanifold + strided) | SLAT flow io-blocks, VAE | gather-scatter via a coord hash map; the biggest new op. |
| **Sparse attention** (serialized + windowed/swin) | SLAT flow + VAE | group voxels into windows (z-order / spatial) → dense SDPA per window (reuse cuDNN). |
| **Flexicubes** | mesh decoder | dual-MC on active cubes from SDF+weights. Defer (GS first). |
| **Gaussian splat rasterizer** (optional) | previews | PLY export exists; rendering is optional. |

## Phased plan (each phase = code + Python reference dump + parity test, corr ~1.0, on CUDA)

- **A. Scaffolding + config + conditioner.** Package skeleton, `TrellisConfig` (dims above), pipeline stub, weight-key
  survey. DINOv2 ViT-L/14-**reg** preset + `x_prenorm`+layer_norm output on the shared `Dinov2VisionEncoder`
  (we have the encoder; need registers + the prenorm tap). Parity vs `dinov2_vitl14_reg`.
- **B. Stage 1 dense.** `Conv3d` backend op → `SparseStructureDecoder`; `SparseStructureFlowModel` (reuse Flux/DiT
  block machinery + 3D APE + patchify). FlowEuler sampler. Gate: **coords match the reference** for a fixed seed.
- **C. Sparse infrastructure.** `SparseTensor` + sparse conv + sparse attention (serialized/windowed) + sparse
  norm/linear. Unit-parity each op vs `trellis.modules.sparse`.
- **D. Stage 2 flow.** `SLatFlowModel` on the sparse infra. Gate: slat feats corr ~1.0 vs reference.
- **E. GS decoder (first full e2e).** `slat_decoder_gs` → `GaussianSplatCloud` → PLY. Gate: splat params corr ~1.0 +
  a coherent splat. Then mesh (flexicubes) + rf.
- **F. Perf + e2e.** Apply the gen-perf playbook once correct (cuDNN SDPA, graphs, F16, host-glue audit).

## Reference / repro
- Code `/tmp/TRELLIS`; weights `/tmp/TRELLIS-weights/ckpts/` (`ss_flow_img_dit_L_16l8`, `slat_flow_img_dit_L_64l8p2`,
  `ss_dec_conv3d_16l8`, `slat_dec_{gs,mesh,rf}_swin8_B_64l8`, + `pipeline.json` with samplers + slat_normalization).
- Python env for dumps: needs `torch`, `spconv`/`torchsparse` (or a dense fallback), `dinov2` hub. Build per-model
  `dump_*.py` oracles under `tests/python-reference/trellis_reference/` (pattern: TripoSR/Hunyuan3D references).

## Status
- [x] Architecture mapped from the reference — full bit-exact spec in [`docs/Research/TRELLIS_ARCHITECTURE.md`](../Research/TRELLIS_ARCHITECTURE.md) (norm eps, adaLN chunk order, qk-rmsnorm √64, sampler 25/cfg5/interval[0.5,1]/rescale3, all weight keys, sparse conv rulebook + Morton/Hilbert serialization algorithms).
- [~] Phase A — `TrellisConfig` (exact image-large dims) landed + compiles. TODO: DINOv2-reg conditioner (x_prenorm tap + reg preset), pipeline skeleton.
- [~] Phase B — **`Conv3d` backend op DONE** (IBackend host reference + CUDA `conv3d_f32` kernel; `CudaOpBisectTests.Conv3d_*` **bit-exact** corr 1.0, k3s1p1 + strided). TODO: `pixel_shuffle_3d` + `ChannelLayerNorm3d` helpers → `SparseStructureDecoder`; `SparseStructureFlowModel` (24-block modulated cross DiT); FlowEuler interval-CFG sampler. Gate: active-voxel coords match reference for a fixed seed.
- [ ] Phase C — sparse infra (SparseTensor + submanifold conv rulebook + full/windowed/serialized attention + Morton/Hilbert). Spec ready in the arch doc; needs a Python `spconv`/`vox2seq` dump for parity.
- [ ] Phase D–F — per above.
