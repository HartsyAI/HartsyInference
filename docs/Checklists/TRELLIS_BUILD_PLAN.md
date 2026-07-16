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
- [x] **Phase A DONE (2026-07-15) — DINOv2-vitl14-reg conditioner, parity-verified (corr 0.99994861 vs the real
  `dinov2_vitl14_reg`).** `TrellisConfig` (image-large dims) + `TrellisImageConditioner` = the shared
  `Dinov2VisionEncoder` with a new `Dinov2Preset.LargeReg` (ViT-L/24L/16H + **4 register tokens**, native 518px so
  pos_embed is used directly, no interpolation) tapped at **`x_prenorm`** (new `applyFinalNorm:false` on `Encode`)
  → **non-affine** `LayerNormNoAffine` (eps 1e-5) = the reference `dino(t,is_training=True)['x_prenorm']` then
  `F.layer_norm(feats,[1024])`. Weights = the torch.hub `dinov2_vitl14_reg4_pretrain.pth` remapped to HF keys
  (`/tmp/trellis_ref/convert_dinov2_reg.py`: splits fused `attn.qkv` [3H,H] → q/k/v rows, `ls*.gamma`→`layer_scale*.lambda1`,
  etc.) → `dinov2_vitl14_reg.safetensors`. **Spec/converter proven exact**: a torch F32 forward with the converted
  weights matches the reference cond to maxAbs 1.9e-4 (corr 1.0); the C# residual (corr 0.99994861, maxAbs ~0.7 on
  DINOv2's high-norm outlier tokens, ref magnitude ~13) is TF32/SDPA GEMM noise over 24 layers — the reference cond
  was itself dumped on CUDA (TF32), so TF32-vs-TF32, not a bug (`corr` is the gate). Test `TrellisConditionerParityTests`.
  `TrellisGenerationTests` now computes the cond **in-engine** when the remapped weights are present (12967 voxels →
  414944 gaussians → valid 26 MB PLY) — TRELLIS is self-contained on the network side; only the raw-image
  premultiply-alpha/LANCZOS crop remains in Python (as with TripoSR/Hunyuan3D background removal).
- [x] **Phase B DONE — stage-1 dense, validated end-to-end vs the REAL TRELLIS.** `Conv3d` (IBackend + CUDA `conv3d_f32`, bit-exact) + `PixelShuffle3d` + `ChannelLayerNorm3d` (IBackend host defaults) → `SparseStructureDecoder`; `SparseStructureFlow` (24-block `SsFlowBlock`: modulated self-attn + cross-attn + tanh-GELU MLP, per-head QK-RMSNorm ×√64) → `TrellisSparseStructureSampler` (25-step FlowEuler interval-CFG). Parity (`TrellisStage1ParityTests`, real ckpt + real-model torch dumps): **SS decoder corr 1.0**, **SS flow velocity corr 0.99999866**, **full pipeline (noise→sampler→decoder→occupancy) corr 0.99964, 99.96 % voxel agreement** (active 2942 vs 2990 — borderline voxels at occ≈0 flip under F32 drift, the true-math gate). GOTCHA fixed: `Transpose2D(out,in,d1,d2)` takes SIZES not dim-indices (bisection: tok corr −0.005 → fixed with `(InCh,Tokens)` → corr 1.0). Reference loads the real dense models via sys.modules stubs (bypass rembg `__init__`) + `ATTN_BACKEND=sdpa` (no flash_attn).
- [x] **Phase C DONE + stage-2 SLAT flow parity-verified vs the REAL `SLatFlowModel`** (corr **0.99999984**). Built
  `SparseTensor`, `SparseOps` (submanifold conv = scatter→`Conv3d`→gather; avg-pool downsample; gather upsample),
  `SlatResBlock3d`, and `SlatFlowModel` (sparse U-Net: input_layer → io ResBlocks w/ 1 downsample → APE(coords) → 24
  transformer blocks reusing `SsFlowBlock` (B=1 full attn = dense SDPA) → io ResBlocks w/ upsample+skip-concat →
  out_layer). GOLD REFERENCE: ran the real `SLatFlowModel` without spconv/flash_attn by monkeypatching `SparseConv3d`
  → the proven dense-equivalent + a fake `spconv.SparseConvTensor` (permissive `__getattr__`) + a dense
  `sparse_scaled_dot_product_attention` (`/tmp/trellis_ref/dump_slat_flow.py`). **BUG the gold reference caught that
  first-principles missed:** TRELLIS `SparseDownsample` = `scatter_reduce(reduce='mean')` with default
  `include_self=True` → divides by **count+1** (the zero self is in the mean), not count. My numpy standalone had the
  same mistake so it passed — only the real model exposed it (bisected: il/b0 corr 1.0 → b1 0.98 → downsample 0.987 →
  fixed). Tests `TrellisStage2ParityTests` + `TrellisSparseOpsParityTests`.
- [~] ~~Phase C keystone~~ superseded — original keystone note: `SparseTensor` (feats[N,C]+coords[N,4]+resolution), `SparseOps.SubmanifoldConv3d` (**= scatter→`Conv3d`→gather; provably equals real spconv since inactive neighbours are 0 — no spconv install needed**; `TrellisSparseOpsParityTests` corr **1.0** vs first-principles numpy), `SparseOps.Downsample` (avg-pool, bit-exact) + `Upsample` (gather via cached idx). **Key finding: the SLAT flow needs NO rulebook and NO Morton/Hilbert** — its attention is `full` (= dense SDPA over all N voxels at B=1, reuse `SsFlowBlock`) and its only coord-change is avg-pool downsample (strided conv is not used). TODO: full/cross sparse attention wrapper (B=1 dense), `AbsolutePositionEmbedder` on coords, `ModulatedSparseTransformerCrossBlock` (reuse SsFlowBlock math), assemble `SLatFlowModel` (U-Net: input_layer→io ResBlocks w/ downsample→APE→24 transformer blocks→out ResBlocks w/ upsample+skips→out_layer) + slat denormalize. Validate against real block/APE (importable, attention-only) + assembled velocity. NOTE: full real `SLatFlowModel` can't run here (needs spconv), so gate per-op + per-block, then assemble.
- [~] **Phase E — GS decoder NETWORK done + parity-verified (corr 0.99999973 vs the real `SLatGaussianDecoder`).**
  `SlatGaussianDecoder` (`GsBlock`): plain sparse transformer (DiT-B 768, 12 blocks, swin window-8, no modulation/
  cross-attn/qk-norm) → layer_norm → out_layer → 448 params/voxel (32 gaussians × 14). **Windowed (swin) attention =
  a single masked dense SDPA** over all voxels with a block-diagonal mask (attend within the same 8³ window; shift
  alternates 0/4 → two precomputed masks) — no sort/gather/serialization needed for B=1. Gold ref: monkeypatch the
  real `sparse_windowed_scaled_dot_product_self_attention` → torch partition (`calc_window_partition`) + per-window
  SDPA (`/tmp/trellis_ref/dump_gs_dec.py`). Test `TrellisGsDecoderParityTests`. **TODO for first e2e:** `to_representation`
  (build `Gaussian` from the 448 params — xyz offset + hammersley perturbation, softplus scaling, sigmoid opacity,
  normalized rotation → `GaussianSplatCloud` → PLY) + Phase A DINOv2-reg conditioner (real image cond). Then mesh/rf decoders.
- [x] **FIRST IMAGE→3D E2E GENERATION (2026-07-15).** `TrellisGaussianRepresentation.Build` (to_representation:
  voxel-centre + tanh(offset + hammersley perturbation)·½·voxel_size/res, softplus scaling+bias, logit opacity,
  y-up→z-up transform [x,−z,y] + quaternion rotate 90°X) → `GaussianSplatCloud` → `PlyWriter.SaveSplats`.
  `TrellisSlatSampler` (FlowEuler over the SparseTensor) + `TrellisGenerationTests` wire the whole path (SS flow →
  SS decoder → active voxels → SLAT flow → denorm(mean/std) → GS decoder → splat). **Dragon: 13115 voxels → 419680
  gaussians → valid 27 MB 3DGS `.ply`** (xyz∈[−0.5,0.5], no NaN, sane opacity/scale). Cond fed pre-computed
  (`/tmp/trellis_ref/dump_cond.py` runs the real `dinov2_vitl14_reg`).
- [x] **Phase F stage-2 perf pass (2026-07-15) — SLAT flow 290 → 35.5 s (8.2×), parity preserved (velocity corr
  0.99999983).** The killer was `SubmanifoldConv3d` running a **dense** `Conv3d` over a full `res³` grid at up to
  2048 channels (a 2.1 GB grid, 3789 ms/conv on profiling) — ~all of stage-2. Replaced with a **spconv rulebook**
  (`SparseOps.SubmanifoldConv3dSparse` + `ConvWeightSlices`): a coord hash builds per-kernel-offset (in,out) index
  pairs → per offset `RowGather` → cuBLAS GEMM (the offset's `[Cout,Cin]` weight slice) → `RowScatterAdd`, ~20–90×
  less compute than the dense grid at high channel counts (only active voxels, no multi-GB grid). New `row_gather_f32`
  / `row_scatter_add_f32` CUDA kernels + `IBackend.RowGather`/`RowScatterAdd` (host defaults). Rulebook conv unit
  parity `TrellisSparseOpsParityTests.SubmanifoldConv3dSparse_MatchesReference` corr 0.99999996 (maxAbs 1.2e-2, F32
  GEMM accumulation). Full generation now **~65 s** (load 5.6 · stage1 20.9 · **stage2 35.5** · gs-decode 2.8),
  same 419680 gaussians / valid 27 MB PLY. **GOTCHA fixed:** `ConvWeightSlices` AV'd at weight-load — a GC-finalizer
  race: `wf` (the F32-cast weight) had its last managed use at `s = wf.DataPointer`, then the 27 per-slice `new
  Tensor` allocations pressured GC into collecting/finalizing the now-dead `wf`, freeing its native buffer mid-loop
  → dangling `s` → AccessViolation. Fix = `GC.KeepAlive(wf)` after the write loops.
- [ ] Remaining: raw-image preprocessing (premultiply-alpha bbox crop + 518 LANCZOS resize) for a fully in-engine
  CLI path (network conditioner is done) + further Phase F (stage-1 20.9 s / GS-decode windowed-SDPA batching, F16,
  graphs — apply the gen-perf playbook) + mesh (flexicubes) & RF decoders + a real render check.
