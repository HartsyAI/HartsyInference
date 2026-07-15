# TRELLIS-image-large — architecture spec (bit-exact port ground-truth, 2026-07-15)

Reconstructed from the reference (`/tmp/TRELLIS`) + real weight keys (`/tmp/TRELLIS-weights/ckpts/*.safetensors`).
Build plan + phasing: [`docs/Checklists/TRELLIS_BUILD_PLAN.md`](../Checklists/TRELLIS_BUILD_PLAN.md). Every numeric
detail below (norm eps, modulation chunk order, qk-rmsnorm scale, sampler params, weight-key names) must be matched
for parity — validate F32-first on CUDA (F16 amplifies tail drift).

## Stage 1 — sparse structure (dense)

**Conditioner:** `dinov2_vitl14_reg` (embed 1024, 518² premultiplied RGB, ImageNet-norm). cond =
`layer_norm(dinov2(img, is_training=True)['x_prenorm'], [C])` → `[1, 1374, 1024]` (1 CLS + 4 reg + 37²). neg=zeros.
Reuse `Dinov2VisionEncoder` — needs the **pre-norm** tap + trailing non-affine LN + the reg preset.

**`SparseStructureFlowModel`** (DiT-L, config `ss_flow_img_dit_L_16l8`): resolution 16, in/out 8, model 1024, cond
1024, 24 blocks, 16 heads (head_dim 64), mlp 4, patch 1 (patchify = identity), pe=ape, qk_rms_norm.
- Head (F32): `input_layer` Linear 8→1024; `pos_emb` is a **saved buffer** `[4096,1024]` (load directly; APE formula:
  channels 1024, in_ch 3, freq_dim 170, freqs 1/10000^(i/170), token n→coords (n//256,(n//16)%16,n%16), per coord
  [sin(340),cos] → 1020 + 4 zero-pad). `t_embedder`: sinusoid(dim 256, **cos-first**, max_period 1e4)→Linear256→1024→
  SiLU→Linear1024→1024. Model is fed **1000·t**. `share_mod=false` (each block owns adaLN).
- Block (F16) = `ModulatedTransformerCrossBlock` (self-attn + cross-attn + MLP):
  - `mod = SiLU→Linear(1024→6144)`, chunk 6: **shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp**.
  - self: `h=LN(x, affine=F, eps 1e-6); h=h*(1+sc_msa)+sh_msa; h=selfattn(h); h=h*g_msa; x+=h`.
  - cross: `h=LN(x, affine=T, eps 1e-6, = norm2.weight/bias); h=crossattn(h,cond); x+=h` (**ungated, unmodulated**).
  - mlp: `h=LN(x, affine=F, eps 1e-6); h=h*(1+sc_mlp)+sh_mlp; h=mlp(h); h=h*g_mlp; x+=h`.
  - self-attn: `to_qkv` Linear 1024→3072 → reshape [B,L,3,16,64] → **MultiHeadRMSNorm on q,k** (`normalize(dim=-1)*
    gamma[16,64]*√64`, eps 0, no bias) → SDPA scale 1/√64 → `to_out` 1024→1024.
  - cross-attn: `to_q` 1024→1024, `to_kv` 1024→2048 (k,v), **no qk-norm**, SDPA, `to_out`.
  - mlp: Linear 1024→4096 → **tanh-GELU** → Linear 4096→1024.
- Out head (F32): `layer_norm(h, [C], eps **1e-5**)` (non-affine) → `out_layer` Linear 1024→8 → reshape [1,8,16³].
- Weight keys: `input_layer/out_layer/pos_emb/t_embedder.mlp.{0,2}`; per block `blocks.{i}.` `adaLN_modulation.1`
  (SiLU is .0), `norm2.{weight,bias}` (only norm2 affine), `self_attn.{to_qkv,to_out,q_rms_norm.gamma,k_rms_norm.gamma}`,
  `cross_attn.{to_q,to_kv,to_out}`, `mlp.mlp.{0,2}`.

**`SparseStructureDecoder`** (conv3d VAE, config `ss_vae_conv3d_16l8`): `[1,8,16³] → [1,1,64³]` occupancy logits.
`input_layer` Conv3d 8→512 (k3s1p1). `middle_block` = 2× `ResBlock3d(512)`. `blocks` = ResBlock(512)×2, Upsample
512→128, ResBlock(128)×2, Upsample 128→32, ResBlock(32)×2. `out_layer` = ChannelLayerNorm32(32) → SiLU → Conv3d
32→1 (k3s1p1). All convs **k3 s1 p1** (✅ our `Conv3d` covers this).
- `ResBlock3d(c)`: `norm1(ChannelLN)→SiLU→conv1(k3)→norm2(ChannelLN)→SiLU→conv2(k3)→+x` (identity skip).
- `UpsampleBlock3d(cin→cout)`: `Conv3d(cin→cout·8, k3) → pixel_shuffle_3d(·, 2)` (3D depth-to-space: reshape
  [B,cout,2,2,2,H,W,D]→permute(0,1,5,2,6,3,7,4)→[B,cout,2H,2W,2D]). **New helper.**
- `ChannelLayerNorm3d(c)`: LN over the channel axis of [N,C,D,H,W], **eps 1e-5** (permute→LN(c)→permute back).
- Threshold: `coords = argwhere(occ > 0)[:, [0,2,3,4]]` → active voxels [M,4] at 64³. (logits, no sigmoid.)

**FlowEuler sampler** (`sparse_structure_sampler.params`): steps **25**, cfg_strength **5.0**, cfg_interval
**[0.5,1.0]**, rescale_t **3.0**, sigma_min 1e-5. Schedule `t=linspace(1,0,26)`; warp `t=3t/(1+2t)`. Per step
`(t,t_prev)`: `t_model=1000·t`; if `0.5≤t≤1.0`: `v=(1+5)·v_cond−5·v_uncond` else `v=v_cond`; `x_prev=x−(t−t_prev)·v`.

## Stage 2 — structured latent (sparse) — SEE full sparse spec below

**`SLatFlowModel`** (sparse DiT-L, `slat_flow_img_dit_L_64l8p2`): resolution 64, in/out 8, model 1024, 24 blocks,
patch 2, io_res_blocks 2 (io_block_channels [128]), pe=ape, qk_rms_norm, **attn_mode='full'** (block-diagonal per
batch — no windowing needed for the flow model). Operates on `SparseTensor(feats[N,8], coords[N,4])`. Same modulated
cross-block as stage 1 but sparse (`ModulatedSparseTransformerCrossBlock`). Denormalize `slat = slat*std + mean`
(from `slat_normalization` in pipeline.json).

**SLat VAE decoders** (swin window-8, DiT-B 768/12/12): `slat_dec_gs` (32 gaussians/voxel), `_mesh` (flexicubes),
`_rf`. Use **windowed/serialized** sparse attention (unlike the flow model's full attn).

## Sparse subsystem (the new infrastructure — `trellis/modules/sparse`)

**SparseTensor**: `feats [N,C]` float + `coords [N,4]` int32 `(b,x,y,z)` (values ≤1023). Metadata: `shape=[B,C…]`,
`layout` = per-batch contiguous row ranges (from bincount(coords[:,0])→cumsum; **INVARIANT: batch-contiguous rows**),
`scale`, `spatial_cache` (dict, side-channel for serialization maps + down/up pairing + conv rulebooks). `replace(feats)`
is the workhorse (reuse coords/layout, swap feats). `sparse_batch_broadcast([B,C]→[N,C])` scatters a per-batch vector
to every voxel of that batch (how adaLN modulation applies).

**Feats-only ops (reuse dense IBackend):** SparseLinear (MatMul), activations (SiLU/tanh-GELU), plain LayerNorm32 over
C (DiT path), elementwise/modulation `h*(1+scale)+shift` (after batch-broadcast). No coord change.

**Sparse attention** = permutation + block-diagonal/batched dense SDPA (reuse cuDNN) + inverse permutation. Modes:
- `full`: per batch item (block-diagonal over layout), cu_seqlens. Used by the SLat **flow** model + all cross-attn.
- `windowed` (swin, ws 8): partition into ws³ spatial cubes (+ optional shift w//2), `window_id = batch·OFFSET0 +
  linearized cube`; `fwd=argsort(window_id)`, gather→varlen SDPA per window→scatter by `bwd`. Windows hold only active
  voxels (usually <8³ → varlen). **New index math** (pure int, cached).
- `serialized`: fixed-size ws-chunks along a space-filling curve. `code = vox2seq.encode(coords, {z_order|hilbert},
  permute)` — **must reimplement Morton bit-interleave + 3D Hilbert exactly** (10-bit coords). Sort by code, split into
  ceil(N/ws) **overlapping padded** windows (uniform length ws → batched SDPA), scatter only the valid center via bwd.
- qk_rms_norm: `normalize(dim=-1)·gamma[H,Cd]·√Cd` fp32.

**Sparse conv** (spconv semantics — reimplement the rulebook; no in-tree reference):
- **Submanifold** (stride 1, the common case, every ResBlock conv1/conv2): **output coords = input coords**. Build
  coord hash `(b,x,y,z)→row`; for each voxel×each of K³ kernel offsets δ, if neighbor `(b,x+δ)` exists emit
  `(k, in_idx, out_idx)`; apply `out[o]=bias+Σ_k Σ feats[i]@W[k]` (gather→GEMM W[k][Cin,Cout]→scatter-add). Weight
  layout spconv `[Kx,Ky,Kz,Cin,Cout]`. Cross-correlation, center-aligned.
- **Strided** (down/up, IO stages): generate+dedup output coords, **re-sort rows by batch id** to restore layout
  invariant (cache unsorted order + bwd perm under `conv_{stride}_*`). `SparseInverseConv3d` reuses the forward
  rulebook transposed (needs the matching forward conv to have cached first).
- Down/Up/Subdivide (`spatial.py`): Downsample = coord//factor → unique+scatter-mean (cache pairing); Upsample =
  gather via cached maps; Subdivide = ×2 coords + 2³ children.

**Sparse{Group,Layer}Norm** (VAE only, NOT the DiT): per-batch `feats[layout[k]].permute(1,0).reshape(1,C,Lk)` →
GroupNorm/LayerNorm over the **voxel** axis → permute back. Unusual — replicate exactly if porting the SLat VAE.

**C# design** (`src/HartsyInference.ThreeD/Sparse/`): `SparseTensor` (feats=engine Tensor [N,C], coords=int Tensor
[N,4], host-side layout/cache) + Replace + batch-broadcast. CPU-first for the two novel algorithms (conv rulebook via
`Dictionary<(int,int,int,int),int>`; Morton/Hilbert codes) — parity-gate vs a Python `spconv`/`vox2seq` dump — then
GPU-accelerate hash+gather+scatter. Priority: SparseTensor → feats-ops → full/cross SDPA → submanifold conv +
down/up → windowed/serialized (+Morton/Hilbert) → GPU kernels.
