# 3D-model e2e + parity grind — session worklog

Mirrors the image-model loop ([E2E_PARITY_WORKLOG.md](E2E_PARITY_WORKLOG.md)) for the `HartsyInference.ThreeD`
package. Goal: take each 🔧 "built, validation-pending" 3D model to real-weight numeric parity (Python
reference diff) + a visual `.glb` mesh e2e, then flip status in [MODEL_STATUS_3D.md](MODEL_STATUS_3D.md).

**Rule:** real gen tests run on GPU (3060 = `CUDA_VISIBLE_DEVICES=0`, 4090 = `=1`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`). CPU only for cheap component parity smoke (see [[gpu-for-real-gen-tests]]).

## Order of attack
1. **TripoSR** (deterministic feed-forward LRM — no diffusion → single clean diff, easiest; ~1.7 GB, fits
   the 3060). Reference oracle already complete. **← in progress.**
2. **Hunyuan3D-2** (flow-match DiT + ShapeVAE + DINOv2-L). Reference dump `dump_hunyuan3d_full_forward.py`
   still has unfilled `TODO[VG]` (model construction + hooks) — build the oracle before running.

---

## TripoSR — in progress (2026-06-30)

Model: `stabilityai/TripoSR` (MIT, ungated). DINO ViT-B/16 (768) → Transformer1D (width 1024, depth 16) →
ConvTranspose upsample → triplane → NeRF MLP (density exp + color sigmoid) → marching cubes. Deterministic.

**Reference oracle — DONE.** Downloaded `model.ckpt` (1.67 GB) + `config.yaml` to `/tmp/triposr`; cloned
`VAST-AI-Research/TripoSR` to `/tmp/TripoSR`. Installed the tsr deps into `/home/hartsy/hfvenv`
(omegaconf 2.3.0, einops 0.7.0, transformers 4.35.0, trimesh 4.0.5, imageio; torchmcubes/rembg stubbed by
the dump script). Ran `tests/python-reference/triposr_reference/dump_triposr_reference.py` (CPU torch) →
wrote `triposr.safetensors` (549 tensors, F32 state dict) + `triposr_ref_io.safetensors` (per-stage IO:
pixels, pos_embed_interp, image_tokens, triplane_tokens, backbone_out, scene_codes, probe density/color,
64³ density grid). `renderer.radius=0.87, density_act=exp, density_bias=-1.0`.

**Component parity (`TripoSrParityTests`, CpuBackend, gated on `TRIPOSR_WEIGHTS`+`TRIPOSR_REF_IO`) — ALL 4 PASS:**
- Decoder probe: density_act maxAbs=1.62e-2, color maxAbs=4.98e-6 — PASS.
- Transformer scene_codes: maxAbs=2.81e-3, corr=1.00000000 — PASS.
- Decoder density grid 64³: maxAbs=0.111, corr=1.00000000 — PASS.
- DINO image_tokens: maxAbs=**0.286 → 1.04e-5**, corr 0.99984 → **1.00000000** — PASS **after fixing the bug below.**

**BUG FOUND + FIXED — DINO ViT position-embedding interpolation missing the +0.1 DINO fudge.**
`DinoViTEncoder.InterpolatePosEmbeddings` used coordinate `scale = src/dst`. HF/DINO `ViTEmbeddings.interpolate_pos_encoding`
adds `+0.1` to the target grid and passes it to `F.interpolate` as a **scale_factor** `(dst+0.1)/src`; with
`align_corners=False` + an explicit scale_factor PyTorch's coordinate scale is `1/scale_factor = src/(dst+0.1)`,
NOT `src/dst`. Off-by-0.1 shifted every bicubic sample → image_tokens maxAbs 0.286 (corr still 0.99984 — a
magnitude/boundary error corr barely catches). Fix: `float scale = src / (dst + 0.1f);` in
`src/HartsyInference.Vision/DinoVit/DinoViTEncoder.cs`. ⚠ **Shared DINO encoder** — also feeds Hunyuan3D's
conditioning path when it uses DinoViT; DINOv2 uses a separate encoder (verify its interp separately).

**GPU e2e (2026-06-30):** `TripoSrGenerationTests.TripoSr_Gpu_ImageToMesh` on the 3060 (chair.png,
background-composited to gray-0.5 + foreground-resized 0.85 via `/tmp/prep_triposr.py` to match TSR's
`ImagePreprocessor`). First run → **empty mesh** (raw RGBA alpha-dropped → no foreground). With correct
composite → mesh renders but is **noise filling the whole [-0.87,0.87]³ box** (3.3M verts / 6.5M tris),
not a chair.

**Localization (Python reference for the *chair*, not the random dump):** TSR chair density grid = min0/
max2707/mean6.2, **1.19% > iso 25** → clean 20k-face chair (skimage). My C# field = mean43.8/**91% > 25**,
capped ≈±60 → noise. Isolation test `TripoSrDiagnosticTests.IsolateChairDinoVsBackbone`:
- (A) my DINO on the chair vs ref chair `image_tokens`: maxAbs 5.4e-4, corr ~1.0 → **DINO correct.**
- (B)/(C) ref/my chair tokens → my **backbone** → scene_codes: maxAbs 3.5e3, corr **−0.004**, std 27 vs
  ref 409, output bounded ≈±60 → **backbone produces uncorrelated garbage.**
- BUT `TripoSrParityTests.Transformer_SceneCodes_MatchReference` (corr 1.0) passed — **on `CpuBackend`**.
  The diagnostic runs `CudaBackend`. ⇒ **suspected CUDA-specific op bug in the Transformer1D backbone**
  (GroupNorm over [1,C,N,1] / ConvTranspose2d upsample / SDPA / Linear), not present on CPU.

**RESOLVED → TripoSR ✅ (2026-06-30).** Per-block std trace vs a Python backbone dump localized the divergence
to *before block 0*: pre-block `h` std **0.036** (CUDA) vs **0.879** (Python) — a 24× collapse. Every backbone
op passed CPU-vs-CUDA in isolation (`CudaOpBisectTests`: GroupNorm 4.8e-7, ConvTranspose2d 0, SDPA cross 2.5e-5,
LayerNorm 4.8e-7, Linear→host-reshape→SDPA chain corr 1.0), so it was a **compose** bug:

> `TripoSrTransformer.Forward` wrote GroupNorm into `gnOut = normed.Reshape([1,C,N,1])` then fed the ORIGINAL
> `normed` to `Transpose2D`. The **CUDA activation cache is keyed by Tensor object identity** — GroupNorm cached
> under `gnOut`, Transpose read `normed` (different object) → cache miss → stale lazily-zeroed host buffer.
> Invisible on CPU (no cache indirection → CPU parity passed). Fix: `gnOut = new([1,C,N,1])` fed straight to
> Transpose2D. See [[cuda-activation-cache-reshape-identity]].

After the fix: backbone scene_codes corr **1.00000000** (std 409.33 vs 409.33); full GPU pipeline → **42k verts /
84k tris**, compact bounds, coherent chair (orthographic-silhouette render confirms seat+back+legs). Also fixed
the DINO **pos-embed interp `+0.1`** bug (image_tokens 0.286→1.04e-5). Docs updated: MODEL_STATUS_3D → ✅,
PARITY_VERIFICATION §Bugs (2 rows), PHASE_11 §3 checked.

**Tests (committable):** `TripoSrParityTests` (CPU numeric, 4/4), `TripoSrGenerationTests` (GPU e2e → `.glb`),
`CudaOpBisectTests` (CPU-vs-CUDA op regression at 3D shapes). Reference oracle: `dump_triposr_reference.py`.

## Hunyuan3D-2 — IN PROGRESS (2026-06-30): architecture confirmed, C# is wrong-architecture → needs rewrite

Downloaded `tencent/Hunyuan3D-2` (ungated): `hunyuan3d-dit-v2-0/model.fp16.safetensors` (4.7 GB, 1645 keys),
`hunyuan3d-vae-v2-0/model.fp16.safetensors` (408 MB), configs; cloned the hy3dgen repo to `/tmp/Hunyuan3D-2`.
Read the authoritative source (`hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`) + dumped real keys.

**⚠ The C# `Hunyuan3DDit` / `Hunyuan3DShapeVae` / `Hunyuan3DConfig` are structural GUESSES that do NOT match the
real architecture. All three need rewrites.** Confirmed spec (checkpoint key prefixes: `model.*`=DiT,
`conditioner.main_image_encoder.model.*`=DINOv2, `vae.*`=ShapeVAE):

### DiT — Flux double/single-stream, **NO RoPE** (`pe=None`; VecSet is permutation-invariant)
config.yaml: `in_channels 64, context_in_dim 1536, hidden_size 1024, mlp_ratio 4.0, num_heads 16, depth 16,
depth_single_blocks 32, theta 10000, qkv_bias True, time_factor 1000`. Structure = **Flux** (reuse the verified
`ChromaDoubleStreamBlock`/single-stream pattern but drop RoPE):
- `latent_in` Lin(64→1024); `time_in` = MLPEmbedder(256→1024→1024, SiLU); `cond_in` Lin(1536→1024).
- timestep_embedding: `t*=1000`, half=128, `cat([cos,sin])`, dim 256 (cos first).
- `double_blocks.{0..15}`: `img_mod.lin`/`txt_mod.lin` (1024→6144, SiLU-then-Linear, `[shift,scale,gate]×2`),
  `{img,txt}_norm1/2` LayerNorm no-affine eps1e-6, `{img,txt}_attn.qkv` (1024→3072, biased), `.norm` =
  **QKNorm = RMSNorm(headDim64)** on q,k (keys `norm.query_norm.scale`/`key_norm.scale`), `.proj`,
  `{img,txt}_mlp.0/2` (1024→4096→1024, **GELU tanh**). Joint attention over **cat([txt,img])** (txt FIRST), split back.
- `single_blocks.{0..31}`: `linear1` (1024→7168 = 3·1024 qkv + 4096 mlp), `linear2` (5120→1024), `modulation.lin`
  (1024→3072, 3-way), `norm` QKNorm, pre_norm LayerNorm no-affine; `out = x + gate·linear2(cat(attn, gelu_tanh(mlp)))`.
- Forward: doubles → `cat([cond,latent],1)` → singles → drop cond prefix → `final_layer`.
- `final_layer` (LastLayer): `adaLN_modulation.1` (SiLU+Lin 1024→2048), **`shift,scale = chunk(2)` — SHIFT FIRST**
  (opposite of Qwen! see [[adaln-continuous-scale-shift-order]]); `x=(1+scale)·norm_final(x)+shift`; `linear` 1024→64.

### Conditioner — DINOv2-**giant** (not Large): 40 layers, hidden 1536, 24 heads, image 518/patch14 (1369 patches
+ cls = 1370 pos), **HAS layerscale** (`layerscale_value 1.0`), NO register tokens, HF ViT key layout. Needs a
`Dinov2Preset.Giant` + verify `Dinov2VisionEncoder` layerscale + 518-px pos-embed interp (own path, cf. the DINOv1
`+0.1` bug — DINOv2 uses antialias interpolate, check separately). Converter must strip `conditioner.main_image_encoder.model.`.

### ShapeVAE decoder (`vae.*`): `num_latents 3072, embed_dim 64, num_freqs 8, include_pi false, heads 16, width 1024,
num_decoder_layers 16, qkv_bias false, qk_norm true, scale_factor 0.9990943042622529`.
- `post_kl` Lin(64→1024); `transformer.resblocks.{0..N}` = self-attn stack over the 3072 latent tokens;
- `geo_decoder`: `query_proj` Lin(**51**→1024) [51 = 3 coords × (1 + 2·8 freqs) Fourier, include_pi false];
  `cross_attn_decoder` = 1 cross-attn block (`c_q` 1024, `c_kv` 1024→2048, q/k `_norm` on headDim64, `ln_1/2/3`,
  `mlp.c_fc` 1024→4096 → `c_proj`); `ln_post`; `output_proj` Lin(1024→**1**) = occupancy. Iso via `scale_factor`.

### Build order (each its own parity-gated step; reference: hy3dgen loaded from the downloaded ckpt)
1. `Dinov2Preset.Giant` + conditioner parity (image → cond tokens [1,1370,1536]).
2. Rewrite `Hunyuan3DDit` → Flux double/single (no RoPE) + config; per-block parity vs a Python dump.
3. Rewrite `Hunyuan3DShapeVae` → post_kl + resblocks + geo_decoder cross-attn; occupancy parity.
4. `Hunyuan3DCheckpointConverter` key tables (`model.`/`vae.`/`conditioner.main_image_encoder.model.` strips).
5. Scheduler (flow-match, shift) + CFG; GPU e2e mesh → inspect `.glb`.
**Watch:** the same [[cuda-activation-cache-reshape-identity]] class (never feed a Reshape of an op's output to the
next op), and QKNorm RMSNorm F32-affine casts. Reference-dump script `dump_hunyuan3d_full_forward.py` `TODO[VG]`
can now be filled from `hy3dgen.shapegen` (pipeline `from_pretrained` → `.model`/`.vae`/`.conditioner`).

**This is a from-scratch architecture build (not a verify) ≈ one focused session per component.**

### Progress (2026-06-30)
- **Component 1 — DINOv2-giant conditioner ✅** (added `Dinov2Preset.Giant` + SwiGLU FFN path to the shared
  `Dinov2VisionEncoder`; last_hidden_state maxAbs 5.3e-3, corr 1.0 vs HF `Dinov2Model`). *(Note: this landed in
  both a user commit `214f19c` and my stash — resolved the stash-pop conflict keeping the committed version.)*
- **Component 2 — DiT ✅** rewritten to Flux double/single-stream (no RoPE): `Hunyuan3DDit.cs` +
  `Hunyuan3DFluxBlocks.cs`, reusing `AdaLNModulation`/`QkNorm`/`SwiGluFfn`. **Velocity corr 0.99999738** vs
  hy3dgen `Hunyuan3DDiT` (CUDA). Bug found+fixed: `timestep_embedding` **max_period=time_factor=1000** (the
  model passes `self.time_factor` as the max_period positional arg), not the 10000 default — wrong base → vec
  corr 0.965 → compounded to velocity corr −0.64. Localized via per-stage element-wise corr trace. See
  [[hunyuan3d-timestep-maxperiod]]. Tests: `Hunyuan3DDitParityTests`; oracle `dump_hunyuan3d_dit.py` (loads the
  hy3dgen module by file path to bypass the diffusers-broken package `__init__`).
- **Component 3 — ShapeVAE ✅** rewritten to the real `vae.*` arch (`Hunyuan3DShapeVae.cs` + `Hunyuan3DVaeBlocks.cs`):
  `post_kl` → 16 `Hunyuan3DVaeResBlock` (fused head-interleaved `c_qkv`, **LayerNorm-64** QK-norm, **erf**-GELU MLP)
  → `Hunyuan3DGeoDecoder` (FourierEmbedder 51-dim `[x, sin(x·2^i), cos(x·2^i)]` no-π, `query_proj`, cross-attn to
  latents via precomputed K/V, `ln_post` eps 1e-5, `output_proj`→1). **Occupancy corr 0.99999518** vs hy3dgen
  `ShapeVAE` on CUDA. Test `Hunyuan3DVaeParityTests`; oracle `dump_hunyuan3d_vae.py` (loads the hy3dgen decode
  classes standalone via fake parent packages, stubbing `hy3dgen.shapegen.utils.logger`, to dodge the
  diffusers-broken package `__init__`). Added `Hunyuan3DConfig.VaeScaleFactor` (0.9990943).
- **Component 4 — pipeline + e2e — IN PROGRESS.** `Hunyuan3DShapePipeline` wired + converter key tables fixed
  (`model.`→DiT, `vae.`→ShapeVAE, `conditioner.main_image_encoder.model.`→Dino) + default preset → Giant.
  **Scheduler = hy3dgen FlowMatchEuler (verified by replicating its math in numpy):** sigmas **ASCEND**
  `linspace(1,1000,steps)/1000` (+ trailing 1.0); DiT fed `t=σ·1000` (and the DiT ×time_factor 1000 again — the
  model's own convention); Euler `x += (σ_next−σ)·noise_pred`; 2-way CFG `uncond+g·(cond−uncond)`; init = pure
  noise; then `latents /= scale_factor` before decode. **Runs e2e on CUDA** (loads 10.5s; steps=1/grid32 → valid
  mesh saved, no crash; **~62s/step** from CPU-resident block glue). Test `Hunyuan3DGenerationTests`.
  **VRAM is the current blocker for multi-step runs:** one DiT forward on a CLEAN 4090 fits, but the CPU-glue DiT's
  transients are heavy and the earlier multi-step OOMs were **residual VRAM from prior crashed runs** (the
  background runner started before the killed orphan released — always `pkill -9 -f testhost; sleep 3` and confirm
  `nvidia-smi` is at baseline before launching). Also lowered the ShapeVAE decode `chunkSize` 32768→4096 (the
  geo-decoder cross-attn scores buffer is `H·chunk·Nlatent` ≈ 6 GB at 32768) and added a per-block `Sync()` in the
  DiT to drain async frees. NEXT: confirm a coherent mesh from a clean-GPU multi-step run, then the GPU-residency
  perf/memory rewrite (mirror Boogu/Qwen device-resident blocks) makes the full 30–50 step run fast + light.
  **ROOT CAUSE FOUND (2026-06-30): async memory-pool race, NOT a logic bug.** Multi-step/large-grid runs crashed
  natively (segfault/hard-abort, no managed exception), but the SAME run with **`CUDA_LAUNCH_BLOCKING=1` passes**
  (steps=4/grid16 → coherent mesh, no crash). The CPU-glue DiT/VAE host ops read `.DataPointer` mid-forward, which
  triggers the lazy `cuStreamSynchronize`+D2H+`cuMemFreeAsync`; interleaved with the stream-ordered pool's
  `cuMemAllocAsync`, this races and corrupts memory once allocations get large/numerous. The verified image models
  don't hit it because they're GPU-resident (never read DataPointer mid-forward). **Workaround: run with
  `CUDA_LAUNCH_BLOCKING=1`** (the DiT is already host-bound so it's not much slower). **Proper fix = the
  GPU-residency block rewrite** (removes the mid-forward host reads → no lazy-sync race AND far less VRAM + ~10×
  faster). Model correctness is DONE (all 3 components corr ~1.0).
  **RESOLVED (2026-07-01): GPU-residency rewrite → Hunyuan3D-2 ✅ fast + robust.** Rewrote the DiT
  (`Hunyuan3DFluxBlocks.cs`) and ShapeVAE (`Hunyuan3DVaeBlocks.cs`) blocks fully device-resident (all glue via
  `IBackend` ops — mirror `ChromaDoubleStreamBlock`; mod params via `SliceLastDim`; fused head-interleaved VAE
  `c_qkv`/`c_kv` split into head-major weights at load). No mid-forward host `DataPointer` reads → **the async race
  is gone (no `CUDA_LAUNCH_BLOCKING` needed) and it's ~13× faster**: 30 steps + grid 128 in **87 s** (was 19 min),
  coherent 160k-tri chair. Parity preserved: DiT corr 0.99999970, VAE corr 0.99999454. **Both 3D models are now
  fully wired, fast, and produce correct meshes.** Only follow-up: the shared C# background-removal tool (§6).
- **Component 4 (orig spec) — pipeline + e2e.** Wire `Hunyuan3DShapePipeline`: DINOv2-giant conditioner (uncond =
  zeros [B,1370,1536]) → FlowMatchEulerDiscreteScheduler (init noise [1,3072,64]; `prev = x + (σ_next−σ)·v`,
  shifted sigmas) with 2-way CFG (guidance ~5) → **latent ÷ scale_factor** → ShapeVAE grid decode → marching
  cubes → `.glb`. Fix `Hunyuan3DCheckpointConverter` key tables (`model.`→DiT, `vae.`→ShapeVAE,
  `conditioner.main_image_encoder.model.`→Dino). Validate on CUDA; inspect the mesh. Then the DiT/VAE
  GPU-residency perf pass (block glue is CPU-resident → 50-step loop is slow).
- **(superseded) old Component-3 spec below, kept for the geo_decoder key detail:** Rewrite `Hunyuan3DShapeVae` to the real `vae.*` arch: `post_kl` Lin(64→1024)
  → `transformer.resblocks.{0..15}` (CLIP-style self-attn: packed `c_qkv`? [dump showed `attention.{q,k}_norm`
  + `c_proj`; confirm packed vs split] + `ln_1/2` + `mlp.c_fc/c_proj`) → `geo_decoder`: Fourier point embed
  (num_freqs 8, include_pi false → 51) → `query_proj` Lin(51→1024) → `cross_attn_decoder` (c_q, packed c_kv
  [2048,1024], q/k_norm, ln_1/2/3, mlp) → `ln_post` → `output_proj` Lin(1024→1) = occupancy. `scale_factor`
  0.9990943 (latent ÷ scale_factor before decode). Oracle: load hy3dgen `ShapeVAE` by file path, dump occupancy
  at fixed query points. Then Component 4: converter key tables + FlowMatchEuler + CFG → e2e mesh.
- **⚠ Perf:** the DiT block glue is CPU-resident (host reshape/QKNorm/layernorm reading DataPointer) — fine for a
  single-forward parity but the 50-step CFG e2e will be slow; do a GPU-residency rewrite (mirror Boogu/Qwen
  double/single blocks) before/after the e2e mesh works.
