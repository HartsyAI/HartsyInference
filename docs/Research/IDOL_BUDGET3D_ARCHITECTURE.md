# Idol, Idol-turbo and Budget3D: architecture, weights, licences

Image → 3D Gaussian splats. Source snapshot 2026-11-17. This records upstream facts the port needs; implementation status lives
in [MODEL_STATUS_3D](../Checklists/MODEL_STATUS_3D.md) and parity evidence in
[PARITY_VERIFICATION](../Checklists/PARITY_VERIFICATION.md). Reference dumps: `tests/python-reference/idol_reference/`.

## Pins

| What | Pin |
|---|---|
| Idol code | github.com/speridlabs/idol-sdk `956439f5e664a7e8c5db25a8055e72d6365aafa5` (Apache-2.0); the commit Budget3D pins |
| Idol weights | huggingface.co/speridlabs/idol `2731c2291fc65a797c5cb627a5e5526fc1490902` (Apache-2.0) |
| Comfy repack (engine source) | huggingface.co/Comfy-Org/Idol `e907d8dca6c465735eaf4ce9b3a3c2c6401d5705` (card: Apache-2.0) |
| Idol-turbo | huggingface.co/speridlabs/idol-turbo `45cdd41f1f4d6d70496daa7d619f57ad79307a37` (Apache-2.0) |
| Budget3D code | github.com/BudgetAny/Budget3D `95bbeab46063375d560f9b20fa7b9d9fedb078f7` (MIT) |
| Budget3D-Idol | huggingface.co/BudgetAny/Budget3D-Idol `2dd6d83c57e7853234dbbb24c54b78edc842d074` (Apache-2.0) |
| Budget3D-Idol-Turbo | huggingface.co/BudgetAny/Budget3D-Idol-Turbo `ba76450efdc888310741659665ec72b8c7d0ebc0` (Apache-2.0) |
| DINOv3 ViT-H+/16 original | facebook/dinov3-vith16plus-pretrain-lvd1689m (gated, DINOv3 License) |
| ComfyUI implementation | comfyanonymous/ComfyUI `comfy/ldm/triposplat/` (Idol is a TripoSplat fine-tune) |

Budget3D is an unofficial community extension of BudgetPix (arXiv 2610.12307, pixel-space image diffusion only); there is no
paper for the 3D method yet, and its quality and speed figures are the authors' own.

## Files

| Engine folder `Models/3D/…` | Source | Bytes | sha256 | Layout |
|---|---|---|---|---|
| `idol/clip_vision/dino_v3_vit_h_bf16.safetensors` | Comfy-Org/Idol | 1,681,247,696 | `a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53` | 615 BF16, HF DINOv3 layout, SwiGLU (`mlp.gate_proj/up_proj/down_proj`, biases) |
| `idol/diffusion_models/idol_fp16.safetensors` | Comfy-Org/Idol | 741,106,994 | `9abf6a99fc6bdad701901b12de8fe1b256f02cae4948f7489d25d86b11b52a93` | 559 F16, idol-sdk keys, no `pos_pe` |
| `idol/vae/flux2-vae.safetensors` | Comfy-Org/Idol | 336,211,292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` | 250 F32 + `bn.num_batches_tracked`, ldm keys (`decoder.mid.attn_1…`), encoder included |
| `idol/vae/idol_decoder_fp16.safetensors` | Comfy-Org/Idol | 576,148,442 | `6a146a1e6b2653f74422f7ea835eb8715c547c7de8fc4545051f15c247ae54a9` | 388 F16, prefixes `octree.` and `gs.` |
| `idol/pos_pe_f32_1x8192x3.bin` | slice of speridlabs/idol `denoiser.safetensors` (`data_offsets [1450889812, 1450988116]`) | 98,304 | `884df4db664a680c100d032c9855397ced6afa28f8984efbee6b77f6ba3efe6b` | raw F32 `[1, 8192, 3]` |
| `idol-turbo/loras/idol_turbo_slow_comfy.safetensors` | speridlabs/idol-turbo | 58,744,176 | `4b27bbbf87d002762d8d0c840fdbf8b48d740c9fac2ba34d4e607196861f843f` | 224 F32, `<module>.lora_down/lora_up.weight`, rank 32, pre-scaled for strength 1 |
| `idol-turbo/loras/idol_turbo_fast_comfy.safetensors` | speridlabs/idol-turbo | 58,744,176 | `1111341fbe27c2e80534fcac236a12c4484b334c407d993d601e9a86f9b36e8a` | as slow |
| `budget3d-idol/budget3d_idol.safetensors` | BudgetAny/Budget3D-Idol | 1,512,422,068 | `8d87669ab02823d6fde862b4f1a0be6628f862ec16bbdb263a0838b5bbc4c1fe` | 641 F32, idol-sdk keys + 82 `bp_*`, no `pos_pe`; metadata `weights: ema, step: 20000` |
| `budget3d-idol-turbo/budget3d_idol_turbo.safetensors` | BudgetAny/Budget3D-Idol-Turbo | 1,629,908,908 | `4f0c31ed6979f80478ce75e5002c81353604bf8560d392fc4ea06687be193c05` | 1089 F32: as above, the four LoRA'd Linears as `.base.weight/bias` + `slow_A/slow_B/fast_A/fast_B` |

Reference-only (not shipped): speridlabs/idol `decoder.safetensors` (203,171,696 B, `9da4d767…`), `decoder_gs.safetensors`
(949,083,036 B, `3087eb8e…`), idol-turbo `idol_turbo_lora.safetensors` (117,484,528 B, `2e4b0892…`, idol-sdk keys
`<module>.slow_A/…`, scale α/r = 128/32 = 4).

**`pos_pe`** is Idol's token positions. It is absent from the Comfy and Budget3D files. The stored buffer equals
`torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(8192)` exactly (ComfyUI computes it that way), and it is
an exact 16³ × 2 stratification (every 1/16 cell holds two points, one per x-half), which Budget3D's octree requires.

The Budget3D checkpoints replace the whole denoiser: the backbone was fine-tuned too (`backbone_lr 5e-5`, adapters ×6), so they
are not an add-on to Idol's weights. `bp_feat_proj.*` is training-only.

## Pipeline (idol-sdk)

1. **Preprocess** (`idol/preprocess.py`): resize the short side to 1024 (Lanczos); BiRefNet mattes an image without real alpha;
   erode alpha by one pixel (`MinFilter(3)`); square crop of side `bbox × margin` (synthetic 1.2, real 1.7) around the bbox
   centre; resize to 1024 (Lanczos); composite on black.
2. **Condition** (`idol/conditioning.py`): `feature1` = DINOv3 ViT-H+/16 `last_hidden_state` on the ImageNet-normalised
   1024² image ([1, 4101, 1280]: CLS, 4 registers, 64×64 patches), then an affine-free `F.layer_norm` (eps 1e-5).
   HF's `last_hidden_state` already includes DINOv3's own final norm, so Idol normalises twice (TRELLIS.2 skips the final norm).
   `feature2` = FLUX.2 VAE encode of `2x − 1`, 2×2 patchify to 128 channels, BatchNorm-normalise with `bn.running_mean/var` and
   `batch_norm_eps` **1e-4** (diffusers config), flatten to 4096 tokens, prefix 5 zero tokens → [1, 4101, 128]. Idol samples the
   posterior; Budget3D's encoder takes the mean, which the engine uses (`VaeEncoder` returns the mean). Negative condition: zeros.
3. **Denoiser** `LatentSeqMMFlowModel` (370.6 M; `config.json` `denoiser.args`): latent 8192 × 16 and camera 1 × 5,
   width 1024, 16 heads × 64, MLP ×4 with tanh GELU. `t_embedder` sinusoidal 256 (cos first, `max_period` 10000) on
   `1000·t` → MLP; `adaLN_modulation = SiLU → Linear(1024, 6144)` shared by all modulated blocks, each block adds its own
   `shift_table [1, 6144]`; order shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. Latent tokens get
   `input_layer(z) + PcdAbsolutePositionEmbedder(pos_pe)` (per axis 170 frequencies: `2^0 … 2^15` then `2^(16·k/154)`,
   angle `2π·x·f`, sin block then cos block per axis, zero-padded to 1024). Two noise-refiner blocks (modulated), two
   context-refiner blocks over `cond_embedder(feature1) + cond_embedder2(feature2)` (not modulated, affine LayerNorm,
   eps 1e-6), `cam_refiner` MLP (5 → 1024 → GELU-tanh → 1024), then 24 joint blocks over `[latent | 4101 image | camera]`.
   Head: affine-free LayerNorm, `(shift, scale) = shift_table[1, 2, 1024] + t_emb`, `x·(1 + scale) + shift`, `out_layer`
   and `cam_out_layer`. The upstream runs block Linears in F16 and norms, embeddings and heads in F32.
4. **RePo learned rotary** (one module per block): `h = LayerNorm(x)` (affine, F32);
   `d = final_map(SiLU(gate_map(h)) · content_map(h))` (1024 → 128 → 48, no biases) reshaped `[L, 16 heads, 3]`;
   angles per axis `d_a · freqs_a · π` with learned `freqs_0 [10]`, `freqs_1 [10]`, `freqs_2 [12]` (`clamp_mul` equals a plain
   product in the forward pass); concatenated to 32 angles per head, applied to q and k as **interleaved** pairs
   `(x[2i], x[2i+1])`. Then the per-head RMS norm `F.normalize(x) · γ[16, 64] · 8` on q and k, **after** the rotation. The angles
   must be computed in F32 (Budget3D measured several percent output drift from BF16 angles).
5. **Sampler**: Euler from t = 1 to 0 on `t' = r·t / (1 + (r − 1)·t)` over `linspace(1, 0, steps + 1)`, r = 3; guided
   velocity `(1 + w)·v(cond) − w·v(zeros)`, w = 3, 50 steps; latent and camera both integrate.
6. **Decode**: `OctreeProbabilityFixedlenDecoder` (4 cross-only adaLN blocks, level embedding `SinusoidalEmbedder(max_period
   1024, two_pi)`) descends to level 8, splitting `num_gaussians / 32` anchor counts among 8 children per cell by systematic
   resampling (torch RNG), anchors uniformly random inside leaf cells. `GaussianFixedlenDecoder` (16 self + cross blocks)
   gives 32 Gaussians per anchor (offset, DC colour, scale, rotation, opacity, offset scale). Activations
   (`idol/gaussian.py`): xyz in `[-0.5, 0.5]³`, scale `sqrt(softplus(raw + softplus⁻¹(0.004))² + 0.0009²)`, rotation
   `normalize(raw + [1, 0, 0, 0])`, opacity `sigmoid(raw + logit(0.1))`; PLY turns z-up to y-up `(x, −z, y)` and
   pre-multiplies the quaternion by `(√½, √½, 0, 0)`.

**Idol-turbo**: rank-32 slow and fast experts on `attn.qkv`, `attn.out`, `mlp.mlp.0`, `mlp.mlp.2` of the noise refiner,
context refiner and 24 joint blocks; slow while `t ≥ 0.5`; no guidance; 5 steps recommended (on the r = 3 grid: four slow
steps at t = 1, 0.923, 0.818, 0.667 and one fast at 0.429). The context refiner is LoRA'd, so the condition depends on the phase.

## Budget3D

The budget picks how many of the 8192 tokens the transformer runs on: `N = 512 + 7·k4 + k2` (512 or 519 … 8192). The Sobol set
is relabelled as an octree: 16³ cells × 2 points, 8³ cubes × 8 cells. Per example, the `k4` most salient cubes open into cells and
the `k2` most salient exposed cells open into points; salience is `log2(1 + sqrt(var + 1e-8) / 0.08)` of the latent over the
merged points; `(k4, k2)` comes from a 24-step bisection of a shared threshold offset (τ = 2.1, 1.5). Token order `[s4 | s2 | s1]`.
Added modules: two `ScaleMixer3D` (mean of children + gated learned correction), `MipNeRFIPE3D` (8 octaves, damped by the
cube's variance), `bp_scale_emb [4, 1024]`, two `RefineStage` (width 128, 4 heads, 2 AdaLN-Zero blocks over a token's 2 or 8
children; the s = 4 path runs stage 4 then stage 2 with a carry) and a parameter-free high-frequency skip
`t / max(t² + 0.25²(1 − t)², 1e-6) · (z − mean over the merged region)`. The dense layout runs Idol's graph unchanged. Sampling:
base model builds the layout at step 0 from noise, rebuilds it from `z0 = z − t·v` at step 5, then freezes it; turbo rebuilds every
step. Full shapes and the key list are in the reference dumps and in the checkpoints' headers.

## Licences

Idol, Idol-turbo and Budget3D weights: Apache-2.0. idol-sdk code: Apache-2.0. Budget3D code: MIT. DINOv3 ViT-H+/16: DINOv3
License (the original is gated; the Comfy repack redistributes it). FLUX.2 VAE: Budget3D sources it from FLUX.2-klein-4B `vae/`
(Apache-2.0); the Comfy file is the FLUX.2 VAE with the same config. BiRefNet matting: MIT.

## Settled in the reference harness (2026-11-17)

- `pos_pe` = `SobolEngine(seed 123)`, exact (all 24,576 values equal) and exactly 16³ × 2 stratified.
- HF `last_hidden_state` is after DINOv3's final norm (the knight image: pre-norm mean |x| 164, after 0.14), so Idol's
  `feature1` is final norm, then affine-free LayerNorm.
- DINOv3 RoPE periods: HF keeps `inv_freq` in float32 even when the model loads in BF16 (`dtype=torch.bfloat16` leaves the
  buffer float32), so Idol's periods are not BF16-rounded; the engine's `Dinov3RopeTables` rounds them for every caller today.
- The Comfy VAE keeps `quant_conv` and `post_quant_conv` under `encoder.` and `decoder.`.
- The CUDA `Linear` takes BF16 and F16 weights directly, so the DINOv3-H+ weights can stay BF16 on the device.

## Open questions

- Whether the Comfy `flux2-vae.safetensors` is byte-identical in values to FLUX.2-klein-4B's `vae/`.
- The TRELLIS and SAM 3D splat builders turn the rotation as `q ⊗ q_R` and Idol's PLY as `q_R ⊗ q` (the second is the orientation of the same splat in the turned frame, consistent with the position turn). They agree only for orientations about x; the engine keeps both (`GaussianSplatActivations.TurnToYUpAfter` and `TurnToYUp`) and has not checked which the TRELLIS reference intends.
- The engine's image recipes pass FLUX.2 BatchNorm eps 1e-5 (`Flux2Recipe.cs`) against the diffusers config's 1e-4; not
  investigated here.
