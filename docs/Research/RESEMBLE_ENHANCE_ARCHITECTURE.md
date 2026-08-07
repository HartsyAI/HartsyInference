# resemble-enhance — Architecture

> Build-ready spec. Sources: `resemble-ai/resemble-enhance` (`denoiser/unet.py`, `enhancer/lcfm/{lcfm,irmae,cfm,wn}.py`,
> `enhancer/univnet/univnet.py`, `melspec.py`, `hparams.py`). Fetched 2026-06-20.

## Pipeline
Two stages @ 44.1 kHz: (1) **Denoiser** — 2D conv UNet over the STFT [mag,cos,sin] → mask + phase rotation →
iSTFT. (2) **Enhancer = LCFM** (IRMAE latent AE + WN CFM) + **UnivNet** vocoder + an internal denoiser as a mel
pre-conditioner. `enhance()`: mel → x = λ·denoised_mel + (1−λ)·mel → LCFM (Gaussian prior τ·randn → CFM solve →
latent ×5 → IRMAE.decode → mel_hat) → UnivNet → wav. `denoise()` runs the denoiser only.

## Denoiser (net-new)
2D UNet (own STFT n_fft 1680 / hop 420). Input [B,3,F,T] = [mag, cos(phase), sin(phase)]. hidden 16→32→64→128→256
(4 down + 2 mid + 4 up), 3×3 Conv2d, `Upsample(0.5/2)`, GroupNorm(dim/16), GELU. Out 3ch: sigmoid mask + tanh
phase residuals → rotate → iSTFT. Keys `denoiser.net.*`.

## Enhancer mel
n_fft 2048, win 2048, hop **420**, n_mels 128, sr 44100, fmax 22050, power 1, slaney norm, custom dB floor −80 +
15 dB headroom, optional preemphasis 0.97.

## LCFM
- **IRMAE** (`lcfm.ae.*`, 1D, no temporal resample): enc mel128→1024 + 4 ResBlocks (dil [1,2,4,8]) + 4× no-bias
  1×1 rank-min → tanh → **latent 64**. dec latent64→1024 + 4 ResBlocks → 1024→128 + head. GroupNorm(32)/GELU.
- **CFM** (`lcfm.cfm.*`): OT-CFM (`μ=t·ψ1+(1−t)·ψ0`, σ 1e-4). Prior `ψ0=τ·randn`, latent ×5. Solvers euler/
  **midpoint**(default)/rk4; steps nfe / (nfe/2) / (nfe/4). Exponential time mapping `divisor=4`. **No CFG.** Time
  emb sinusoidal (128, global); conditioning mel (local).
- **WN estimator** (`lcfm.cfm.net.*`): DiffWave-style **30** dilated gated 1D layers, hidden **512**, kernel 3,
  dilation cycle 5 ([1,2,4,8,16]); local cond = InstanceNorm(mel) → proj 2·hidden added; global = time-emb proj;
  skips summed ×1/√30 → out latent. **NOT the CosyVoice UNet1D estimator.**

## UnivNet vocoder (net-new)
LVCNet kernel-predictor GAN, noise-excited (d_noise 128). 4 LVCBlocks, upsample [7,5,4,3]=420, nc 96, cond =
mel128 + 32 extra. Tanh out. weight_norm (fold at load). Keys `vocoder.*`. (Not HiFi-GAN — `VitsHiFiGan` won't fit.)

## Inference defaults
nfe 64 (hparam) / 32 (API), solver midpoint, lambd 0.5, tau 0.5, 30 s chunks + 1 s xfade. Ckpt
`enhancer_stage2/ds/.../mp_rank_00_model_states.pt` (DeepSpeed). Keys: `denoiser.net`, `lcfm.ae`, `lcfm.cfm.{net,emb}`,
`vocoder`, `normalizer`.

## C# build status (`Models/ResembleEnhance/`)
- [x] [`ResembleWnEstimator`](../../src/HartsyInference.Audio/Models/ResembleEnhance/ResembleWnEstimator.cs) — the WN CFM velocity net (30 dilated gated layers, InstanceNorm local cond, sinusoidal time global), `ICfmEstimator`. Synthetic-forward verified.
- [x] [`ResembleIrmaeDecoder`](../../src/HartsyInference.Audio/Models/ResembleEnhance/ResembleIrmae.cs) — latent→mel (res-stacks + head). Synthetic-forward verified.
- [x] [`ResembleEnhancePipeline`](../../src/HartsyInference.Audio/Pipelines/ResembleEnhancePipeline.cs) — **reuses `ConditionalCfm`** (CFG off) to solve the latent CFM → IRMAE decode → enhanced mel. Synthetic-forward verified (mel → mel).
- [ ] **Staged:** the 2D-STFT **denoiser** UNet (mel pre-conditioner / denoise-only), the **UnivNet** LVCNet vocoder (mel → 44.1 kHz waveform), the slaney-dB mel front-end, midpoint/rk4 + exponential time-mapping (currently euler), τ-scaled prior, the mel `Normalizer`, and the DeepSpeed `.pt` loader.
