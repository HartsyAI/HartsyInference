# Algorithmic Step-Level Inference Acceleration — Research Notes

## Summary

Survey of algorithmic / step-level acceleration techniques for diffusion models (2023-2026), scoped to what HartsyInference can adopt: changes to the **sampling loop** and the **backbone forward pass**, plus **weight-loadable** distilled LoRAs/checkpoints. The engine runs its own UNet/DiT/MM-DiT backbones on custom PTX kernels (no PyTorch at runtime), so every technique here is evaluated for whether it is implementable as C# loop/forward code or as a loadable weight artifact, never as a library call.

Five families, ordered by adoption priority for a custom backbone:

1. **Step distillation / few-step samplers** — collapse 20-50 steps into 1-8. Best ones are LoRA-loadable (LCM-LoRA, Hyper-SD, PCM, TCD) and need only a custom scheduler; the rest need distilled checkpoints (same architecture, just new weights + CFG off).
2. **Feature / activation caching** — training-free reuse of forward-pass tensors across steps. Highest leverage because it is pure loop code with no weight changes. The unifying trick for DiT is **residual caching**: cache `Delta = out - in`, on a hit compute `out = in + Delta_cached`.
3. **CFG acceleration** — eliminate or cache the unconditional pass (a direct ~2x). Includes the guidance-scalar embedding that Flux-dev/SD3 already use (one pass instead of two).
4. **Solver / scheduler efficiency** — training-free integration schemes that cut steps at equal quality (DPM-Solver++, UniPC, DEIS, SA-Solver, AYS schedule, restart).
5. **Video / audio / 3D specific** — temporal caching (PAB, AdaCache), causal autoregressive few-step (CausVid, Self-Forcing), and the distilled Wan/LTX/Hunyuan/ACE-Step checkpoints that already exist.

A core engineering observation that recurs across families: build **one exponential-integrator solver core in log-SNR space** plus **one residual-cache mechanism** plus **one consistency-scheduler (predict-x0 then re-noise)**, and most of the 2023-2026 literature drops in as parameterizations of those three primitives.

Notation used throughout: noisy latent `x_t = alpha_t * x_0 + sigma_t * eps`; log-SNR `lambda_t = log(alpha_t / sigma_t)`; step size `h = lambda_t - lambda_s`; guidance scale `w`; relative L1 `L1_rel(a,b) = ||a - b||_1 / ||b||_1`.

---

## 1. Step Distillation / Few-Step Samplers

Distillation collapses the sampling loop into 1-8 steps. The decisive axis for a C# engine is **adoption cost**: a *LoRA-loadable* method needs only a LoRA merge into weights you already load plus a custom scheduler and CFG off; a *new-checkpoint* method needs distilled weights (still your existing architecture, so the forward pass is unchanged); almost none require you to train anything.

Only **two schedulers** unlock the entire LoRA-loadable set:
- **LCM multistep scheduler**: predict-x0, then re-noise *fully* to the next scheduled timestep with fresh noise.
- **TCD scheduler**: predict-x0, then re-noise a *tunable fraction* (`gamma`/`eta`) toward an intermediate time, which is why it scales to 8-50 steps where LCM degrades past ~8.

Both share the same consistency parameterization `f_theta = c_skip(t) * x + c_out(t) * x0_est`. Build it once.

### 1.1 LCM (Latent Consistency Models)
- **Mechanism**: distill the guided PF-ODE so the network predicts the ODE solution (clean latent) directly, enabling 1-4 step jumps.
- **Loop change**: LCMScheduler. At each scheduled `t_n`: (a) one UNet eval gives `x0 = f_theta(z, c, t_n)`; (b) if not last, re-noise forward to `t_{n-1}`: `z = alpha_{t_{n-1}} * x0 + sigma_{t_{n-1}} * eps`, `eps ~ N(0,I)`. Use a short skipping-step schedule (~4 timesteps spread over the 1000-step range, skip `k ~ 20`). CFG is baked into the distilled weights; run a single forward pass with guidance off.
- **Speed/quality**: 2-4 steps (even 1); 4-step competitive with 8-16 step DPM-Solver++; some softness vs the 50-step teacher.
- **Adoption**: REQUIRES-NEW-CHECKPOINT (architecture unchanged; only scheduler + CFG handling change).
- **Cite**: *Latent Consistency Models*, arXiv:2310.04378 — https://arxiv.org/abs/2310.04378

### 1.2 LCM-LoRA
- **Mechanism**: capture the LCM distillation delta as a **LoRA** instead of full weights; a universal plug-in acceleration module for any fine-tune of the same base.
- **Loop change**: identical to LCM (LCMScheduler, single-pass CFG off). The only load-time difference is merging the LoRA `W' = W + alpha * B * A` before inference.
- **Speed/quality**: 4 steps; reported FID 8.76 @ 4 steps / 0.8s vs DPM-Solver++ 8.83 @ 8 steps / 1.6s (LAION-5B-Aesthetics). Softer at 1-2 steps.
- **Adoption**: WEIGHT-LOADABLE (LoRA). Cheapest possible path: LoRA merge + LCMScheduler + CFG off. Universal across SD1.5/SDXL fine-tunes.
- **Cite**: *LCM-LoRA: A Universal Stable-Diffusion Acceleration Module*, arXiv:2311.05556 — https://arxiv.org/abs/2311.05556

### 1.3 SDXL-Turbo / ADD (Adversarial Diffusion Distillation)
- **Mechanism**: score distillation (teacher as soft target) + adversarial loss (discriminator) keeps 1-4 step output sharp.
- **Loop change**: trivial at inference — fixed small set of large timesteps (1-step uses a single high-noise step), CFG off, predict-x0/v then optionally re-noise for 2-4 steps. No consistency math. All complexity is training-only.
- **Speed/quality**: SOTA 1-step; matches SDXL teacher at ~4 steps; beats LCM/prior GANs in 1-2 step regime.
- **Adoption**: REQUIRES-NEW-CHECKPOINT (distilled SDXL-Turbo UNet, architecture unchanged).
- **Cite**: *Adversarial Diffusion Distillation*, arXiv:2311.17042 — https://arxiv.org/abs/2311.17042

### 1.4 SD3.5-Turbo / LADD (Latent Adversarial Diffusion Distillation)
- **Mechanism**: ADD moved into latent space (discriminator on latents, no RGB decode in training), applied to the SD3 MM-DiT.
- **Loop change**: 4 large timesteps, guidance off, plain flow-matching/v-prediction step on the MM-DiT. Same MM-DiT forward as the SD3.5 path; only timestep schedule + CFG-off differ.
- **Speed/quality**: 4 steps matches the SD3.5-Large teacher; ~6-8x faster.
- **Adoption**: REQUIRES-NEW-CHECKPOINT.
- **Cite**: *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation*, arXiv:2403.12015 — https://arxiv.org/abs/2403.12015

### 1.5 DMD / DMD2 (Distribution Matching Distillation)
- **Mechanism**: train a one-step generator whose **output distribution** matches the teacher via the gradient of a KL between two score nets (real-data score, fake-sample score). DMD2 adds a GAN loss, drops the dataset-construction regression step, and supports multi-step.
- **Loop change**: inference is one (or a few) forward passes, no CFG, no special scheduler; 1-step maps noise->image directly, multi-step DMD2 uses a small fixed timestep set with predict-x0 + re-noise (LCM-style). The two auxiliary score nets are training-only.
- **Speed/quality**: 1-step; FID 1.28 ImageNet-64, 8.35 zero-shot COCO-2014, surpassing the teacher at ~500x lower inference cost.
- **Adoption**: REQUIRES-NEW-CHECKPOINT (architecture unchanged).
- **Cite**: *Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)*, arXiv:2405.14867 — https://arxiv.org/abs/2405.14867 (original DMD: arXiv:2311.18828)

### 1.6 Hyper-SD
- **Mechanism**: Trajectory-Segmented Consistency Distillation (split the ODE into segments, enforce per-segment consistency) + human-feedback + score distillation, packaged as a unified LoRA for 1/2/4/8 steps.
- **Loop change**: load the Hyper-SD LoRA (SD1.5 / SDXL / SD3-Medium / FLUX.1-dev variants exist). CFG-preserved variants keep normal CFG (guidance 5-8) with a normal/TCD scheduler; the 1-step unified LoRA uses the TCDScheduler at any of 1/2/4/8 steps with CFG off.
- **Speed/quality**: SOTA 1-8 steps; Hyper-SDXL beats SDXL-Lightning by +0.68 CLIP, +0.51 Aesthetic at 1-step.
- **Adoption**: WEIGHT-LOADABLE (LoRA; full checkpoints also offered). Second-cheapest path after LCM-LoRA, better 1-step quality.
- **Cite**: *Hyper-SD: Trajectory Segmented Consistency Model*, arXiv:2404.13686 — https://arxiv.org/abs/2404.13686

### 1.7 PCM (Phased Consistency Models)
- **Mechanism**: fix LCM's error accumulation by phasing the ODE into sub-trajectories and enforcing self-consistency *per sub-trajectory*, giving deterministic multi-step sampling; optional adversarial loss for the low-step regime.
- **Loop change**: same predict-x0 / re-noise loop as LCM, but timesteps are partitioned at trained phase boundaries and each jump targets the start of its phase. Deterministic by default. LCM-style scheduler, CFG off.
- **Speed/quality**: outperforms LCM across 1-16 steps; stable as steps increase (LCM degrades past ~8). Also applies to few-step T2V.
- **Adoption**: WEIGHT-LOADABLE (LoRAs for SD1.5/SDXL plus full weights).
- **Cite**: *Phased Consistency Models*, arXiv:2405.18407 — https://arxiv.org/abs/2405.18407

### 1.8 SDXL-Lightning
- **Mechanism**: progressive (1->2->4->8) + adversarial distillation; x0-style prediction so it slots into discrete schedulers.
- **Loop change**: pick the checkpoint matching the step budget, fixed timestep subset, CFG off. The 1-step model needs a specific prediction-type handling; 2/4/8-step run with an ordinary discrete scheduler (Euler) over the reduced schedule. Full-UNet and LoRA forms released.
- **Speed/quality**: SOTA 1-step/few-step at 1024px SDXL; 2/4/8-step near-teacher; 1-step weakest.
- **Adoption**: WEIGHT-LOADABLE (LoRA for few-step) or REQUIRES-NEW-CHECKPOINT (full UNet for 1-step).
- **Cite**: *SDXL-Lightning: Progressive Adversarial Diffusion Distillation*, arXiv:2402.13929 — https://arxiv.org/abs/2402.13929

### 1.9 TCD (Trajectory Consistency Distillation)
- **Mechanism**: broaden the LCM consistency boundary with a trajectory consistency function (exponential-integrator form of the PF-ODE) + strategic stochastic sampling that re-noises only partway, controlled by `gamma`, cutting accumulated multi-step error.
- **Loop change — TCDScheduler (implement concretely)**: at `t_n`, one model eval gives `x0 = f_theta(z, c, t_n)`. Instead of jumping straight to `t_{n-1}`, pick an intermediate time `t_s` set by `gamma` (diffusers exposes this as `eta`): `gamma=0 -> t_s = t_{n-1}` (deterministic, no added noise); `gamma=1 -> t_s = t_n` (max re-noise). Form the next latent by a deterministic step to `t_{n-1}` plus fresh noise scaled by the `t_{n-1}`-vs-`t_s` gap (injected noise proportional to `gamma`). Recommended `gamma/eta ~ 0.3`, raised with more steps. CFG off. The key difference from LCM: LCM always re-noises *fully*; TCD re-noises a *tunable fraction*.
- **Speed/quality**: 2-4 steps high quality; unlike LCM it keeps improving to 8-20 steps and can beat its teacher; `gamma` trades detail/stochasticity.
- **Adoption**: WEIGHT-LOADABLE (TCD-SD15 / SD21-base / SDXL LoRAs; same base reuse as LCM-LoRA).
- **Cite**: *Trajectory Consistency Distillation*, arXiv:2402.19159 — https://arxiv.org/abs/2402.19159

### 1.10 FLUX.1 schnell vs dev (guidance + timestep distillation)
- **Mechanism**: **dev** is guidance-distilled from Pro (CFG baked into a guidance-scalar embedding, one pass). **schnell** is timestep-distilled from dev via LADD (1-4 step, Apache-2.0).
- **Loop change**: dev's MM-DiT forward takes a **guidance embedding** as extra conditioning (no double-batch CFG, single forward per step); sampler is flow-matching Euler over ~20-50 steps. schnell uses the same forward but flow-match Euler over 1-4 timesteps only. See 3.2 for the embedding implementation.
- **Speed/quality**: schnell 1-4 steps; dev ~20-50; schnell trades some fidelity for licensing/speed.
- **Adoption**: REQUIRES-NEW-CHECKPOINT (separate schnell/dev weights). Engine note: both require a guidance-scalar embedding input and single-pass (no classic CFG double batch). Hyper-SD also ships a FLUX.1-dev few-step LoRA if a LoRA path on Flux is wanted.
- **Cite**: FLUX.1 model cards — https://huggingface.co/black-forest-labs/FLUX.1-schnell , https://huggingface.co/black-forest-labs/FLUX.1-dev (lineage: ADD/LADD, arXiv:2311.17042 / arXiv:2403.12015)

### 1.11 Rectified Flow / Reflow / InstaFlow
- **Mechanism**: train straight-line flow-matching trajectories, apply **reflow** (re-train on the model's own (noise, image) pairs) to straighten paths so one Euler step is accurate, then distill to one step (InstaFlow).
- **Loop change**: rectified-flow sampling is flow-matching Euler `x_{t-dt} = x_t - dt * v_theta(x_t, t)` over a few uniform `t` steps (already what SD3/Flux use); straighter paths just mean fewer steps suffice. InstaFlow's distilled model is one pass noise->image, CFG off. Reflow data-generation is training-only.
- **Speed/quality**: InstaFlow 1 step, FID 23.3 MS-COCO-2017-5k (vs progressive distillation 37.2).
- **Adoption**: REQUIRES-NEW-CHECKPOINT (InstaFlow) / reflow is REQUIRES-TRAINING. The rectified-flow Euler sampler itself is TRAINING-FREE and is the native sampler for SD3/Flux.
- **Cite**: *InstaFlow: One Step is Enough...*, arXiv:2309.06380 — https://arxiv.org/abs/2309.06380

### 1.12 Shortcut Models
- **Mechanism**: condition the network on both noise level and desired **step size** `d`, with a self-consistency constraint "one step of size 2d = two steps of size d". One training run, no teacher.
- **Loop change**: forward pass gets an extra scalar input `d` alongside `t`. Sampling: choose budget `N`, set `d = 1/N`, iterate `x <- x - d * s_theta(x, t, d)`. `N=1` is one giant leap.
- **Speed/quality**: across 1->N budgets beats consistency models and reflow at equal steps; one model serves all budgets (mostly DiT-scale benchmarks).
- **Adoption**: REQUIRES-NEW-CHECKPOINT / REQUIRES-TRAINING. Needs an added scalar conditioning channel for `d` (a real forward-pass change, unlike the LoRA methods). Only relevant if you train your own backbone.
- **Cite**: *One Step Diffusion via Shortcut Models*, arXiv:2410.12557 — https://arxiv.org/abs/2410.12557

### Family 1 summary

| Technique | Steps | Adoption | Loop change | CFG | arXiv |
|---|---|---|---|---|---|
| LCM | 2-4 (1) | New ckpt | LCM multistep (full re-noise) | baked, off | 2310.04378 |
| **LCM-LoRA** | 4 | **LoRA** | LCMScheduler | off | 2311.05556 |
| SDXL-Turbo / ADD | 1-4 | New ckpt | big-timestep subset | off | 2311.17042 |
| SD3.5-Turbo / LADD | 4 | New ckpt | flow-match Euler, 4 steps | off | 2403.12015 |
| DMD / DMD2 | 1 (multi) | New ckpt | single pass / LCM-style multi | off | 2405.14867 |
| **Hyper-SD** | 1-8 | **LoRA** (also ckpt) | TCD or LCM scheduler | off (some on) | 2404.13686 |
| **PCM** | 1-16 | **LoRA** | phased deterministic | off | 2405.18407 |
| SDXL-Lightning | 1/2/4/8 | LoRA / ckpt | discrete Euler, reduced sched | off | 2402.13929 |
| **TCD** | 2-50 | **LoRA** | TCDScheduler (gamma partial re-noise) | off | 2402.19159 |
| Flux schnell/dev | 1-4 / 20-50 | New ckpt | flow Euler + guidance embed | embedded | model card |
| InstaFlow/reflow | 1 | New ckpt (+training) | flow Euler | off | 2309.06380 |
| Shortcut models | 1-N | New ckpt + arch change | step-size-conditioned | base | 2410.12557 |

---

## 2. Feature / Activation Caching

Training-free reuse of forward-pass tensors across steps. **Highest-priority family** because every technique is implemented entirely inside the sampling loop: intercept a block/feature output, store it, and on a cache hit inject the stored tensor instead of recomputing. No weight changes.

The unifying trick for DiT/MM-DiT: cache the **residual** `Delta = out - in` (the network's contribution that step), and on a hit compute `out = in + Delta_cached`. Correct because the residual changes slowly across adjacent steps even when the input does not. TeaCache, FBCache, and Delta-DiT are all variants of this.

### 2.1 DeepCache (UNet)
- **Cached**: the deep high-level UNet feature at one skip-connection depth `F_cache = U_{m+1}(...)`. The shallow encoder/decoder still run every step.
- **Reuse rule**: fixed 1:N schedule (full compute at steps `{iN}`, reuse otherwise). Optional non-uniform schedule packs full steps where features change fastest via power-spaced indices centered at a timestep `c`.
- **Forward work**: at a cached step run only down-block `m`, then `Concat(D_m(x), F_cache)` into the up-path from the skip point, skipping the deep encoder + bottleneck + deep decoder.
- **Speed/cost**: 2.3x SD1.5 (CLIP -0.05); 4.1x LDM-4-G ImageNet (FID +0.22); up to ~10x aggressive.
- **Training-free**: yes. **UNet-only** (exploits the skip-connection dual path); maps to SD1.5/SDXL backbones, not DiT.
- **Cite**: *DeepCache*, arXiv:2312.00858 — https://arxiv.org/abs/2312.00858

### 2.2 Block Caching ("Cache Me if You Can")
- **Cached**: individual UNet block outputs (attention/resnet), each with its own schedule.
- **Reuse rule**: calibration-derived per-block schedule. Measure each block's relative output change `L1(out_t, out_{t-1}) / norm` across steps (smooth U-shaped curve), cache where below a threshold. Adds a small calibration-fit scale-shift affine correction to reduce artifacts.
- **Forward work**: per block, store last output; on cached steps emit stored (optionally scale-shift corrected). One-time offline calibration to fix the schedule.
- **Speed/cost**: 1.5-1.8x, quality maintained.
- **Training-free**: yes (calibration only).
- **Cite**: *Cache Me if You Can: Accelerating Diffusion Models through Block Caching*, arXiv:2312.03209 — https://arxiv.org/abs/2312.03209

### 2.3 TeaCache (DiT/MM-DiT) — recommended runtime gate
- **Cached**: the full-model residual `Delta = out - in`.
- **Reuse rule (load-bearing)**: proxy = the timestep-embedding-modulated input `F` (the input after the first modulation/AdaLN scale-shift, ~free at the top of the network). Compute `L1_rel(F_t, F_{t+1})`, pass it through a fitted polynomial `f(x) = a0 + a1 x + ... + a4 x^4` (per-model coefficients, corrects input->output scaling bias; gains saturate ~4th order), and **accumulate** `f(L1_rel)` across skipped steps. Keep reusing `Delta_cached` while the running sum `< delta`; the step it crosses `delta`, do a full compute and reset. Thresholds `delta = 0.1` (slow/quality), `0.2` (fast). Per-model coefficients ship for Flux, HunyuanVideo, CogVideoX, etc.
- **Forward work**: compute the first modulation to get `F`; compute `L1_rel` vs previous `F`; polynomial; accumulate. Under threshold -> skip the whole backbone, return `in + Delta_cached`. Else run + overwrite `Delta_cached`. Storage: one residual tensor + previous `F` + scalar accumulator.
- **Speed/cost**: ~4.4x Open-Sora-Plan, ~2x Flux-class image; works for video/image/audio.
- **Training-free**: yes (polynomial is a cheap offline fit).
- **Cite**: *Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model*, arXiv:2411.19108 — https://arxiv.org/abs/2411.19108 ; code https://github.com/ali-vilab/TeaCache

### 2.4 FBCache / First-Block-Cache (DiT/MM-DiT) — recommended first implementation
- **Cached**: (a) first block's residual `r1 = block1(x) - x` from the previous step (the indicator); (b) `hidden_states_residual` = accumulated residual across all blocks after block 1 (`out - block1_out`), injected on a hit.
- **Reuse rule**: compute block 1 this step to get `r1_cur`; `diff = mean(|r1_cur - r1_prev|) / mean(|r1_prev|)`. If `diff < residual_diff_threshold`, cache hit: skip blocks 2..N and set `out = block1_out + hidden_states_residual_cached`. Else run all blocks and refresh both caches. Default threshold ~0.05; ranges 0.02-0.06 (conservative) to 0.06-0.12 (fast); 0.12 cited for Flux-dev fp8 @ 28 steps.
- **Forward work**: always run block 1 (cheap vs full stack); maintain `r1_prev` and `hidden_states_residual`; inject by a single tensor add. The easiest DiT cache to implement and the natural default.
- **Speed/cost**: up to ~2x, quality degrades smoothly with threshold.
- **Training-free**: yes.
- **Cite**: no standalone arXiv; ParaAttention / Comfy-WaveSpeed (chengzeyi): https://github.com/chengzeyi/ParaAttention , https://github.com/chengzeyi/Comfy-WaveSpeed ; HF Diffusers cache docs https://huggingface.co/docs/diffusers/api/cache . Conceptually the measured-first-block-residual special case of TeaCache.

### 2.5 Delta-DiT
- **Cached**: the feature *change* (Delta) of a contiguous DiT block range, using the prior step's input to reduce bias.
- **Reuse rule**: stage-dependent fixed scheme — early steps cache/skip the rear block range (detail), later steps the front range (outline). Split point + interval are per-model hyperparameters, not a runtime threshold.
- **Forward work**: intercept a chosen block span; store its Delta between steps; on cached steps add the stored Delta to the span's input instead of running it.
- **Speed/cost**: ~1.6x at 20 steps (PixArt-alpha / DiT-XL), often equal-or-better quality.
- **Training-free**: yes.
- **Cite**: *Delta-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers*, arXiv:2406.01125 — https://arxiv.org/abs/2406.01125

### 2.6 FORA (Fast-Forward Caching)
- **Cached**: attention output and MLP output within each DiT block.
- **Reuse rule**: fixed interval (static). Recompute at indices divisible by the interval (sweet spot 3), reuse otherwise.
- **Forward work**: per block, store last attn + last MLP output; on cached steps skip those sublayers (residual adds/norms still run).
- **Speed/cost**: several-x with minimal IS/FID change; the baseline static cache.
- **Training-free**: yes.
- **Cite**: *FORA: Fast-Forward Caching in Diffusion Transformer Acceleration*, arXiv:2407.01425 — https://arxiv.org/abs/2407.01425

### 2.7 L2C (Learning-to-Cache)
- **Cached**: individual transformer layer outputs (skipped layers reuse prior step).
- **Reuse rule**: a **trained** input-invariant, timestep-variant router (learned per-layer gate per timestep), frozen into a static mask after offline optimization.
- **Forward work**: load the per-timestep layer mask; reuse prior output for masked layers. Trivial in the loop; cost is the offline router training.
- **Speed/cost**: U-ViT-H/2 up to 93.7% of cache-step compute removable (46.8% overall) at <0.01 FID.
- **Training-free**: **NO** (only non-training-free entry here). Deprioritize unless a specific model justifies per-model training.
- **Cite**: *Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching*, arXiv:2406.01733 — https://arxiv.org/abs/2406.01733

### 2.8 SmoothCache
- **Cached**: per-layer output feature maps per a calibration-derived schedule.
- **Reuse rule**: calibration threshold `alpha`. From a short calibration pass, where a layer's L1 relative representation error across adjacent steps `< alpha`, mark it cacheable at that step.
- **Forward work**: like FORA but with a per-layer, per-step mask.
- **Speed/cost**: ~1.5-1.8x across DiT image/video/audio.
- **Training-free**: yes (calibration).
- **Cite**: *SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers*, arXiv:2411.10510 — https://arxiv.org/abs/2411.10510 ; code https://github.com/Roblox/SmoothCache

### 2.9 ToCa / DuCa / ClusCa (token-level)
- **ToCa**: cache features for most tokens, recompute an important subset every step. Token scores: influence on other tokens, self/control ability, caching frequency, spatial distribution; per-layer cache ratios. 2.36x OpenSora / 1.93x PixArt-alpha. arXiv:2410.05317 — https://arxiv.org/abs/2410.05317
- **DuCa**: alternate aggressive and conservative cache steps (fixed dual cycle), recomputed tokens chosen **randomly** (matches/beats score-based). Higher acceleration than ToCa at matched quality. arXiv:2412.18911 — https://arxiv.org/abs/2412.18911
- **ClusCa**: orthogonal spatial axis — cluster tokens each step, compute ~1 representative per cluster, broadcast to members. 4.96x Flux. Composes on top of a temporal cache. arXiv:2509.10312 — https://arxiv.org/abs/2509.10312
- **Forward work**: all three need gather/scatter + variable-length/partial attention in the PTX kernels. **Most kernel-invasive; defer** until residual-cache wins are banked.
- **Training-free**: yes.

### 2.10 TaylorSeer (cache-then-forecast)
- **Cached**: a few recent fully-computed feature values per block (samples along the time-trajectory) + finite-difference derivatives.
- **Reuse rule**: fixed compute interval N + Taylor prediction. On skipped steps predict `F(t) ~ F(t0) + F'(t0) dt + (1/2) F''(t0) dt^2 + ...` with derivatives from finite differences of the cached full-compute points (no extra forward passes).
- **Forward work**: maintain a short ring buffer per block; compute FD derivatives; evaluate the Taylor polynomial on skipped steps and inject. **Forecast the residual instead of reusing it — composes with FBCache/TeaCache.**
- **Speed/cost**: ~3.53x latency on Flux.1-dev (near lossless), ~4.65x on HunyuanVideo; strictly better quality-per-speedup than reuse-only at high ratios.
- **Training-free**: yes. The recommended quality upgrade once basic residual caching works.
- **Cite**: *From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers*, arXiv:2503.06923 — https://arxiv.org/abs/2503.06923 ; code https://github.com/Shenyi-Z/TaylorSeer

### 2.11 PAB (Pyramid Attention Broadcast) — video
- **Cached**: attention outputs, separately for spatial / temporal / cross attention (FFN still runs).
- **Reuse rule**: fixed per-attention-type broadcast ranges (a pyramid). Lowest-variance attention broadcast over the widest step range, highest-variance over the narrowest. Reference (VideoSys): spatial_range=2, temporal_range=3, cross_range=5, only within stable middle windows spatial/temporal [100,800], cross [100,900] (never first/last steps).
- **Forward work**: per attention module cache the last output; on broadcast steps skip QKV+softmax+proj and reuse. Three independent broadcast counters. Also offers broadcast sequence parallelism for multi-GPU.
- **Speed/cost**: 1.26-1.32x single-GPU; up to 10.5x on 8 GPUs (large factor is the parallelism, not caching alone); negligible quality loss.
- **Training-free**: yes. Portable to any DiT with separable spatial/temporal/cross attention (Wan, LTX, HunyuanVideo).
- **Cite**: *Real-Time Video Generation with Pyramid Attention Broadcast*, arXiv:2408.12588 — https://arxiv.org/abs/2408.12588

### 2.12 AdaCache — video
- **Cached**: DiT block residuals with a per-video, per-region schedule.
- **Reuse rule**: runtime content-adaptive. Measure a block-change metric, consult a rate-distortion table for the next cache interval (recompute on large change). **MoReg** (Motion Regularization) computes a motion score from latent frame differences and scales caching aggressiveness by motion (high motion -> recompute more).
- **Forward work**: track block-change metric + motion score each step; set how many upcoming steps reuse cache; inject cached block residuals on skips.
- **Speed/cost**: up to 4.7x (Open-Sora 720p 2s), no reported quality drop.
- **Training-free**: yes.
- **Cite**: *Adaptive Caching for Faster Video Generation with Diffusion Transformers*, arXiv:2411.02397 — https://arxiv.org/abs/2411.02397 ; code https://github.com/AdaCache-DiT/AdaCache

### Family 2 summary

| Technique | Backbone | What's cached | Reuse rule | Speedup | Train-free | arXiv |
|---|---|---|---|---|---|---|
| DeepCache | UNet | deep skip feature | fixed 1:N | 2.3-4.1x | yes | 2312.00858 |
| Block Caching | UNet | per-block outputs | calibration schedule | 1.5-1.8x | yes | 2312.03209 |
| **TeaCache** | DiT | full-model residual | accumulated poly-rescaled L1, delta 0.1/0.2 | 2-4.4x | yes | 2411.19108 |
| **FBCache** | DiT | stack residual + r1 | rel-absmean of r1 < ~0.05 | ~2x | yes | (ParaAttn) |
| Delta-DiT | DiT | block-span delta | stage-based (rear/front) | 1.6x | yes | 2406.01125 |
| FORA | DiT | attn+MLP outputs | fixed interval ~3 | several-x | yes | 2407.01425 |
| L2C | DiT | per-layer outputs | trained router | ~46% | **no** | 2406.01733 |
| SmoothCache | DiT | per-layer features | calibration L1 < alpha | 1.5-1.8x | yes | 2411.10510 |
| ToCa/DuCa/ClusCa | DiT | per-token features | token scores / cycle / cluster | 1.9-4.96x | yes | 2410.05317 etc |
| TaylorSeer | DiT | features + derivatives | interval + Taylor forecast | 3.5-4.7x | yes | 2503.06923 |
| PAB | video DiT | attention outputs | pyramid broadcast ranges | up to 10.5x | yes | 2408.12588 |
| AdaCache | video DiT | block residuals | runtime content + MoReg | up to 4.7x | yes | 2411.02397 |

---

## 3. CFG Acceleration

CFG computes `pred = uncond + w * (cond - uncond)`, normally **two forward passes per step**. Eliminating, caching, or skipping the uncond pass is a direct ~2x. Three families: guidance-distilled checkpoints, training-free CFG skipping/interval, training-free uncond caching/extrapolation.

### 3.1 Guidance Distillation (w-conditioning, Meng et al.)
- **Mechanism**: distill the two-pass guided prediction into a single network that takes `w` as input.
- **Loop change**: embed scalar `w` with a Fourier/sinusoidal embedding (same construction as the timestep), MLP, add to the conditioning vector alongside the timestep embedding. After distillation the loop calls the model once per step (no separate uncond pass, no `w*(cond-uncond)` subtraction). To implement: add a `guidance_in` embedder parallel to `time_in`, sum into the global modulation vector.
- **Speed/quality**: ~2x per-step (halves NFEs), quality comparable to two-pass CFG.
- **Adoption**: requires a guidance-distilled checkpoint (training). NOT training-free.
- **Cite**: *On Distillation of Guided Diffusion Models*, arXiv:2210.03142 — https://arxiv.org/abs/2210.03142

### 3.2 Flux-dev / SD3 guidance embedding (the concrete implementation)
- **Mechanism**: Flux-dev and SD3 are already guidance-distilled; the guidance scale is fed as a timestep-like scalar embedding, so the model runs one pass and there is no uncond pass.
- **Exact implementation (Flux reference)**:
```
vec = time_in(timestep_embedding(t, 256))
vec = vec + guidance_in(timestep_embedding(guidance, 256))   # guidance-distilled only
vec = vec + vector_in(pooled_text)
```
  - `timestep_embedding(guidance, 256)` is a 256-dim sinusoidal embedding of the guidance scalar, exactly like the timestep.
  - `guidance_in` is an MLPEmbedder (256 -> hidden, e.g. 3072): Linear, SiLU, Linear.
  - The result is added into the global modulation vector `vec` that drives AdaLN in every double- and single-stream block.
  - **For the backbone**: implement CFG as a constructor flag. If `guidance_embed == true`, add a `guidance_in` MLP, accept a `guidance` float in the forward signature, embed it, add to `vec`. The sampling loop passes the user's guidance scale (e.g. 3.5 for Flux-dev) and never duplicates the batch for uncond. CFG becomes a single scalar parameter.
  - Flux-**schnell** does NOT take a guidance embedding (guidance fixed); Flux-**dev** does. SD3 follows the same pattern.
- **Speed/quality**: strict 2x vs naive CFG, no extra quality cost (intended operating mode).
- **Adoption**: requires the distilled checkpoint; the engine side is just implementing the embedding path. Caveat: a true negative-prompt branch is no longer free; supporting negative prompts on a distilled model means re-introducing a second pass ("true CFG").
- **Cite**: Flux — https://github.com/black-forest-labs/flux ; *Demystifying Flux Architecture*, arXiv:2507.09595 — https://arxiv.org/abs/2507.09595 . SD3 — *Scaling Rectified Flow Transformers...*, arXiv:2403.03206 — https://arxiv.org/abs/2403.03206

### 3.3 AGD (Adapter Guidance Distillation)
- **Mechanism**: approximate CFG in one pass via lightweight adapters (~2% params) on a frozen base, trained on CFG-guided trajectories.
- **Loop change**: run the model once per step with adapters active.
- **Speed/quality**: ~2x; comparable or superior FID to CFG; distillable on a single 24GB GPU for ~2.6B models.
- **Adoption**: requires an adapter-distillation pass (cheap, not training-free).
- **Cite**: *Efficient Distillation of Classifier-Free Guidance using Adapters*, arXiv:2503.07274 — https://arxiv.org/abs/2503.07274

### 3.4 Limited Interval Guidance (Kynkaanniemi et al.) — best training-free win
- **Mechanism**: apply CFG only in a middle band of noise levels; disable at high noise (early) and low noise (late). Guidance is harmful early, unnecessary late, beneficial in the middle.
- **Loop change**: per step, if sigma is outside `[sigma_lo, sigma_hi]`, **skip the uncond pass** and use the conditional alone (`w = 1`); if inside, normal two-pass CFG. A per-step boolean gate around the CFG block, using the sigma schedule the sampler already has. EDM2 setting: interval [0.19, 1.61] normalized sigma, `w = 2.0` inside. Practical SDXL port: enable CFG only in the middle ~60-80% of steps.
- **Speed/quality**: free quality gain *and* speedup; ImageNet-512 FID 1.81 -> 1.40. Quality improves (a Pareto win), so no quality cost. Speedup scales with how much of the schedule sits outside the interval.
- **Adoption**: TRAINING-FREE; works on any pretrained CFG model and any sampler. Single highest-value training-free CFG win.
- **Cite**: *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models*, arXiv:2404.07724 — https://arxiv.org/abs/2404.07724 ; code https://github.com/kynkaat/guidance-interval

### 3.5 Adaptive Guidance / LinearAG
- **Mechanism**: skip the uncond pass once guided and conditional directions have converged (mostly the second half); optionally replace early uncond evals with an affine extrapolation of past score estimates.
- **Loop change**: AG drops the uncond pass for the remaining steps after convergence. LinearAG reuses past score estimates with a cheap affine transform to approximate uncond instead of a full eval.
- **Speed/quality**: ~25% compute reduction with CFG-level quality (~50% of distillation's speedup), and **retains negative-prompt support**.
- **Adoption**: TRAINING-FREE (the NAS search is offline/one-time).
- **Cite**: *Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models*, arXiv:2312.12487 — https://arxiv.org/abs/2312.12487

### 3.6 CFG-Cache (FasterCache)
- **Mechanism**: within a timestep the uncond output is highly redundant with the cond output; cache the cond->uncond relationship in the frequency domain and reuse it on cached steps.
- **Loop change**: on compute steps run both passes and store the frequency-domain bias; on reuse steps run only the conditional pass and reconstruct uncond by applying the stored bias (with dynamic high/low-frequency enhancement). Pairs with a cross-timestep feature cache in the full FasterCache.
- **Speed/quality**: up to 1.67x (Vchitect-2.0, full method); quality comparable.
- **Adoption**: TRAINING-FREE.
- **Cite**: *FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality*, arXiv:2410.19355 — https://arxiv.org/abs/2410.19355

Note: CADS (arXiv:2310.17347) and online-feedback dynamic CFG (arXiv:2509.16131) are diversity/quality CFG scheduling, **not** acceleration — they do not remove the uncond pass.

### Family 3 summary

| Technique | Loop change | Speedup | Quality | Train-free | arXiv |
|---|---|---|---|---|---|
| Guidance distill (w-cond) | add w embed; 1 pass/step | ~2x | == CFG | no (ckpt) | 2210.03142 |
| **Flux-dev/SD3 guidance embed** | `vec += guidance_in(...)`; 1 pass | 2x | == | no (ckpt); engine just implements | 2403.03206 |
| AGD (adapters) | adapters; 1 pass | ~2x | == or better | no | 2503.07274 |
| **Limited interval guidance** | gate CFG on sigma; else 1 pass | up to ~2x + FID gain | improves | **yes** | 2404.07724 |
| Adaptive Guidance / LinearAG | drop uncond late; affine early | ~25% | == CFG, keeps neg | **yes** | 2312.12487 |
| CFG-Cache (FasterCache) | cache cond->uncond freq bias | up to 1.67x | == | **yes** | 2410.19355 |

---

## 4. Solver / Scheduler Efficiency

Training-free changes to the integration scheme or the timestep/sigma schedule that cut NFE at fixed quality. Backbone weights untouched. Solvers and schedules **compose** (multiplicative).

**Prediction types and log-SNR time**: exponential-integrator solvers (DPM-Solver, DEIS, UniPC, SA-Solver) work in log-SNR time `lambda = log(alpha/sigma)`, step `h = lambda_t - lambda_s`, and convert the model output to a consistent prediction:
- eps-prediction (DDPM/SD1.5): `x0 = (x_t - sigma_t * eps) / alpha_t`
- v-prediction (SD2.x): `v = alpha_t * eps - sigma_t * x0`, convert analytically
- x0/data-prediction: DPM-Solver++ form, recommended for guided sampling (stable under large guidance)
- flow-matching velocity (SD3/Flux): `u = x_1 - x_0`, equivalent to a diffusion ODE with `alpha_t = 1-t`, `sigma_t = t`; convert velocity->x0 (`use_flow_sigmas`/`flow_prediction`).

### 4.1 Euler / Heun (baseline)
- Euler (1st): `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * d_i`, `d_i = (x_i - x0_hat)/sigma_i`. ~30-50 steps for SD quality.
- Heun (2nd RK, EDM): adds a corrector eval at the predicted point and averages slopes; **2 NFE/step**. Lower per-step error; the clean higher-order option for flow-matching (Flux/SD3).
- Training-free. **Cite**: *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM), arXiv:2206.00364 — https://arxiv.org/abs/2206.00364

### 4.2 DPM-Solver
- Exponential integrator: integrate the linear term exactly in lambda-space, approximate only the eps-term. Single-step orders 1/2/3 (1st = DDIM). High quality in ~10-20 steps.
- Training-free. **Cite**: arXiv:2206.00927 — https://arxiv.org/abs/2206.00927

### 4.3 DPM-Solver++ (2M, 2S, SDE) — highest priority solver
- Data-prediction (x0) reformulation for CFG stability + multistep (Adams-type) mode.
- **Update equations** (x0 form, `h_i = lambda_i - lambda_{i-1}`):
  - First-order (= DDIM): `x_t = (sigma_t/sigma_s) * x_s + alpha_t * (1 - e^{-h}) * x0_hat(x_s)`
  - **2M** (step i, `r_i = h_{i-1}/h_i`): `D_i = (1 + 1/(2 r_i)) * x0_hat^{(i)} - (1/(2 r_i)) * x0_hat^{(i-1)}`, then `x_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) * x_{t_{i-1}} + alpha_{t_i} * (1 - e^{-h_i}) * D_i`
  - **2S** (single-step): evaluate at an intermediate lambda, same finite-difference form, 2 NFE/step
  - **SDE-2M**: add a noise term scaled by `sigma_t * sqrt(1 - e^{-2h})` (analytic reverse-SDE variance); orders 1-2 only
- **Buffers**: store the previous step's x0_hat (one tensor for 2M; 3M stores two). Stabilizers: `lower_order_final`, `euler_at_final` (SDE), `final_sigmas_type="zero"`.
- **Steps**: 2M high-quality guided at ~15-20 (ok at ~10); de-facto SDXL default. SDE variant better detail/diversity at slight speed cost. `order=2` for CFG, `3` unconditional.
- Training-free. **Cite**: *DPM-Solver++*, arXiv:2211.01095 — https://arxiv.org/abs/2211.01095

### 4.4 DEIS
- Exponential integrator + Adams-Bashforth multistep. rhoAB variant reparameterizes to shrink error.
- eps-prediction; buffer of last (order-1) eps; update `x_t = (linear) * x_s + sum_j c_j * eps_{i-j}` with precomputable Lagrange-integral coefficients. 1 NFE/step.
- 4.17 FID @ 10 NFE, 3.37 FID @ 15 NFE (CIFAR-10); slightly ahead of single-step DPM-Solver at very low NFE.
- Training-free. **Cite**: *Fast Sampling of Diffusion Models with Exponential Integrator (DEIS)*, arXiv:2204.13902 — https://arxiv.org/abs/2204.13902

### 4.5 UniPC — best very-low-step
- Unified predictor (arbitrary order) + unified corrector. UniC refines the current step using the model eval already computed for the next step, so it raises effective order with **no extra NFE**.
- Data-prediction skeleton (shares ~90% with DPM-Solver++). Predictor = generalized multistep update via a coefficient matrix from the last p points; corrector = redo the prior step's update with one more known data point folded in. Buffers: last (order-1) x0_hat. Corrector coefficients from a small Vandermonde-like solve over log-SNR offsets (precompute per schedule).
- 3.87 FID CIFAR-10, 7.51 FID ImageNet-256 @ 10 NFE; matches/beats 2M at 8-10 steps.
- Training-free. **Cite**: *UniPC*, arXiv:2302.04867 — https://arxiv.org/abs/2302.04867

### 4.6 SA-Solver
- Stochastic Adams: Adams-Bashforth predictor + Adams-Moulton corrector on a variance-controlled SDE. Default predictor_order=3, corrector_order=4. Noise-scale `tau(t)`: tau=0 deterministic ODE, tau>0 injects analytic noise. Data-prediction. Buffers: past data-predictions.
- SOTA/comparable few-step; the stochasticity buys detail/diversity at moderate NFE (~15-30).
- Training-free. **Cite**: *SA-Solver*, arXiv:2309.05019 — https://arxiv.org/abs/2309.05019

### 4.7 Restart Sampling
- A loop strategy: alternate a deterministic backward ODE for several steps with a large forward-noise burst to jump back up ("restart"), then re-descend. The noise injection contracts accumulated ODE error.
- Wrap any deterministic solver; define restart intervals `[t_min, t_max]` and repeat K; within: ODE down to t_min, then `x_{t_max} = x_{t_min} + sqrt(sigma^2_{t_max} - sigma^2_{t_min}) * z`, repeat K, continue.
- ~10x on CIFAR-10, ~2x on ImageNet-64 vs prior SDE; better quality-vs-diversity on SD. Best at mid-NFE.
- Training-free. **Cite**: *Restart Sampling for Improving Generative Processes*, arXiv:2306.14878 — https://arxiv.org/abs/2306.14878

### 4.8 AYS (Align Your Steps) — optimized schedule
- A schedule, not a solver. Choose discretization timesteps by minimizing a KL upper bound between true and discretized SDE, solved offline per model. Output: a fixed model-specific list of optimal sigmas, interpolated to your step count.
- **Implement**: ship the precomputed sigma tables + log-linear interpolation to N steps (log reversed sigmas, linear interp, exp back); feed to any solver. Published 11-point tables:
  - SD1.5: `[14.615, 6.475, 3.864, 2.695, 1.884, 1.394, 0.964, 0.652, 0.398, 0.152, 0.029]`
  - SDXL: `[14.615, 6.318, 3.768, 2.181, 1.341, 0.862, 0.555, 0.380, 0.233, 0.111, 0.029]`
  - SVD (video): `[700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002]`
- Strong at ~10 steps; prediction-type agnostic. No published table for Flux/SD3 (use native shifted schedule or compute your own).
- Training-free. **Cite**: *Align Your Steps*, arXiv:2404.14507 — https://arxiv.org/abs/2404.14507

### 4.9 Flow-matching application (Flux, SD3)
- Default sampler is Euler (`FlowMatchEulerDiscreteScheduler`) with a resolution-dependent `shift` warping the schedule (more steps at high noise for larger images). Near-straight paths mean few Euler steps go far (1-4 distilled, ~20-28 not).
- Higher order: **Heun** is the clean RF option. DPM-Solver++/UniPC apply by converting velocity->x0 and expressing the RF schedule in (alpha, sigma) via flow-sigmas + flow_shift. Use the flow-sigma path, not Karras sigmas (naive DPM++ tuned for VP diffusion is not ideal for RF).

### Family 4 summary

| Technique | Type | Buffers | Pred | Steps->quality | NFE/step | arXiv |
|---|---|---|---|---|---|---|
| Euler | ODE 1st | none | any | ~30-50 | 1 | 2206.00364 |
| Heun | ODE 2nd | none | any | ~2N NFE | 2 | 2206.00364 |
| DPM-Solver | exp-int 1/2/3 | none | eps | ~10-20 | 1-3 | 2206.00927 |
| **DPM-Solver++ 2M** | exp-int multistep | prev x0 | x0 | ~15-20 (10 ok) | 1 | 2211.01095 |
| SDE-DPM++ 2M | SDE multistep | prev x0 | x0 | ~20 better detail | 1 | 2211.01095 |
| DEIS | exp-int AB | last k-1 eps | eps | 4.17 FID@10 | 1 | 2204.13902 |
| **UniPC** | pred+free corr | last k-1 x0 | x0 | **8-10** | 1 | 2302.04867 |
| SA-Solver | stoch Adams | past x0 | x0 | ~15-30 | 1 | 2309.05019 |
| Restart | loop wrapper | none | any | 2-10x vs SDE | varies | 2306.14878 |
| **AYS** | schedule | const table | any | strong @10 | n/a | 2404.14507 |

**Engine note**: build one exponential-integrator core in log-SNR space (the (alpha,sigma)->lambda machinery + eps/v/x0/velocity->x0 converter). DPM-Solver++(2M) and UniPC then share ~90% of the code; DEIS/SA-Solver reuse the buffers/coefficient infra; Restart is a thin wrapper; AYS is a drop-in sigma table.

---

## 5. Video / Audio / 3D Specific Acceleration

### 5.1 Video

**CausVid** (arXiv:2412.07772 — https://arxiv.org/abs/2412.07772): distill a bidirectional teacher into a causal autoregressive student via DMD, run few-step with KV caching. Loop: causal attention masking (frame i attends only to <=i); per-layer KV cache so each new frame/chunk reuses prior K/V; 4 denoising steps per chunk, stream frames out. 50->4 steps, 9.4 FPS, VBench-Long 84.27. WEIGHT-LOADABLE (load a CausVid-distilled Wan checkpoint/LoRA) / REQUIRES-TRAINING to distill new; the causal-mask + KV-cache rollout is loop code to add.

**Self-Forcing** (arXiv:2506.08009 — https://arxiv.org/abs/2506.08009): AR self-rollout with a rolling KV cache + few-step student, removing exposure bias. Same loop as CausVid (causal mask + rolling KV cache + few-step). Real-time, sub-second latency on one GPU. WEIGHT-LOADABLE / REQUIRES-TRAINING. This is the technique behind Matrix-Game 2.0 (see 5.4), directly relevant to the Interactive package.

**FastWan / FastVideo** (https://github.com/hao-ai-lab/FastVideo): joint DMD step-distillation + Video Sparse Attention. Load the distilled 1.3B Wan checkpoint, run 3 steps. 5s 480p in ~1s (H200). WEIGHT-LOADABLE (FastWan2.1 diffusers checkpoint exists).

**Loadable distilled Wan artifacts** (no training needed, just LoRA merge + 4-step scheduler + CFG off):
- Wan2.1 CausVid LoRA (lightx2v/Wan2.1-T2V-14B-CausVid; linoyts/causvid-distilled-wan-21) — 4-step, ~12x. Also `lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v`.
- **Wan2.2-Lightning** (github.com/ModelTC/Wan2.2-Lightning; lightx2v/Wan2.2-Distill-Loras) — native 4-step, dual-noise rank-64 LoRA, ~20x, T2V+I2V. Current best loadable Wan2.2 accelerator.

**LTX-Video distilled** (Lightricks/LTX-Video-0.9.7-distilled, 0.9.8-13B-distilled): guidance + timestep distilled. Set guidance_scale=1.0 (skip the CFG double-forward), num_inference_steps 4-10 (8 recommended). 0.9.7+ adds a spatial latent upscaler (generate low-res then upscale+refine, a two-pass loop). ~6-8x. WEIGHT-LOADABLE.

**PAB, TeaCache, AdaCache** for video: see sections 2.11, 2.3, 2.12. All training-free, portable to Wan/LTX/HunyuanVideo/GameCraft.

**Sliding Tile Attention (STA)** (arXiv:2502.04507 — https://arxiv.org/html/2502.04507v1): replace dense 3D attention with hardware-aligned tile-local sliding windows (a PTX/CUDA kernel). Attention 2.8-17x over FA2; HunyuanVideo 945->685s training-free. TRAINING-FREE as a kernel swap; best with sparse-distill finetune. Relevant to HunyuanVideo/GameCraft.

**Block Cascading** (arXiv:2511.20426): for block-causal models, start a future block from a partially-denoised predecessor so multiple blocks denoise in parallel. ~2.2x but needs ~5 GPUs. TRAINING-FREE; only for multi-GPU block-causal setups.

### 5.2 Audio / Music

**ACE-Step distillation** (arXiv:2506.00045 — https://arxiv.org/abs/2506.00045; v1.5 arXiv:2602.00744): ACE-Step supports few-step. A distillation protocol compresses the DiT trajectory from 50 to 4-8 steps (authors report it also improves SNR). v1.5 <2s/song on A100, >100x vs 50-step, <4GB VRAM. WEIGHT-LOADABLE (distilled checkpoint). HartsyInference already implements the ACE-Step DiT, so the work is loading the distilled weights + a short flow schedule.

**Presto!** (arXiv:2410.05167 — https://arxiv.org/abs/2410.05167): dual distillation — score-based DMD (steps) + layer distillation (per-step cost). 10-18x (230/435ms), improved diversity. WEIGHT-LOADABLE / REQUIRES-TRAINING.

**Music Consistency / AudioLCM** (arXiv:2406.00356 — https://arxiv.org/abs/2406.00356; MusicCM arXiv:2404.13358): latent consistency distillation, 2-5 step (AudioLCM ~2 steps). Consistency sampler (predict x0, optionally re-noise, repeat). WEIGHT-LOADABLE. Caveat: pure consistency on long audio historically needs up to ~16 steps and struggles past ~10s clips.

**ARC post-training** (arXiv:2505.08175 — https://arxiv.org/abs/2505.08175; Stable Audio Open Small): distillation-free adversarial acceleration (relativistic GAN + contrastive discriminator for prompt adherence). ~12s of 44.1kHz stereo in ~75ms (H100). WEIGHT-LOADABLE (public weights); no teacher to host at inference.

### 5.3 3D

TripoSR / feed-forward triplane regression is already single forward pass — no sampling loop to shorten; acceleration there is kernel/precision-level. Few-step methods apply only to diffusion-based 3D (Hunyuan3D-2 DiT).

**FlashVDM** (arXiv:2503.16302 — https://arxiv.org/abs/2503.16302): accelerate both halves of a vecset diffusion model — Progressive Flow Distillation for ~5-step DiT sampling + a Lightning vecset decoder (Adaptive KV Selection: decode each query against only a selected KV subset; Hierarchical Volume Decoding: coarse-to-fine occupancy skipping empty space). Applied to Hunyuan3D-2 -> Turbo: 32x generation, 45x reconstruction, sub-1s shapes. WEIGHT-LOADABLE (distilled DiT) + TRAINING-FREE decoder algorithms (the KV/volume-decode tricks work regardless of weights).

**Hunyuan3D-2 Turbo / Fast** (github.com/Tencent-Hunyuan/Hunyuan3D-2): Turbo = step-distilled DiT (FlashVDM); Fast = guidance distillation removing the CFG branch (load Fast weights, skip the CFG double-forward, ~2x). Turbo ~1s on a 4090. WEIGHT-LOADABLE. Since HartsyInference already implements the Hunyuan3D-2 pipeline, **Fast is the cheapest win** (drop CFG + load Fast weights).

### 5.4 World models (Matrix-Game / Oasis)

**Matrix-Game 2.0** (arXiv:2508.13009 — https://arxiv.org/abs/2508.13009; Skywork/Matrix-Game-2.0): an action-controlled video DiT distilled into a causal few-step AR model via Self-Forcing -> real-time streaming. The loop is exactly the Self-Forcing/CausVid pattern (causal mask + rolling KV cache + few-step denoise per chunk) plus per-frame action injection. 25 FPS on one H100, minute-long consistency. WEIGHT-LOADABLE. HartsyInference already implements Matrix-Game; acceleration = load the few-step causal weights + run the rolling-KV AR loop. The same machinery is reusable for Oasis-style action-conditioned world models.

### Family 5 summary

| Technique | Domain | Speedup | Class | arXiv |
|---|---|---|---|---|
| CausVid | video | 50->4 steps, 9.4 FPS | WEIGHT-LOADABLE / train | 2412.07772 |
| Self-Forcing | video | real-time, sub-second | WEIGHT-LOADABLE / train | 2506.08009 |
| FastWan | video | 5s 480p in ~1s | WEIGHT-LOADABLE | (FastVideo) |
| Wan2.1 CausVid LoRA | video | ~12x | WEIGHT-LOADABLE (LoRA) | 2412.07772 |
| **Wan2.2-Lightning** | video | ~20x | WEIGHT-LOADABLE | (ModelTC) |
| LTX distilled 0.9.7/0.9.8 | video | ~6-8x | WEIGHT-LOADABLE | (Lightricks) |
| PAB | video | up to 10.5x | TRAINING-FREE | 2408.12588 |
| TeaCache | video/audio | up to 4.4x | TRAINING-FREE | 2411.19108 |
| AdaCache | video | up to 4.7x | TRAINING-FREE | 2411.02397 |
| STA | video | attn 2.8-17x | TRAINING-FREE (kernel) | 2502.04507 |
| **ACE-Step distill** | audio | 50->4-8 steps, >100x | WEIGHT-LOADABLE | 2506.00045 |
| Presto! | audio | 10-18x | WEIGHT-LOADABLE / train | 2410.05167 |
| AudioLCM | audio | ~2 steps | WEIGHT-LOADABLE | 2406.00356 |
| ARC (Stable Audio Small) | audio | 12s in ~75ms | WEIGHT-LOADABLE | 2505.08175 |
| FlashVDM | 3D | gen 32x, recon 45x | WEIGHT-LOADABLE + train-free decoder | 2503.16302 |
| Hunyuan3D-2 Fast/Turbo | 3D | Fast ~2x, Turbo ~1s | WEIGHT-LOADABLE | (Tencent) |
| Matrix-Game 2.0 | world model | 25 FPS | WEIGHT-LOADABLE | 2508.13009 |

---

## Cross-Family Ranking (speedup x ease-of-implementation for a custom C# backbone)

Ease scored for *this* engine: trivial = pure loop code, no weights, one threshold/formula; easy = loop code + small per-model constants or a LoRA merge; moderate = new forward-pass input or a new scheduler core; hard = kernel changes (gather/scatter, sparse attention) or training.

| Rank | Technique | Speedup | Ease | Why it ranks here |
|---|---|---|---|---|
| 1 | **FBCache** | ~2x | trivial | run block 1, one rel-absmean compare, one cached residual add. Smallest possible loop change for every DiT/MM-DiT. |
| 2 | **Limited interval guidance** | up to ~2x + quality gain | trivial | per-step sigma gate around CFG; free quality AND speed; any non-distilled CFG model. |
| 3 | **DPM-Solver++ 2M / UniPC** | 2-3x (steps 50->10-20) | easy | one shared exp-integrator core; training-free; works on every eps/v/x0 model. UniPC for sub-10-step. |
| 4 | **TeaCache** | 2-4.4x | easy | same residual-inject machinery as FBCache + a per-model polynomial; robust runtime gate; video/image/audio. |
| 5 | **LCM-LoRA / Hyper-SD / TCD / PCM LoRAs** | 4-12x (50->4) | easy | LoRA merge + LCM/TCD scheduler (build the two schedulers once); no training; universal across SD fine-tunes. |
| 6 | **AYS schedule** | ~1.5-2x | easy | drop-in sigma table (SD1.5/SDXL/SVD published) + log-lin interp; composes with any solver. |
| 7 | **Distilled checkpoints** (Turbo/Lightning/DMD2/schnell, Wan2.2-Lightning, LTX-distilled, ACE-Step distill, Hunyuan3D Fast) | 6-20x | easy | architecture unchanged; load weights + short schedule + CFG off. Per-model weight availability is the only gate. |
| 8 | **Guidance embedding** (Flux-dev/SD3) | 2x | moderate | constructor flag + one MLP adds guidance scalar to the modulation vec; mandatory for Flux-dev/SD3 anyway. |
| 9 | **PAB** (video) | up to 10.5x (mostly multi-GPU) | easy-moderate | per-attention-type broadcast counters; single-GPU gain modest (~1.3x) without parallelism. |
| 10 | **AdaCache** (video) | up to 4.7x | moderate | residual cache + latent motion-score + adaptive interval policy. |
| 11 | **TaylorSeer** | 3.5-4.7x | moderate | upgrade to FBCache/TeaCache: forecast the residual via FD derivatives; more math per skipped step. |
| 12 | **DEIS / SA-Solver / Restart** | 1.5-3x | easy-moderate | reuse the exp-integrator core / a thin loop wrapper; secondary to 2M/UniPC. |
| 13 | **CausVid / Self-Forcing** (video AR) | real-time | moderate | causal mask + rolling KV cache + few-step; needs loadable causal weights (Matrix-Game 2.0, FastWan). |
| 14 | **DeepCache** (UNet) | 2.3-4.1x | easy | UNet-only skip-feature cache; applies to SD1.5/SDXL backbones, not DiT. |
| 15 | **ToCa / DuCa / ClusCa** | 1.9-4.96x | hard | needs gather/scatter + variable-length/partial attention in PTX. Defer. |
| 16 | **STA** (video) | attn 2.8-17x | hard | new sparse-attention PTX kernel; biggest wins need sparse-distill finetune. |
| 17 | **Shortcut models / L2C** | varies | hard | require training (step-size conditioning / a router). Only if training a backbone. |

---

## Top Recommendations for HartsyInference

Adopt in this order. The first four are training-free, weight-free loop code and together compound to roughly an order of magnitude on existing models with no new weights.

1. **Build the two consistency schedulers (LCM + TCD) and the residual-cache mechanism first.** These three primitives unlock the largest swath of the literature. The LCM/TCD schedulers (predict-x0 + parameterized re-noise, sharing `f_theta = c_skip*x + c_out*x0`) immediately enable LCM-LoRA, Hyper-SD, PCM, and TCD LoRAs across SD1.5/SDXL with only a LoRA merge and CFG off (4-step generation, no training). The residual cache (`out = in + Delta_cached`) is the substrate for FBCache, TeaCache, Delta-DiT, and TaylorSeer.

2. **FBCache as the default DiT/MM-DiT cache.** Run block 1, compute `mean(|r1_cur - r1_prev|)/mean(|r1_prev|)`, and on `< ~0.05` skip the rest of the stack and add the cached `hidden_states_residual`. One threshold, ~2x, no weights. Then layer **TeaCache** (polynomial-rescaled accumulated gate on the timestep-modulated input, `delta = 0.1`/`0.2`) for a more robust runtime decision, and **TaylorSeer** to forecast (not just reuse) the residual at high cache ratios.

3. **Limited-interval guidance + the guidance-scalar embedding.** The sigma-gated CFG skip (3.4) is a free quality-and-speed win on any non-distilled model and is a few lines around the existing per-step CFG enabler. The `guidance_in` MLP embedding (3.2) is mandatory for Flux-dev/SD3 (single-pass CFG) and turns guidance into a scalar input instead of a doubled batch. Add Adaptive Guidance (3.5) where negative prompts must be kept.

4. **A shared exponential-integrator solver core in log-SNR space**, exposing DPM-Solver++(2M), UniPC (sub-10-step), DEIS, SA-Solver, and a Restart wrapper from one converter (eps/v/x0/flow-velocity -> x0). Ship the **AYS sigma tables** (SD1.5/SDXL/SVD) as a drop-in schedule. For Flux/SD3 start with Euler + resolution shift, reuse the core via flow-sigmas.

5. **Load distilled checkpoints where they exist** (architecture-identical, so no new forward code beyond CFG-off / guidance-embedding): SDXL-Turbo/Lightning/DMD2, Flux-schnell, **Wan2.2-Lightning** (4-step, ~20x) and **LTX-0.9.8-distilled** (8-step, CFG=1) for video, **ACE-Step distilled** (4-8 step) for music, **Hunyuan3D-2 Fast** (CFG-skip) for 3D.

6. **Video caching layer reusable across every video DiT** (Wan, LTX, HunyuanVideo/GameCraft): TeaCache (simplest) and/or PAB (per-attention-type broadcast, windows [100,800]/[100,900], ranges 2/3/5), AdaCache for per-clip adaptivity.

7. **Interactive / world models**: the rolling-KV causal AR loop (Self-Forcing/CausVid) is the core primitive (already half-built in the Interactive package); loading Matrix-Game-2.0's few-step causal weights yields 25 FPS.

**Defer**: token-level caches (ToCa/DuCa/ClusCa) and STA until residual-cache wins are banked (they need gather/scatter + sparse-attention PTX). L2C and shortcut models need training. DeepCache only matters for the remaining UNet (SD1.5/SDXL) backbones.
