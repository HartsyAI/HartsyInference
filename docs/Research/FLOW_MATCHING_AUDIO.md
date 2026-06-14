# Flow Matching for Audio — Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio + HartsyInference.Diffusion (schedulers)

## Summary

Modern open-source audio generation models — F5-TTS, E2-TTS, ACE-Step, DiffRhythm, Stable Audio Open (1.0 / Small / 2) — are almost all built on **rectified flow / Conditional Flow Matching (CFM)**. The model predicts a constant velocity field `v_theta(x_t, t, cond)` along a straight-line path between Gaussian noise (`t=1`) and the data (`t=0`), and a first-order Euler ODE solver integrates it. The math is essentially identical to image flow-matching (Flux, SD3), with three audio-specific wrinkles worth dedicated scheduler code: (1) **F5/DiffRhythm "Sway Sampling"** — a one-shot cosine remap of the timestep grid that biases sampling toward early steps where the model lays out coarse structure; (2) **ACE-Step "omega mean-shift"** — a mean-preserving rescale applied to each Euler delta to control variance of the trajectory; (3) ACE-Step's optional **APG (Adaptive Projected Guidance)** and **CFG-Zero\*** in place of vanilla CFG. AudioLDM 2 is included as a counter-example: it is *not* flow matching — it is classic ε-prediction latent diffusion with DDIM/DPM-Solver++ at ~200 steps.

For HartsyInference, every model in this batch except AudioLDM 2 can be served by a single C# `FlowMatchEulerDiscreteScheduler` (which we already have for Flux/SD3) plus three small additions: a Sway-Sampling timestep remap helper, an Omega mean-shift `Step` overload, and a CFG combiner that operates on the velocity field exactly like ε-prediction CFG. AudioLDM 2 reuses our existing `DpmPlusPlus2MScheduler` / `DdimScheduler`.

---

## Detailed Findings

### 1. Rectified Flow / Conditional Flow Matching — the base method

**Papers:** Lipman et al. 2023 ("Flow Matching for Generative Modeling", [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)); Liu et al. 2022 ("Flow Straight and Fast", rectified flow).

Flow matching defines a probability path from a simple noise distribution `p_1 = N(0, I)` to the data distribution `p_0` and trains a neural net to regress the **time-conditional vector field** that transports samples along this path. For the **linear/optimal-transport (OT) path** — which every model in this document uses — the forward interpolation is:

```
x_t = (1 - t) * x_0 + t * x_1        where x_0 ~ data, x_1 ~ N(0,I), t in [0,1]
```

The ground-truth velocity along this path is the constant `dx/dt = x_1 - x_0`. The training loss is

```
L = E_{t, x_0, x_1} || v_theta(x_t, t) - (x_1 - x_0) ||^2
```

At inference time the ODE `dx/dt = v_theta(x, t)` is integrated **backward in time** from `t=1` (pure noise) to `t=0` (clean data). With a first-order **Euler step** and a decreasing grid `1 = t_0 > t_1 > ... > t_N = 0`:

```
x_{i+1} = x_i + (t_{i+1} - t_i) * v_theta(x_i, t_i, cond)   # note (t_{i+1} - t_i) is negative
```

This is what our `FlowMatchEulerDiscreteScheduler.Step` already does: `out = sample + model_output * (sigma_next - sigma)`.

**Connection to diffusion.** Flow matching with the linear/OT path is mathematically equivalent to v-prediction diffusion under the variance-preserving schedule at the limit; in practice they differ in the loss weighting and the trajectory straightness (FM is straighter, which is why ≤32 steps usually suffice).

**Why audio chose flow matching.** Two reasons dominate the literature:
1. **Fewer NFE for the same quality** — F5-TTS reports near-saturation at 16–32 NFE vs. 200+ for AudioLDM 2.
2. **Straight trajectories play well with in-context conditioning** — the prompt-audio + masked-target paradigm (Section "Conditioning patterns") needs the model to "snap" the unmasked region back to the prompt; straighter paths mean less drift.

### 2. F5-TTS — CFM with Sway Sampling

**Repo:** [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) — `src/f5_tts/model/cfm.py`. **Paper:** "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" ([arXiv:2410.06885](https://arxiv.org/abs/2410.06885), ACL 2025).

**Sampler** (exact code from `cfm.py`):

```python
t = torch.linspace(t_start, 1, steps + 1, device=...)        # default steps=32, t_start=0
if sway_sampling_coef is not None:
    t = t + sway_sampling_coef * (torch.cos(torch.pi/2 * t) - 1 + t)
# ...
odeint(fn, y0, t, method="euler")                            # first-order Euler
```

The CFG-aware vector field inside `fn(t, x)` is:

```python
pred = pred + (pred - null_pred) * cfg_strength              # CFG on the velocity
```

**Defaults** (from `src/f5_tts/infer/utils_infer.py`):

| Parameter | Default | Notes |
|---|---|---|
| `nfe_step` | **32** | 16 also evaluated and ~OK |
| `cfg_strength` | **2.0** | Applied to velocity (see formula above) |
| `sway_sampling_coef` | **−1.0** | Negative biases toward early `t` |
| `ode_method` | **"euler"** | First-order, fixed step |

Note F5-TTS internally integrates `t: 0 → 1`. The relationship to our `sigma` convention (`sigma = 1 - t`, sigma decreasing from 1 to 0) is just a sign flip; the per-step update is mathematically identical.

#### Sway Sampling — exact formula

```
f_sway(u; s) = u + s * (cos(pi/2 * u) - 1 + u)            # u in [0,1]
```

This is applied **after** generating a uniform linspace `[0, 1]` and **before** feeding to the ODE solver, so it does *not* change the endpoints (`f_sway(0;s) = 0`, `f_sway(1;s) = 1`) — only the interior density. With **s < 0**, more samples cluster near `u = 0` (i.e., near the noise end at inference, when integrating from u=1 down to u=0 the model spends more steps on the early/coarse-structure portion). The paper's ablation finds s = −1.0 optimal; more negative s degrades.

Concrete numbers — for `steps=32`, `s=-1.0`:

| `i` | uniform `t_i` | sway-remapped `t_i` |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 4 | 0.1250 | 0.0259 |
| 8 | 0.2500 | 0.0795 |
| 16 | 0.5000 | 0.2929 |
| 24 | 0.7500 | 0.5957 |
| 28 | 0.8750 | 0.7672 |
| 32 | 1.0000 | 1.0000 |

(Pure scalar math, easy to validate in C# bit-for-bit.)

### 3. E2-TTS — predecessor to F5

**Paper:** "E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS" ([arXiv:2406.18009](https://arxiv.org/abs/2406.18009), Microsoft).

E2-TTS uses **plain CFM with Euler integration, no sway sampling, no EPSS, no special shift**. Vector field, CFG formula and per-step update are identical to F5-TTS. The conditioning trick — concatenate padded character sequence with input speech, predict the masked portion — is what F5-TTS inherited. E2-TTS evaluated at NFE = 32 with `cfg_strength ≈ 2.0`.

For HartsyInference, **E2-TTS == F5-TTS scheduler with `sway_sampling_coef = 0`**.

### 4. ACE-Step — flow matching for music with mean-shift step

**Repo:** [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step). Scheduler at `acestep/schedulers/scheduling_flow_match_euler_discrete.py`.

ACE-Step is a fork of HuggingFace `FlowMatchEulerDiscreteScheduler` with one major addition: an **omega mean-shift** in the step:

```python
# sigma schedule: linspace, then shifted
sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)  # shift=3.0

# step (per-sample, audio latent shape):
dx  = (sigma_next - sigma) * model_output
m   = dx.mean()
dx_ = (dx - m) * omega + m                # mean-preserving variance rescale
prev_sample = sample + dx_
```

The raw `omega` passed in (default `omega_scale=10.0`) goes through a logistic squash into a bounded multiplier:

```python
new_x = L + (U - L) / (1 + exp(-k * (x - x_0)))
# L=0.9, U=1.1, x_0=0.0, k=0.1
# omega=10.0 -> logistic(1.0) -> roughly 1.05
```

The intent: keep the mean of the velocity update intact but slightly amplify (or dampen) the per-element variance around it — empirically helps musical detail / transient sharpness.

**Pipeline defaults** (from `pipeline_ace_step.py`):

| Parameter | Default |
|---|---|
| `infer_steps` | **60** |
| `shift` | **3.0** (same form as SD3 shift; **not** dynamic) |
| `guidance_scale` | **15.0** (high — APG is recommended) |
| `cfg_type` | **"apg"** (alternatives: "cfg", "cfg_zero_star") |
| `omega_scale` | **10.0** (logistic-squashed → ~1.05 effective) |
| `scheduler_type` | Euler (default), Heun, PingPong supported |

CFG variants:
- `cfg`: standard `v = v_uncond + s * (v_cond - v_uncond)`.
- `apg`: **Adaptive Projected Guidance** — at each step, projects `(v_cond - v_uncond)` onto a momentum buffer, applies guidance only along the orthogonal complement; reduces over-saturation at high `s`.
- `cfg_zero_star`: zeros out CFG for the first `zero_steps` (typically the first 1–3 of 60), then standard CFG; reduces artifacts at very high guidance.

### 5. DiffRhythm — flow matching for full-length songs

**Repo:** [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm). Sampler at `model/cfm.py`.

DiffRhythm uses F5-TTS's `CFM` class essentially unchanged:

```python
t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
if sway_sampling_coef is not None:
    t = t + sway_sampling_coef * (torch.cos(torch.pi/2 * t) - 1 + t)

odeint_kwargs = dict(method="euler")
return pred + (pred - null_pred) * cfg_strength
```

**Defaults** (from `infer/infer.py` and project README):

| Parameter | Default |
|---|---|
| `steps` | **32** |
| `cfg_strength` | **4.0** (higher than F5 — music benefits) |
| `sway_sampling_coef` | **None / 0** in main inference path (user-toggleable; community docs note it activates "below 32 steps") |
| `ode_method` | **euler** (UI also exposes midpoint, RK4, implicit Adams) |
| `max_frames` (95 s mode) | **2048** |
| `max_frames` (285 s mode) | extended config |

Functionally **DiffRhythm == F5-TTS scheduler with cfg_strength=4 and (typically) sway off**.

### 6. Stable Audio Open (1.0 / Small / 2) — rectified flow at 44.1 kHz

**Repo:** [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). Sampler at `stable_audio_tools/inference/sampling.py`.

The newer Stable Audio Open checkpoints train a DiT as a **rectified flow** (`diffusion_objective="rectified_flow"`). The dispatch in `generation.py` selects `sample_rf(...)` for these models, which then calls `sample_discrete_euler` (or `sample_flow_pingpong`, `sample_flow_dpmpp`):

```python
@torch.no_grad()
def sample_discrete_euler(model, x, steps=None, sigma_max=1, sigmas=None,
                          callback=None, dist_shift=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""
    ts = x.new_ones([x.shape[0]])
    if sigmas is None:
        t = torch.linspace(sigma_max, 0, steps + 1)
        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    else:
        t = sigmas
    for i, (t_curr, t_prev) in enumerate(zip(t[:-1], t[1:])):
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        dt = t_prev - t_curr           # negative
        v = model(x, t_curr_tensor, **extra_args)
        x = x + dt * v
    return x
```

CFG is applied **inside** the model wrapper (`DiTWrapper`) when `batch_cfg=True`: noise is duplicated, cond and uncond passes are batched into one forward, then split and combined via `v = v_uncond + cfg_scale * (v_cond - v_uncond)`. When `rescale_cfg=True`, the combined `v` is rescaled by `std(v_cond) / std(v_guided)` (the SD3-style rescale from Lin et al. 2024) — this is important above `cfg_scale ≈ 4`.

**Defaults per checkpoint:**

| Checkpoint | Sample rate | Max duration | `steps` | `cfg_scale` | `sampler_type` | Objective |
|---|---|---|---|---|---|---|
| Stable Audio Open 1.0 | 44.1 kHz stereo | 47 s | 100 (recipe), 250 (diffusers default) | 7.0 | `dpmpp-3m-sde` (`sigma_min=0.3, sigma_max=500`) | **v-prediction** (legacy k-diffusion) |
| Stable Audio Open Small | 44.1 kHz stereo | 11 s | **8** | **1.0** (CFG disabled — distilled) | **`pingpong`** | rectified flow |
| Stable Audio Open 2 | 44.1 kHz stereo | (model card) | ~50–100 | ~5–7 | `euler` (sample_discrete_euler) or `dpmpp-3m-sde` | **rectified flow** |

(The "Open 2" defaults are the family practice — Stability publishes the inference recipe per-checkpoint; exact numbers depend on the release. The Open 2 model is the first in the family trained from scratch as rectified flow rather than v-prediction.)

**Sigma_max convention.** Note Stable Audio Open 1.0 uses `sigma_max=500` (k-diffusion-style EDM-σ space — model is v-prediction). The rectified-flow paths use `sigma_max=1` (flow time `t` in [0,1]) — same convention as F5/SD3.

Optional `dist_shift.time_shift(t, seq_len)` applies a length-dependent shift exactly like SD3's dynamic shift — relevant when generating audio of unusual length.

### 7. AudioLDM 2 — counter-example (NOT flow matching)

**HF pipeline:** [AudioLDM2Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2). **Paper:** [arXiv:2308.05734](https://arxiv.org/abs/2308.05734).

AudioLDM 2 is a **classic ε-prediction latent diffusion** (variance-preserving DDPM schedule on mel-spectrogram latents, then HiFi-GAN-like vocoder).

| Parameter | Default | Notes |
|---|---|---|
| Default scheduler | **DDIM** | works at 200 NFE |
| Alternative scheduler | **DPM-Solver++ multistep** | 20–25 NFE for similar quality |
| `num_inference_steps` | **200** | (yes — 6–10× more than flow matching) |
| `guidance_scale` | **3.5** | applied to ε via `eps = eps_uncond + s*(eps_cond - eps_uncond)` |
| `audio_length_in_s` | 10.24 | |
| sample rate | 16 kHz | mel-spec latents → HiFi-GAN |

For HartsyInference, AudioLDM 2 routes through our **existing** `DdimScheduler` (200 steps) or `DpmPlusPlus2MScheduler` (~25 steps), no new code needed.

### Why flow matching for audio? (the intuition)

The probability path in CFM is a straight line between noise and data. The neural net only ever has to predict a single constant direction `x_1 - x_0` for a given `(x_t, t)`. Compare to ε-prediction diffusion, where the loss-weighted prediction target varies in scale across `t` and the path through state space is curved. Two consequences:

1. **Fewer NFE.** A straight-line path means a first-order Euler integrator accumulates almost no truncation error, so 16–32 steps produce the same quality that ε-prediction needs 100–200 for. This matters enormously for audio because audio latents are *long* (a 95 s DiffRhythm song = 2048 latent frames; Stable Audio Open 1.0 = 1024) and each forward pass is expensive.
2. **In-context conditioning is well-posed.** Audio TTS routinely concatenates `[prompt_audio | masked_target]` and asks the model to predict only the masked half. Straight trajectories make the unmasked half "snap" back to the prompt at every step without drift — the velocity for those positions is approximately zero.

### Conditioning patterns — in-context audio inference

F5-TTS / E2-TTS / DiffRhythm all use the same paradigm: concatenate a **reference audio prompt** (mel features or VAE latents) with a **noise-initialized target region**, in a single sequence. Predict the velocity over the whole sequence; ignore the predictions in the prompt region and overwrite them with the prompt at every step.

In F5-TTS code (`cfm.py`):

```python
# cond:           [B, prompt_len, n_mel]            — reference audio mel features
# Pad to total target length max_duration:
cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
# Build per-step conditioning: keep cond where mask is True, zeros elsewhere
step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
# After the ODE solve:
out = torch.where(cond_mask, cond, out)            # paste prompt back over the prompt region
```

Shape summary:
- `cond`: `[B, prompt_len, n_mel]` initially, padded to `[B, max_duration, n_mel]`
- `cond_mask`: `[B, max_duration]` boolean (True over prompt region)
- `step_cond`: `[B, max_duration, n_mel]` — same shape, zeros over target region
- `y0` (initial noise): `[B, max_duration, n_mel]`
- `out` after sampling: `[B, max_duration, n_mel]` — prompt region overwritten with original `cond`

For ACE-Step (music) the same idea applies at the latent level, with the prompt being optional (text-only generation just feeds noise). Stable Audio Open uses traditional text/timing conditioning rather than audio-prompt infilling.

### Sway Sampling — detailed analysis

The remap `f(u; s) = u + s*(cos(pi/2*u) - 1 + u)` has these properties:

| Property | Value / Formula |
|---|---|
| Fixed points | `f(0; s) = 0`, `f(1; s) = 1` for any `s` |
| Identity case | `f(u; 0) = u` (uniform) |
| First derivative | `f'(u; s) = 1 + s*(1 - (pi/2)*sin(pi/2*u))` |
| Monotonic? | `f'(u; s) > 0` for all `u in [0,1]` iff `s > -1/(1 - pi/2) ≈ -2.75` (so `s = -1.0` is well-inside) |
| Effect at `s < 0` | Density compresses near `u = 0`, expands near `u = 1` |
| F5-TTS default | **`s = -1.0`** |

When the ODE solver integrates from `t=1` (noise) down to `t=0` (data), with `s = -1.0` the **steps are taken in larger leaps near t=1** (the model bootstraps a coarse contour fast) and **smaller leaps near t=0** (it spends NFE polishing fine detail). Empirically this raised F5-TTS's SIM-O on LibriSpeech-PC by ~0.02 and lowered WER by ~10% relative at 32 NFE. Negative `s` strictly improves WER/SIM in the F5 paper ablation; `s = 0` (uniform), `−0.5`, `−1.0` are tested with `−1.0` winning.

---

## Key Numbers / Constants

| Constant | Value | Where used |
|---|---|---|
| F5-TTS default NFE | 32 | `cfm.py` |
| F5-TTS `cfg_strength` | 2.0 | `utils_infer.py` |
| F5-TTS `sway_sampling_coef` | **−1.0** | `utils_infer.py` |
| F5-TTS `ode_method` | "euler" | `utils_infer.py` |
| E2-TTS NFE | 32 | paper §4 |
| E2-TTS `cfg_strength` | ~2.0 | paper |
| ACE-Step default NFE | 60 | `pipeline_ace_step.py` |
| ACE-Step `shift` | **3.0** | `scheduling_flow_match_euler_discrete.py` |
| ACE-Step `guidance_scale` | 15.0 (with APG) | `pipeline_ace_step.py` |
| ACE-Step `omega_scale` | 10.0 → ~1.05 effective | `step()` logistic squash |
| ACE-Step omega logistic | L=0.9, U=1.1, x_0=0.0, k=0.1 | `scheduling_flow_match_euler_discrete.py` |
| ACE-Step `cfg_type` default | "apg" | options: cfg / apg / cfg_zero_star |
| DiffRhythm NFE | 32 | `infer.py` |
| DiffRhythm `cfg_strength` | 4.0 | `infer.py` |
| DiffRhythm `sway_sampling_coef` | None / 0 by default | `cfm.py` |
| DiffRhythm max_frames (95 s) | 2048 | |
| Stable Audio Open 1.0 NFE | 100 (recipe), 250 (diffusers) | v-prediction, dpmpp-3m-sde |
| Stable Audio Open 1.0 `cfg_scale` | 7.0 | |
| Stable Audio Open 1.0 `sigma_max` | 500 | EDM σ-space |
| Stable Audio Open 1.0 `sigma_min` | 0.3 | |
| Stable Audio Open Small NFE | 8 | distilled, pingpong sampler |
| Stable Audio Open Small `cfg_scale` | 1.0 (off) | |
| Stable Audio Open (RF) `sigma_max` | 1.0 | flow time space |
| AudioLDM 2 NFE | 200 (DDIM default), 20-25 (DPM++) | |
| AudioLDM 2 `guidance_scale` | 3.5 | ε-prediction CFG |
| AudioLDM 2 sample rate | 16 kHz | mel-latent + HiFi-GAN |
| All flow-matching `num_train_timesteps` | 1000 | conditioning embedding range |

---

## Data Layouts / Formats

**Velocity-prediction tensor shape contract (C# perspective):**
All flow-matching audio models predict velocity `v` with the **same shape as the input latent** `x`. There is no separate noise/x_0 head.

- F5/E2/DiffRhythm latent: `[B, T_mel, n_mel]` where `n_mel` = 80 or 100 (model-dependent), `T_mel` = mel frames at the model's hop rate. F5-TTS uses **100-channel mels at 24 kHz / hop 256** by default.
- ACE-Step latent: `[B, C_lat, T_lat]` — VAE-compressed music latents.
- Stable Audio Open latent: `[B, 64, T_lat]` — 64-channel stereo VAE latents at ~21.5 Hz (1024 tokens = 47 s).

**Timestep encoding (model input):**
- F5/E2/DiffRhythm: `t in [0, 1]` (float per-sample, broadcast to batch). Some implementations multiply by 1000 to reuse standard sinusoidal embeddings; our existing `FlowMatchEulerDiscreteScheduler.Timesteps` already returns `sigma * 1000`.
- ACE-Step: `t in [0, 1000]` (shift-applied sigma * 1000).
- Stable Audio Open RF: `t in [0, 1]` (raw flow time).

**Sigma table (CPU-side, what the C# scheduler emits):**
- Length: `N + 1` where `N = num_inference_steps`. First entry is `sigma_max` (1.0 for FM, 500 for SAO 1.0), last entry is 0.0.
- Float32. Strictly monotonically decreasing.
- For F5/DiffRhythm with sway: this is the post-sway sequence (1 − sway-remapped uniform).

---

## Algorithm Steps

### CFM Euler step (universal — F5, E2, DiffRhythm, ACE-Step "cfg", Stable Audio Open RF)

```text
Inputs:
  x        : current latent, shape [B, ...]
  sigma    : float, current flow time (t)            # in [0,1], decreasing toward 0
  sigma_next : float, next flow time
  cond, uncond : conditioning tensors
  cfg_strength : float

Step:
  1. v_cond   = model(x, sigma, cond)
  2. v_uncond = model(x, sigma, uncond)              # batched if batch_cfg=True
  3. v        = v_uncond + cfg_strength * (v_cond - v_uncond)
  4. dt       = sigma_next - sigma                    # negative
  5. x_next   = x + v * dt
```

Note: F5-TTS writes the CFG combination as `v = v_cond + (v_cond - v_uncond) * cfg_strength`, which is `(1 + s) * v_cond - s * v_uncond`. This is **equivalent** to the standard form `v = v_uncond + (s+1)*(v_cond - v_uncond)` with `guidance_scale = s + 1`. So F5's `cfg_strength = 2.0` corresponds to `guidance_scale = 3.0` in the diffusers convention.

### ACE-Step Euler step with omega mean-shift

```text
Inputs: x, sigma, sigma_next, v, omega_raw
1. dt   = sigma_next - sigma
2. dx   = dt * v                                      # shape [B, ...]
3. m    = mean(dx)                                    # scalar, mean over ALL elements
4. omega_eff = 0.9 + 0.2 / (1 + exp(-0.1 * omega_raw))   # logistic squash, L=0.9, U=1.1, k=0.1, x_0=0
5. dx_  = (dx - m) * omega_eff + m
6. x_next = x + dx_
```

For `omega_raw = 10.0` (default): `omega_eff = 0.9 + 0.2 / (1 + exp(-1.0)) = 0.9 + 0.2 * 0.7311 = 1.0462`.

### Sway sampling remap (F5/DiffRhythm)

```text
Input:  N steps, sway_coef s
Output: timestep array t[0..N], all in [0,1]

1. for i in 0..N:
     u = i / N
     t[i] = u + s * (cos(pi/2 * u) - 1 + u)
2. (optional, depending on integration direction) sigma[i] = 1 - t[N - i]   # for sigma-from-1-to-0 convention
```

### Stable Audio Open `pingpong` sampler (Open Small, distilled)

Used because 8 steps of straight Euler underintegrates the SDE for a distilled model; pingpong re-injects noise after each step to keep the trajectory on-manifold:

```text
for i in 0..N-1:
  v = model(x, t[i], cond)                          # no CFG (cfg=1.0)
  dt = t[i+1] - t[i]
  x = x + dt * v                                    # Euler step
  if i < N - 1:                                     # don't re-noise on final step
    noise = randn_like(x)
    x = (1 - t[i+1]) * x_denoised + t[i+1] * noise  # re-noise back to t[i+1]
                                                    # x_denoised = x - t[i+1]*v (predicted clean)
```

(This is the same recipe Stability uses for distilled image models like SDXL-Turbo. PingPong adds stochasticity; with `cfg_scale=1.0` it is the only source of diversity.)

---

## Reference Implementations

| Model | File | What to look at |
|---|---|---|
| F5-TTS | [`src/f5_tts/model/cfm.py`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py) | `sample()` method: linspace, sway remap, odeint(euler), CFG |
| F5-TTS | [`src/f5_tts/infer/utils_infer.py`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py) | Defaults: `nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0` |
| E2-TTS | [arXiv:2406.18009](https://arxiv.org/abs/2406.18009) | §3 (CFM training), §4 (Euler @ NFE=32) |
| ACE-Step | [`acestep/schedulers/scheduling_flow_match_euler_discrete.py`](https://github.com/ace-step/ACE-Step/blob/main/acestep/schedulers/scheduling_flow_match_euler_discrete.py) | `set_timesteps()`, `step()` with omega, logistic squash |
| ACE-Step | [`acestep/pipeline_ace_step.py`](https://github.com/ace-step/ACE-Step/blob/main/acestep/pipeline_ace_step.py) | `cfg_forward`, `apg_forward`, `cfg_zero_star`, defaults |
| DiffRhythm | [`model/cfm.py`](https://github.com/ASLP-lab/DiffRhythm/blob/main/model/cfm.py) | `sample()` — basically F5's CFM |
| DiffRhythm | [`infer/infer.py`](https://github.com/ASLP-lab/DiffRhythm/blob/main/infer/infer.py) | Defaults: `steps=32, cfg_strength=4.0` |
| Stable Audio Open | [`stable_audio_tools/inference/sampling.py`](https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/inference/sampling.py) | `sample_rf`, `sample_discrete_euler`, `sample_flow_pingpong` |
| Stable Audio Open | [`stable_audio_tools/inference/generation.py`](https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/inference/generation.py) | `generate_diffusion_cond`: dispatches to `sample_rf` when `diffusion_objective == "rectified_flow"`, passes `batch_cfg=True, rescale_cfg=True` |
| Stable Audio Open paper | [arXiv:2407.14358](https://arxiv.org/abs/2407.14358) | §3 architecture, §4 inference recipe |
| AudioLDM 2 | [diffusers AudioLDM2Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2) | 200 NFE DDIM default; recommend `DPMSolverMultistepScheduler` for ~25 NFE |
| Lipman et al. CFM | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | §3 OT path, §4 conditional FM loss |
| Existing C# baseline | `src/HartsyInference.Diffusion/Schedulers/FlowMatchEulerDiscreteScheduler.cs` | Already implements the SD3-shift sigma schedule + Euler step |

---

## Differences Between Implementations

1. **Time direction.** F5 / DiffRhythm / E2 integrate `t : 0 → 1` (t = "noise level" going up; integrate forward but the noise level goes from 0 at clean to 1 at pure noise). Stable Audio Open's `sample_discrete_euler` integrates from `sigma_max → 0` (decreasing). The Euler update is **the same** because `dt` carries the sign. Our C# scheduler uses `sigma` decreasing from 1 → 0; convert F5/DiffRhythm by `sigma = 1 - t`.
2. **CFG formula written two ways:**
   - F5/DiffRhythm: `v = v_cond + cfg_strength * (v_cond - v_uncond)`
   - Stable Audio Open / ACE-Step `cfg_forward`: `v = v_uncond + guidance_scale * (v_cond - v_uncond)`
   - Equivalent under `guidance_scale = cfg_strength + 1`. **Be careful with parameter naming in C# API** — use diffusers convention (`guidance_scale = 3.0` for F5 default) and document the mapping.
3. **Sigma shift.** ACE-Step uses static `shift=3.0` (same form SD3 Medium uses for images). F5 / DiffRhythm use no shift (`shift=1`, identity). Stable Audio Open supports optional `dist_shift` (length-dependent) but the default recipes don't use it.
4. **Omega.** Only ACE-Step has it. Setting `omega_eff = 1.0` reduces step 5 to plain Euler.
5. **Number of NFE.** Flow matching audio settles between 8 (distilled) and 60. AudioLDM 2 needs 200 (or 25 with DPM++) — order of magnitude difference.
6. **Sway sampling.** Only F5 / DiffRhythm. ACE-Step and Stable Audio Open don't use it.
7. **Sampler in Stable Audio Open Small.** `pingpong` is required for distilled `cfg=1.0` 8-step inference. Plain Euler at 8 steps produces audible artifacts.

---

## Open Questions

1. **EPSS (Empirically Pruned Step Sampling)** — F5-TTS optionally replaces the linspace with a learned non-uniform schedule (paper "Accelerating Flow-Matching-Based TTS via EPSS", Interspeech 2025). Not in the default inference path; revisit only if we ship EPSS-trained checkpoints.
2. **ACE-Step `cfg_zero_star` `zero_steps`** — exact default count of zeroed steps (1 vs. 2 vs. 3 of 60) not yet confirmed from code. Read `acestep/schedulers/cfg_zero_star.py` when implementing.
3. **Stable Audio Open 2 exact released defaults.** Stability has released multiple Open-family checkpoints; "Open 2" appears informally in literature for the rectified-flow generation but Stability's HF naming is `stable-audio-open-1.0` and `stable-audio-open-small`. Confirm the exact RF checkpoint name and its recipe when we wire it up.
4. **DiffRhythm 285 s mode** — does `max_frames` change the model's positional encoding configuration, or only the sigma table length? Re-read `model/cfm.py` __init__.
5. **Higher-order solvers (midpoint / Heun / RK4).** All these audio models default to first-order Euler, but DiffRhythm exposes midpoint/RK4/Implicit-Adams in its UI. Quality difference at NFE=32 is reportedly negligible; if we ever need higher-order, implement Heun as `v_half = model(x + dt*v_curr, t_next); v_avg = 0.5*(v_curr + v_half); x_next = x + dt*v_avg`. Defer.
6. **APG implementation details.** ACE-Step's `apg_forward` uses a `MomentumBuffer` carried across steps — exact decay and projection formula need to be read line-by-line from `acestep/schedulers/scheduling_flow_match_apg.py`. APG is optional (vanilla CFG works); defer until ACE-Step quality is shown to need it.
7. **CFG rescale exact formula in `sample_rf`.** Stability passes `rescale_cfg=True`; the rescale lives inside the model wrapper not the sampler. Confirm whether it's the SD3 zero-terminal-SNR rescale (`v *= sqrt(var(v_cond)/var(v_guided))`) or a per-sample variant. Matters above `cfg_scale ≈ 5`.

---

## Implementation Notes for HartsyInference

### Reuse what we have

- `FlowMatchEulerDiscreteScheduler` (existing, `src/HartsyInference.Diffusion/Schedulers/FlowMatchEulerDiscreteScheduler.cs`) **already handles**:
  - `sigma = shift * t / (1 + (shift - 1) * t)` schedule (set `shift=3.0` for ACE-Step, `shift=1.0` for F5/DiffRhythm/Stable Audio Open RF).
  - The Euler step `out = sample + v * (sigma_next - sigma)`.
  - The flow-matching forward `AddNoise` for img2img / audio2audio.
- `DdimScheduler` and `DpmPlusPlus2MScheduler` cover **AudioLDM 2** with no changes.

### What to add

**1. Sway sampling timestep helper.** A static method on (or factory variant of) `FlowMatchEulerDiscreteScheduler`:

```csharp
// Apply F5-TTS sway sampling AFTER computing the standard linspace sigmas
public void SetTimestepsWithSway(int numInferenceSteps, float swayCoef)
```

Implementation: for `i = 0..N`, compute `u = (float)i / N`, then `t_remap = u + swayCoef * (MathF.Cos(MathF.PI/2 * u) - 1 + u)`, then `sigma[i] = 1 - t_remap[N - i]` (reverse to match our 1→0 convention). Verify monotonicity in debug builds.

Default `swayCoef = -1.0f` for F5-TTS; expose to API.

**2. Omega mean-shift step variant.** Add overload (or `IScheduler` flag) to `Step` that performs the mean-preserving rescale. Naive serial loop is fine (sigma table is CPU, but `dx` is GPU). Two options:
- (a) Do mean reduction + rescale on GPU as a fused kernel (~30 lines of PTX), since `dx` is already there from the model forward.
- (b) Do it on CPU after a DtoH copy of `dx` — fine if the audio latents are small (~MBs not GBs). Recommend (a) for ACE-Step throughput.

The logistic squash on `omega_raw` is a scalar — do it once in C# when setting up the step:

```csharp
float omegaEff = 0.9f + 0.2f / (1.0f + MathF.Exp(-0.1f * omegaRaw));
```

**3. Velocity-field CFG combiner.** Identical to the ε-prediction CFG combiner we already have (per `CFG_AND_GUIDANCE.md`). Just rename the tensor labels in code paths that handle audio: `vCond`, `vUncond`, `vGuided`. The math `vGuided = vUncond + guidanceScale * (vCond - vUncond)` is one fused element-wise kernel — reuse the existing image kernel verbatim.

When wiring F5-TTS/DiffRhythm, **document in API surface** that their `cfg_strength` parameter equals `guidance_scale - 1.0` in our convention. Probably easiest: accept the user-facing name they expect, internally normalize to `guidance_scale`.

**4. PingPong sampler.** New scheduler class `FlowMatchPingPongScheduler` for Stable Audio Open Small. After Euler step, compute `x_clean = x - sigma_next * v`, re-noise to `x = (1 - sigma_next) * x_clean + sigma_next * randn`. Needs an RNG handle on the device (we already have one for img2img noise).

**5. SchedulerFactory entries.** Extend `SchedulerFactory.Create` to recognize:

| Name string | Returns |
|---|---|
| `"flow_match_euler"` | existing `FlowMatchEulerDiscreteScheduler` (shift=1.0) |
| `"flow_match_euler_sway"` | new — F5/E2/DiffRhythm |
| `"flow_match_euler_omega"` | new — ACE-Step (shift=3.0) |
| `"flow_match_pingpong"` | new — Stable Audio Open Small |

Add a constructor arg (or builder) for shift / sway_coef / omega so the same class can serve multiple model families without subclassing.

### Validation strategy

For each new scheduler, generate the sigma table for representative configs in Python (pin a checkpoint of each upstream repo) and dump as a `.json` golden file under `tests/HartsyInference.Diffusion.Tests/golden/audio/`. The C# scheduler must reproduce the table within `1e-6` absolute. This validates everything except the model forward pass — the sigma math is small and deterministic.

Concrete golden tables to produce:
- F5-TTS: `(steps=32, shift=1.0, sway_coef=-1.0)` → 33 sigma values
- F5-TTS: `(steps=16, shift=1.0, sway_coef=-1.0)` → 17 sigma values
- E2-TTS: `(steps=32, shift=1.0, sway_coef=0.0)` → 33 sigma values (== plain flow-match Euler)
- ACE-Step: `(steps=60, shift=3.0)` → 61 sigma values
- DiffRhythm: `(steps=32, shift=1.0, sway_coef=None)` → 33 sigma values
- Stable Audio Open (RF): `(steps=100, shift=1.0)` → 101 sigma values
- Stable Audio Open Small (RF): `(steps=8, shift=1.0)` → 9 sigma values

Then validate the full per-step `x_next` output bit-for-bit against the Python sampler using a fixed `v` tensor (replace the model forward with a deterministic fixture) and a fixed `randn` seed.

### Performance notes

- Sigma table is `O(N)` floats, computed once per generation. Don't worry about it.
- Omega mean-reduction across the latent tensor is `O(B * C * T)` — single fused GPU kernel, irrelevant compared to a transformer forward.
- The CFG kernel is the same fused `axpy` we use for image CFG — no perf work needed.
- All audio model latents are 1D in time (`[B, C, T]` or `[B, T, C]`); existing image-oriented `[B, C, H, W]` kernels work as long as we treat `T` as `H*W` and `C` as channels. Verify our element-wise kernels are layout-agnostic.

### What does NOT need new C# code

- AudioLDM 2 ε-prediction sampling — already covered by `DdimScheduler` / `DpmPlusPlus2MScheduler`.
- The flow-matching forward `AddNoise` for audio inpainting — identical formula to image (`x_t = (1-sigma)*x_0 + sigma*noise`), existing implementation works.
- Velocity prediction CFG — identical to ε-prediction CFG at the kernel level.
- Conditioning concat / masking — handled by the audio pipeline class, not the scheduler.

Sources:
- [Flow Matching for Generative Modeling (Lipman et al.)](https://arxiv.org/abs/2210.02747)
- [F5-TTS paper (ACL 2025)](https://aclanthology.org/2025.acl-long.313/)
- [F5-TTS arXiv](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [E2-TTS arXiv](https://arxiv.org/abs/2406.18009)
- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [DiffRhythm GitHub](https://github.com/ASLP-lab/DiffRhythm)
- [DiffRhythm arXiv](https://arxiv.org/abs/2503.01183)
- [stable-audio-tools GitHub](https://github.com/Stability-AI/stable-audio-tools)
- [Stable Audio Open paper](https://arxiv.org/abs/2407.14358)
- [stable-audio-open-1.0 HF](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [stable-audio-open-small HF](https://huggingface.co/stabilityai/stable-audio-open-small)
- [AudioLDM 2 pipeline (diffusers)](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2)
- [AudioLDM 2 — faster (HF blog)](https://huggingface.co/blog/audioldm2)
- [EPSS (Interspeech 2025)](https://arxiv.org/abs/2505.19931)
