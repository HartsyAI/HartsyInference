# Diffusion Schedulers — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (Schedulers)

## Summary

Research for implementing three diffusion noise schedulers: Euler Discrete, DDIM, and DPM++ 2M. All formulas verified against HuggingFace diffusers source code (main branch, April 2026).

## Common Beta Schedule

All schedulers share the same noise schedule initialization:

### Scaled Linear (default for Stable Diffusion)
```
beta_start = 0.00085, beta_end = 0.012, num_train_timesteps = 1000
betas = linspace(sqrt(beta_start), sqrt(beta_end), T)^2
alphas = 1 - betas
alphas_cumprod = cumprod(alphas)
```

### Linear
```
betas = linspace(beta_start, beta_end, T)
```

### AddNoise (Forward Process)
```
noisy = sqrt(alphas_cumprod[t]) * sample + sqrt(1 - alphas_cumprod[t]) * noise
```

## Euler Discrete Scheduler

Source: scheduling_euler_discrete.py (Katherine Crowson's k-diffusion)

### Sigmas
```
sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod)  (reversed, high to low)
sigmas = [sigmas..., 0.0]  (append terminal zero)
```

### Karras Sigmas
```
rho = 7.0
ramp = linspace(0, 1, num_inference_steps)
sigmas = (sigma_max^(1/rho) + ramp * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
```

### InitNoiseSigma
```
linspace/trailing: max(sigmas)
leading: sqrt(max(sigmas)^2 + 1)
```

### Step (epsilon prediction)
```
pred_x0 = sample - sigma * model_output
derivative = (sample - pred_x0) / sigma
dt = sigma_next - sigma
prev_sample = sample + derivative * dt
```

### Step (v_prediction)
```
pred_x0 = model_output * (-sigma / sqrt(sigma^2 + 1)) + (sample / (sigma^2 + 1))
derivative = (sample - pred_x0) / sigma
dt = sigma_next - sigma
prev_sample = sample + derivative * dt
```

## DDIM Scheduler

Source: scheduling_ddim.py (Song et al. 2020)

### Step Formula
```
alpha_prod_t = alphas_cumprod[t]
alpha_prod_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else final_alpha_cumprod
beta_prod_t = 1 - alpha_prod_t

# Epsilon prediction
pred_x0 = (sample - sqrt(beta_prod_t) * model_output) / sqrt(alpha_prod_t)
pred_eps = model_output

# V prediction
pred_x0 = sqrt(alpha_prod_t) * sample - sqrt(beta_prod_t) * model_output
pred_eps = sqrt(alpha_prod_t) * model_output + sqrt(beta_prod_t) * sample

# Variance
variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
std_dev_t = eta * sqrt(variance)

# DDIM update
pred_sample_direction = sqrt(1 - alpha_prod_t_prev - std_dev_t^2) * pred_eps
prev_sample = sqrt(alpha_prod_t_prev) * pred_x0 + pred_sample_direction
if eta > 0:
    prev_sample += std_dev_t * random_noise
```

### final_alpha_cumprod
```
set_alpha_to_one=True: final_alpha_cumprod = 1.0
set_alpha_to_one=False: final_alpha_cumprod = alphas_cumprod[0]
```

## DPM++ 2M Scheduler

Source: scheduling_dpmsolver_multistep.py (Lu et al. 2022)

### Alpha/Sigma/Lambda
```
alpha_t = sqrt(alphas_cumprod)
sigma_t = sqrt(1 - alphas_cumprod)
lambda_t = log(alpha_t) - log(sigma_t)   // log-SNR
```

### Sigma-to-alpha conversion (for sigma-based timestep indexing)
```
alpha_t = 1 / sqrt(sigma^2 + 1)
sigma_t = sigma * alpha_t
```

### Convert model output to x0 (DPM++ data prediction)
```
# epsilon prediction
x0_pred = (sample - sigma_t * model_output) / alpha_t

# v_prediction
x0_pred = alpha_t * sample - sigma_t * model_output
```

### First-order update (used on first step)
```
h = lambda_t - lambda_s
x_t = (sigma_t / sigma_s) * sample - alpha_t * (exp(-h) - 1) * D0
where D0 = x0_pred (data prediction from current step)
```

### Second-order multistep update (DPM++ 2M)
```
h = lambda_t - lambda_s0
h_0 = lambda_s0 - lambda_s1
r0 = h_0 / h
D0 = m0  (current model output in data prediction form)
D1 = (1/r0) * (m0 - m1)  (first-order difference)

# Midpoint solver (default, recommended)
x_t = (sigma_t / sigma_s0) * sample
    - alpha_t * (exp(-h) - 1) * D0
    - 0.5 * alpha_t * (exp(-h) - 1) * D1
```

## Reference Implementations

- [EulerDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py)
- [DDIMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
- [DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
- Original DDIM paper: Song et al., "Denoising Diffusion Implicit Models" (2020)
- Original DPM-Solver++ paper: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling" (2022)

## Key Numbers / Constants

- num_train_timesteps: 1000 (Stable Diffusion default)
- beta_start: 0.00085
- beta_end: 0.012
- beta_schedule: "scaled_linear"
- Karras rho: 7.0
- prediction_type: "epsilon" (SD1.5/SDXL), "v_prediction" (some models)
- DDIM eta: 0.0 (deterministic), 1.0 (DDPM equivalent)
