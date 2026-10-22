# Attention kernels — HartsyInference vs PyTorch SDPA and Comfy Kitchen INT8

Attention timed on the shapes the flagship models run, frozen from `diagnostics.profileShapes` logs into
[`attention_shapes.json`](../python-baseline/attention_shapes.json): our kernels against PyTorch 2.13's SDPA backends and
Comfy Kitchen 0.2.37's `int8_attention`, on the same card. Recorded 2026-10-20 at the start of the attention overhaul
(Phase 0). Later phases add dated rows beside these, never in place of them.

**Method.** Median of 20 timed trials after 3 warm-ups (CUDA events), one shape at a time on an idle card. Error is
relative L2 against an F32 reference with TF32 off, over sampled query rows. An arm is timed as `kernel` (the attention
kernel alone) or `path` (the whole entry point, casts, copies and permutes included). "Production default" is the F16
entry point as the models call it. Ratios are ours over theirs, so below 1× is faster. Cells read
`arm timing: kernel ms (relL2)`. The full tables, probe results, environment and ledger are in
[`results/2026-10-20_attention_phase0/`](../results/2026-10-20_attention_phase0/); per-arm logs and GPU telemetry are not
committed.

**Hardware:** RTX 4090 (450 W) and RTX 3060 12 GB (170 W), driver 595.99.02, CUDA 13, cuDNN 9.24.0.

## RTX 4090

| shape | ours: fastest arm | ours: production default | torch: fastest | Comfy Kitchen INT8 | ours ÷ torch | ours ÷ CK |
|---|---|---|---|---|---|---|
| shape | ours: fastest arm | ours: production default | torch: fastest | Comfy Kitchen INT8 | ours ÷ torch | ours ÷ CK |
|---|---|---|---|---|---|---|
| krea2-1024 | sage-f16 kernel: sage-int8 2.97 ms (1.9e-03) | cudnn 3.54 ms (2.9e-04) | def-bf16 2.80 ms (2.3e-03) | 1.32 ms (5.8e-03) | 1.06× | 2.25× |
| krea2-2048 | sage-f16 kernel: sage-int8 40.15 ms (1.8e-03) | sage-int8 42.48 ms (1.8e-03) | flash-bf16 40.99 ms (2.4e-03) | 15.15 ms (6.6e-03) | 0.98× | 2.65× |
| flux-1024 | def-f16 kernel: cudnn 1.70 ms (2.9e-04) | cudnn 1.70 ms (2.9e-04) | def-f16 1.72 ms (2.9e-04) | 0.95 ms (6.0e-03) | 0.98× | 1.78× |
| chroma-keypad | cudnn-f16 kernel: cudnn 1.78 ms (2.9e-04) | cudnn 1.79 ms (2.9e-04) | cudnn-f16 1.67 ms (2.9e-04) | 0.92 ms (6.0e-03) | 1.07× | 1.93× |
| zimage-1024 | def-f16 kernel: cudnn 1.76 ms (2.9e-04) | cudnn 1.76 ms (2.9e-04) | def-f16 1.76 ms (2.9e-04) | 1.00 ms (6.0e-03) | 1.01× | 1.77× |
| qwenimage-1024 | sage-f16 kernel: sage-int8 1.58 ms (1.9e-03) | cudnn 1.65 ms (3.0e-04) | def-bf16 1.54 ms (2.3e-03) | 0.88 ms (6.1e-03) | 1.03× | 1.79× |
| wan14b-480p | def-f16 kernel: sage-int8 128.73 ms (1.9e-03) | sage-int8 128.73 ms (1.9e-03) | flash-bf16 134.99 ms (2.4e-03) | 50.56 ms (5.9e-03) | 0.95× | 2.55× |
| wan14b-xattn | def-f16 kernel: cudnn 2.33 ms (2.9e-04) | cudnn 2.33 ms (2.9e-04) | def-bf16 2.20 ms (2.3e-03) | 1.90 ms (6.3e-03) | 1.06× | 1.22× |
| wan5b | sage-f16 kernel: sage-int8 2.17 ms (1.9e-03) | cudnn 2.17 ms (3.0e-04) | def-f16 2.15 ms (2.9e-04) | 1.12 ms (6.0e-03) | 1.01× | 1.93× |
| ltx25-video | sage-f16 kernel: sage-int8 29.39 ms (1.9e-03) | cudnn 32.08 ms (3.0e-04) | def-f16 31.27 ms (2.9e-04) | 12.11 ms (5.9e-03) | 0.94× | 2.43× |
| ltx25-a2v | def-f16 kernel: cudnn 0.58 ms (3.1e-04) | cudnn 0.62 ms (3.1e-04) | def-f16 0.27 ms (3.0e-04) | 0.75 ms (6.2e-03) | 2.19× | 0.78× |
| d40-sd15 | def-f16 kernel: materialized 1.23 ms (3.7e-04) | materialized 1.23 ms (3.7e-04) | def-bf16 0.28 ms (2.3e-03) | 0.40 ms (5.4e-03) | 4.37× | 3.12× |
| d40-sd15-x | mat-f16 kernel: materialized 0.17 ms (3.6e-04) | materialized 0.17 ms (3.6e-04) | def-f16 0.09 ms (2.9e-04) | 0.37 ms (5.4e-03) | 1.95× | 0.45× |
| d64-sd35m | sage-f16 kernel: sage-int8 1.04 ms (1.9e-03) | cudnn 1.05 ms (3.0e-04) | def-bf16 0.87 ms (2.4e-03) | 0.59 ms (6.2e-03) | 1.19× | 1.77× |
| d64-sdxl | def-f16 kernel: cudnn 0.67 ms (3.0e-04) | cudnn 0.67 ms (3.0e-04) | def-f16 0.58 ms (3.0e-04) | 0.45 ms (5.9e-03) | 1.15× | 1.48× |
| d64-sdxl-x | cudnn-f16 kernel: cudnn 0.10 ms (3.0e-04) | cudnn 0.10 ms (3.0e-04) | def-bf16 0.08 ms (2.4e-03) | 0.26 ms (5.9e-03) | 1.18× | 0.38× |
| d96-lumina2 | cudnn-f16 kernel: cudnn 1.64 ms (2.9e-04) | cudnn 1.72 ms (2.9e-04) | def-bf16 1.15 ms (2.4e-03) | 0.87 ms (5.8e-03) | 1.42× | 1.89× |
| d256-ideogram4 | flash-f16 path: flash-f16 2.56 ms (2.9e-04) | flash-f16 2.75 ms (2.9e-04) | def-f16 2.39 ms (3.0e-04) | 1.43 ms (6.5e-03) | 1.07× | 1.79× |
| t5-mask | cast-f16 kernel: materialized 0.61 ms (2.1e-04) | materialized 0.63 ms (1.0e-03) | def-f16 0.09 ms (6.3e-04) | 0.28 ms (7.0e-03) | 6.53× | 2.18× |
| ck-parity-4k | def-f16 kernel: cudnn 1.40 ms (2.9e-04) | cudnn 1.40 ms (2.9e-04) | def-bf16 1.33 ms (2.3e-03) | 0.80 ms (6.2e-03) | 1.05× | 1.76× |
| ck-parity-8k | sage-f16 kernel: sage-int8 5.14 ms (1.9e-03) | sage-int8 5.15 ms (1.9e-03) | def-f16 5.11 ms (3.0e-04) | 2.25 ms (6.5e-03) | 1.01× | 2.29× |
| ck-parity-12k | sage-f16 kernel: sage-int8 10.71 ms (1.9e-03) | sage-int8 10.85 ms (1.9e-03) | def-bf16 11.44 ms (2.3e-03) | 4.56 ms (6.0e-03) | 0.94× | 2.35× |
| llm-prefill-4k | fa-nogqa kernel: fa2-causal 1.19 ms (5.4e-05) | cudnn-causal 2.87 ms (7.7e-05) | def-bf16 1.00 ms (6.3e-04) | — | 1.19× | — |
| llm-prefill-8k | fa-splitoff kernel: fa2-causal 4.50 ms (4.1e-05) | cudnn-causal 10.61 ms (5.9e-05) | def-bf16 3.59 ms (4.7e-04) | — | 1.25× | — |
| llm-decode-512 | fa-def kernel: flash-decode 0.03 ms (1.9e-07) | flash-decode 0.03 ms (1.9e-07) | def-bf16 0.09 ms (2.2e-03) | — | 0.30× | — |
| llm-decode-4k | fa-f16kv kernel: flash-decode 0.04 ms (2.2e-07) | flash-decode 0.04 ms (2.6e-07) | def-bf16 0.09 ms (2.1e-03) | — | 0.40× | — |
| llm-decode-32k | fa-nofa2 kernel: flash-decode 0.17 ms (4.7e-07) | flash-decode 0.31 ms (6.1e-07) | def-f16 0.22 ms (2.8e-04) | — | 0.81× | — |
| wan5b-xattn | def-f16 kernel: cudnn 0.25 ms (3.0e-04) | cudnn 0.25 ms (3.0e-04) | def-f16 0.27 ms (2.9e-04) | 0.34 ms (6.1e-03) | 0.94× | 0.74× |
| ltx25-video-xattn | sage-f16 path: cudnn 1.95 ms (3.0e-04) | cudnn 2.07 ms (3.0e-04) | def-f16 1.89 ms (3.0e-04) | 1.26 ms (6.1e-03) | 1.03× | 1.54× |
| ltx25-v2a | def-f16 kernel: cudnn 0.40 ms (2.9e-04) | cudnn 0.48 ms (2.9e-04) | eff-bf16 0.29 ms (2.4e-03) | 0.44 ms (5.9e-03) | 1.36× | 0.90× |
| ltx25-audio | cast-f16 kernel: cudnn 0.04 ms (2.9e-04) | cudnn 0.23 ms (2.9e-04) | def-bf16 0.08 ms (2.4e-03) | 0.27 ms (5.8e-03) | 0.53× | 0.15× |
| h3-gold | def-f16 kernel: cudnn 1.53 ms (2.9e-04) | cudnn 1.53 ms (2.9e-04) | def-f16 1.42 ms (3.0e-04) | 0.92 ms (6.4e-03) | 1.08× | 1.67× |
| d80-sd15 | def-f16 kernel: cudnn 0.08 ms (2.9e-04) | cudnn 0.08 ms (2.9e-04) | def-f16 0.09 ms (3.0e-04) | 0.36 ms (5.7e-03) | 0.88× | 0.22× |
| d160-sd15 | def-f32 kernel: materialized 0.07 ms (2.1e-04) | materialized 0.25 ms (3.7e-04) | def-bf16 0.09 ms (2.3e-03) | 0.35 ms (5.5e-03) | 0.69× | 0.18× |
| d64-sdxl-l3 | def-f16 kernel: cudnn 0.14 ms (2.9e-04) | cudnn 0.14 ms (2.9e-04) | def-bf16 0.14 ms (2.3e-03) | 0.26 ms (5.8e-03) | 1.03× | 0.55× |
| d64-sdxl-l3-x | def-f16 kernel: cudnn 0.14 ms (3.0e-04) | cudnn 0.14 ms (3.0e-04) | def-f16 0.09 ms (3.0e-04) | 0.24 ms (6.0e-03) | 1.59× | 0.55× |

## RTX 3060

| shape | ours: fastest arm | ours: production default | torch: fastest | Comfy Kitchen INT8 | ours ÷ torch | ours ÷ CK |
|---|---|---|---|---|---|---|
| shape | ours: fastest arm | ours: production default | torch: fastest | Comfy Kitchen INT8 | ours ÷ torch | ours ÷ CK |
|---|---|---|---|---|---|---|
| krea2-1024 | cudnn-f16 kernel: cudnn 17.43 ms (2.9e-04) | cudnn 19.83 ms (2.9e-04) | flash-f16 17.08 ms (2.9e-04) | 8.75 ms (5.9e-03) | 1.02× | 1.99× |
| krea2-2048 | sage-f16 kernel: sage-int8 235.30 ms (1.8e-03) | sage-int8 247.91 ms (1.8e-03) | def-f16 264.86 ms (2.9e-04) | 121.58 ms (5.7e-03) | 0.89× | 1.94× |
| flux-1024 | cudnn-f16 kernel: cudnn 10.47 ms (2.9e-04) | cudnn 10.49 ms (2.9e-04) | def-f16 10.55 ms (3.0e-04) | 5.67 ms (6.1e-03) | 0.99× | 1.85× |
| chroma-keypad | def-f16 kernel: cudnn 9.66 ms (2.9e-04) | cudnn 9.66 ms (2.9e-04) | cudnn-f16 9.49 ms (3.0e-04) | 5.21 ms (6.0e-03) | 1.02× | 1.85× |
| zimage-1024 | sage-f16 kernel: sage-int8 11.19 ms (1.8e-03) | cudnn 11.81 ms (2.9e-04) | def-f16 11.00 ms (2.9e-04) | 6.04 ms (6.1e-03) | 1.02× | 1.85× |
| qwenimage-1024 | sage-f16 kernel: sage-int8 8.90 ms (1.9e-03) | cudnn 9.63 ms (3.0e-04) | def-f16 8.88 ms (2.9e-04) | 4.88 ms (6.1e-03) | 1.00× | 1.83× |
| wan14b-480p | sage-f16 kernel: sage-int8 761.64 ms (1.9e-03) | sage-int8 761.81 ms (1.9e-03) | def-f16 908.93 ms (2.9e-04) | 418.82 ms (5.7e-03) | 0.84× | 1.82× |
| wan14b-xattn | sage-f16 kernel: sage-int8 14.60 ms (1.9e-03) | cudnn 14.72 ms (2.9e-04) | def-bf16 14.68 ms (2.3e-03) | 10.75 ms (5.9e-03) | 0.99× | 1.36× |
| wan5b | sage-f16 kernel: sage-int8 12.87 ms (1.9e-03) | cudnn 13.21 ms (3.0e-04) | flash-f16 13.75 ms (2.9e-04) | 7.50 ms (5.8e-03) | 0.94× | 1.72× |
| ltx25-video | sage-f16 kernel: sage-int8 179.26 ms (1.9e-03) | cudnn 202.17 ms (3.0e-04) | def-f16 209.26 ms (2.9e-04) | 100.63 ms (5.7e-03) | 0.86× | 1.78× |
| ltx25-a2v | sage-f16 path: cudnn 2.00 ms (3.1e-04) | cudnn 2.14 ms (3.1e-04) | eff-bf16 2.13 ms (2.3e-03) | 3.78 ms (5.7e-03) | 0.94× | 0.53× |
| d40-sd15 | def-f16 kernel: materialized 3.93 ms (3.7e-04) | materialized 3.93 ms (3.7e-04) | def-bf16 1.65 ms (2.3e-03) | 1.38 ms (5.2e-03) | 2.38× | 2.84× |
| d40-sd15-x | def-f16 kernel: materialized 0.84 ms (3.6e-04) | materialized 0.84 ms (3.6e-04) | def-bf16 0.11 ms (2.2e-03) | 0.37 ms (5.3e-03) | 7.40× | 2.26× |
| d64-sd35m | sage-f16 kernel: sage-int8 5.58 ms (1.9e-03) | cudnn 5.76 ms (3.0e-04) | def-bf16 5.18 ms (2.3e-03) | 3.79 ms (5.4e-03) | 1.08× | 1.47× |
| d64-sdxl | sage-f16 kernel: sage-int8 3.90 ms (1.9e-03) | cudnn 3.92 ms (3.0e-04) | flash-bf16 3.69 ms (2.3e-03) | 2.92 ms (5.6e-03) | 1.06× | 1.33× |
| d64-sdxl-x | cudnn-f16 kernel: cudnn 0.23 ms (3.0e-04) | cudnn 0.24 ms (3.0e-04) | def-f16 0.20 ms (3.0e-04) | 0.34 ms (5.9e-03) | 1.16× | 0.70× |
| d96-lumina2 | def-f16 kernel: cudnn 8.95 ms (2.9e-04) | cudnn 9.29 ms (2.9e-04) | def-bf16 6.88 ms (2.3e-03) | 5.77 ms (5.7e-03) | 1.30× | 1.55× |
| d256-ideogram4 | def-f16 path: flash-f16 15.61 ms (2.9e-04) | flash-f16 15.61 ms (2.9e-04) | def-bf16 15.31 ms (2.3e-03) | 8.47 ms (5.8e-03) | 1.02× | 1.84× |
| t5-mask | def-f32 kernel: materialized 2.05 ms (2.1e-04) | materialized 2.17 ms (1.0e-03) | def-f16 0.35 ms (6.2e-04) | 0.82 ms (8.5e-03) | 5.87× | 2.50× |
| ck-parity-4k | sage-f16 kernel: sage-int8 8.64 ms (1.9e-03) | cudnn 9.00 ms (2.9e-04) | flash-f16 8.81 ms (3.0e-04) | 4.88 ms (6.0e-03) | 0.98× | 1.77× |
| ck-parity-8k | def-f16 kernel: sage-int8 31.37 ms (1.9e-03) | sage-int8 31.37 ms (1.9e-03) | def-f16 34.60 ms (2.9e-04) | 17.02 ms (5.9e-03) | 0.91× | 1.84× |
| ck-parity-12k | sage-f16 kernel: sage-int8 67.00 ms (1.9e-03) | sage-int8 67.02 ms (1.9e-03) | def-f16 77.46 ms (2.9e-04) | 37.66 ms (6.0e-03) | 0.87× | 1.78× |
| llm-prefill-4k | fa-f16kv kernel: fa2-causal 7.30 ms (5.4e-05) | cudnn-causal 16.11 ms (7.7e-05) | def-f16 5.91 ms (8.1e-05) | — | 1.24× | — |
| llm-prefill-8k | fa-f16kv kernel: fa2-causal 26.17 ms (4.1e-05) | cudnn-causal 59.88 ms (5.9e-05) | flash-bf16 23.51 ms (4.7e-04) | — | 1.11× | — |
| llm-decode-512 | fa-f16kv kernel: flash-decode 0.03 ms (1.9e-07) | flash-decode 0.04 ms (1.9e-07) | def-f16 0.09 ms (3.0e-04) | — | 0.33× | — |
| llm-decode-4k | fa-f16kv kernel: flash-decode 0.08 ms (3.4e-07) | flash-decode 0.13 ms (4.1e-07) | def-f16 0.14 ms (3.0e-04) | — | 0.62× | — |
| llm-decode-32k | fa-nofa2 kernel: flash-decode 0.44 ms (8.5e-07) | flash-decode 0.85 ms (1.2e-06) | flash-bf16 0.54 ms (2.5e-03) | — | 0.82× | — |
| wan5b-xattn | sage-f16 kernel: sage-int8 1.51 ms (2.0e-03) | cudnn 1.53 ms (3.0e-04) | def-bf16 1.42 ms (2.3e-03) | 1.18 ms (5.6e-03) | 1.06× | 1.28× |
| ltx25-video-xattn | sage-f16 kernel: sage-int8 11.25 ms (2.0e-03) | cudnn 12.06 ms (3.0e-04) | def-f16 12.47 ms (2.9e-04) | 8.23 ms (5.9e-03) | 0.90× | 1.37× |
| ltx25-v2a | flash-f16 path: cudnn 1.62 ms (2.9e-04) | cudnn 1.81 ms (2.9e-04) | cudnn-bf16 1.47 ms (2.4e-03) | 1.34 ms (5.9e-03) | 1.10× | 1.21× |
| ltx25-audio | cast-f16 kernel: cudnn 0.07 ms (2.9e-04) | cudnn 0.26 ms (2.9e-04) | def-bf16 0.09 ms (2.4e-03) | 0.26 ms (5.6e-03) | 0.82× | 0.28× |
| h3-gold | cudnn-f16 kernel: cudnn 9.07 ms (2.9e-04) | cudnn 9.12 ms (2.9e-04) | def-f16 8.59 ms (2.9e-04) | 5.87 ms (6.0e-03) | 1.06× | 1.55× |
| d80-sd15 | cudnn-f16 kernel: cudnn 0.30 ms (2.9e-04) | cudnn 0.30 ms (2.9e-04) | def-f16 0.24 ms (2.9e-04) | 0.46 ms (5.3e-03) | 1.26× | 0.65× |
| d160-sd15 | def-f32 kernel: materialized 0.14 ms (2.1e-04) | materialized 0.32 ms (3.7e-04) | eff-f16 0.12 ms (2.9e-04) | 0.38 ms (5.3e-03) | 1.10× | 0.35× |
| d64-sdxl-l3 | def-f16 kernel: cudnn 0.53 ms (2.9e-04) | cudnn 0.53 ms (2.9e-04) | def-bf16 0.52 ms (2.4e-03) | 0.62 ms (6.0e-03) | 1.01× | 0.85× |
| d64-sdxl-l3-x | cudnn-f16 kernel: cudnn 0.20 ms (3.0e-04) | cudnn 0.21 ms (3.0e-04) | def-bf16 0.13 ms (2.3e-03) | 0.26 ms (5.5e-03) | 1.51× | 0.76× |

## Attention's share of a denoising step (RTX 4090)

From `diagnostics.profile` with `profileSync` over the attention labels, steps 2 to N. "Adjacent" is the permutes, K/V
widening and casts around attention. The last three columns are the step-time cut a kernel `r` times faster would give,
`f·(1 − 1/r)`, with the overall speedup in brackets. A 15 % cut at `r = 1.9` needs a share of at least 31.7 %.

| case | step ms (plain) | step ms (synced) | attention ms/step | share | adjacent ms/step | adj share | Δstep @ r=1.5 | Δstep @ r=1.9 | Δstep @ r=2.5 |
|---|---|---|---|---|---|---|---|---|---|
| chroma-hd | 483.6 | 892.7 | 182.4 | 20.4% | 37.3 | 4.2% | 6.8% (×1.07) | 9.7% (×1.11) | 12.3% (×1.14) |
| flux1-dev | 474.3 | 488.8 | 95.6 | 19.6% | 21.9 | 4.5% | 6.5% (×1.07) | 9.3% (×1.10) | 11.7% (×1.13) |
| h3-fl2va-fp8 | 1226.7 | 1153.2 | 616.2 | 53.4% | 66.9 | 5.8% | 17.8% (×1.22) | 25.3% (×1.34) | 32.1% (×1.47) |
| ideogram4 (fixed 12 steps) | 822.4 | 850.3 | 173.7 | 20.4% | 25.4 | 3.0% | 6.8% (×1.07) | 9.7% (×1.11) | 12.3% (×1.14) |
| krea2 | 537.4 | 553.2 | 79.1 | 14.3% | 17.8 | 3.2% | 4.8% (×1.05) | 6.8% (×1.07) | 8.6% (×1.09) |
| ltx25-dev | 783.9 | 1050.8 | 142.2 | 13.5% | 61.6 | 5.9% | 4.5% (×1.05) | 6.4% (×1.07) | 8.1% (×1.09) |
| lumina2 | 686.3 | 749.8 | 98.4 | 13.1% | 22.5 | 3.0% | 4.4% (×1.05) | 6.2% (×1.07) | 7.9% (×1.09) |
| qwenimage-q4k | 1620.1 | 1716.5 | 225.8 | 13.2% | 82.1 | 4.8% | 4.4% (×1.05) | 6.2% (×1.07) | 7.9% (×1.09) |
| sd15 | 71.1 | 111.8 | 35.6 | 31.8% | 2.7 | 2.4% | 10.6% (×1.12) | 15.1% (×1.18) | 19.1% (×1.24) |
| sdxl | 275.4 | 289.3 | 28.2 | 9.7% | 8.1 | 2.8% | 3.2% (×1.03) | 4.6% (×1.05) | 5.8% (×1.06) |
| wan21-14b | 1739.6 | 1852.2 | 306.0 | 16.5% | 132.2 | 7.1% | 5.5% (×1.06) | 7.8% (×1.08) | 9.9% (×1.11) |
| wan5b | 1121.4 | 1189.7 | 181.2 | 15.2% | 68.1 | 5.7% | 5.1% (×1.05) | 7.2% (×1.08) | 9.1% (×1.10) |
| zimage | 422.9 | 306.0 | 66.1 | 21.6% | 12.9 | 4.2% | 7.2% (×1.08) | 10.2% (×1.11) | 13.0% (×1.15) |

## cuDNN 9.24 capability probes

Each probe built a fused SDPA graph in its own process, ran it and compared it with an F32 reference
([`tests/cudnn-probes.sh`](../../tests/cudnn-probes.sh)).

| Card | Result |
|---|---|
| RTX 3060 (SM 86) | All 15 probes pass. Supported and correct: BF16 I/O, grouped K/V (48/12, 16/8), head dims 32 to 256 in steps of 8 (40 and 160 included; 36 and 44 refused), per-head, per-batch and key-only bias, causal as one band op, sliding window, LSE output, sinks, softcap, ALiBi, padding, a by-value scale. A unified block mask is not supported. Plan build about 45 ms. |
| RTX 4090 (SM 89) | The same outcome as the RTX 3060 on every probe. |

## End to end against ComfyUI

RTX 4090, 2026-10-21, engine at the Phase 0 baseline (e845dcd9), ComfyUI from the Swarm venv with and without
`--use-ck-attention` (commit in `results/2026-10-20_attention_phase0/4090/e2e/comfy.txt`). Per-step time is the slope
(T(2S) − T(S)) / S after a discarded warm-up per arm and model, median of two rounds, so model load cancels
([`bench_comfy_e2e.py`](../python-baseline/bench_comfy_e2e.py), rows in `e2e/e2e.json`).

| model (steps S) | ComfyUI ms/step | ComfyUI + CK attention | engine | engine ÷ CK |
|---|---|---|---|---|
| Krea2-Turbo (8) | 683.8 | 647.4 | 531.1 | 0.82× |
| Flux.1-dev (20) | 563.8 | 510.6 | 454.5 | 0.89× |
| Wan 2.2 5B (20) | 779.7 | 742.3 | 1140.9 | 1.54× |
| LTX-2.5 distilled (8) | not comparable | | | |

- LTX-2.5's slope moves ±25% between rounds on both engines (the two-stage recipe's refine stage does not scale with S)
  and the engine's first round went negative, so no ratio is quoted.
- The engines compute at different precisions (ComfyUI bf16, the engine F16/F32 by layer), so these rows are speed only.

## Reading the numbers

- On the DiT shapes our best path ties PyTorch's best SDPA backend on both cards.
- Comfy Kitchen's INT8 attention beats our best wherever it applies (unmasked, head dim 128 class), at about 6e-3 relative
  L2 against our 2e-3 to 3e-4.
- The biggest gaps to PyTorch are paths that still materialize the score matrix: SD1.5's head dim 40 and the T5 per-head
  bias. cuDNN 9.24 serves both shapes.
- Only MiniMax-H3 and SD1.5 spend enough of a step in attention for a 1.9× faster kernel to cut a step by 15 %.
