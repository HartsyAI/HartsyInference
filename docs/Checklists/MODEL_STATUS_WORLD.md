# World / Interactive Models — status

Concise status for interactive world models (action-conditioned, real-time frame generation). Build
detail lives in [PHASE_10_INTERACTIVE.md](PHASE_10_INTERACTIVE.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Oasis-500m** (Decart + Etched) | Real-weight numeric parity vs `etched-ai/open-oasis` on **CUDA/3060**: ViT-VAE encode corr **0.99999999**, decode corr **1.0**, DiT-S/2 v-pred corr **1.0** (maxAbs 3e-5). Fixed a DiT unpatchify vec-order bug (`[py,px,c]` not `[c,py,px]` — see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)). E2E AR rollout (DDIM v-pred + Diffusion Forcing) follows by composition (all cores corr 1.0). Weights: `camenduru/oasis-500m` mirror (the `Etched` repo is gated). |

## Built, validation-pending (🔧)

All built end-to-end with structural tests passing; numeric parity pending.

| Model | Notes |
|---|---|
| **Matrix-Game 3.0** (Skywork) | Flagship. 5B Wan2.2-TI2V finetune + `ActionModule` (mouse self-attn / keyboard cross-attn) + FOV memory + DMD 3-step. UMT5-XXL + Wan2.2 VAE. DiT reuses `WanVideoBlock`. |
| **Matrix-Game 2.0** (Skywork) | Entry-level. 1.8B Wan2.1-lineage; sliding-window KV cache; three per-domain variants. Wan2.1 16ch VAE; CLIP-ViT-H/14 seed. |
| **Hunyuan-GameCraft 1.0** (Tencent) | 12.5B HunyuanVideo MM-DiT + CameraNet (Plücker rays) + 33-ch composite history. PCM+CFG 8-step. Reusable `.pt` pickle loader + N-axis rope. No license gate (engine MIT, user-supplied weights). |

## Not started (❌)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W** | AR world-model substrate (FSQ tokenizer + AR transformer); tracked under video. |

## Notes

The `HartsyInference.Interactive` package (sessions / actions / camera / FOV memory) + the FlowUniPC and
FlowMatchDmd schedulers + a shared `ActionModule` back these models. World models share the 3D-video VAE
foundation with [the video models](MODEL_STATUS_VIDEO.md).
