# World / Interactive Models — status

Concise status for interactive world models (action-conditioned, real-time frame generation). Build
detail lives in [PHASE_10_INTERACTIVE.md](PHASE_10_INTERACTIVE.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

None yet. World models are built structurally; none has a real-weight, output-confirmed run.

## Built, validation-pending (🔧)

All built end-to-end with structural tests passing; numeric parity pending.

| Model | Notes |
|---|---|
| **Matrix-Game 3.0** (Skywork) | Flagship. 5B Wan2.2-TI2V finetune + `ActionModule` (mouse self-attn / keyboard cross-attn) + FOV memory + DMD 3-step. UMT5-XXL + Wan2.2 VAE. DiT reuses `WanVideoBlock`. |
| **Matrix-Game 2.0** (Skywork) | Entry-level. 1.8B Wan2.1-lineage; sliding-window KV cache; three per-domain variants. Wan2.1 16ch VAE; CLIP-ViT-H/14 seed. |
| **Oasis-500m** (Decart + Etched) | Tiny 500M AR Minecraft model; continuous Gaussian ViT-VAE (incl. encoder), DDIM v-pred + Diffusion Forcing, axial-attention DiT. The action-conditioning CI smoke model (fits the 3060). |
| **Hunyuan-GameCraft 1.0** (Tencent) | 12.5B HunyuanVideo MM-DiT + CameraNet (Plücker rays) + 33-ch composite history. PCM+CFG 8-step. Reusable `.pt` pickle loader + N-axis rope. No license gate (engine MIT, user-supplied weights). |

## Not started (❌)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W** | AR world-model substrate (FSQ tokenizer + AR transformer); tracked under video. |

## Notes

The `HartsyInference.Interactive` package (sessions / actions / camera / FOV memory) + the FlowUniPC and
FlowMatchDmd schedulers + a shared `ActionModule` back these models. World models share the 3D-video VAE
foundation with [the video models](MODEL_STATUS_VIDEO.md).
