# Image conditioning features — verification & flagship spot-bench, 2026-07-16

Engine `1.0.0-alpha.49.7/49.8-local`, extension `29fc1e2`, RTX 4090 24GB, live SwarmUI API
(`exact same request path as the published image benchmarks`). All rows visually verified from the
saved outputs; wall times are single warm runs (this is a correctness/feature campaign snapshot, not
a perf round — medians unchanged from the 07-11 table where re-measured).

## Flagship regression gate (per-deploy requirement)

| Model | Bar | Measured | Result |
|---|---|---|---|
| Krea2-Turbo 1024²/8st | < 6.5 s warm + clean | 4.58 s, clean apple | PASS |
| Z-Image-Turbo 1024²/8st | ≤ 3.2 s warm + clean | 2.81 s, clean | PASS |
| Flux-Dev warm repeat (step graph ON) | gen2 == gen1 | mean abs diff 0.0 (bit-identical) | PASS |

## Feature verifications (first-ever real-weight e2e for most rows)

| Feature | Checkpoint | Evidence |
|---|---|---|
| FLUX.1 Kontext edit | flux1-dev-kontext_fp8_scaled | apple recolor, composition preserved, clean textures (after VaeEncoder asym-padding fix; encode corr vs ComfyUI 0.871 → 0.999993) |
| FLUX.1 Fill inpaint | flux1-fill-dev-fp8 (Academia-SD) | masked apple → strawberry bowl, seam-free (binarized pixel-zero mask) |
| FLUX.1 Canny | flux1-canny-dev-fp8, guidance 30 | marble-apple contour-exact |
| FLUX.1 Depth | flux1-depth-dev-fp8 + in-engine DA-V2 ViT-L | synthetic-map ball + cropped-reference apple, depth-faithful; note: maps with flat caption-like bands trigger the model's web-image prior (not an engine bug) |
| FLUX.1 Redux | flux1-redux-dev + sigclip_vision_384 | reference-dominant variation; multiply×merge strength applied |
| SDXL ControlNet canny | diffusers_xl_canny_full | bronze-apple contour-exact |
| SDXL ControlNet stack | canny 0.7 + depth 0.5 | museum statue following reference |
| SDXL ControlNet start/end | canny end=0.4 | loose-contour partial-window behavior |
| SD1.5 ControlNet canny | control_v11p_sd15_canny_fp16 (LDM layout via new converter) | watercolor apple on contour |
| SD1.5 ControlNet depth | control_v11f1p_sd15_depth_fp16 | vase at reference position |
| SD1.5 ControlNet openpose | control_v11p_sd15_openpose_fp16 + in-engine YOLO11-pose→BODY-18 | robot in reference pose |
| IP-Adapter SDXL standard | ip-adapter_sdxl_vit-h @0.8 | subject/style transfer from portrait |
| IP-Adapter SDXL Plus | ip-adapter-plus_sdxl_vit-h @0.8 | near-identity transfer |
| IP-Adapter SD1.5 standard | ip-adapter_sd15 @0.8 | strong reference influence |

## Regressions found by this campaign's checks (fixed same day unless noted)

- Fused BF16/F16 GEMV misread 16-bit biases as F32 → Krea2 black (fixed).
- Flux step-graph replayed a freed latent buffer (preview hook) → warm-gen noise (fixed).
- VaeEncoder symmetric downsample padding → every img2img/Fill/Tools/Kontext encode off-grid (fixed).
- IP-Adapter consumed checkpoint K/V lists in the wrong layer order → black output (fixed).
- Qwen-Image-Edit 2511: ref VAE encode OOMs beside the resident DiT at 768²+ on 24GB (staging fix in flight).

## Evening additions (49.13→49.21-local, all live-verified via SwarmUI)

| Feature | Evidence |
|---|---|
| Lance 3B T2I | real-checkpoint reconciliation (7 arch fixes), velocity parity corr 1.000000; live clean apple 768² |
| Lens Turbo + Base | full bring-up (tokenizer/OOM/CUDA-MoE/4 correctness bugs); DiT parity corr 1.0; live astronaut-on-moon (turbo cfg 1) + lighthouse (base cfg 4); ~25-30s/step host-bound (perf follow-up) |
| Boogu-Image-Edit | VLM 384² token budget fix; live yellow-pear edit, scene preserved |
| Qwen-Image-Edit 2511 | VRAM staging regression fixed (ref-latent cache + eviction guards); live 1024² red-suit edit |
| Flux DiT ControlNet | union Pro-2.0 adapter, parity 3.7e-9 vs diffusers; live contour-locked glass apple |
| HED/Lineart/NormalBAE annotators | parity ≤7.9e-6 (several bit-exact); live lineart oil painting + normalbae clay apple |
| OmniGen2 edit | upstream dual-guidance math (shared ApplyDualCfg); live green-apple edit 1024² |
| IP-Adapter FaceID | ArcFace IR-50 parity cosine 1.000000; live identity-carrying portrait @0.9 |
| Final flagship gate | Krea2-Turbo 4.56s warm, Z-Image-Turbo 2.81s warm, Kontext spot clean |
