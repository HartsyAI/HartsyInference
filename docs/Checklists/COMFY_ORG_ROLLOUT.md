# Comfy-Org rollout

Open work: bring every [Comfy-Org](https://huggingface.co/Comfy-Org) Hugging Face repo to a final state. That means
either verified end to end per the [add-model skill](../Agents/skills/add-model/SKILL.md), or recorded as partial or
blocked with the reason.

- Each row's evidence moves into [model status](MODEL_STATUS.md) and [parity](PARITY_VERIFICATION.md) as its stack lands.
- This file is deleted once every row is final.
- Shared infrastructure the rollout needs is tracked in [roadmap §13](ROADMAP.md#13-comfy-org-rollout-shared-infrastructure).

## Pins

- **Engine provenance pin:** `origin/main` 9204de77a (2.0.0-alpha.398), 2026-11-01. This is provenance only. Each stack
  rebases on current `main` and verifies its merge candidate there. Its status row records the HF revision, this pin, and
  the verified merge-candidate SHA.
- **HF revisions:** the `@rev` in each row is the repo `sha` from the Hugging Face API on 2026-11-01. Moving a pin means
  re-running that row's verification.

| Role | Machine | Confirmed |
|---|---|---|
| Research, code, CPU lane | Windows 11 dev box: .NET SDK 10.0.402, Python 3.12, `gh`; no GPU, no CUDA toolkit | 2026-11-01 |
| Real-weight, GPU and Swarm verification | Linux box per the add-model skill: RTX 4090 24 GB + RTX 3060 12 GB, models under `/mnt/model-storage/Models`, Swarm user unit, bench ComfyUI, the `gpu4090.lock`/`vk_3060.lock` locks | Not yet. Confirm before the first GPU batch |

## Completion states

A refusal never counts as support.

| State | Meaning |
|---|---|
| ✅ | Every feature the official release and ComfyUI ship is implemented and passes the gates below on real weights |
| 🟡 | Core path verified. Each named feature throws `NotSupportedException` and is listed in the status row |
| ⛔ | Blocked or unverified (gated weights, VRAM, or no parity). The row names the blocker and what unblocks it |

These three are the targets. The "Current / evidence" column records today's state with the
[model status legend](MODEL_STATUS.md), where 🔧 means built with verification incomplete and ⚠️ means mixed coverage.
Those are current states, not extra completion states.

Gates for ✅:

- **Hygiene.** Every GPU or Swarm batch runs under the device lock and `tests/swarm-quiet-window.sh`, with foreign-process
  sampling during the batch and `--verify-since` after it.
- **Tests actually ran.** Mandatory tests run with `HARTSY_REQUIRE_REAL_WEIGHTS=1` against a nonempty expected-test
  manifest. A `SKIPPED` line, an unavailable-backend guard, zero passed, or a missing manifest test fails the run. The
  check will live in `tests/require-executed.sh`, which does not exist yet (see roadmap §13). Until it lands, use the
  `run_class` check in `tests/run-multigpu-campaign.sh`.
- **CLI-vs-API identity**, compared per modality:
  - image: decoded pixels, including every layer;
  - video: decoded frames;
  - audio: decoded PCM samples;
  - vision: raw arrays;
  - 3D: vertex, face and splat arrays;
  - LLM: token ids.
- **No Swarm route.** Vision, 3D and frame-interpolation rows are verified through CLI and HTTP API and say "no Swarm
  route yet", by decision on 2026-11-01.

## Phases

| Phase | Scope | Shared infrastructure first |
|---|---|---|
| P0 | Pins, this inventory, independent catalog fixes, extension mapping sweep, test-execution helper | `tests/require-executed.sh` (to be written) |
| P1 | Quick wins: catalog assets, first real-weight runs, small fixes | — |
| P2 | Additive request/output contracts | vision batch request, new `VisionMode` values, structured `VisionResult`, `ImageResult.Layers` |
| P3 | Finish the partial models | ByT5 encoder, Stable Audio CFG + v-pred EDM |
| P4 | Control adapters and model patches | control base + ComfyUI `model_patches` loader |
| P5 | SD2.x | SD2 UNet presets, OpenCLIP-H text, v-prediction, `.ckpt` |
| P6 | New image DiTs | Cosmos MiniTrainDIT generalization, Jina-CLIP-v2 tower |
| P7 | Vision and post-process (CLI/API) | `IBackend.GridSample2D`, ViT RoPE with view-grouped attention |
| P8 | Video | causal frame KV cache (investigation first) |
| P9 | Audio | T5Gemma2 hoist |
| P10 | 3D (CLI/API) | DINOv3 |
| P11 | Quant, LLM, Ming-Image | Comfy w4a8/w6a8, AWQ `pre_quant_scale`, BailingMoeV2 config |

## Target repos (49)

"—" means no code yet.

| Repo @rev | Files / variants | Depends on | Phase | Target | Current / evidence |
|---|---|---|---|---|---|
| Fibo-2.0 @4742d4bec4 | dev and edit-turbo × bf16/int8; turbo LoRA; qwen3vl-4b; flux2-vae | — | P1 | ✅ Swarm | 🔧 dev bf16 ran 2026-10-22; int8 and edit-turbo not run ([image](MODEL_STATUS_IMAGE.md)) |
| MOSS-Image @530a4c94f9 | base and sft × bf16/int8; edit and colour/depth LoRAs; qwen3.5-9b bf16/int8/w4a8-gptq; flux2-vae (+small) | shared ref-KV refactor | P1, P3 | ✅ Swarm | 🔧 CUDA real-weight run 2026-10-20; catalog side models differ from recipe ([image](MODEL_STATUS_IMAGE.md#moss-image)) |
| MiniMax-H3 @912b6e38b2 | 39 files, checklist below | P4 model patches; P11 AWQ, w6a8 | P3, P4, P11 | ✅ Swarm | 🟡 dense FL2VA verified; the rest gated ([video](MODEL_STATUS_VIDEO.md#minimax-h3)) |
| Real-ESRGAN @5fd49b7b27 | RealESRGAN_x4plus | — | P1 | ✅ Swarm upscale | ⚠️ structural tests and CPU smoke only ([vision](MODEL_STATUS_VISION.md#real-esrgan)) |
| Flux-2-Klein-9B @1138ce884c | qwen3-8b bf16/fp8mixed/fp4mixed; flux2-vae; DiT from BFL | — | P1 | ✅ Swarm | 🔧 `Flux2Variants.Klein9B` wired, never run ([image](MODEL_STATUS_IMAGE.md)) |
| OneReward @77a4042ca2 | Fill transformer bf16/fp16/fp8; object-removal LoRA | P0 Fill entry | P1 | ✅ Swarm | — (Flux Fill path exists) |
| Qwen-Image-InstantX-ControlNets @90cf72c3b0 | Union, Inpainting | — | P1 | ✅ Swarm | 🔧 `QwenImageControlNetResolver`, unverified |
| Bernini-R @f7fd9812e5 | wan2.1 1.3B; wan2.2 high/low × fp16/fp8/int8/mxfp8 | P0 A14B role | P1 (P8 if not plain Wan) | ✅ Swarm | — |
| stable-audio-open-1.0 @3eb100708b | single checkpoint | Stable Audio CFG/EDM | P3 | ✅ Swarm music | 🔧 `StableAudioDitConfig.Open1_0` scaffolded, not weight-verified |
| Qwen-Image-DiffSynth-ControlNets @5ad28a1b22 | canny, depth, inpaint patches; union LoRA | control base | P4 | ✅ Swarm | — |
| stable-diffusion-3.5-controlnets @714fda3718 | large blur, canny, depth | control base | P4 | ✅ Swarm | — (SD3.5 verified, no ControlNet) |
| USO-1.0 @5665f42499 | DiT LoRA; projector patch | control base, Redux | P4 | ✅ Swarm | — |
| T2I-Adapter @db8c05084d | 18 files: sd14v1, sd15v2, coadapter-* | control base | P4 | ✅ Swarm | — |
| stable-diffusion-2.1 @a3feb9bf86 | 768 ema/nonema (.ckpt, .safetensors); 512-inpainting | SD2 path | P5 | ✅ Swarm | — |
| stable-diffusion-2.1-unclip @e1d3c1d908 | unclip-h, unclip-l (.ckpt) | SD2 path | P5 | ✅ Swarm | — |
| lotus @39a199245c | lotus-depth-d-v1-1 | SD2 path, vision contracts | P5 | ✅ CLI/API | — |
| Cosmos-Predict2 @bc1946fd79 | 2B and 14B × t2i and video2world (480/720 × 10/16 fps) | Cosmos DiT | P6 t2i, P8 v2w | ✅ Swarm | — (Anima 2B only, [image](MODEL_STATUS_IMAGE.md#anima)) |
| Anima-LLLite @5b79999eea | 10 patches: any-test ×4, depth, inpaint v1/v2, lineart, pose, scribble | Cosmos DiT, control base | P6 | ✅ Swarm | — |
| Qwen-Image-Layered @02130716aa | bf16, fp8mixed; RGBA VAE | layered output | P6 | ✅ Swarm | — |
| LongCat-Image @d49457a376 | base, edit, edit-turbo | — | P6 | ✅ Swarm | — |
| JoyAI-Image-Edit @8010e7b4c2 | DiT bf16/int8; qwen3vl-8b bf16/int8; wan2.1 VAE | — | P6 | ✅ Swarm | — |
| NewBie-Image-Exp0.1 @ae96af8db0 | DiT; gemma-3-4b; jina-clip-v2 | Jina-CLIP-v2 tower | P6 | ✅ Swarm | — |
| Ovis-Image @b7d0cb855e | DiT; ovis-2.5 | — | P6 | ✅ Swarm | — |
| HiDream-O1-Image @7b75f4cd7e | full and dev × bf16/fp8/mxfp8 | — | P6 | ✅ Swarm | — |
| PixelDiT @f1aa9118da | 24 PiD decoders (flux1, flux2, qwenimage, sd3; 512/1024; 4-step; bf16/int8/mxfp8) | — | P6 | ✅ Swarm | — |
| Depth-Anything-3 @8265be0701 | small, base, mono-large, metric-large | ViT RoPE, vision contracts | P7 | ✅ CLI/API | — (Depth-Anything-V2 verified) |
| Frame-Interpolation @aa7009be4d | rife 4.25/4.26 (lite, heavy), film_net | GridSample2D, vision contracts | P7 | ✅ CLI/API | — |
| MoGe @1484985258 | moge-1 vitl, moge-2 vitl-normal, moge-3 vitl/vitg | vision contracts | P7 | ✅ CLI/API | — |
| BiRefNet @884af709e9 | birefnet; lucida | vision contracts | P7 | ✅ CLI/API | — |
| mediapipe @f787efc1ad | face detector fp32 | vision contracts | P7 | ✅ CLI/API | — |
| marigold-v2-0 @c304e4370a | Qwen-Image-Edit-2509 int8; albedo, depth, normals × (LoRA + embedding + VAE) | vision contracts | P7 | ✅ CLI/API | — (Edit-2511 int8 verified) |
| SDPose @acb43fbe81 | wholebody; rt_detr_v4-x-hgnet fp16/fp32 | SD2 path, HGNetv2 backbone, vision contracts | P7 | ✅ CLI/API | — |
| sam3.1 @7bb8374780 | multiplex fp16 | vision contracts (temporal) | P7 | ✅ CLI/API | — (SAM 2 image verified) |
| HuMo @41dae81448 | 1.7B fp16; 17B fp16/fp8; whisper-large-v3 | Whisper per-layer taps | P8 | ✅ Swarm | — |
| SCAIL-2 @fe3c728bc7 | 14B × fp16/fp8/int8/mxfp8/nvfp4-mix; DPO and relight LoRAs | — | P8 | ✅ Swarm | — |
| Wan-Dancer @507e392c8e | 14B global, local (fp8) | — | P8 | ✅ Swarm | — |
| Causal-Forcing-Framewise @d9f57daa65 | single DiT | causal KV investigation | P8 | ✅ Swarm | — |
| Hunyuan-Video-1.5 @1405a2b738 | 25 DiTs: 480/720 t2v/i2v, cfg- and step-distilled, 1080p SR, capybara; sigclip | ByT5 | P8 | ✅ Swarm | — (HunyuanVideo 13B verified) |
| Mochi-Preview @ed4b8585dc | DiT bf16/fp8; all-in-one fp8; t5xxl; VAE | — | P8 | ✅ Swarm | — |
| void-model @f9c9e56592 | pass1, pass2; RAFT; t5xxl; CogVideoX VAE | GridSample2D | P8 | ✅ CLI/API | — |
| stable-audio-3 @75aad88362 | medium, small-music, small-sfx (+ base); t5gemma | Stable Audio CFG/EDM, T5Gemma2 | P9 | ✅ Swarm music | — |
| Ming-3.0-TTS @fde007af92 | tts-tiny int8 ConvRot | — | P9 | ✅ AudioLab | — |
| Hunyuan-3D-2.1 @d28dcff0eb | single checkpoint | GridSample2D, 2.0 mesh fix | P10 | ✅ CLI/API | — (2.0 has the open mesh defect, [3D](MODEL_STATUS_3D.md#hunyuan3d-2)) |
| TRELLIS.2 @430a9d09b2 | DiT bf16/int8; shape and texture VAEs; dinov3-L | DINOv3 | P10 | ✅ CLI/API | — ([TRELLIS v1](MODEL_STATUS_3D.md#trellis) splats only) |
| Pixal3D @f37641be37 | single and multiview × bf16/int8; TRELLIS.2 VAEs; dinov3-L | TRELLIS.2 | P10 | ✅ CLI/API | — |
| Idol @edb29d75b6 | DiT; dinov3-H; flux2-vae; decoder (TripoSplat fine-tune) | DINOv3 | P10 | ✅ CLI/API | — |
| sam-3d-body @7c974aea58 | dinov3 bf16/int8 | DINOv3 | P10 | ✅ CLI/API | — |
| Qwen3.8-27B @15daf09dbe | w4a8 | Comfy w4a8 | P11 | ✅ LLMAssistant | — |
| Ming-Image @93804ff374 | design and design-layer × bf16/int8; ling-mini-2.0 bf16/int8/w4a8; VAE | Comfy w4a8, BailingMoeV2, layered output | P11 | ✅ Swarm | — |

### MiniMax-H3 files (@912b6e38b2)

| Group | Files | Phase |
|---|---|---|
| FL2VA DiT | bf16 (verified path), int8 ConvRot, pruned bf16, pruned fp8-scaled, pruned int8 ConvRot, pruned w6a8 | P3; w6a8 P11 |
| Ref2VA DiT | bf16, int8 ConvRot, pruned bf16, pruned fp8-scaled, pruned int8 ConvRot, pruned w6a8 | P3; w6a8 P11 |
| Effect embeddings (10) | art_is_explosion, blooming_flowers, bullet_time, dark_magic, fire_breath, four_seasons, kiss_camera, spiral_ascent, storm_magic, truman_show | P3 |
| LoRAs (5) | fl2v turbo 4-step v1.0 768p, fl2v turbo 8-step v1.0, ref2v turbo 4-step v0.1, pika flex fl2va v1, pika flex ref2va v1 | P3 |
| Fun ControlNet patches (4) | union pruned bf16/int8, union 2.0 pruned bf16/int8 | P4 |
| Text encoders (3) | qwen3vl-32b bf16, int8 ConvRot, nvfp4-awq | P3; AWQ P11 |
| VAEs (5) | audio fp32, sol fp16/int8, video fp16/int8 | P3 |

## Already supported (44): P0 catalog and Swarm-mapping sweep

Alphabetical, read row by row.

| Repo @rev (1) | Repo @rev (2) | Repo @rev (3) | Repo @rev (4) |
|---|---|---|---|
| ACE-Step @ea4f3d349e | ACE-Step-1.5 @371bb1680f | Boogu-Image @8606e7df2f | Chroma1-HD @595f6c4194 |
| Chroma1-Radiance @a94a9efb87 | CLIP-ViT-H-14-laion2B-s32B-b79K_repackaged @b1aee1a783 | ERNIE-Image @50509a88bc | Flux-1-Dev @10cf201a3d |
| Flux-1-Kontext-Dev @795e390e4a | Flux-1-Krea-Dev @d9d7c421e6 | Flux-1-Redux-Dev @b23a47a2b6 | Flux-1-Schnell @d31e040670 |
| Flux-2-Dev @9630c1b4b6 | Flux-2-Klein-4B @5b3abca5c0 | gemma-4 @63d0f7c476 | HiDream @942970a008 |
| Hunyuan-3D-2.0 @15c2937a84 | Hunyuan-Image-2.1 @78cfe74376 | Hunyuan-Video @533a43813f | Ideogram-4 @2aa6c75ce6 |
| Krea-2 @1171f4dc7e | Lens @393618fd7d | ltx-2 @ccde4ba417 | ltx-2.3 @f20f3a5437 |
| Lumina-Image-2 @5b072540ef | Mage-Flow @6ff68fbf3c | MiniMax-Music-3 @9e364ccd29 | OmniGen-2 @babe908cd4 |
| Qwen-Image @1f12b17be1 | Qwen-Image-2.1 @aacf31a6d4 | Qwen-Image-Edit @f68ace85e6 | Qwen3-VL @02f0d3eefb |
| Qwen3.5 @5d50a2252b | RT-DETR @8d913979f8 | SeedVR2 @df48879708 | sigclip_vision_384 @5421dab312 |
| stable-diffusion-1.5 @9cfd069101 | stable-diffusion-3.5 @934251d1e6 | Wan-2.1 @123acf1cc7 | Wan-2.2 @ee6f4a4073 |
| Wan-Animate-2 @e7181fe189 | YuE2 @2f76ca75e6 | Z-Image @94d47a468e | Z-Image-Turbo @6fc90a3b1b |

Known gaps from the sweep audit:

- **Catalog:**
  - no Flux.1 Fill entry;
  - no Wan 2.2 A14B entry or low-noise expert role;
  - `flux2` lists Dev only;
  - Chroma Radiance is not CLI-drivable.
- **Hunyuan-Image-2.1:** the ByT5 glyph branch is never run because the pipeline passes null. That is fixed in P3 and
  shared with Hunyuan-Video-1.5.
- **Hunyuan-3D-2.0:** the open mesh defect is fixed in P10, before 2.1.
