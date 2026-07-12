# Wan-Animate in-engine driving preprocessing — build plan

**Status:** confirmed scope 2026-07-11. Feature = auto-derive the pose skeleton + cropped face from a single
driving video, in the HartsyInference backend (pure C#). ComfyUI requires the user to wire DWPose + face-crop
nodes manually; doing it one-click in-backend is the new UX feature.

**Confirmed decisions:** Full auto (pose + face). Pose keypoints = **YOLOv11-pose now (17 COCO kpts), DWPose
(133 kpts) as a later fidelity upgrade.**

## Why it's needed

Wan-Animate's checkpoint was trained on two *preprocessed* driving inputs:
- **Pose video** = a rendered OpenPose/DWPose **skeleton** (colored limbs), VAE-encoded → `pose_patch_embedding`.
- **Face video** = a **detected + cropped, aligned** face at 512² → motion-encoder → face-adapter.
- *(replace mode, unmodeled)* background video + character mask.

`WanAnimateLoader` currently feeds the **raw** driving clip to both (`DecodeControlClip`). The pose branch then
sees raw RGB (out-of-distribution vs. skeletons) → degraded motion following; the face branch resamples the whole
frame (only OK for head-shots). Fixing this is the last Animate parity gap (correctness/quality, item 3).

## What exists to build on

- `HartsyInference.Vision/Detection` — YOLO backbone + FPN/PAN neck + decoupled DFL `DetectHead` (`YoloV8/V11`
  configs, `C2f`/`Sppf`/`ConvBnSilu` blocks). **Reusable for pose + face heads.**
- `CannyPreprocessor.Process(Image, w, h)` — the pure-C# extension-preprocessor pattern to mirror.
- `ControlVideoDecoder.DecodeControlClip` — driving container → `[1,3,T,H,W]` frames.
- **Conditioning cache (44.86)** — preprocessing is deterministic per driving video, so it runs once on a cache
  MISS and is cached with the pose/face latents. No per-gen cost after the first.
- ❌ `FaceDetection/{FaceDetector,LandmarkExtractor}` are `NotImplementedException` stubs. No pose-keypoint head yet.

## Build order

### Phase 0 — contract + power-user path (no vision models)
- Param **"Animate Driving Mode"** (enum): `Auto` (default) · `Pre-rendered` · `Raw`.
- Params **"Animate Pose Video"** + **"Animate Face Video"** (optional `Image` overrides) — supply pre-rendered
  inputs now; decouples the two branches from the single Init-Image clip.
- `WanAnimateLoader`: branch on mode. `Raw` = today's behavior. `Pre-rendered` = use the override clips (or the
  Init-Image clip) verbatim. `Auto` = run the preprocessors below; until they land, fall back to `Raw` + a WARN.
- Pipeline already accepts separate pose/face clips — no engine change. Ships immediately.

### Phase 1 — face crop (YOLOv8-face; reuses YOLO backbone)
- Implement `FaceDetector` for real: YOLO backbone + 1-class detect head + 5-point landmark branch (yolov8n-face).
  Weight source: akanametov/yolov8-face (or lindevs) → convert `.pt`→safetensors; auto-download via a new
  `SideModels.Entry`. (Check the existing YOLO weight-load path — add a `.pt` converter if there's no safetensors.)
- `WanAnimateFacePreprocessor`: per driving frame → largest face → 5-pt landmark-aligned padded **square** crop →
  resize 512² → `[1,3,T−1,512,512]` face clip (replaces the naïve whole-frame resample).

### Phase 2 — pose skeleton (YOLOv11-pose)
- `PoseHead` = detect head + keypoint branch `[N, 17*3]`; keypoint decode (anchor+stride offset/scale, sigmoid on
  visibility). New `YoloV11nPose` config (`kpt_shape=(17,3)`, 1 class). Weights: ultralytics `yolo11n-pose`.
- `SkeletonRenderer` (pure-C# raster): COCO-17 → OpenPose-18 (synthesize neck = shoulder midpoint), draw colored
  limbs (OpenPose palette) + joints → RGB frames. **Palette/limb convention must match what Wan expects — validate.**
- `WanAnimatePosePreprocessor`: frames → keypoints → skeleton render → `[1,3,T,H,W]` pose clip (engine VAE-encodes).

### Phase 3 — replace mode (optional, later)
- Character/background masks via existing `YoloSeg` (or SAM2) → wire `background_video`/`character_mask`
  conditioning (also needs `WanAnimatePipeline` work; currently unmodeled).

## Integration & validation

- Preprocessors run in `WanAnimateLoader.Generate` on the **conditioning-cache MISS** only (one-time per driving
  video). Output feeds the existing `GenerateAnimation(referenceRgb, poseClip, faceClip, …)` path.
- **Validation (discriminator-first):** (a) same-seed A/B raw-clip vs skeleton output — motion fidelity should
  jump; (b) cross-check the skeleton render + face crop against ComfyUI's DWPose/face-crop on the same clip;
  (c) YOLOv11-pose keypoints + YOLOv8-face bbox/landmarks vs the ultralytics reference on a test image (real-weight
  parity per `PARITY_VERIFICATION.md`). GPU-gate + flagship-gate as always.

## Open decisions / risks
- YOLO weight path: is there a `.pt`→safetensors converter, or do existing YOLO models ship safetensors? (Phase 1.)
- Skeleton palette/format the Wan checkpoint expects (OpenPose colored vs. DWPose) — verify against a comfy render.
- COCO-17 lacks hands/face kpts → v1 skeleton is body-only; DWPose (Phase 2b) closes that for expressive hands.
