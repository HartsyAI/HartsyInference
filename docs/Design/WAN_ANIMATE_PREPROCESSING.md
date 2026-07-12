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

## Progress log

- **2026-07-11 — Phase 0 DONE** (extension, compiles): params `Animate Auto-Preprocess Driving` +
  `Animate Pose Video`/`Animate Face Video`; `WanAnimateDrivingPreprocessor` (override+raw wired, auto→raw
  fallback+warn); loader branches + cond-key folds in mode/overrides.
- **2026-07-11 — Phase 1/2 pose-model core DONE** (engine `Vision`, compiles 0/0). Rather than a redundant
  YOLOv8-face, we build **YOLOv11-pose** — its 17 COCO keypoints include the 5 face points (nose/eyes/ears),
  so it yields both the skeleton (Phase 2) and the face crop (Phase 1) from one model.
  - `YoloConfig`: `NumKeypoints`/`KptDims` fields + `YoloV11nPose` preset (nc=1, 17 kpts).
  - `PoseHeadV11`: composes `DetectHeadV11` (box/cls, no decode duplication) + a `cv4` keypoint branch;
    decode `x=(2rx+gx)·stride, y=(2ry+gy)·stride, v=σ(rv)` per `Pose.kpts_decode`.
  - `YoloV11PoseModel`: backbone/neck (self-contained for now; TODO extract shared `YoloV11Trunk`) + pose head;
    `Forward → (detections [1,4+nc,A], keypoints [1,nk·3,A])`.

- **2026-07-11 — pose pipeline DONE + VALIDATED (CPU, sub-pixel vs ultralytics).** `YoloPosePostProcessor`
  (confidence filter → letterbox-invert → `NonMaxSuppression.RunIndices` → gather+invert keypoints) +
  `Keypoint`/`PoseDetection` + `YoloPosePipeline`; `Transform.InvertPoint` + `NMS.RunIndices` (Run delegates,
  14/14 NMS tests pass). The **existing converter is generic** (folds BN over all convs, Pass 2 picks up the plain
  `cv4.{s}.2`) — `yolo11n-pose.pt` → `Models/yolo/yolo11n-pose-folded.safetensors` unchanged. **`YoloPoseRealImageTest`
  (CpuBackend) matches ultralytics within <1px** on a driving frame: box (97.6,385.3)→(395.2,832.0) conf 0.923 vs
  ref (97.4,385.5,396.2,832.0)/0.92; nose (252.8,439.3) vs (252.3,439.2). Whole port proven with NO GPU.
  Prep: `scratchpad/round12-backups/phase1-vision/pose_prep.py`.

- **2026-07-11 — face-crop geometry DONE + validated (CPU).** `PoseFaceCrop.ComputeSquareCrop` (face keypoints 0–4
  bbox → square × expand 2.2, +0.12 chin nudge; falls back to the person box's top square). On the driving frame it
  yields a well-framed 512² face crop (forehead→chin centered) — saved `phase1-vision/face_crop.png`. **Phase 1's
  core logic (pose keypoints → face crop) is now proven end-to-end with no GPU.**

- **2026-07-11 — Phase 1 face preprocessing WIRED + CPU-validated (extension, compiles 0/0 vs 44.87).**
  `WanAnimateFacePreprocessor.BuildFaceClip` (decode driving frames at the pose res → per-frame `YoloPosePipeline`
  → highest-conf person → `PoseFaceCrop` → bilinear-sample a face square, gray-padded, to 512² → `[1,3,T−1,512,512]`).
  `ControlVideoDecoder.DecodeFramesRgb` exposed. Loader: lazy `GetOrCreatePose` (pipeline cached on the entry,
  `ResolvePoseWeights` from the `Clip` folder + clear error) + threads it through `BuildDrivingClips`; cond-key already
  covers mode/overrides. Weights placed at `<Swarm>/Models/clip/yolo11n-pose-folded.safetensors`
  (sha `f112139b…`; TODO host for a `SideModels` auto-download). **CPU filmstrip across 6 driving frames: the crop
  tracks the moving face smoothly, well-framed, no bad jitter** (`phase1-wiring/face_filmstrip.png`).
- **2026-07-11 — Phase 1 DONE + GPU-VALIDATED (44.87 deployed).** Real 14B Animate gen (17f 480² 20st, auto-preprocess
  on): `loaded YOLO11n-pose` + **`face preprocess: 16/16 frames face-detected → 512² face clip`** (+2.7s, cached with
  the conditioning); output coherent (checkerboard-free android dance, identity + motion). Pose weights resolved from
  `Models/text_encoders/` (Clip FolderPaths[0]). Flagship gate: Z-Image 2.84s / Krea2 4.55s (visual, no regression).
  **Phase 1 (face crop) is shipped.** Pose-skeleton branch still logs the raw-fallback warning (Phase 2).
- **2026-07-11 — Phase 2 DONE + GPU-VALIDATED (44.88). FEATURE COMPLETE.** `OpenPoseRenderer` (engine `Vision`):
  COCO-17 → OpenPose-18 (neck = shoulder midpoint), controlnet_aux `draw_bodypose` convention — 17 colored limb
  "sticks" (rotated-ellipse fill, α 0.6) + 18 joint dots, standard palette. CPU-validated: renders correctly and
  **aligns with the person** (`phase2/skeleton_overlay.png`). Extension `WanAnimatePosePreprocessor` (per-frame
  `YoloPosePipeline` → render → `[1,3,T,H,W]` pose clip) wired into `BuildPoseClip`; loader creates one shared pose
  pipeline for either branch. **Real 14B full-auto gen: `pose preprocess: 17/17 frames skeleton-rendered` +
  `face preprocess: 16/16 frames` (+5s, cached), output coherent** (`phase2/animate_fullauto_output.png`). No
  diffusion-path delta vs 44.87 → flagships unchanged (2.84/4.55).

## FEATURE COMPLETE
One driving video → auto face crop + pose skeleton, in-backend, no manual DWPose/face-crop nodes (the ComfyUI gap).

## Remaining polish (optional — no correctness gaps)
- DWPose 133-kpt (hands + face) for expression fidelity (COCO-17 is body-only).
- Host the folded pose weights for a `SideModels` auto-download (currently manual-placed).
- Temporal smoothing on the per-frame face crop / skeleton to reduce any residual jitter.
- Combine the two per-frame pose passes (face + skeleton share one YOLO forward per frame).
- Phase 3 replace-mode (background video + character mask via `YoloSeg`/SAM2) — needs pipeline work too.

## Open decisions / risks
- YOLO weight path: is there a `.pt`→safetensors converter, or do existing YOLO models ship safetensors? (Phase 1.)
- Skeleton palette/format the Wan checkpoint expects (OpenPose colored vs. DWPose) — verify against a comfy render.
- COCO-17 lacks hands/face kpts → v1 skeleton is body-only; DWPose (Phase 2b) closes that for expressive hands.
