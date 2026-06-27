# Vision Models — status

Concise status for vision models: CLIP embeddings, object detection, and segmentation / face. Build
detail lives in [PHASE_6_VISION.md](PHASE_6_VISION.md). Vision-tower parity for VLM use is tracked in
[MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md) and [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend:
[MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Status | Notes |
|---|---|---|
| **CLIP** (ViT-L/14, H/14, bigG/14) | ✅ | Real `openai/clip-vit-large-patch14` loads; L2 norm == 1.0, self-sim == 1.0, semantic ranking matches published OpenAI numbers. Bit-exact HF diff is the one remaining diagnostic. |
| **YOLOv8** (n/s/m/l/x) | ✅ | End-to-end on real `yolov8n-folded.safetensors`; bus.png detects 4 persons + 1 bus, matching Ultralytics one-for-one. Output shape `[1,84,8400]`. |
| **YOLO11** (n/s/m/l/x) | ✅ | End-to-end on real `yolo11n.pt` → safetensors; bus.png output exactly matches Ultralytics YOLO11n (bus 0.940 + 4 persons). |

## Scaffold / stub (🚧)

| Model | Status | Notes |
|---|---|---|
| **SAM 2 / MobileSAM** | 🚧 | `SamPipeline` + `SamMaskDecoder` API reserved; throws until the Hiera encoder + two-way mask decoder are built. |
| **Face detection / landmarks** | 🚧 | `FaceDetector` + `LandmarkExtractor` placeholders (YOLOv8-Face / RetinaFace / InsightFace, backbone-agnostic). |

## Deferred (❌)

YOLO segmentation head (Proto + mask-coefficient branch), SigLIP / SigLIP 2 standalone, DINOv2 / DINOv3,
RT-DETR, Grounding DINO, EVA-CLIP / MetaCLIP / AM-RADIO, and GPU-native MaxPool2D / depthwise Conv kernels.
See [PHASE_6_VISION.md § 8](PHASE_6_VISION.md) for the stretch list.

## Notes

CLIP and SigLIP **vision towers used inside diffusion and VLMs** are validated separately (SigLIP tower
corr 1.0 for Gemma-3 / SmolVLM); see [MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md). The standalone Vision
package wraps the existing `ClipVisionEncoder` rather than duplicating the math.
