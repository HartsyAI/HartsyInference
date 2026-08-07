# YOLO Architecture — Research Notes

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

YOLO (You Only Look Once) is an anchor-free, single-stage object detection model used in HartsyInference for subject detection, auto-cropping, and content moderation. This document covers both YOLOv8 and YOLO11 (also called YOLOv11) from Ultralytics. Both share a three-part architecture: a CSPDarknet-inspired **backbone** using C2f blocks (YOLOv8) or C3k2 blocks (YOLO11) with an SPPF module, a **neck** implementing a bidirectional Feature Pyramid Network (FPN + PAN) for multi-scale feature fusion, and a **decoupled detection head** that predicts bounding boxes via Distribution Focal Loss (DFL) and class probabilities through separate branches. The architecture produces detections at three scales (P3/8, P4/16, P5/32) yielding 8400 candidate boxes at 640x640 input. Post-processing uses Non-Maximum Suppression (NMS) with default confidence threshold 0.25 and IoU threshold 0.45. Both detection and instance segmentation (YOLOv8-seg / YOLO11-seg) variants exist; HartsyInference should prioritize the `n` (nano) variant for speed-sensitive use cases and support `s`/`m` for accuracy-sensitive scenarios.

## Key Numbers/Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Default input size | 640x640 | Square, letterboxed |
| Letterbox pad color | (114, 114, 114) | Gray |
| Stride | 32 | Backbone max downsample factor |
| Detection scales | P3/8, P4/16, P5/32 | 80x80, 40x40, 20x20 grids |
| Total anchors (640x640) | 8400 | 6400 + 1600 + 400 |
| reg_max | 16 | DFL distribution bins |
| Default classes (COCO) | 80 | |
| NMS conf_thres | 0.25 | |
| NMS iou_thres | 0.45 | |
| NMS max_det | 300 | |
| NMS agnostic | False | Class-specific by default |
| Normalization | /255.0 | No mean/std subtraction |
| Mask prototypes (seg) | 32 | nm parameter |
| Proto internal channels | 256 | npr parameter |
| Seg prototype resolution | 160x160 | At 640 input (1/4 scale) |

### Model Variant Scaling Parameters

| Variant | depth_multiple | width_multiple | max_channels |
|---------|---------------|----------------|--------------|
| **YOLOv8n** | 0.33 | 0.25 | 1024 |
| **YOLOv8s** | 0.33 | 0.50 | 1024 |
| **YOLOv8m** | 0.67 | 0.75 | 768 |
| **YOLOv8l** | 1.00 | 1.00 | 512 |
| **YOLOv8x** | 1.00 | 1.25 | 512 |
| **YOLO11n** | 0.50 | 0.25 | 1024 |
| **YOLO11s** | 0.50 | 0.50 | 1024 |
| **YOLO11m** | 0.50 | 1.00 | 512 |
| **YOLO11l** | 1.00 | 1.00 | 512 |
| **YOLO11x** | 1.00 | 1.50 | 512 |

- `depth_multiple` scales repeat count: `n_actual = max(round(n_base * depth_multiple), 1)`
- `width_multiple` scales channel count: `ch_actual = make_divisible(ch_base * width_multiple, 8)` capped at `max_channels`

### YOLOv8 Detection Model Performance (COCO val2017)

| Model | Params (M) | FLOPs (B) | mAP50-95 | CPU ONNX (ms) | A100 TRT (ms) |
|-------|-----------|-----------|----------|---------------|---------------|
| YOLOv8n | 3.2 | 8.7 | 37.3 | 80.4 | 0.99 |
| YOLOv8s | 11.2 | 28.6 | 44.9 | 128.4 | 1.20 |
| YOLOv8m | 25.9 | 78.9 | 50.2 | 234.7 | 1.83 |
| YOLOv8l | 43.7 | 165.2 | 52.9 | 375.2 | 2.39 |
| YOLOv8x | 68.2 | 257.8 | 53.9 | 479.1 | 3.53 |

### YOLO11 Detection Model Performance (COCO val2017)

| Model | Params (M) | FLOPs (B) | mAP50-95 | CPU ONNX (ms) | T4 TRT10 (ms) |
|-------|-----------|-----------|----------|---------------|---------------|
| YOLO11n | 2.6 | 6.5 | 39.5 | 56.1 | 1.5 |
| YOLO11s | 9.4 | 21.5 | 47.0 | 90.0 | 2.5 |
| YOLO11m | 20.1 | 68.0 | 51.5 | 183.2 | 4.7 |
| YOLO11l | 25.3 | 86.9 | 53.4 | 238.6 | 6.2 |
| YOLO11x | 56.9 | 194.9 | 54.7 | 462.8 | 11.3 |

### YOLOv8 Segmentation Model Performance (COCO val2017)

| Model | Params (M) | FLOPs (B) | mAPbox50-95 | mAPmask50-95 | CPU ONNX (ms) |
|-------|-----------|-----------|-------------|--------------|---------------|
| YOLOv8n-seg | 3.4 | 12.6 | 36.7 | 30.5 | 96.1 |
| YOLOv8s-seg | 11.8 | 42.6 | 44.6 | 36.8 | 155.7 |
| YOLOv8m-seg | 27.3 | 110.2 | 49.9 | 40.8 | 317.0 |
| YOLOv8l-seg | 46.0 | 220.5 | 52.3 | 42.6 | 572.4 |
| YOLOv8x-seg | 71.8 | 344.1 | 53.4 | 43.4 | 712.1 |

### YOLO11 Segmentation Model Performance (COCO val2017)

| Model | Params (M) | FLOPs (B) | mAPbox50-95 | mAPmask50-95 | CPU ONNX (ms) |
|-------|-----------|-----------|-------------|--------------|---------------|
| YOLO11n-seg | 2.9 | 9.7 | 38.9 | 32.0 | 65.9 |
| YOLO11s-seg | 10.1 | 33.0 | 46.6 | 37.8 | 117.6 |
| YOLO11m-seg | 22.4 | 113.2 | 51.5 | 41.5 | 281.6 |
| YOLO11l-seg | 27.6 | 132.2 | 53.4 | 42.9 | 344.2 |
| YOLO11x-seg | 62.1 | 296.4 | 54.7 | 43.8 | 664.5 |

## Data Layouts/Formats

### Input Tensor
```
Shape: [batch, 3, 640, 640]
Format: NCHW (batch, channels, height, width)
Channel order: RGB
Value range: [0.0, 1.0] (pixel / 255.0)
Dtype: float32
```

### ONNX Output Tensor — Detection
```
Shape: [batch, 4 + nc, 8400]   e.g., [1, 84, 8400] for COCO
Layout (before transpose):
  - [0:4, :] = cx, cy, w, h (center-format bounding box)
  - [4:4+nc, :] = class probabilities (sigmoid-activated)
After transpose to [8400, 84]: each row is one detection candidate.
```

### ONNX Output Tensor — Segmentation
```
Output 0: [batch, 4 + nc + nm, 8400]  e.g., [1, 116, 8400]
  - [0:4, :] = cx, cy, w, h
  - [4:84, :] = class probabilities
  - [84:116, :] = mask coefficients (32 values)

Output 1: [batch, nm, proto_h, proto_w]  e.g., [1, 32, 160, 160]
  - Mask prototype feature maps
```

### NMS Output (Post-Processing Result)
```
Per detection: [x1, y1, x2, y2, confidence, class_id]
  - (x1, y1): top-left corner (xyxy format, in original image coordinates)
  - (x2, y2): bottom-right corner
  - confidence: max class probability
  - class_id: integer class index
Maximum max_det=300 detections per image
```

### Effective Channel Counts per Variant (After Scaling)

**YOLOv8n** (width=0.25, depth=0.33, max_ch=1024):
- Backbone: 16→32→32→64→64→128→128→256→256→256 (SPPF)
- C2f repeats: [1, 2, 2, 1] (base [3,6,6,3] * 0.33, rounded, min 1)

**YOLOv8s** (width=0.50, depth=0.33, max_ch=1024):
- Backbone: 32→64→64→128→128→256→256→512→512→512 (SPPF)
- C2f repeats: [1, 2, 2, 1]

**YOLOv8m** (width=0.75, depth=0.67, max_ch=768):
- Backbone: 48→96→96→192→192→384→384→576*→576→576 (SPPF)
- C2f repeats: [2, 4, 4, 2]
- *1024 * 0.75 = 768, but capped by max_channels = 768

**YOLOv8l** (width=1.00, depth=1.00, max_ch=512):
- Backbone: 64→128→128→256→256→512→512→512*→512→512 (SPPF)
- C2f repeats: [3, 6, 6, 3]
- *1024 * 1.0 = 1024, capped by max_channels = 512

**YOLOv8x** (width=1.25, depth=1.00, max_ch=512):
- Backbone: 80→160→160→320→320→512*→512→512→512→512 (SPPF)
- C2f repeats: [3, 6, 6, 3]
- *All stages >= 512 base channels get capped to 512

## Reference Implementations

- [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics) — Official implementation in PyTorch
- [YOLOv8 model config YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml) — Backbone/head layer definitions and scaling params
- [YOLO11 model config YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml) — YOLO11 architecture definition
- [ultralytics/nn/modules/block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) — C2f, C3k2, SPPF, Bottleneck source code
- [ultralytics/nn/modules/head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py) — Detect, Segment head implementations
- [ultralytics/utils/ops.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py) — NMS implementation (`non_max_suppression`)
- [Ultralytics YOLO Docs — YOLOv8](https://docs.ultralytics.com/models/yolov8/) — Official performance benchmarks
- [Ultralytics YOLO Docs — YOLO11](https://docs.ultralytics.com/models/yolo11/) — YOLO11 benchmarks and docs
- [MMYOLO YOLOv8 description](https://mmyolo.readthedocs.io/en/latest/recommended_topics/algorithm_descriptions/yolov8_description.html) — Third-party detailed architecture walkthrough
- [YOLOv11 architectural enhancements (arXiv:2410.17725)](https://arxiv.org/html/2410.17725v1) — Academic analysis of YOLO11 changes
- [What is YOLOv8 (arXiv:2408.15857)](https://arxiv.org/html/2408.15857v1) — Comprehensive YOLOv8 analysis paper

## Differences Between Implementations

### YOLOv8 vs YOLO11

| Aspect | YOLOv8 | YOLO11 |
|--------|--------|--------|
| Core block | C2f | C3k2 (inherits C2f) |
| Spatial attention | None | C2PSA after SPPF |
| Backbone layers | 10 | 11 (extra C2PSA) |
| Default depth for `n` | 0.33 | 0.50 |
| `n` variant params | 3.2M | 2.6M (19% fewer) |
| `m` variant params | 25.9M | 20.1M (22% fewer) |
| `m` variant mAP | 50.2 | 51.5 (+1.3) |
| C3k2 expansion ratio | N/A | 0.25 in early stages, 0.5 in deep |
| SPPF n parameter | 3 (default) | 3 (default) |
| Head block | C2f | C3k2 |

### ONNX Export Considerations

- By default, ONNX export includes DFL decoding but **not** NMS
- The `nms` export flag can optionally embed NMS into the ONNX graph
- Without embedded NMS: output is `[1, 84, 8400]` (raw detections)
- With embedded NMS: output is `[1, max_det, 6]` (post-NMS, xyxy + conf + class)
- HartsyInference should implement NMS in C# for flexibility (adjustable thresholds at runtime)

### PyTorch vs ONNX Output Differences

- PyTorch `Detect.forward()` returns different formats for training vs inference
- ONNX export uses the inference path: decoded boxes + sigmoid class scores
- The ONNX model has DFL softmax and dist2bbox baked in; no need to reimplement DFL in C#

## Implementation Notes

### Priority Order for HartsyInference.Vision

1. **ONNX inference with pre-trained YOLO11n** — load model, preprocess, run, decode output
2. **NMS in C#** — implement greedy NMS with configurable conf/IoU thresholds
3. **Letterbox preprocessing** — implement exact Ultralytics-compatible letterbox with stride alignment
4. **Coordinate scaling** — undo letterbox transform to map detections back to original image
5. **Segmentation support** — add Proto mask decoding path
6. **Additional model sizes** — support `s`/`m` via same code path (architecture is identical, only weights differ)

### C# Implementation Considerations

- **NMS**: Use `Span<T>` or `Memory<T>` for zero-allocation box processing. IoU is purely arithmetic — no GPU needed.
- **Letterbox**: Can use SkiaSharp, ImageSharp, or raw pixel manipulation. Padding is trivial.
- **Transpose**: The `[1, 84, 8400]` → `[8400, 84]` transpose can be done as a view/stride change rather than copying.
- **ONNX Runtime**: Use `Microsoft.ML.OnnxRuntime` NuGet package. Input name is typically `"images"`, output name is `"output0"`.
- **Channel scaling**: All variants share the same code; the ONNX model already has the correct channel counts baked in. No need to implement depth/width multipliers in C#.
- **SIMD for NMS**: The IoU computation over many box pairs is a good candidate for `System.Runtime.Intrinsics` vectorization, especially the max/min/clamp operations.
- **Memory layout**: ONNX output is row-major. After transpose, iterate boxes in order for cache-friendly access during NMS.
- **Threshold for masks**: Use 0.5 for binary mask thresholding after sigmoid. This is the Ultralytics default.

### Weight Sources

- Pre-trained ONNX models can be exported from Ultralytics: `yolo export model=yolo11n.pt format=onnx`
- Or downloaded directly from [Ultralytics releases](https://github.com/ultralytics/ultralytics)
- COCO pre-trained models detect 80 object classes (person, bicycle, car, ..., toothbrush)
- For content moderation, the COCO "person" class (id=0) is the primary detection target
