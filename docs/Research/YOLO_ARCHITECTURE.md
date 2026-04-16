# YOLO Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Vision

## Summary

YOLO (You Only Look Once) is an anchor-free, single-stage object detection model used in SharpInference for subject detection, auto-cropping, and content moderation. This document covers both YOLOv8 and YOLO11 (also called YOLOv11) from Ultralytics. Both share a three-part architecture: a CSPDarknet-inspired **backbone** using C2f blocks (YOLOv8) or C3k2 blocks (YOLO11) with an SPPF module, a **neck** implementing a bidirectional Feature Pyramid Network (FPN + PAN) for multi-scale feature fusion, and a **decoupled detection head** that predicts bounding boxes via Distribution Focal Loss (DFL) and class probabilities through separate branches. The architecture produces detections at three scales (P3/8, P4/16, P5/32) yielding 8400 candidate boxes at 640x640 input. Post-processing uses Non-Maximum Suppression (NMS) with default confidence threshold 0.25 and IoU threshold 0.45. Both detection and instance segmentation (YOLOv8-seg / YOLO11-seg) variants exist; SharpInference should prioritize the `n` (nano) variant for speed-sensitive use cases and support `s`/`m` for accuracy-sensitive scenarios.

## Detailed Findings

### 1. Backbone Architecture

The backbone is a modified CSPDarknet that progressively downsamples the input image through five stages (P1/2 through P5/32), producing multi-scale feature maps.

**YOLOv8 backbone** (10 layers):

| Layer | Module | Output Channels | Repeats (base) | Stride | Resolution |
|-------|--------|----------------|-----------------|--------|------------|
| 0 | Conv (3x3) | 64 | 1 | 2 | P1/2 |
| 1 | Conv (3x3) | 128 | 1 | 2 | P2/4 |
| 2 | C2f | 128 | 3 | - | P2/4 |
| 3 | Conv (3x3) | 256 | 1 | 2 | P3/8 |
| 4 | C2f | 256 | 6 | - | P3/8 |
| 5 | Conv (3x3) | 512 | 1 | 2 | P4/16 |
| 6 | C2f | 512 | 6 | - | P4/16 |
| 7 | Conv (3x3) | 1024 | 1 | 2 | P5/32 |
| 8 | C2f | 1024 | 3 | - | P5/32 |
| 9 | SPPF | 1024 | 1 | - | P5/32 |

Note: Channel counts shown are base (pre-scaling). The `True` argument on C2f layers enables shortcut (residual) connections in the bottleneck blocks. The repeat count is scaled by `depth_multiple`.

**YOLO11 backbone** (11 layers — adds C2PSA after SPPF):

| Layer | Module | Output Channels | Repeats (base) | Args |
|-------|--------|----------------|-----------------|------|
| 0 | Conv (3x3) | 64 | 1 | stride=2 |
| 1 | Conv (3x3) | 128 | 1 | stride=2 |
| 2 | C3k2 | 256 | 2 | c3k=False, e=0.25 |
| 3 | Conv (3x3) | 256 | 1 | stride=2 |
| 4 | C3k2 | 512 | 2 | c3k=False, e=0.25 |
| 5 | Conv (3x3) | 512 | 1 | stride=2 |
| 6 | C3k2 | 512 | 2 | c3k=True |
| 7 | Conv (3x3) | 1024 | 1 | stride=2 |
| 8 | C3k2 | 1024 | 2 | c3k=True |
| 9 | SPPF | 1024 | 1 | k=5 |
| 10 | C2PSA | 1024 | 2 | — |

### 2. C2f Block (YOLOv8)

C2f ("Cross Stage Partial with 2 convolutions — Faster") is the core building block. It is an evolution of the C3 block from YOLOv5/v7 that retains outputs from every bottleneck for richer gradient flow.

**Structure:**

```
Input (c1 channels)
  |
  cv1: Conv(c1, 2*c, 1x1)          # expand to 2*hidden_channels
  |
  chunk(2, dim=1)                   # split into two halves of c channels
  |       \
  y[0]    y[1]
           |
        Bottleneck_0 → y[2]
           |
        Bottleneck_1 → y[3]
           |
          ...
        Bottleneck_{n-1} → y[n+1]
  |
  cat([y[0], y[1], y[2], ..., y[n+1]], dim=1)   # (2+n)*c channels
  |
  cv2: Conv((2+n)*c, c2, 1x1)      # compress back to output channels
  |
Output (c2 channels)
```

Where:
- `c = int(c2 * e)` with expansion ratio `e = 0.5` by default
- `n` = number of bottleneck blocks (determined by `repeats` scaled by `depth_multiple`)
- Each **Bottleneck** has: `Conv(c, c_, 3x3)` → `Conv(c_, c, 3x3)` with `c_ = int(c * 1.0)` (expansion=1.0 inside bottleneck, so c_ = c)
- When `shortcut=True` and `c1 == c2`: output = input + bottleneck(input)

**Key difference from C3**: C3 only used the *last* bottleneck output. C2f concatenates *all* intermediate outputs, providing denser feature reuse.

### 3. C3k2 Block (YOLO11)

C3k2 inherits from C2f and replaces it in YOLO11. The key change: when `c3k=True`, each bottleneck is replaced with a deeper C3k module (which itself contains 2 bottleneck layers with 3x3 kernels); when `c3k=False`, it uses standard Bottleneck blocks identical to C2f.

```python
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```

In the YOLO11 YAML, early backbone stages use `c3k=False` with `e=0.25` (narrower hidden channels), while deeper stages (layers 6, 8) and the final head stage (layer 22) use `c3k=True` with default `e=0.5` for richer feature extraction.

### 4. C2PSA Block (YOLO11 Only)

Cross Stage Partial with Spatial Attention. Added after SPPF in YOLO11 backbone (layer 10). Applies spatial attention pooling to enable the model to focus on key regions, improving detection of small and partially occluded objects. Not present in YOLOv8.

### 5. SPPF (Spatial Pyramid Pooling — Fast)

Produces multi-scale context by applying sequential max-pooling operations:

```
Input (c1 channels)
  |
  cv1: Conv(c1, c1//2, 1x1)          # halve channels
  |
  y[0] = cv1_output
  y[1] = MaxPool2d(k=5, s=1, p=2)(y[0])
  y[2] = MaxPool2d(k=5, s=1, p=2)(y[1])
  y[3] = MaxPool2d(k=5, s=1, p=2)(y[2])
  |
  cat([y[0], y[1], y[2], y[3]], dim=1)   # 4 * (c1//2) = 2*c1 channels
  |
  cv2: Conv(2*c1, c2, 1x1)
  |
Output (c2 channels)
```

Three sequential MaxPool2d with kernel=5 is equivalent to SPP with kernels (5, 9, 13) but approximately 2x faster. The spatial dimensions are preserved (stride=1, padding=k//2).

### 6. Neck: FPN + PAN (Path Aggregation Network)

The neck fuses features from multiple backbone scales using a top-down FPN path followed by a bottom-up PAN path.

**YOLOv8 Neck** (layers 10-21):

```
Top-down (FPN) path:
  Upsample(P5) + Concat(backbone_P4) → C2f → P4_fused (512ch)
  Upsample(P4_fused) + Concat(backbone_P3) → C2f → P3_out (256ch)    # Layer 15

Bottom-up (PAN) path:
  Conv_downsample(P3_out) + Concat(P4_fused) → C2f → P4_out (512ch)  # Layer 18
  Conv_downsample(P4_out) + Concat(P5) → C2f → P5_out (1024ch)       # Layer 21
```

Output feature maps fed to detection head:
- P3/8: 80x80 grid (small objects), 256 base channels
- P4/16: 40x40 grid (medium objects), 512 base channels
- P5/32: 20x20 grid (large objects), 1024 base channels

**YOLO11 Neck**: Identical structure but uses C3k2 blocks instead of C2f, and concatenates from layer 10 (C2PSA output) instead of layer 9 for the P5 path.

### 7. Decoupled Detection Head

YOLOv8/YOLO11 use a decoupled head design where classification and bounding box regression are handled by separate convolutional branches. This replaces the coupled head and removes the objectness branch from YOLOv5.

**For each of the three scale levels (P3, P4, P5):**

**Box regression branch (cv2):**
```
Conv(in_ch, c2, 3x3) → Conv(c2, c2, 3x3) → Conv2d(c2, 4*reg_max, 1x1)
where c2 = max(16, ch[0]//4, reg_max*4)
```

**Classification branch (cv3):**
```
Conv(in_ch, c3, 3x3) → Conv(c3, c3, 3x3) → Conv2d(c3, nc, 1x1)
where c3 = max(ch[0], min(nc, 100))
```

**reg_max = 16** (default). Each box coordinate is predicted as a discrete probability distribution over 16 bins, then converted to a continuous value via DFL (Distribution Focal Loss). The DFL layer computes a weighted sum: `sum(softmax(logits) * [0, 1, 2, ..., 15])` for each of the 4 coordinates (left, top, right, bottom distances from anchor point).

The head produces predictions at all three scales. Total anchor points at 640x640: `80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400`.

### 8. Instance Segmentation Head (YOLOv8-seg / YOLO11-seg)

The segmentation variant extends the Detect head with:

1. **Proto module**: Generates `nm=32` mask prototypes at 1/4 resolution (160x160 for 640x640 input)
   - Input: P3 features (first/shallowest feature map)
   - Architecture: Conv layers upsampling to produce `[batch, 32, 160, 160]`
   - `npr=256` (number of internal prototype channels)

2. **Mask coefficient branch (cv4)**: Per-scale convolution branches producing 32 mask coefficients per detection
   - Output per detection: 32 float values

3. **Mask assembly**: Final mask = sigmoid(coefficients @ prototypes), cropped to bounding box, then upsampled to original image size

**Segmentation output tensor shape (ONNX):**
- Detection: `[1, 116, 8400]` where 116 = 4 (bbox) + 80 (classes) + 32 (mask coefficients)
- Prototypes: `[1, 32, 160, 160]`

### 9. Input Preprocessing

**Letterbox resize:**
1. Compute scale ratio: `r = min(target_size / img_height, target_size / img_width)`
2. Resize image by ratio `r` (preserving aspect ratio)
3. Compute padding: `dw = target_size - new_width`, `dh = target_size - new_height`
4. When `auto=True`: round padding to nearest multiple of stride (32): `dw = dw % 32`, `dh = dh % 32`
5. Split padding evenly: `top = dh/2`, `bottom = dh - top`, `left = dw/2`, `right = dw - left`
6. Apply padding with `cv2.copyMakeBorder` using color `(114, 114, 114)` (gray)

**Normalization:**
- Divide pixel values by 255.0 to normalize to `[0, 1]`
- No ImageNet mean/std subtraction (no centering)

**Channel ordering:**
- Convert BGR (OpenCV default) to RGB
- Convert HWC to CHW format: `[3, 640, 640]`
- Add batch dimension: `[1, 3, 640, 640]`

**Data type:** float32

### 10. Output Tensor Format and Decoding

**ONNX detection output shape:** `[1, 84, 8400]` for 80-class COCO model
- Rows 0-3: bounding box coordinates (cx, cy, w, h) — center-x, center-y, width, height (already decoded from DFL distributions; sigmoid already applied)
- Rows 4-83: class probabilities (sigmoid already applied, no softmax needed)

**Post-processing steps:**
1. Transpose: `[1, 84, 8400]` → `[8400, 84]`
2. Extract boxes: columns 0-3 (cx, cy, w, h)
3. Extract class scores: columns 4-83
4. Convert xywh to xyxy: `x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2`
5. Get max class score and class index per box
6. Filter by confidence threshold
7. Apply NMS
8. Scale coordinates back to original image dimensions (undo letterbox transform)

**For segmentation models**, the output is `[1, 116, 8400]` plus a separate prototype tensor `[1, 32, 160, 160]`. The extra 32 channels (columns 84-115) are mask coefficients. After NMS, multiply mask coefficients by prototypes, apply sigmoid, crop to bounding box, and upsample to original resolution.

### 11. Non-Maximum Suppression (NMS) Algorithm

**Default parameters:**
- `conf_thres = 0.25` (confidence score threshold)
- `iou_thres = 0.45` (IoU threshold for suppression)
- `agnostic = False` (class-specific NMS by default)
- `max_det = 300` (maximum detections per image)

**Algorithm steps:**

```
1. Filter candidates: keep boxes where max_class_score > conf_thres
2. For each remaining box: compute (class_index, class_score) from argmax of class scores
3. If class-specific NMS (agnostic=False):
     Offset boxes by class_index * max_box_dimension to separate classes spatially
   If class-agnostic NMS (agnostic=True):
     No offset (all classes compete with each other)
4. Sort boxes by confidence score (descending)
5. Greedy NMS loop:
   a. Pick the box with highest score → add to results
   b. Compute IoU of this box with all remaining boxes
   c. Remove boxes with IoU > iou_thres
   d. Repeat until no boxes remain or max_det reached
6. Return at most max_det detections per image
```

**IoU calculation (axis-aligned boxes):**
```
intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) *
               max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
union = area_a + area_b - intersection
iou = intersection / union
```

**Class-specific NMS trick**: Ultralytics implements class-specific NMS by adding a large offset (`class_id * 7680`) to box coordinates before NMS, so boxes of different classes never overlap spatially during the IoU computation. This avoids running NMS separately per class.

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

## Algorithm Steps

### Complete Inference Pipeline

```
1. PREPROCESS
   a. Read image → BGR uint8 HWC
   b. BGR → RGB
   c. Letterbox resize to 640x640:
      - Scale to fit, pad with (114,114,114)
      - Padding aligned to stride 32
   d. HWC → CHW
   e. uint8 → float32, divide by 255.0
   f. Add batch dim → [1, 3, 640, 640]

2. INFERENCE
   Run ONNX model → output tensor(s)

3. POSTPROCESS (Detection)
   a. Transpose output: [1, 84, 8400] → [1, 8400, 84]
   b. Extract boxes (cx, cy, w, h) and class scores
   c. Convert xywh → xyxy:
      x1 = cx - w/2, y1 = cy - h/2
      x2 = cx + w/2, y2 = cy + h/2
   d. Per box: max_score = max(class_scores), class_id = argmax(class_scores)
   e. Filter: keep where max_score > conf_thres (0.25)
   f. NMS:
      - If class-specific: offset boxes by class_id * 7680
      - Sort by score descending
      - Greedy: pick top, remove overlapping (IoU > 0.45), repeat
      - Cap at max_det (300)
   g. Scale boxes back to original image coordinates:
      - Subtract padding offset
      - Divide by letterbox scale ratio
      - Clip to image bounds

4. POSTPROCESS (Segmentation — additional steps)
   a. After NMS, extract mask coefficients for surviving detections
   b. Matrix multiply: masks = coefficients @ prototypes → [N, 160, 160]
   c. Apply sigmoid activation
   d. Crop each mask to its bounding box
   e. Upsample masks to original image resolution
   f. Threshold masks at 0.5 for binary masks
```

### Letterbox Resize Algorithm (Detailed)

```
function letterbox(image, target_size=640, stride=32, auto=True):
    h, w = image.shape[:2]
    r = min(target_size / h, target_size / w)    # scale ratio
    new_w = round(w * r)
    new_h = round(h * r)

    image = resize(image, (new_w, new_h))         # bilinear interpolation

    dw = target_size - new_w
    dh = target_size - new_h

    if auto:
        dw = dw % stride    # minimum padding (stride-aligned)
        dh = dh % stride

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    image = copyMakeBorder(image, top, bottom, left, right,
                           BORDER_CONSTANT, value=(114, 114, 114))
    return image, (r, (left, top))    # return scale and offset for later
```

### DFL Decoding (Inside Model — For Understanding)

```
function dfl_decode(raw_box_output):
    # raw_box_output shape: [batch, 4*reg_max, anchors] = [B, 64, 8400]
    # Reshape to [B, 4, reg_max, anchors] = [B, 4, 16, 8400]

    for each of 4 coordinates (left, top, right, bottom):
        distribution = softmax(logits[0:16])           # 16 bins
        value = sum(distribution[i] * i for i in 0..15) # weighted sum

    # Convert (left, top, right, bottom) distances to (cx, cy, w, h):
    # cx = anchor_x + (right - left) / 2
    # cy = anchor_y + (bottom - top) / 2
    # w = left + right
    # h = top + bottom
```

Note: In the exported ONNX model, DFL decoding is already performed inside the model. The ONNX output contains decoded (cx, cy, w, h) values directly.

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
- SharpInference should implement NMS in C# for flexibility (adjustable thresholds at runtime)

### PyTorch vs ONNX Output Differences

- PyTorch `Detect.forward()` returns different formats for training vs inference
- ONNX export uses the inference path: decoded boxes + sigmoid class scores
- The ONNX model has DFL softmax and dist2bbox baked in; no need to reimplement DFL in C#

## Open Questions

- [x] Exact C2f block structure — documented: split → n bottlenecks (each 2x Conv3x3) → concat all → 1x1 conv
- [x] Number of bottleneck layers per stage — documented with depth_multiple scaling per variant
- [x] Whether to support instance segmentation head — **Recommendation: Yes**. The seg variant adds minimal complexity (Proto module + mask coefficients + matrix multiply + sigmoid). It enables content moderation masking and precise subject extraction for auto-cropping.
- [x] Model size variants and priority — **Recommendation**: Prioritize YOLO11n (2.6M params, fastest) as default, support YOLO11s/m for users needing higher accuracy. YOLOv8 support is also straightforward since YOLO11 is structurally similar. The `l` and `x` variants offer diminishing returns on accuracy for significantly more compute.

### Remaining Questions for Implementation

- [ ] Should SharpInference support the raw PyTorch `.pt` format, or only ONNX? ONNX is simpler (no need for DFL decoding) but `.pt` is more common in the community.
- [ ] For segmentation mask assembly, should the C# implementation use GPU (CUDA) matrix multiply for the coefficients-prototypes product, or is CPU sufficient for 32x160x160?
- [ ] Should we support oriented bounding box (OBB) detection? YOLO11 supports it but it is a niche use case.
- [ ] Batched inference: should NMS support batch_size > 1? This requires per-image offset tricks or separate NMS per image.

## Implementation Notes

### Priority Order for SharpInference.Vision

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
