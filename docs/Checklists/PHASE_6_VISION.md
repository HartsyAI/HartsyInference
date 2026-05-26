# Phase 6 — Vision (CLIP + YOLO)

> **Goal:** CLIP embeddings and object detection working end-to-end.
> **Packages:** SharpInference.Vision

---

## 1. Research — COMPLETE

- [x] CLIP architecture — covered in [TEXT_ENCODERS.md § CLIP](../Research/TEXT_ENCODERS.md) (there is no separate CLIP_ARCHITECTURE.md; the doc covers both the text and vision towers including ViT-L/14, ViT-H/14, ViT-bigG/14, OpenAI vs OpenCLIP differences, and the diffusion usage path)
- [x] [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) — YOLOv8 + YOLO11, C2f/C3k2/SPPF/C2PSA, FPN+PAN neck, decoupled detect head, DFL, letterbox preprocessing, NMS

## 2. Planning

- [x] CLIP image preprocessing (resize, center crop, CLIP normalize — *not* ImageNet)
- [ ] YOLO preprocessing (letterbox resize, 0-1 normalize), NMS algorithm
- [x] Embedding API surface (single, batch, similarity scoring)
- [x] **Reuse decision**: `ClipTextEncoder` + `ClipVisionEncoder` already exist in `SharpInference.Diffusion/Models/TextEncoders/` (originally for IP-Adapter / SDXL conditioning). The standalone Vision pipeline wraps these rather than duplicating the math. Vision references Diffusion for now; long-term these encoders should be hoisted to a shared sub-package so pure-Vision installs don't pull the full diffusion stack.
- [x] **YOLO format decision**: Pure-C# native — read weights from converted safetensors (Ultralytics `.pt` → safetensors via a converter), execute through `IBackend`. **No `Microsoft.ML.OnnxRuntime` dependency.** This overrides the suggestion in `YOLO_ARCHITECTURE.md § Implementation Notes` which assumed ONNX Runtime; that section was written before the engine's pure-C# rule was codified.

## 3. CLIP — Standalone Embedding Pipeline

- [x] `Clip/ClipImagePreprocessor.cs` — bicubic resize (shortest edge → 224 by default) + center crop + CLIP normalize (mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
- [x] `Clip/ClipImageEncoder.cs` — wraps `ClipVisionEncoder` from Diffusion, exposes `Encode(IBackend, Tensor pixelValues) → embedding`
- [x] `Clip/ClipTextEmbedder.cs` — wraps `ClipTextEncoder` + `ClipTokenizer`, exposes `Encode(string) → embedding`
- [x] `Clip/ClipScorer.cs` — L2-normalize + cosine similarity (single pair, text↔image matrix, top-K)
- [x] `Embeddings/ImageEmbedding.cs` / `TextEmbedding.cs` — strongly-typed value records carrying the embedding tensor + source metadata
- [x] `Embeddings/ImageEmbeddingPipeline.cs` — public façade; constructs preprocessor + encoder; implements `IEmbeddingPipeline`
- [x] `Embeddings/TextEmbeddingPipeline.cs` — public façade for text-only embedding
- [x] `Clip/ClipModelLoader.cs` — loads safetensors + builds preset config (CLIP-L/14, CLIP-H/14, CLIP-bigG/14) from a single directory path

## 4. YOLO — Native Object Detection (✅ complete for YOLOv8)

**Algorithm-side pieces (model-independent):**
- [x] `Detection/YoloPreprocessor.cs` — letterbox resize with stride-32 alignment + `/255` normalize + HWC→CHW. Bilinear (matches Ultralytics `cv2.INTER_LINEAR`). Returns a `Transform` for box un-projection after NMS.
- [x] `Detection/NonMaxSuppression.cs` — greedy NMS, class-specific by default with optional class-agnostic mode. `RunWithConfidenceFilter` mirrors Ultralytics defaults (conf=0.25, IoU=0.45, max_det=300).
- [x] `Detection/YoloDetection.cs` — xyxy + confidence + class id; built-in `Iou()`.
- [x] `Detection/CocoLabels.cs` — 80-class COCO label table; `Get(classId)` falls back to `class_N` for non-COCO heads.

**Model-side pieces:**
- [x] `Detection/Blocks/ConvBnSilu.cs` — basic block. BatchNorm folded into Conv weights at conversion time, so runtime is just Conv2D + SiLU — no new kernel needed.
- [x] `Detection/Blocks/Bottleneck.cs` — two 3×3 Conv-BN-SiLU stages with optional residual.
- [x] `Detection/Blocks/C2f.cs` — YOLOv8 cross-stage-partial block. Splits into halves, chains bottlenecks, concats all intermediate outputs.
- [x] `Detection/Blocks/Sppf.cs` — spatial pyramid pooling (fast). Uses the new `IBackend.MaxPool2D` (added to Core with CPU-loop default).
- [x] `Detection/Blocks/DetectHead.cs` — decoupled box (cv2) + class (cv3) branches per scale + DFL decoding + anchor synthesis + sigmoid. Outputs `[B, 4 + numClasses, totalAnchors]` directly in input-pixel xywh.
- [x] `Detection/YoloModel.cs` — full assembly: 10-layer CSPDarknet backbone + 12-layer FPN+PAN neck + detect head. Channel counts derived from `YoloConfig` width/depth multipliers; weight keys match Ultralytics `model.{layer}.*` so the converter is a near-identity remap.
- [x] `Detection/YoloConfig.cs` — `YoloV8n` / `s` / `m` / `l` / `x` presets with the correct `make_divisible(min(c_base, max_channels) * width, 8)` formula.
- [x] `Detection/YoloPostProcessor.cs` — confidence filter → xywh→xyxy → NMS → letterbox-invert.
- [x] `Detection/YoloPipeline.cs` — `IDetectionPipeline` implementation. PNG bytes route through `PngDecoder`; raw RGB bytes go through `Detect(...)`.
- [x] `tests/python-reference/convert_yolov8_pt_to_safetensors.py` — Ultralytics `.pt` → safetensors with BN folded into Conv weights. Decided against a pure-C# pickle parser; Python conversion is the path of least friction and runs once per checkpoint.

**Validation results (CPU end-to-end, against `Models/yolo/yolov8n-folded.safetensors`):**
- Output tensor shape `[1, 84, 8400]` (4 box + 80 classes × 8400 anchors at 640×640) — matches Ultralytics exactly.
- Real-image test (`tests/SharpInference.Vision.Tests/TestData/bus.png`, 810×1080) detects:
  - **4 persons** (confidence 0.44, 0.88, 0.88, 0.89) at the correct locations
  - **1 bus** (confidence 0.84) spanning the image
  - This matches Ultralytics' canonical demo output on this image one-for-one.
- ~6s/image at 640×640 on CPU; CUDA / Vulkan acceleration is a future perf slice.

**YOLO11 support — ✅ complete and validated:**
- [x] `Detection/Blocks/C3k.cs` — CSP-with-3-convs block (cv1 + cv2 parallel + cv3 final project; inner Bottlenecks applied sequentially on cv1's branch). Used as the inner unit of <see cref="C3k2"/> when c3k=True.
- [x] `Detection/Blocks/C3k2.cs` — outer C2f-style block with switchable inner units (Bottleneck for c3k=False, C3k for c3k=True). Critically, when c3k=False the inner Bottlenecks use `expansion=0.5` (compress by half) — Ultralytics' default, distinct from C2f which explicitly passes e=1.0.
- [x] `Detection/Blocks/PsaAttention.cs` — multi-head self-attention with depthwise-conv positional encoding. Custom shapes (Q/K have `key_dim` channels per head while V has `head_dim`) so we hand-code the matmuls; `IBackend.ScaledDotProductAttention`'s standard layout doesn't apply.
- [x] `Detection/Blocks/PsaBlock.cs` — attention + FFN with residual connections.
- [x] `Detection/Blocks/C2psa.cs` — cross-stage with PSA: cv1 expand → split → PsaBlocks on partB → concat → cv2 project.
- [x] `Detection/Blocks/DwConvBnSilu.cs` — depthwise variant of `ConvBnSilu` (used by v11's class branch + C2psa's positional encoding).
- [x] `Detection/Blocks/DetectHeadV11.cs` — same box branch as v8 but cv3 (class) uses depthwise-separable: `[DwConv 3×3 + Conv 1×1] → [DwConv 3×3 + Conv 1×1] → Conv 1×1`. Final 1×1 is plain Conv2d (no BN/SiLU).
- [x] `Detection/YoloV11Model.cs` — 11-layer backbone (Conv → C3k2 × 4 → SPPF → C2PSA) + 12-layer FPN+PAN neck + DetectHeadV11. Implements `IYoloDetectModel`.
- [x] `Detection/IYoloDetectModel.cs` — interface so `YoloPipeline` can hold either v8 or v11.
- [x] YOLO11 presets: `YoloConfig.YoloV11n / s / m / l / x`.
- [x] `IBackend.Conv2dDepthwise` (default CPU fallback, override-ready).
- [x] `YoloPipeline.LoadV11(...)` static factory.
- [x] Python converter handles both v8 and v11 — picks up plain (non-BN) Conv2d weights anywhere via a generic suffix-based fallback (not hard-coded to the v8 detect-head layer index).
- [x] **End-to-end validation against `yolo11n.pt` (5.5 MB) → `yolo11n-folded.safetensors` (10 MB)**: on Ultralytics' bus.png test image, my YOLO11n produces:
  - bus (conf 0.940), person (0.902), person (0.849), person (0.833), person (0.396)
  - Exactly matches Ultralytics' canonical YOLO11n output

**Out of scope (deferred):**
- YOLO segmentation head (YOLOv8-seg / YOLO11-seg) — adds a Proto module + 32-mask coefficient branch.
- GPU-native MaxPool2D / Conv2dDepthwise — currently use the IBackend defaults (CPU loops). Worth dedicated Vulkan/CUDA kernels for inference perf.

## 5. Segmentation / Face (stubs ✅)

- [x] `Segmentation/SamPipeline.cs` — type + API shape reserved. Throws `NotImplementedException` until SAM 2 (Hiera image encoder + two-way mask decoder) is built.
- [x] `Segmentation/SamMaskDecoder.cs` — placeholder for the two-way transformer + multi-mask head.
- [x] `FaceDetection/FaceDetector.cs` — placeholder for YOLOv8-Face / RetinaFace / InsightFace (intentionally backbone-agnostic).
- [x] `FaceDetection/LandmarkExtractor.cs` — placeholder; doc notes the 5-point (detector side-output) vs 68/468-point (dense, separate model) tradeoff.

## 6. Testing

- [x] CLIP embedding shape sanity (`ImageEmbeddingPipelineSurfaceTests`, `ClipPresetSurfaceTests`)
- [x] CLIP checkpoint loads cleanly (`ClipCheckpointTests.ClipL_Loads_AndProducesNormalizedEmbeddings`) — `openai/clip-vit-large-patch14` (1.71 GB) downloaded to `Models/clip/`
- [x] CLIP embedding L2 norm == 1.0 within float precision (image: 1.00000002, text: 1.00000000)
- [x] CLIP self-similarity == 1.0 within tolerance (both modalities)
- [x] CLIP batch consistency (single-pass vs batch-pass cosine sim 1.000000 on the cat/fox pair)
- [x] CLIP text-to-text semantic ranking — *"a photograph of a cat"* ↔ *"an image of a kitten"* = **0.8999** vs *"a diagram of a rocket ship"* = **0.6596** (matches published OpenAI CLIP-L numbers; gap 0.24)
- [ ] CLIP bit-exact Python reference comparison (cosine sim > 0.999 against HF `CLIPModel`) — needs a Python dump harness; tracked as a future diagnostic following the SD3.5 layer-by-layer pattern
- [ ] YOLO detection IoU > 0.95 vs Ultralytics
- [ ] YOLO NMS edge cases (zero detections, all-overlap, max_det clamp)
- [ ] Multi-class COCO sanity check
- [ ] Performance: CLIP embeddings/sec, YOLO detections/sec
- [ ] All tests pass on CI

## 7. Review & Merge

- [ ] Code review (preprocessing values, NMS edge cases, encoder lifetime / disposal)
- [ ] Merge to main branch

## 9. Image Codec (✅ shipped)

- [x] `Codec/PngDecoder.cs` — pure-C# PNG decoder. Supports 8-bit RGB (color type 2) and RGBA (color type 6, alpha dropped). DEFLATE via `System.IO.Compression.DeflateStream`. All 5 PNG filter algorithms (None / Sub / Up / Average / Paeth) implemented and validated against a real test fixture (Ultralytics `bus.png`, 810×1080 RGB).
- [x] `YoloPipeline.DetectAsync(ReadOnlyMemory<byte> imageBytes, ...)` — routes PNG bytes through the decoder so the `IDetectionPipeline` contract works end-to-end without a third-party image library.
- [ ] JPEG / WebP decoders (deferred — much larger lift; the engine isn't blocked since callers can pre-decode).

## 8. Stretch Goals (deferred to Phase 6.5+)

The roadmap lists these in Phase 2 but they fit naturally as Phase 6 extensions once CLIP + YOLO ship. None of them require new infra beyond what Phase 6 lands:

- **SigLIP / SigLIP 2** — same `ClipVisionEncoder` shape but with **sigmoid loss** (no temperature softmax), **no CLS token** (mean-pool the patch tokens). Smaller config + tokenizer change.
- **DINOv2 / DINOv3** — pure ViT (no contrastive pair), exposes dense patch features. Reuses our ViT block.
- **RT-DETR** — first transformer-based detector; no NMS needed (Hungarian matcher). Pairs well with the CLIP wrapper as an alternative to YOLO.
- **Grounding DINO** — open-vocab detection. Needs the BERT text tower; we already have CLIP text but BERT is a different architecture.
- **SAM 2 / MobileSAM** — needs new attention variant (windowed) + mask decoder. Stub already in §5.
- **EVA-CLIP / MetaCLIP / AM-RADIO** — drop-in CLIP replacements; same `IEmbeddingPipeline` surface, different preset configs.
