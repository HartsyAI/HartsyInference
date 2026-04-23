# Phase 6 — Vision (CLIP + YOLO)

> **Goal:** CLIP embeddings and object detection working end-to-end.
> **Packages:** SharpInference.Vision

---

## 1. Research — COMPLETE

- [x] CLIP_ARCHITECTURE (image encoder), YOLO_ARCHITECTURE

## 2. Planning

- [ ] CLIP image preprocessing (resize, center crop, ImageNet normalize)
- [ ] YOLO preprocessing (letterbox resize, 0-1 normalize), NMS algorithm
- [ ] Embedding API surface (single, batch, similarity scoring)

## 3. CLIP

- [ ] `ClipImageEncoder.cs` — ViT: patch embed → positional → transformer → projection
- [ ] `ClipScorer.cs` — cosine similarity text↔image
- [ ] Image preprocessing, `ImageEmbeddingPipeline.cs`, `TextEmbeddingPipeline.cs`

## 4. YOLO

- [ ] `YoloPipeline.cs` — image → preprocess → model → post-process → detections
- [ ] Backbone (C2f, SPPF), neck (FPN), detection head (box regression + class)
- [ ] `YoloPostProcessor.cs` — decode boxes, NMS, confidence filter
- [ ] `DetectionResult.cs`

## 5. Segmentation (stubs)

- [ ] `SamPipeline.cs`, `SamMaskDecoder.cs`, `FaceDetector.cs`, `LandmarkExtractor.cs`

## 6. Testing

- [ ] CLIP embedding cosine sim > 0.999 vs OpenAI, batch consistency, text embeddings
- [ ] YOLO detection IoU > 0.95 vs Ultralytics, NMS, multi-class COCO
- [ ] Performance: CLIP embeddings/sec, YOLO detections/sec
- [ ] All tests pass on CI

## 7. Review & Merge

- [ ] Code review (preprocessing values, NMS edge cases)
- [ ] Merge to main branch
