# Phase 6 — Vision (CLIP + YOLO)

> **Goal:** CLIP embeddings and object detection working end-to-end.
> **Packages:** SharpInference.Vision

---

## 1. Research

- [x] Complete [CLIP_ARCHITECTURE.md](../Research/CLIP_ARCHITECTURE.md) research (image encoder) — done and verified
- [x] Complete [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) research — done and verified

## 2. Planning

- [ ] Plan CLIP image preprocessing (resize, center crop, normalize with ImageNet stats)
- [ ] Plan YOLO input preprocessing (letterbox resize, normalize to 0–1)
- [ ] Plan NMS algorithm implementation (IoU threshold, confidence threshold)
- [ ] Plan embedding API surface (single image, batch, similarity scoring)
- [ ] Write agent instructions for Phase 6

## 3. Implementation — CLIP

- [ ] `ClipImageEncoder.cs` — ViT: patch embedding → positional encoding → transformer → projection
- [ ] `ClipScorer.cs` — cosine similarity between text and image embeddings
- [ ] Image preprocessing — resize, center crop, normalize (ImageNet mean/std)
- [ ] `ImageEmbeddingPipeline.cs` — image path/bytes → embedding vector
- [ ] `TextEmbeddingPipeline.cs` — text → embedding vector (Nomic-Embed / E5 / BGE)

## 4. Implementation — YOLO

- [ ] `YoloPipeline.cs` — image → preprocessing → model forward → post-processing → detections
- [ ] YOLO backbone — C2f blocks, SPPF module, convolutional layers
- [ ] YOLO neck — FPN multi-scale feature fusion
- [ ] YOLO detection head — box regression + class prediction
- [ ] `YoloPostProcessor.cs` — decode boxes, apply NMS, filter by confidence
- [ ] `DetectionResult.cs` — bounding box, class label, confidence score

## 5. Implementation — Segmentation (stubs)

- [ ] `SamPipeline.cs` — stub for SAM 2 (Phase 2 model)
- [ ] `SamMaskDecoder.cs` — stub
- [ ] `FaceDetector.cs` — stub for face detection
- [ ] `LandmarkExtractor.cs` — stub for facial landmarks

## 6. Testing & Validation

- [ ] CLIP image encoder — same image → same embedding as OpenAI CLIP (cosine sim > 0.999)
- [ ] CLIP scorer — text-image similarity scores match reference
- [ ] CLIP batch — batch of 10 images produces same embeddings as individual
- [ ] Text embeddings — same text → same embedding as reference model
- [ ] YOLO detection — same image → same bounding boxes as Ultralytics (IoU > 0.95)
- [ ] YOLO NMS — verify correct suppression of overlapping detections
- [ ] YOLO multi-class — verify correct class assignment on COCO test images
- [ ] Performance test — CLIP embedding latency for single image
- [ ] Performance test — YOLO detection latency for single image
- [ ] All tests pass on CI

## 7. Review & Merge

- [ ] Code review — image preprocessing correctness (normalization values)
- [ ] Code review — NMS edge cases (overlapping boxes, single detection)
- [ ] Benchmark: CLIP embeddings/sec, YOLO detections/sec
- [ ] Merge to main branch
