# Phase 9 — Video (LTX-Video + Wan)

> **Goal:** Video generation starting with LTX-Video.
> **Packages:** SharpInference.Video

---

## 1. Research

- [ ] LTX-Video architecture (temporal attention, 3D VAE, conditioning)
- [ ] Wan 2.1 architecture
- [ ] Temporal attention, video VAE decoder (3D convolutions), video output encoding

## 2. Planning

- [ ] LTX-Video model structure/weights, temporal attention integration
- [ ] VRAM management for video, frame streaming output, video encoding (FFmpeg or pure C# mp4)

## 3. Implementation

- [ ] `TemporalAttention.cs`, `VideoVaeDecoder.cs`
- [ ] Video-specific PTX kernels (3D conv, temporal attention)
- [ ] `LtxVideoPipeline.cs`, `WanPipeline.cs`
- [ ] Frame output (PNGs or video), video encoding, progress streaming

## 4. Server Integration

- [ ] Video generation endpoint, SSE streaming, video file serving

## 5. Testing

- [ ] Temporal attention consistency, video VAE vs reference
- [ ] Pipeline quality (manual check), VRAM usage, server endpoint
- [ ] All tests pass on GPU CI

## 6. Review & Merge

- [ ] Code review (temporal correctness, frame buffer memory)
- [ ] Benchmark frames/sec
- [ ] Merge to main branch
