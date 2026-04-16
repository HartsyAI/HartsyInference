# Phase 9 — Video (LTX-Video + Wan)

> **Goal:** Video generation starting with LTX-Video.
> **Packages:** SharpInference.Video

---

## 1. Research

- [ ] Research LTX-Video architecture (temporal attention, 3D VAE, conditioning)
- [ ] Research Wan 2.1 architecture
- [ ] Research temporal attention mechanisms (cross-frame consistency)
- [ ] Research video VAE decoder (3D convolutions, temporal dimension)
- [ ] Research video output encoding (frame sequence → mp4)

## 2. Planning

- [ ] Map LTX-Video model structure and weight layout
- [ ] Plan temporal attention integration with existing attention kernels
- [ ] Plan VRAM management for video (much higher than single image)
- [ ] Plan frame-by-frame streaming output
- [ ] Plan video encoding pipeline (raw frames → mp4/webm)
- [ ] Write agent instructions for Phase 9

## 3. Implementation

- [ ] `TemporalAttention.cs` — cross-frame attention for video consistency
- [ ] `VideoVaeDecoder.cs` — 3D VAE decoder for video latents
- [ ] Video-specific PTX kernels (3D convolution, temporal attention)
- [ ] `LtxVideoPipeline.cs` — text/image → video generation pipeline
- [ ] `WanPipeline.cs` — Wan video generation pipeline
- [ ] Frame sequence output — save as individual PNGs or combined video
- [ ] Video encoding integration — FFmpeg or pure C# mp4 muxing
- [ ] Progress streaming — frame-by-frame progress events

## 4. Implementation — Server Integration

- [ ] Video generation endpoint in SharpInference.Server
- [ ] SSE streaming for video generation progress
- [ ] Video file serving (generated mp4 available via URL)

## 5. Testing

- [ ] Temporal attention — verify cross-frame consistency
- [ ] Video VAE — decode video latents, compare to reference
- [ ] LTX-Video pipeline — generate short video, manual quality check
- [ ] Wan pipeline — generate short video, manual quality check
- [ ] Memory test — verify VRAM usage during video generation
- [ ] Server test — video generation via HTTP endpoint
- [ ] All tests pass on GPU CI

## 6. Review & Merge

- [ ] Code review — temporal attention correctness
- [ ] Code review — video memory management (frame buffers properly freed)
- [ ] Benchmark: frames/sec for LTX-Video, Wan
- [ ] Merge to main branch
