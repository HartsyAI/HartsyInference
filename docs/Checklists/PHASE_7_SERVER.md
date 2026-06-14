# Phase 7 — Server (OpenAI-Compatible API)

> **Goal:** OpenAI-compatible REST API serving image generation and audio endpoints.
> **Packages:** HartsyInference.Server

---

## 1. Research

- [x] OPENAI_IMAGE_API
- [ ] OpenAI audio API (transcriptions, speech)
- [ ] Existing OpenAI-compatible servers (LocalAI, vLLM) for patterns

## 2. Planning

- [ ] Request/response schemas, SSE streaming format, chunked audio format
- [ ] Request queue design, model management lifecycle, auth middleware

## 3. Setup

- [ ] `HartsyInferenceServiceExtensions.cs` — `AddHartsyInference()` + `MapHartsyInferenceEndpoints()`
- [ ] `HartsyInferenceServerOptions.cs`

## 4. Image Endpoints

- [ ] `POST /v1/images/generations` — text-to-image (JSON body)
- [ ] `POST /v1/images/edits` — img2img + inpainting (multipart form)
- [ ] Response formats (b64_json, url), size parsing, seed/steps/cfg/sampler, batch `n`

## 5. Audio Endpoints

- [ ] `POST /v1/audio/transcriptions` — Whisper STT (multipart upload)
- [ ] `POST /v1/audio/speech` — Kokoro TTS (JSON body, audio stream response)
- [ ] Accept wav/mp3/m4a/webm input; output wav/mp3/opus/flac

## 6. Streaming

- [ ] `SseProgressStream.cs` — step progress + preview + complete events
- [ ] `AudioChunkStream.cs` — chunked transfer for TTS

## 7. Model Management

- [ ] `GET /v1/models`, `POST /v1/models/load`, `DELETE /v1/models/{id}`, `POST /v1/models/pull`

## 8. Infrastructure

- [ ] `InferenceQueue.cs` (FIFO, configurable depth, 429 on full)
- [ ] `ApiKeyMiddleware.cs` (optional), health/readiness probes
- [ ] OpenAI error format, request validation, rate limiting

## 9. Testing

- [ ] Image API round-trips (t2i, i2i, inpaint, SSE streaming)
- [ ] Audio API round-trips (transcription, TTS streaming)
- [ ] Model lifecycle, queue overflow (429), auth (401), health probes
- [ ] Real SD1.5 via HTTP (GPU CI), OpenAI Python SDK compatibility
- [ ] All tests pass on CI

## 10. Review & Merge

- [ ] Code review (security, cancellation propagation, error handling)
- [ ] Load test concurrent requests
- [ ] Merge to main branch
