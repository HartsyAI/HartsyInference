# Phase 7 — Server (OpenAI-Compatible API)

> **Goal:** OpenAI-compatible REST API serving image generation and audio endpoints.
> **Packages:** SharpInference.Server

---

## 1. Research

- [x] Complete [OPENAI_IMAGE_API.md](../Research/OPENAI_IMAGE_API.md) research — done and verified
- [ ] Review OpenAI audio API docs (transcriptions, speech)
- [ ] Review existing OpenAI-compatible servers (LocalAI, vLLM) for patterns

## 2. Planning

- [ ] Define full request/response schemas for all endpoints
- [ ] Plan SSE streaming format for image generation progress
- [ ] Plan chunked audio streaming format for TTS
- [ ] Plan request queue design (max depth, timeout, rejection policy)
- [ ] Plan model management lifecycle (load, unload, hot-swap)
- [ ] Plan authentication middleware (API key validation)
- [ ] Write agent instructions for Phase 7

## 3. Implementation — Setup

- [ ] `SharpInferenceServiceExtensions.cs` — `AddSharpInference()` DI registration
- [ ] `SharpInferenceServiceExtensions.cs` — `MapSharpInferenceEndpoints()` route mapping
- [ ] `SharpInferenceServerOptions.cs` — configuration (models dir, default model, auth, queue depth)

## 4. Implementation — Image Endpoints

- [ ] `POST /v1/images/generations` — text-to-image with JSON body
- [ ] `POST /v1/images/edits` — img2img and inpainting with multipart form
- [ ] Response format — base64 JSON or URL depending on `response_format`
- [ ] Size parameter parsing — "1024x1024", "512x512", etc.
- [ ] Seed, steps, cfg_scale, sampler as optional parameters
- [ ] Batch generation — `n` parameter for multiple images

## 5. Implementation — Audio Endpoints

- [ ] `POST /v1/audio/transcriptions` — Whisper STT with multipart audio upload
- [ ] `POST /v1/audio/speech` — Kokoro TTS with JSON body, audio stream response
- [ ] Audio format handling — accept wav, mp3, m4a, webm input
- [ ] Audio output formats — wav, mp3, opus, flac

## 6. Implementation — Streaming

- [ ] `SseProgressStream.cs` — SSE events: `{"step": N, "total": M, "preview": "<base64>"}`
- [ ] SSE final event — `{"status": "complete", "image": "<base64>"}`
- [ ] `AudioChunkStream.cs` — chunked transfer encoding for TTS audio

## 7. Implementation — Model Management

- [ ] `GET /v1/models` — list loaded models with metadata
- [ ] `POST /v1/models/load` — trigger model load into VRAM
- [ ] `DELETE /v1/models/{id}` — unload model from VRAM
- [ ] `POST /v1/models/pull` — trigger HuggingFace download

## 8. Implementation — Infrastructure

- [ ] `InferenceQueue.cs` — FIFO request queue with configurable concurrency
- [ ] `InferenceQueueEntry.cs` — queue entry with cancellation token
- [ ] `ApiKeyMiddleware.cs` — optional API key header validation
- [ ] `GET /health` — health check endpoint
- [ ] `GET /ready` — readiness probe (model loaded and ready)
- [ ] OpenAI error response format — `{"error": {"message": ..., "type": ..., "code": ...}}`
- [ ] Request validation — return 400 for malformed requests
- [ ] Rate limiting — per-client request throttling

## 9. Testing

- [ ] `ImageApiTests.cs` — text-to-image round-trip (mock backend)
- [ ] `ImageApiTests.cs` — img2img with multipart form
- [ ] `ImageApiTests.cs` — inpainting with mask
- [ ] `ImageApiTests.cs` — SSE streaming events format
- [ ] `AudioApiTests.cs` — transcription round-trip (mock backend)
- [ ] `AudioApiTests.cs` — TTS streaming audio response
- [ ] `ModelApiTests.cs` — list, load, unload lifecycle
- [ ] `QueueTests.cs` — concurrent requests queued correctly
- [ ] `QueueTests.cs` — queue overflow returns 429
- [ ] `AuthTests.cs` — valid key passes, invalid key returns 401
- [ ] `HealthTests.cs` — health and readiness probes
- [ ] Integration test — real SD1.5 generation via HTTP (GPU CI)
- [ ] OpenAI Python SDK compatibility test — use `openai` library to call our server
- [ ] All tests pass on CI

## 10. Review & Merge

- [ ] Code review — no security vulnerabilities (injection, path traversal)
- [ ] Code review — proper request cancellation propagation
- [ ] Code review — error handling (no unhandled exceptions, clean error responses)
- [ ] Load test — concurrent image generation requests
- [ ] Merge to main branch
