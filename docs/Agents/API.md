# API Agent

> **Role:** Build the SharpInference.Server REST API — OpenAI-compatible endpoints, SSE streaming, model management, request queue, authentication, and health probes.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — overall architecture
- `docs/Design/IMPLEMENTATION_DETAILS.md` — server section
- `docs/Design/FILE_STRUCTURE.md` — server package structure
- `docs/Research/OPENAI_IMAGE_API.md` — exact OpenAI request/response schemas
- `docs/Checklists/PHASE_7_SERVER.md` — what needs to be built
- Existing pipeline code in `src/SharpInference.Diffusion/` and `src/SharpInference.Audio/` — understand what you're wrapping

## Your Workflow

1. **Read the OpenAI API spec** — understand exact request/response formats
2. **Design the endpoint** — ASP.NET minimal API pattern
3. **Implement the endpoint** — translate HTTP request → pipeline call → HTTP response
4. **Add streaming** — SSE for image progress, chunked transfer for audio
5. **Add infrastructure** — queue, auth, health probes
6. **Test with OpenAI SDK** — verify an OpenAI client library can call our endpoints
7. **Update checklist** — mark server items complete

## API Design Principles

### OpenAI Compatibility
- Match OpenAI's request/response schemas exactly — any OpenAI client library should work
- Support the same query parameters, headers, and content types
- Return errors in OpenAI's error format: `{"error": {"message": "...", "type": "...", "code": "..."}}`
- Support both `response_format: "b64_json"` and `response_format: "url"`

### ASP.NET Minimal API Pattern
```csharp
// Registration in SharpInferenceServiceExtensions.cs
public static IServiceCollection AddSharpInference(this IServiceCollection services, Action<SharpInferenceServerOptions>? configure = null)
{
    // Register pipelines, queue, model registry
}

public static WebApplication MapSharpInferenceEndpoints(this WebApplication app)
{
    app.MapPost("/v1/images/generations", ImageGenerationEndpoints.Generate);
    app.MapPost("/v1/images/edits", ImageGenerationEndpoints.Edit);
    app.MapPost("/v1/audio/transcriptions", AudioTranscriptionEndpoints.Transcribe);
    app.MapPost("/v1/audio/speech", AudioTranscriptionEndpoints.Speak);
    app.MapGet("/v1/models", ModelManagementEndpoints.List);
    // ...
}
```

### Streaming
- Image generation progress: SSE (`text/event-stream`)
  - Each event: `data: {"step": 5, "total": 20, "preview": "<base64>"}\n\n`
  - Final event: `data: {"status": "complete", "image": "<base64>"}\n\n`
- TTS audio: chunked transfer encoding (`audio/wav` or `audio/mp3`)
  - Stream PCM chunks as they're synthesized

### Request Queue
- FIFO queue with configurable max depth
- One GPU inference at a time (configurable concurrency)
- Return `429 Too Many Requests` when queue is full
- Support request cancellation via `CancellationToken`
- Track queue position and estimated wait time

### Authentication
- Optional — disabled by default for local use
- API key via `Authorization: Bearer sk-...` header
- Configurable via `SharpInferenceServerOptions.ApiKey`

## Security Checklist

- [ ] Validate all input sizes (image dimensions, audio length, prompt length)
- [ ] Validate file uploads (check content type, enforce size limits)
- [ ] No path traversal in model file paths
- [ ] Rate limiting per client IP
- [ ] Request body size limits
- [ ] Proper CORS configuration
- [ ] No secrets in error messages or logs

## Related Docs
- `docs/Research/OPENAI_IMAGE_API.md` — exact API schemas
- `docs/Checklists/PHASE_7_SERVER.md` — implementation checklist
- `docs/Design/FILE_STRUCTURE.md` — server file layout
- `docs/Agents/TESTER.md` — testing the API endpoints
