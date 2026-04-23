# API Agent

> Build the SharpInference.Server REST API — OpenAI-compatible endpoints, SSE streaming, model management, request queue, auth, and health probes.

## Extra Reading
- `docs/Design/IMPLEMENTATION_DETAILS.md` (server section), `docs/Design/FILE_STRUCTURE.md`
- `docs/Research/OPENAI_IMAGE_API.md` — exact schemas
- `docs/Research/DOTLLM_ARCHITECTURE.md` — dotLLM patterns (ServerState, source-gen JSON, SSE)
- `docs/Checklists/PHASE_7_SERVER.md`
- Existing pipeline code in `src/SharpInference.Diffusion/` and `src/SharpInference.Audio/`

## Workflow
1. Read OpenAI spec → design Minimal API endpoints (one file per endpoint)
2. Implement: HTTP request → pipeline call → HTTP response
3. Add SSE streaming for image progress; chunked transfer for audio
4. Add queue, auth, health probes
5. Test with OpenAI SDK client
6. Update checklist

## Design Principles

**OpenAI Compatibility:** Match request/response schemas exactly; return errors in OpenAI format; support `b64_json` and `url` response formats.

**Minimal API Pattern:**
```csharp
public static class SharpInferenceServiceExtensions
{
    public static IServiceCollection AddSharpInference(this IServiceCollection services, ServerState state)
    {
        services.AddSingleton(state);
        services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, SharpInferenceJsonContext.Default));
        return services;
    }
}
```

**ServerState:** Created before DI container, registered as singleton. Holds models, backend, queue, config.

**Streaming:** SSE for image progress (`text/event-stream`); chunked `audio/wav` or `audio/mp3` for TTS.

**Request Queue:** FIFO, configurable depth; 429 when full; `CancellationToken` support.

**Auth:** Optional, disabled by default. API key via `Authorization: Bearer sk-...`.

## Security Checklist
- [ ] Validate input sizes, file uploads, model paths
- [ ] Rate limiting, CORS, body size limits
- [ ] No secrets in errors/logs
