using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native text-generation routes: a byte-for-byte pass-through of <see cref="TextRequest"/> to <see cref="TextResult"/>, plus real incremental-text streaming via <see cref="TextChunk"/>.</summary>
public static class TextEndpoints
{
    /// <summary>Maps <c>/v1/native/text</c>, its SSE streaming variant, and <c>/count-tokens</c>. POST rather than the plan's originally-sketched GET for count-tokens — GET requests with a JSON body are poorly supported by HTTP tooling and ASP.NET Core minimal APIs don't bind one by default.</summary>
    public static void MapTextEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/text", async (NativeTextRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Text);
            try
            {
                TextResult result = await GenerateAsync(engine, queue, spec, req.Request, ct);
                return Results.Ok(result);
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/text/stream", async (NativeTextRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Text);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs inside the queue's held slot — call the service directly, not GenerateAsync below.
                await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, req.Request, ct))
                    writer.TryWrite(SseHelpers.Event("chunk", chunk, jsonOptions));
            }, ct);
        });

        app.MapPost("/v1/native/text/count-tokens", (CountTokensRequest req, IInferenceEngine engine) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Text);
            try
            {
                // Tokenization only — no GPU generation, so it doesn't go through the InferenceQueue gate.
                return Results.Ok(new { tokens = engine.Text.CountTokens(spec, req.Text) });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }

    /// <summary>Queue-gated generation, shared with the OpenAI-compat <c>/v1/chat/completions</c> wrapper.</summary>
    internal static Task<TextResult> GenerateAsync(
        IInferenceEngine engine, InferenceQueue queue, ModelSpec spec, TextRequest request, CancellationToken ct) =>
        queue.EnqueueAsync(() => engine.Text.GenerateAsync(spec, request, ct), ct);
}
