using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native embedding route. One-shot, no step-preview progress (same shape as <see cref="AudioEndpoints"/>), so no <c>/stream</c> variant.</summary>
public static class EmbeddingEndpoints
{
    /// <summary>Maps <c>/v1/native/embeddings</c>.</summary>
    public static void MapEmbeddingEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/embeddings", async (NativeEmbeddingRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Embedding);
            try
            {
                EmbeddingResult result = await queue.EnqueueAsync(() => engine.Embeddings.GenerateAsync(spec, req.Request, ct), ct);
                return Results.Ok(result);
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }
}
