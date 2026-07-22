using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Model catalog, disk cache, resident-model, and backend/queue admin endpoints.</summary>
public static class AdminEndpoints
{
    /// <summary>Maps every <c>/admin/*</c> route.</summary>
    public static void MapAdminEndpoints(this WebApplication app)
    {
        app.MapGet("/admin/catalog", (string? modality) =>
        {
            if (string.IsNullOrWhiteSpace(modality))
                return Results.Ok(ModelCatalog.All);
            if (!Modalities.TryParse(modality, out Modality parsed))
            {
                return HartsyInferenceServiceExtensions.Problem(
                    StatusCodes.Status400BadRequest, $"Unknown modality '{modality}'.", "invalid_request_error");
            }
            return Results.Ok(ModelCatalog.ForModality(parsed));
        });

        app.MapGet("/admin/models", (IInferenceEngine engine) =>
            Results.Ok(new { loaded = engine.LoadedPipelineKeys }));

        // Standard-download families only (image/video/3D/most LLM catalog entries). Audio-cache-backed models
        // (TTS/STT/music/voice-convert/fx) self-download through AudioModelCache during generation instead — see
        // ModelAcquisition.EnsureAudioAssetsPresent in the CLI for the same distinction.
        app.MapPost("/admin/models/pull", async (PullModelRequest req, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(req.Model))
            {
                return HartsyInferenceServiceExtensions.Problem(
                    StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            }

            CatalogEntry? entry = ModelCatalog.Find(req.Model);
            if (entry is null)
            {
                return HartsyInferenceServiceExtensions.Problem(
                    StatusCodes.Status404NotFound, $"'{req.Model}' is not a catalog id.", "invalid_request_error");
            }
            if (entry.Assets.Count == 0)
                return Results.Ok(new { model = entry.Id, pulled = false, reason = "no preset assets for this catalog entry" });

            IReadOnlyList<ModelAsset> missing = ModelDownloader.MissingAssets(entry);
            if (missing.Count == 0)
                return Results.Ok(new { model = entry.Id, pulled = false, reason = "already present" });

            try
            {
                await ModelDownloader.DownloadAsync(missing, onProgress: null, ct);
            }
            catch (Exception ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status502BadGateway, ex.Message, "download_error");
            }

            return Results.Ok(new { model = entry.Id, pulled = true, files = missing.Count });
        });

        app.MapGet("/admin/cache", (HartsyInferenceServerOptions options) =>
        {
            ModelCacheStore cache = OpenCache(options);
            List<ModelInfo> entries = [.. cache.CachedModelIds.Select(cache.Get).OfType<ModelInfo>()];
            return Results.Ok(new { directory = cache.CacheDirectory, models = entries });
        });

        app.MapDelete("/admin/cache/{id}", (string id, HartsyInferenceServerOptions options) =>
        {
            ModelCacheStore cache = OpenCache(options);
            return cache.Remove(id)
                ? Results.Ok(new { removed = id })
                : HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, $"'{id}' is not in the cache.", "invalid_request_error");
        });

        app.MapPost("/admin/memory/free", (IInferenceEngine engine) =>
        {
            engine.FreeMemory();
            return Results.Ok(new { freed = true });
        });

        // Routed through the queue so the switch waits for any in-flight work on the OLD backend to finish
        // (SetBackend disposes every loaded pipeline) before it runs, and nothing new starts against a
        // half-switched engine — holding the single concurrency slot IS the drain, no separate lock needed.
        app.MapPost("/admin/backend", async (SetBackendRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            string selector = (req.Backend ?? "").Trim().ToLowerInvariant();
            if (!BackendFactory.ValidSelectors.Contains(selector))
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                    $"Unknown backend '{req.Backend}'. Valid: {string.Join(", ", BackendFactory.ValidSelectors)}.", "invalid_request_error");
            }

            try
            {
                await queue.EnqueueAsync(() =>
                {
                    engine.SetBackend(selector);
                    return Task.FromResult(true);
                }, ct);
            }
            catch (QueueFullException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error");
            }

            return Results.Ok(new { backend = engine.BackendSelector, resolved = engine.BackendDescription });
        });

        app.MapGet("/admin/queue", (InferenceQueue queue, HartsyInferenceServerOptions options) =>
            Results.Ok(new { pending = queue.PendingCount, maxConcurrency = options.MaxConcurrency, maxQueueDepth = options.MaxQueueDepth }));
    }

    private static ModelCacheStore OpenCache(HartsyInferenceServerOptions options) =>
        string.IsNullOrWhiteSpace(options.ModelCacheDirectory) ? new ModelCacheStore() : new ModelCacheStore(options.ModelCacheDirectory);
}
