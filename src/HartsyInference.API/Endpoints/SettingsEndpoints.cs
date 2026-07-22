namespace HartsyInference.API.Endpoints;

/// <summary>Read-only view of the server's active configuration.</summary>
public static class SettingsEndpoints
{
    /// <summary>Maps <c>GET /settings</c>. Startup-config-only (env vars / <c>appsettings.json</c>) is the
    /// deliberate default — no live PUT/PATCH here; <c>/admin/backend</c> is the one runtime-mutable exception
    /// because switching backends has real operational consequences the inference queue must coordinate.</summary>
    public static void MapSettingsEndpoints(this WebApplication app)
    {
        app.MapGet("/settings", (HartsyInferenceServerOptions options) => Results.Ok(new
        {
            backend = options.Backend,
            kernelDirectory = options.KernelDirectory,
            maxConcurrency = options.MaxConcurrency,
            maxQueueDepth = options.MaxQueueDepth,
            modelCacheDirectory = options.ModelCacheDirectory,
            apiKeyConfigured = !string.IsNullOrEmpty(options.ApiKey),
        }));
    }
}
