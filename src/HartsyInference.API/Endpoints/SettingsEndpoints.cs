using HartsyInference.Core.Configuration;

namespace HartsyInference.API.Endpoints;

/// <summary>The server's configuration, and the engine settings a caller may read and change.</summary>
public static class SettingsEndpoints
{
    /// <summary>Maps the settings endpoints.</summary>
    /// <remarks>Two things live under one route on purpose, because they are genuinely different: <c>server</c> is
    /// startup config owned by the ASP.NET host (ports, backend, API key) and stays read-only here —
    /// <c>/admin/backend</c> is the one runtime-mutable exception, because switching backends has operational
    /// consequences the inference queue must coordinate. <c>engine</c> is the knob surface, which a caller may
    /// change and which persists to the engine's settings file.</remarks>
    public static void MapSettingsEndpoints(this WebApplication app)
    {
        app.MapGet("/settings", (HartsyInferenceServerOptions options) => Results.Ok(new
        {
            server = new
            {
                backend = options.Backend,
                kernelDirectory = options.KernelDirectory,
                maxConcurrency = options.MaxConcurrency,
                maxQueueDepth = options.MaxQueueDepth,
                modelCacheDirectory = options.ModelCacheDirectory,
                apiKeyConfigured = !string.IsNullOrEmpty(options.ApiKey),
            },
            engine = new
            {
                file = KnobFile.Path,
                settings = Describe(),
            },
        }));

        app.MapGet("/settings/engine/{id}", (string id) =>
            KnobRegistry.Find(id) is null
                ? Results.NotFound(new { error = $"Unknown setting '{id}'." })
                : Results.Ok(One(id)));

        // PUT rather than PATCH: the body carries the whole value of one setting, and re-sending it is idempotent.
        app.MapPut("/settings/engine/{id}", (string id, EngineSettingUpdate body) =>
        {
            if (body?.Value is null)
            {
                return Results.BadRequest(new { error = "Body must be {\"value\": \"...\"}." });
            }
            try
            {
                KnobFile.Save(id, body.Value);
            }
            catch (InvalidOperationException ex)
            {
                // Unknown id, wrong type and out-of-range all surface here, from the same parse the file load uses.
                return Results.BadRequest(new { error = ex.Message });
            }
            return Results.Ok(One(id));
        });
    }

    private static object One(string id)
    {
        object knob = KnobRegistry.Find(id)!;
        (string _, string type, object? declared, KnobScope scope, KnobDomain domain, string summary) = KnobRegistry.Describe(knob);
        return new
        {
            id,
            value = KnobRegistry.ValueOf(knob),
            source = KnobStore.SourceOf(id),
            type,
            @default = declared,
            domain = domain.ToString(),
            // A Construction setting is written now but bound when the engine is built, so a caller is told to restart.
            appliesAt = scope == KnobScope.Construction ? "restart" : "next run",
            summary,
        };
    }

    private static List<object> Describe()
        => [.. KnobRegistry.All
            .Select(KnobRegistry.Describe)
            .Where(k => !k.Id.StartsWith("test.", StringComparison.Ordinal))
            .OrderBy(k => k.Id, StringComparer.Ordinal)
            .Select(k => One(k.Id))];
}

/// <summary>Body of <c>PUT /settings/engine/{id}</c>.</summary>
/// <remarks>The value is a string so one shape covers every knob type; it is parsed with the same rules the
/// settings file uses, so <c>"true"</c>, <c>"1"</c> and <c>"256"</c> mean there exactly what they mean in the file.</remarks>
public sealed record EngineSettingUpdate(string? Value);
