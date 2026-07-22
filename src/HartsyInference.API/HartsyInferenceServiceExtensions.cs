using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Diagnostics;

namespace HartsyInference.API;

/// <summary>DI registration and endpoint mapping for the HartsyInference server.</summary>
public static class HartsyInferenceServiceExtensions
{
    /// <summary>Registers the inference engine facade and its concurrency gate.</summary>
    public static IServiceCollection AddHartsyInference(this IServiceCollection services, Action<HartsyInferenceServerOptions>? configure = null)
    {
        HartsyInferenceServerOptions options = new HartsyInferenceServerOptions();
        configure?.Invoke(options);
        services.AddSingleton(options);

        if (!string.IsNullOrWhiteSpace(options.KernelDirectory))
            BackendFactory.KernelDirOverride = options.KernelDirectory;

        // One long-lived facade for the process, exactly like the CLI's one-per-invocation InferenceEngine —
        // just scoped to app lifetime instead of one command. IInferenceEngine is NOT safely re-entrant per
        // backend, so every call site must go through the InferenceQueue below rather than calling it directly.
        services.AddSingleton<IInferenceEngine>(_ => new InferenceEngine(options.Backend));
        services.AddSingleton(new InferenceQueue(options.MaxConcurrency, options.MaxQueueDepth));

        services.ConfigureHttpJsonOptions(o =>
            o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter()));

        return services;
    }

    /// <summary>Maps health/settings/admin probes and optional API-key auth. Generation endpoints (native +
    /// OpenAI-compat) are wired onto <see cref="IInferenceEngine"/> in a follow-up phase.</summary>
    public static void MapHartsyInferenceEndpoints(this WebApplication app)
    {
        HartsyInferenceServerOptions options = app.Services.GetRequiredService<HartsyInferenceServerOptions>();

        // Last-resort safety net: catches any exception a route handler doesn't already handle itself and turns
        // it into a structured, logged 500 instead of a bare/blank response. Registered first so it wraps every
        // middleware/route below it. This does NOT and cannot catch a corrupted-state exception (e.g.
        // AccessViolationException from native/unsafe code) — the CLR terminates the process before any handler,
        // including this one, gets a chance to run; that class of failure needs process-level supervision (see
        // deploy/ for a restart policy), not in-process handling.
        app.UseExceptionHandler(errApp =>
        {
            errApp.Run(async ctx =>
            {
                Exception? ex = ctx.Features.Get<IExceptionHandlerPathFeature>()?.Error;
                ILogger log = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("HartsyInference.UnhandledException");
                log.LogError(ex, "Unhandled exception on {Method} {Path}", ctx.Request.Method, ctx.Request.Path);
                await WriteError(ctx, StatusCodes.Status500InternalServerError, "An unexpected server error occurred.", "server_error");
            });
        });

        // Optional API-key gate over everything except the liveness/readiness/version probes — those need to
        // stay reachable for an orchestrator's health checks regardless of auth config.
        if (!string.IsNullOrEmpty(options.ApiKey))
        {
            app.Use(async (ctx, next) =>
            {
                bool isProbe = ctx.Request.Path.StartsWithSegments("/health")
                    || ctx.Request.Path.StartsWithSegments("/ready")
                    || ctx.Request.Path.StartsWithSegments("/version");
                if (!isProbe && !IsAuthorized(ctx, options.ApiKey!))
                {
                    await WriteError(ctx, StatusCodes.Status401Unauthorized, "Invalid or missing API key.", "invalid_request_error");
                    return;
                }
                await next();
            });
        }

        app.MapHealthEndpoints();
        app.MapSettingsEndpoints();
        app.MapAdminEndpoints();
    }

    private static bool IsAuthorized(HttpContext ctx, string apiKey)
    {
        if (ctx.Request.Headers.TryGetValue("x-api-key", out Microsoft.Extensions.Primitives.StringValues k) && k == apiKey) return true;
        if (ctx.Request.Headers.Authorization.ToString() is { } auth &&
            auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) &&
            auth["Bearer ".Length..] == apiKey)
        {
            return true;
        }
        return false;
    }

    /// <summary>Structured error response, OpenAI-shaped for continuity with the compat routes that will land
    /// alongside generation endpoints. Internal so the <c>Endpoints/*.cs</c> route groups can share it.</summary>
    internal static IResult Problem(int status, string message, string type) =>
        Results.Json(OpenAiError.Make(message, type), statusCode: status);

    private static async Task WriteError(HttpContext ctx, int status, string message, string type)
    {
        ctx.Response.StatusCode = status;
        await ctx.Response.WriteAsJsonAsync(OpenAiError.Make(message, type));
    }
}
