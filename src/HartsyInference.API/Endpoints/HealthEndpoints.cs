using System.Reflection;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Liveness, readiness, and version probes.</summary>
public static class HealthEndpoints
{
    /// <summary>Maps <c>/health</c>, <c>/ready</c>, and <c>/version</c>.</summary>
    public static void MapHealthEndpoints(this WebApplication app)
    {
        // Deliberately cheap and dependency-free: if it did real checks, a transient backend hiccup would make
        // an orchestrator kill+restart a process that was actually fine. See /ready for the real check.
        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));

        // BackendDescription resolves "auto" via a live CUDA device query (cheap: no allocation, no VRAM touched)
        // but never constructs the backend itself — the first real generation request still pays that cost.
        // Generalizes today's chat-only readiness (DynamicBatchScheduler.IsLoopAlive) with an engine-wide check;
        // per-loaded-model health hooks are reintroduced once generation endpoints exist (Phase 3).
        app.MapGet("/ready", (IInferenceEngine engine) =>
        {
            try
            {
                return Results.Ok(new { status = "ready", backend = engine.BackendDescription });
            }
            catch (Exception ex)
            {
                return Results.Json(
                    new { status = "degraded", error = ex.Message },
                    statusCode: StatusCodes.Status503ServiceUnavailable);
            }
        });

        app.MapGet("/version", (IInferenceEngine engine) =>
        {
            AssemblyName asm = typeof(Program).Assembly.GetName();
            return Results.Ok(new
            {
                version = asm.Version?.ToString() ?? "unknown",
                backendSelector = engine.BackendSelector,
                backend = engine.BackendDescription,
            });
        });
    }
}
