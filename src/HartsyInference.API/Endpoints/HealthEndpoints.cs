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
        app.MapGet("/ready", (string? model, IInferenceEngine engine) =>
        {
            if (!string.IsNullOrWhiteSpace(model)) return ReadyFor(model, engine.Text);
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

    /// <summary>The readiness of one model for <c>/ready?model=</c>: 200 only when a deployment of it is Ready. A model with no deployment is 503 with
    /// <c>not_deployed</c>, and a deployment that is not ready says which state it is in. With none Ready, the latest Loading one answers, else the latest of any state
    /// (latest in the listing's order), so a stale Unloaded or Failed record never hides a load in progress. A deployment matches when the requested string equals its
    /// deployment id or its model name, exactly; an alias, or a path that resolves to the same model, does not match.</summary>
    internal static IResult ReadyFor(string model, ITextService text)
    {
        List<DeploymentStatus> matches = [.. text.Deployments.Where(d => d.DeploymentId == model || d.Model == model)];
        DeploymentStatus? pick = matches.FindLast(d => d.State == DeploymentState.Ready)
            ?? matches.FindLast(d => d.State == DeploymentState.Loading) ?? matches.LastOrDefault();
        if (pick is null)
            return Results.Json(new { status = "not_deployed", model }, statusCode: StatusCodes.Status503ServiceUnavailable);
        bool ready = pick.State == DeploymentState.Ready;
        return Results.Json(
            new { status = ready ? "ready" : "not_ready", model, deployment_id = pick.DeploymentId, state = DeploymentDto.ToWire(pick.State), problem = pick.Problem },
            statusCode: ready ? StatusCodes.Status200OK : StatusCodes.Status503ServiceUnavailable);
    }
}
