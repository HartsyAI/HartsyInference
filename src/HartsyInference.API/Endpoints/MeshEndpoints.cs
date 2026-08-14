using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native 3D-mesh route. Unlike the other Phase 3 additions, <see cref="IMeshService.GenerateAsync"/>
/// takes an <see cref="IProgress{T}"/> just like images do, so this gets the same <c>/stream</c> variant.</summary>
public static class MeshEndpoints
{
    /// <summary>Maps <c>/v1/native/mesh</c> and its SSE step-preview variant.</summary>
    public static void MapMeshEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/mesh", async (NativeMeshRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
            try
            {
                MeshResult result = await queue.EnqueueAsync(() => engine.Mesh.GenerateAsync(spec, req.Request, progress: null, ct), ct);
                return Results.Ok(WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, req.Request.Prompt, result.Format)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/mesh/stream", async (NativeMeshRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs inside the queue's held slot — call the service directly, not through a second EnqueueAsync.
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", new { step = p.Step, total = p.TotalSteps }, jsonOptions)));
                MeshResult result = await engine.Mesh.GenerateAsync(spec, req.Request, progress, ct);
                writer.TryWrite(SseHelpers.Event("complete",
                    WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, req.Request.Prompt, result.Format)), jsonOptions));
            }, ct);
        });
    }

    /// <summary>Re-emits the native mesh fields plus where the file landed; property names match
    /// <see cref="MeshResult"/>, so this is additive for existing clients.</summary>
    private static object WithSavedPath(MeshResult result, string? savedPath) => new
    {
        result.Data,
        result.Format,
        savedPath,
    };
}
