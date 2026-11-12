using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Operations;

namespace HartsyInference.API.Endpoints;

/// <summary>Native 3D-mesh route. Unlike the other Phase 3 additions, <see cref="IMeshService.GenerateAsync"/> takes an <see cref="IProgress{T}"/> just like images do, so this gets the same <c>/stream</c> variant.</summary>
public static class MeshEndpoints
{
    /// <summary>The thumbnail renders that may run at once.</summary>
    private static readonly SemaphoreSlim ThumbnailGate = new(2);

    /// <summary>The model edits that may run at once.</summary>
    private static readonly SemaphoreSlim OperateGate = new(2);

    /// <summary>Maps <c>/v1/native/mesh</c> and its SSE step-preview variant.</summary>
    public static void MapMeshEndpoints(this WebApplication app)
    {
        app.MapGet("/v1/native/mesh/loaded", IResult (IInferenceEngine engine) => Results.Json(engine.MeshLifecycle.Loaded()));

        app.MapPost("/v1/native/mesh/free", IResult (NativeMeshFreeRequest? req, IInferenceEngine engine) =>
            Results.Ok(new { Freed = engine.MeshLifecycle.Free(req?.Name) }));

        app.MapGet("/v1/native/mesh/describe", IResult (IInferenceEngine engine) => Results.Json(engine.MeshPlanning.DescribeAll()));

        app.MapGet("/v1/native/mesh/describe/{id}", IResult (string id, IInferenceEngine engine) =>
        {
            try
            {
                return Results.Json(engine.MeshPlanning.Describe(id));
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, ex.Message, "not_found_error");
            }
        });

        app.MapGet("/v1/native/mesh/assets/{id}", IResult (string id, string? modelPath, IInferenceEngine engine) =>
        {
            try
            {
                return Results.Json(engine.MeshAssets.GetStatus(ModelResolver.Resolve(id, modelPath, Modality.Mesh)));
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, ex.Message, "not_found_error");
            }
        });

        app.MapPost("/v1/native/mesh/prefetch", async Task<IResult> (
            NativeMeshPrefetchRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
                ModelPrefetchResult result = await queue.EnqueueAsync(() => engine.ModelPrefetch.PrefetchAsync(spec, null, ct), ct);
                return Results.Ok(new { result.Supported, result.Message, result.Blockers, Installed = engine.MeshAssets.GetStatus(spec).Complete });
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, ex.Message, "not_found_error");
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/mesh/prepare", async Task<IResult> (
            NativeMeshRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
                HartsyInference.Engine.Requests.ImageData image = req.ToMeshRequest().Image ?? throw new ArgumentException("Send the image as ImagePng or Request.Image.");
                PreparedMeshImage prepared = await queue.EnqueueAsync(() => engine.MeshPreparation.PrepareImageAsync(spec, image, ct), ct);
                return Results.Ok(new
                {
                    Image = Convert.ToBase64String(PngEncoder.Encode(prepared.Image)),
                    prepared.Image.Width,
                    prepared.Image.Height,
                    Cutout = prepared.Cutout is null ? null : Convert.ToBase64String(PngEncoder.Encode(prepared.Cutout)),
                    prepared.Matted,
                    prepared.Notes,
                });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapGet("/v1/native/mesh/formats", IResult (IInferenceEngine engine) =>
            Results.Ok(new { Sources = engine.MeshConversion.Sources, Targets = engine.MeshConversion.Targets }));

        app.MapPost("/v1/native/mesh/convert", async Task<IResult> (NativeMeshConvertRequest req, IInferenceEngine engine, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(req.Data) || string.IsNullOrWhiteSpace(req.Format) || string.IsNullOrWhiteSpace(req.Target))
                    throw new ArgumentException("Send Data (base64), Format and Target.");
                byte[] data = MeshBase64.Decode(req.Data);
                MeshConversion result = await Task.Run(() => engine.MeshConversion.Convert(data, req.Format, req.Target, req.Name), ct);
                return Results.Ok(new
                {
                    result.Target,
                    // With an archive the files travel inside it once; their names and types are still listed.
                    Files = result.Files.Select(f => new { f.Name, f.ContentType, Data = req.Archive ? null : System.Convert.ToBase64String(f.Data) }),
                    result.Notes,
                    result.Vertices,
                    result.Triangles,
                    result.StaticOnly,
                    Archive = req.Archive ? System.Convert.ToBase64String(result.Archive()) : null,
                });
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // The caller went away: nobody is left to read a response, and it is not a server error.
                return Results.StatusCode(499);
            }
            catch (NotSupportedException ex)
            {
                // A format the converter does not read or write is the caller's request, not a missing server feature.
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapGet("/v1/native/mesh/operations", IResult (IInferenceEngine engine) =>
            Results.Ok(engine.MeshOperations.Operations.Select(o => new { o.Id, o.Description, o.Parameters })));

        app.MapPost("/v1/native/mesh/operate", async Task<IResult> (NativeMeshOperateRequest req, IInferenceEngine engine, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(req.Data) || string.IsNullOrWhiteSpace(req.Format) || req.Operations is null || req.Operations.Count == 0)
                    throw new ArgumentException("Send Data (base64), Format and at least one entry in Operations.");
                if (req.Operations.Any(o => string.IsNullOrWhiteSpace(o.Operation)))
                    throw new ArgumentException("Every entry of Operations needs an Operation id.");
                byte[] data = MeshBase64.Decode(req.Data);
                List<MeshOperationStep> steps = [.. req.Operations.Select(o => new MeshOperationStep(o.Operation!, o.Parameters))];
                // Editing a large model is CPU and memory heavy and runs next to model work, so only a couple run at once; the rest wait here and give up if their caller does.
                await OperateGate.WaitAsync(ct);
                MeshOperationResult result;
                try
                {
                    result = await Task.Run(() => engine.MeshOperations.Apply(data, req.Format, steps, req.Target ?? "glb", req.Name, ct), ct);
                }
                finally
                {
                    OperateGate.Release();
                }
                return Results.Ok(new
                {
                    result.Output.Target,
                    Files = result.Output.Files.Select(f => new { f.Name, f.ContentType, Data = req.Archive ? null : System.Convert.ToBase64String(f.Data) }),
                    result.Output.Notes,
                    result.Output.Vertices,
                    result.Output.Triangles,
                    result.Output.StaticOnly,
                    Archive = req.Archive ? System.Convert.ToBase64String(result.Output.Archive()) : null,
                    result.Steps,
                });
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                return Results.StatusCode(499);
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/mesh/thumbnail", async Task<IResult> (NativeMeshThumbnailRequest req, IInferenceEngine engine, CancellationToken ct) =>
        {
            try
            {
                if (string.IsNullOrWhiteSpace(req.Data) || string.IsNullOrWhiteSpace(req.Format))
                    throw new ArgumentException("Send Data (base64) and Format.");
                byte[] data = MeshBase64.Decode(req.Data);
                // A render is CPU and memory heavy and runs next to model work, so only a couple run at once; the rest wait here, and give up if their caller does.
                await ThumbnailGate.WaitAsync(ct);
                MeshThumbnail thumbnail;
                try
                {
                    thumbnail = await Task.Run(() => engine.MeshThumbnails.Render(data, req.Format, req.Options, ct), ct);
                }
                finally
                {
                    ThumbnailGate.Release();
                }
                return Results.Ok(new { Png = System.Convert.ToBase64String(thumbnail.Png), thumbnail.Width, thumbnail.Height, thumbnail.Kind, thumbnail.Notes });
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                return Results.StatusCode(499);
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/mesh/plan", async Task<IResult> (NativeMeshRequest req, IInferenceEngine engine, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
                MeshPlan plan = await engine.MeshPlanning.PlanAsync(spec, req.ToMeshRequest(), ct);
                NativeMeshPlanResponse response = NativeMeshPlanResponse.Create(plan);
                return plan.IsValid ? Results.Ok(response) : Results.Json(response, statusCode: StatusCodes.Status422UnprocessableEntity);
            }
            catch (NotSupportedException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, ex.Message, "not_found_error");
            }
            catch (ArgumentException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
        });

        app.MapPost("/v1/native/mesh", async (NativeMeshRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Mesh);
            try
            {
                MeshRequest request = req.ToMeshRequest();
                MeshResult result = await queue.EnqueueAsync(() => engine.Mesh.GenerateAsync(spec, request, progress: null, ct), ct);
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
                    writer.TryWrite(SseHelpers.Event("progress", StepPreviewPayload.Create(p), jsonOptions)));
                MeshRequest request = req.ToMeshRequest();
                MeshResult result = await engine.Mesh.GenerateAsync(spec, request, progress, ct);
                writer.TryWrite(SseHelpers.Event("complete",
                    WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, req.Request.Prompt, result.Format)), jsonOptions));
            }, ct);
        });
    }

    /// <summary>Re-emits the native mesh fields plus where the file landed; property names match <see cref="MeshResult"/>, so this is additive for existing clients.</summary>
    private static object WithSavedPath(MeshResult result, string? savedPath) => new
    {
        result.Data,
        result.Format,
        result.Parts,
        result.Preview,
        result.Info,
        result.Frame,
        result.Warnings,
        result.Settings,
        savedPath,
    };
}
