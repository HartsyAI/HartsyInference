using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
// The enclosing HartsyInference.API namespace declares its own ImageData (the OpenAI-compat b64_json DTO in
// OpenAiDtos.cs) — enclosing-namespace member lookup takes priority over BOTH `using` and `using`-alias
// directives for a bare "ImageData", so an alias reusing that name doesn't disambiguate. This one uses a
// different alias name instead of fighting that resolution order.
using NativeImageData = HartsyInference.Engine.Requests.ImageData;

namespace HartsyInference.API.Endpoints;

/// <summary>Native vision route: embed/detect/segment/depth/edge/lineart/normal/segmap/background-removal, one
/// call for all nine modes (<see cref="VisionRequest.Mode"/> selects which). One-shot, no progress/streaming.</summary>
public static class VisionEndpoints
{
    /// <summary>Maps <c>/v1/native/vision</c>.</summary>
    public static void MapVisionEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/vision", async (NativeVisionRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Vision);
            try
            {
                VisionResult result = await queue.EnqueueAsync(() => engine.Vision.RunAsync(spec, req.Request, ct), ct);
                return Results.Ok(ToResponse(result));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }

    // Only Embedding/Detections travel as-is; Masks/Image carry raw RGB24 (ImageData) like ImageResult does, so
    // they get the same PNG-encode-for-transport treatment as ImageEndpoints.ToResponse.
    private static object ToResponse(VisionResult result) => new
    {
        embedding = result.Embedding,
        detections = result.Detections,
        masks = result.Masks?.Select(EncodePng).ToList(),
        image = result.Image is { } img ? EncodePng(img) : null,
    };

    private static string EncodePng(NativeImageData image) => Convert.ToBase64String(PngEncoder.Encode(image.Rgb, image.Width, image.Height));
}
