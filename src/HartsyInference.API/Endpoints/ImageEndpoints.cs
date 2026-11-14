using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native image-generation routes: a byte-for-byte pass-through of <see cref="ImageRequest"/> (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional composition included) to <see cref="ImageResult"/>.</summary>
public static class ImageEndpoints
{
    /// <summary>Maps <c>/v1/native/images</c> and its SSE step-preview variant.</summary>
    public static void MapImageEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/images", async (NativeImageRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Image);
            try
            {
                ImageResult result = await GenerateAsync(engine, queue, spec, ApplyImageComposition(req), progress: null, ct);
                return Results.Ok(ToResponse(result, Persist(req, result, req.Request.Prompt)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/images/stream", async (NativeImageRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Image);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs INSIDE the queue's held slot (SseHelpers wraps this whole delegate in EnqueueAsync) — call
                // the service directly here, never the GenerateAsync helper below, or the second EnqueueAsync
                // would deadlock waiting on the semaphore this frame already holds.
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", StepPreviewPayload.Create(p), jsonOptions)));
                ImageResult result = await engine.Images.GenerateAsync(spec, ApplyImageComposition(req), progress, ct);
                writer.TryWrite(SseHelpers.Event("complete", ToResponse(result, Persist(req, result, req.Request.Prompt)), jsonOptions));
            }, ct);
        });

        // The people-removal recipe (PeopleRemoval): the caller sends a photo and a mask, and nothing else.
        app.MapPost("/v1/native/images/remove-people", async (RemovePeopleRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(PeopleRemoval.ModelId(req.Precision), null, Modality.Image);
                HartsyInference.Engine.Requests.ImageData image = ImageDataCodec.DecodeBase64(req.InitImageBase64);
                HartsyInference.Engine.Requests.ImageData mask = ImageDataCodec.DecodeBase64(req.MaskBase64);
                if (mask.Width != image.Width || mask.Height != image.Height)
                {
                    return Results.BadRequest(new { error = $"The mask is {mask.Width}x{mask.Height} but the photo is {image.Width}x{image.Height}. They must match." });
                }
                ImageRequest request = PeopleRemoval.Build(image, mask, req.Seed ?? -1);
                ImageResult result = await GenerateAsync(engine, queue, spec, request, progress: null, ct);
                return Results.Ok(ToResponse(result, Persist(req, result, PeopleRemoval.Prompt)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }

    /// <summary>Folds the envelope's base64 init image / mask conveniences into the native request. Returns the request untouched when none are set, so a client that populates <c>request.img2img</c> with raw RGB24 directly keeps working.</summary>
    public static ImageRequest ApplyImageComposition(NativeImageRequest req)
    {
        ImageRequest request = req.Request;
        if (req.InitImageBase64 is not null)
        {
            Img2Img img2img = new Img2Img { InitImage = ImageDataCodec.DecodeBase64(req.InitImageBase64) };
            request = request with { Img2Img = req.Creativity is null ? img2img : img2img with { Creativity = req.Creativity.Value } };
        }
        else if (req.Creativity is not null && request.Img2Img is not null)
        {
            request = request with { Img2Img = request.Img2Img with { Creativity = req.Creativity.Value } };
        }

        if (req.MaskBase64 is not null)
        {
            Inpaint inpaint = new Inpaint { Mask = ImageDataCodec.DecodeBase64(req.MaskBase64) };
            if (req.MaskGrow is not null) { inpaint = inpaint with { Grow = req.MaskGrow.Value }; }
            if (req.MaskBlur is not null) { inpaint = inpaint with { Blur = req.MaskBlur.Value }; }
            if (req.MaskShrinkGrow is not null) { inpaint = inpaint with { ShrinkGrow = req.MaskShrinkGrow.Value }; }
            request = request with { Inpaint = inpaint };
        }
        return request;
    }

    /// <summary>Queue-gated generation, shared with the OpenAI-compat <c>/v1/images/generations</c> wrapper.</summary>
    internal static Task<ImageResult> GenerateAsync(
        IInferenceEngine engine, InferenceQueue queue, ModelSpec spec, ImageRequest request,
        IProgress<StepPreview>? progress, CancellationToken ct) =>
        queue.EnqueueAsync(() => engine.Images.GenerateAsync(spec, request, progress, ct), ct);

    /// <summary>PNG-encodes the raw RGB result for HTTP transport (base64 JSON — the native contract carries no codec of its own), and reports where it was saved.</summary>
    /// <remarks>A layered result also carries <c>layers</c>: <c>{ index, name, png }</c> per layer, bottom first, each an
    /// RGBA PNG with straight alpha; <c>png</c> stays the composite. The OpenAI-compatible route returns the composite only.
    /// A result with depth beside its image (Iris-Z <c>rgbd</c>) carries <c>depthPng</c>, the colour-coded depth map.
    /// <para>For a request seed of 2^31 or more, <c>seed</c> is the 31-bit alias that drove the noise, and <c>meta.seed</c>
    /// carries the full request seed. Send <c>meta.seed</c> back to reproduce the image.</para></remarks>
    internal static object ToResponse(ImageResult result, string? savedPath = null) => new
    {
        png = Convert.ToBase64String(EncodePng(result)),
        result.Width,
        result.Height,
        result.Seed,
        result.Meta,
        savedPath,
        depthPng = DepthImage.Companion(result) is { } depthRgb
            ? Convert.ToBase64String(PngEncoder.Encode(depthRgb, null, result.Width, result.Height)) : null,
        layers = result.Layers?.Select(l => new
        {
            index = l.Index,
            name = l.Name,
            png = Convert.ToBase64String(PngEncoder.Encode(l.Rgb, l.Alpha, l.Width, l.Height)),
        }).ToList(),
    };

    /// <summary>Persists the PNG unless the request opted out; null when nothing was written.</summary>
    internal static string? Persist(NativeArtifactRequest req, ImageResult result, string slugSource) =>
        ArtifactPersistence.Save(req, EncodePng(result), slugSource, "png");

    /// <summary>Color-type-6 PNG when a background-removal pass left a matte behind, color-type-2 otherwise.</summary>
    private static byte[] EncodePng(ImageResult result) =>
        PngEncoder.Encode(result.Rgb, result.HasAlpha ? result.Alpha : null, result.Width, result.Height);
}
