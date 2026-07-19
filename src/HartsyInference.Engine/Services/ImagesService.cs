using System.Globalization;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-generation service. Phase 1 drives the existing SDXL path over the flat request fields; the
/// composition features (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional/variation-seed) land with the
/// architecture-recipe and feature-resolver phases and are rejected until then rather than silently ignored.</summary>
public sealed class ImagesService : IImagesService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal ImagesService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<ImageResult> GenerateAsync(ModelSpec spec, ImageRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        RejectUnsupported(request);

        ParamState parameters = new ParamState(Modality.Image);
        parameters.Put("negative", request.NegativePrompt ?? "");
        parameters.Put("width", request.Width.ToString(CultureInfo.InvariantCulture));
        parameters.Put("height", request.Height.ToString(CultureInfo.InvariantCulture));
        parameters.Put("steps", request.Steps.ToString(CultureInfo.InvariantCulture));
        parameters.Put("cfg", request.CfgScale.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", request.Seed.ToString(CultureInfo.InvariantCulture));

        SinkBridge sink = new SinkBridge(progress, null);
        return Task.Run(
            () =>
            {
                GeneratedArtifact artifact = _engine.RunHandler(spec, request.Prompt, parameters, sink, cancel);
                long seed = artifact.Meta.TryGetValue("seed", out string? s) && long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out long parsed)
                    ? parsed
                    : request.Seed;
                return new ImageResult
                {
                    Rgb = artifact.PreviewRgb ?? Array.Empty<byte>(),
                    Width = artifact.PreviewWidth,
                    Height = artifact.PreviewHeight,
                    Seed = seed,
                    Meta = new Dictionary<string, string>(artifact.Meta),
                };
            },
            cancel);
    }

    private static void RejectUnsupported(ImageRequest request)
    {
        if (request.Loras is not null
            || (request.ControlNets is { Count: > 0 })
            || request.IpAdapter is not null
            || request.Refiner is not null
            || request.Img2Img is not null
            || request.Inpaint is not null
            || request.Regional is not null
            || request.VariationSeed is not null)
        {
            throw new NotSupportedException(
                "Image composition features (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional/variation-seed) " +
                "are wired by the architecture-recipe (E-IMG-3) and feature-resolver (E-IMG-4) phases; not yet available.");
        }
    }
}
