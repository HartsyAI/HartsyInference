using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-generation service: detects the checkpoint's architecture, resolves its recipe from the
/// <see cref="RecipeRegistry"/>, constructs (and caches) the pipeline, and generates. Composition features
/// (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional/variation-seed) are applied by the feature-resolver
/// phase (E-IMG-4); until then a request that sets one is rejected rather than silently ignored.</summary>
public sealed class ImagesService : IImagesService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal ImagesService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<ImageResult> GenerateAsync(ModelSpec spec, ImageRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        RejectUnsupported(request);
        return Task.Run(
            () =>
            {
                IRecipePipeline pipeline = _engine.GetOrConstructRecipe(spec);
                return pipeline.Generate(request, progress, cancel);
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
                "are applied by the feature-resolver phase (E-IMG-4); not yet available.");
        }
    }
}
