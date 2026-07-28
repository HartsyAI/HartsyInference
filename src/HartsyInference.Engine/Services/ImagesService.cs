using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-generation service: detects the checkpoint's architecture, resolves its recipe from the
/// <see cref="RecipeRegistry"/>, constructs (and caches) the pipeline, and generates. Every composition object the
/// request sets is mapped to an <see cref="ImageFeatures"/> bit and checked against the resolved recipe's
/// <see cref="IArchitectureRecipe.Supports"/>, so an unwired feature is rejected by name rather than silently ignored.
/// Every tunable the caller left null is filled from the resolved recipe's <see cref="ImageDefaults"/> before the
/// pipeline is driven, so each family runs at its creator's recommended settings unless the caller overrides them.</summary>
public sealed class ImagesService : IImagesService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal ImagesService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<ImageResult> GenerateAsync(ModelSpec spec, ImageRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        RejectUnsupported(spec, request);
        return Task.Run(
            () =>
            {
                IRecipePipeline pipeline = _engine.GetOrConstructRecipe(spec, request);
                ImageRequest resolved = _engine.DefaultsFor(spec, pipeline).Apply(request);
                return _engine.GenerateWithVramCleanup(() => pipeline.Generate(resolved, progress, cancel));
            },
            cancel);
    }

    /// <summary>The features <paramref name="request"/> asks for, one bit per composition object actually set.</summary>
    private static ImageFeatures RequestedFeatures(ImageRequest request)
    {
        ImageFeatures features = ImageFeatures.None;
        if (request.Loras is { Entries.Count: > 0 })
        {
            features |= ImageFeatures.Lora;
        }
        if (request.ControlNets is { Count: > 0 })
        {
            features |= ImageFeatures.ControlNet;
        }
        if (request.IpAdapter is not null)
        {
            features |= ImageFeatures.IpAdapter;
        }
        if (request.Refiner is not null)
        {
            features |= ImageFeatures.Refiner;
        }
        if (request.Img2Img is not null)
        {
            features |= ImageFeatures.Img2Img;
        }
        if (request.Inpaint is not null)
        {
            features |= ImageFeatures.Inpaint;
        }
        if (request.Regional is not null)
        {
            features |= ImageFeatures.Regional;
        }
        if (request.VariationSeed is not null)
        {
            features |= ImageFeatures.VariationSeed;
        }
        return features;
    }

    /// <summary>Throws naming the family and the exact features it cannot apply; features the recipe declares pass through.</summary>
    private void RejectUnsupported(ModelSpec spec, ImageRequest request)
    {
        ImageFeatures requested = RequestedFeatures(request);
        if (requested == ImageFeatures.None)
        {
            return;
        }
        ImageFeatures missing = requested & ~_engine.SupportedFeatures(spec);
        if (missing != ImageFeatures.None)
        {
            throw new NotSupportedException(
                $"Model family '{InferenceEngine.FamilyIdFor(spec)}' does not support: {missing}.");
        }
    }
}
