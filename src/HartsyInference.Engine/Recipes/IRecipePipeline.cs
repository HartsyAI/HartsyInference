using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes;

/// <summary>A constructed, ready-to-run pipeline for one architecture family. Owns the loaded components (transformer,
/// text encoders, VAE) and drives that family's bespoke encode + denoise + decode. Cached per loaded model and reused
/// across requests; disposed when the model is evicted.</summary>
public interface IRecipePipeline : IDisposable
{
    /// <summary>Generates one image for <paramref name="request"/>, reporting step progress.</summary>
    ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel);

    /// <summary>The official defaults for the specific variant this pipeline was constructed for (Flux schnell vs dev,
    /// Krea 2 Turbo vs Base, Lens Turbo vs standard), or null when the recipe's family-level
    /// <see cref="IArchitectureRecipe.Defaults"/> already describe it. Resolved from the loaded checkpoint, so it can
    /// never disagree with the weights that will actually run.</summary>
    ImageDefaults? VariantDefaults => null;
}
