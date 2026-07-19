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
}
