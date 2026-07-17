using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>An image encoder for the SAM family: turns a preprocessed image into the dense image
/// embedding the mask decoder consumes. SAM 2 uses the <see cref="HieraImageEncoder"/> (windowed-attention
/// Hiera trunk + FPN neck); MobileSAM swaps in a TinyViT trunk while keeping the same prompt encoder and
/// mask decoder — hence this seam so either backbone plugs into <see cref="SamPipeline"/>.</summary>
public interface ISamImageEncoder : IDisposable
{
    /// <summary>Identifier for the concrete backbone (e.g. <c>"hiera"</c>, <c>"tinyvit"</c>).</summary>
    string EncoderName { get; }

    /// <summary>Downsampling factor from input pixels to the returned embedding grid (SAM 2 = 16).</summary>
    int FeatureStride { get; }

    /// <summary>Runs the backbone on <paramref name="pixelValues"/> <c>[1, 3, H, W]</c> and returns the stride-16
    /// image embedding <c>[1, promptDim, H/16, W/16]</c> plus the optional stride-4/stride-8 high-res FPN levels
    /// SAM 2 fuses into the mask upscaler. Caller owns the returned tensors.</summary>
    SamImageFeatures Forward(IBackend backend, Tensor pixelValues);

    /// <summary>Loads backbone weights from a (component-scoped) checkpoint slice under <paramref name="prefix"/>.</summary>
    void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix);

    /// <summary>Yields every loaded weight tensor (for GPU preload / lifetime management).</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
