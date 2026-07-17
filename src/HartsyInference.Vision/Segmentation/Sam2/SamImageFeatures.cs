using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>Output of a SAM image encoder: the stride-16 image embedding the mask decoder attends over, plus
/// the two higher-resolution FPN levels (stride-4 and stride-8) SAM 2 fuses into the mask upscaler when
/// <c>use_high_res_features_in_sam</c> is set. <see cref="HighResStride4"/>/<see cref="HighResStride8"/> are
/// <c>null</c> for backbones that do not expose them (the decoder then falls back to the plain upscaler).</summary>
public sealed record SamImageFeatures
{
    /// <summary>Stride-16 image embedding <c>[1, promptDim, H/16, W/16]</c> (the decoder's key/value source).</summary>
    public required Tensor VisionFeatures { get; init; }

    /// <summary>Stride-4 neck level <c>[1, promptDim, H/4, W/4]</c>, or <c>null</c> if the backbone omits it.</summary>
    public Tensor? HighResStride4 { get; init; }

    /// <summary>Stride-8 neck level <c>[1, promptDim, H/8, W/8]</c>, or <c>null</c> if the backbone omits it.</summary>
    public Tensor? HighResStride8 { get; init; }
}
