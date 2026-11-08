using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Image-path SAM 3 weights grouped by owning component, keyed relative to each component prefix.</summary>
/// <remarks>The tensors are owned by the caller that supplied the state dict to <see cref="Sam3Converter.Convert"/>.</remarks>
public sealed class Sam3WeightGroups
{
    /// <summary>ViT trunk weights, relative to <c>detector.backbone.vision_backbone.trunk.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> VisionTrunk { get; init; }

    /// <summary>FPN neck weights for the kept scale levels, relative to <c>detector.backbone.vision_backbone.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> VisionNeck { get; init; }

    /// <summary>Text tower and resizer weights, relative to <c>detector.backbone.language_backbone.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> TextEncoder { get; init; }

    /// <summary>Fusion encoder weights, relative to <c>detector.transformer.encoder.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> FusionEncoder { get; init; }

    /// <summary>Decoder weights, relative to <c>detector.transformer.decoder.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> Decoder { get; init; }

    /// <summary>Geometry encoder weights, relative to <c>detector.geometry_encoder.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> GeometryEncoder { get; init; }

    /// <summary>Dot-product scorer weights, relative to <c>detector.dot_prod_scoring.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> Scoring { get; init; }

    /// <summary>Segmentation head weights, relative to <c>detector.segmentation_head.</c>.</summary>
    public required IReadOnlyDictionary<string, Tensor> SegmentationHead { get; init; }

    /// <summary>Full names of checkpoint keys the image path deliberately does not load, sorted ordinally.</summary>
    public required IReadOnlyList<string> UnusedKeys { get; init; }
}
