using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Per-query mask and class logits from the SAM 3 segmentation head and scorer (batch 1, F32).</summary>
/// <param name="MaskLogits">Low-resolution mask logits, <c>[1, NumQueries, MaskResolution, MaskResolution]</c>.</param>
/// <param name="ClassLogits">Dot-product class logits (clamped), <c>[1, NumQueries]</c>.</param>
public sealed record Sam3InstancePredictions(Tensor MaskLogits, Tensor ClassLogits);
