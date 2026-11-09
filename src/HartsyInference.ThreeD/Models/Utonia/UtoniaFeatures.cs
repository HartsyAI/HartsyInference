using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Encoder output: features <c>[M, C]</c> the caller must dispose, and their grid coords <c>[M, 3]</c> with no batch index.</summary>
public readonly record struct UtoniaFeatures(Tensor Feats, int[] GridCoord);
