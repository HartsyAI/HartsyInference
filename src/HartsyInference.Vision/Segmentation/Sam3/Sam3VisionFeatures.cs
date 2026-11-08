using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Output of the SAM 3 vision backbone for one image. All tensors are F32, batch-first.</summary>
/// <param name="Features">FPN maps kept after scalp, each <c>[1, DModel, H_i, W_i]</c>, finest first (288, 144, 72).</param>
/// <param name="Positions">Sine position encodings matching <paramref name="Features"/>, same shapes.</param>
public sealed record Sam3VisionFeatures(Tensor[] Features, Tensor[] Positions);
