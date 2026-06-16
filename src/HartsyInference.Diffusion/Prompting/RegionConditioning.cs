using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>One region's conditioning: its already-encoded text embedding, spatial <see cref="Mask"/>, blend <see cref="Weight"/>, and the inclusive-exclusive step window <c>[StartStep, EndStep)</c> over which it is active.</summary>
public readonly record struct RegionConditioning(Tensor Cond, RegionMask Mask, float Weight, int StartStep, int EndStep);
