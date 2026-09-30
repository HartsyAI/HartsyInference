namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The EXL3 side tensors a trellis weight needs to decode. Borrowed, like every recipe tensor.</summary>
public sealed record Exl3Companions(Tensor Suh, Tensor Svh, Tensor Mcg, int Bits);
