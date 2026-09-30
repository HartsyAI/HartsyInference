namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The precision activations are multiplied at, so a plan can say W4A16 rather than let a 4-bit weight imply 4-bit math.</summary>
public enum QuantActivationPrecision
{
    /// <summary>BF16 activations: the weight is dequantized and the GEMM is an ordinary BF16 one.</summary>
    Bf16,

    /// <summary>4-bit block-scaled activations (W4A4), which needs the Blackwell block-scaled GEMM.</summary>
    Fp4,
}
