namespace HartsyInference.Core.Tensors.Quant;

/// <summary>Chooses how a recipe-carrying weight runs today: dequantize to BF16 into a bounded workspace, then a BF16 GEMM.</summary>
/// <remarks>NVFP4 checkpoints carry an <see cref="QuantRecipe.InputScale"/> for W4A4, but no shipped backend has a block-scaled FP4 GEMM (Blackwell only), so it is
/// left unread and the plan is W4A16. EXL3 needs a trellis decode that is not written yet.</remarks>
public static class QuantExecutionPolicy
{
    /// <summary>The BF16 dequant plan for <paramref name="recipe"/>.</summary>
    /// <exception cref="NotSupportedException">The encoding has no dequant path yet.</exception>
    public static QuantExecutionPlan Plan(QuantRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        if (!SupportsBf16Dequant(recipe.Encoding))
            throw new NotSupportedException($"{recipe.Encoding} has no BF16 dequant path.");
        return new QuantExecutionPlan(QuantExecutionKind.DequantBf16, recipe.LogicalRows * recipe.LogicalCols * 2, QuantActivationPrecision.Bf16);
    }

    /// <summary>Whether <paramref name="encoding"/> can be dequantized to BF16 by <see cref="Plan"/>.</summary>
    public static bool SupportsBf16Dequant(QuantEncoding encoding) =>
        encoding is QuantEncoding.Fp8E4M3BlockE8M0 or QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4
            or QuantEncoding.AffineInt4 or QuantEncoding.AffineInt8;
}
