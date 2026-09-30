namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The execution path a backend picked for a recipe, the scratch memory it needs and the activation precision it multiplies at.</summary>
public readonly record struct QuantExecutionPlan(QuantExecutionKind Kind, long WorkspaceBytes, QuantActivationPrecision Activation = QuantActivationPrecision.Bf16)
{
    /// <summary>Weight-and-activation bits as a checkpoint author writes them, e.g. <c>W4A16</c> for 4-bit weights dequantized under BF16 activations.</summary>
    public string Label(QuantRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        int weightBits = recipe.Encoding switch
        {
            QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4 or QuantEncoding.AffineInt4 => 4,
            _ => 8,
        };
        return $"W{weightBits}A{(Activation == QuantActivationPrecision.Fp4 ? 4 : 16)}";
    }
}
