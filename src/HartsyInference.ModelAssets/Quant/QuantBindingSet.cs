namespace HartsyInference.ModelAssets.Quant;

/// <summary>What <see cref="QuantCompanionBinder.Bind"/> decided for a whole inventory.</summary>
/// <param name="Bindings">Weight key to binding, for every block-scaled weight.</param>
/// <param name="PerTensorFp8">FP8 weights whose per-tensor scalar scale the existing <c>ApplyFp8ScaledDequant</c> path folds.</param>
public sealed record QuantBindingSet(IReadOnlyDictionary<string, QuantBinding> Bindings, IReadOnlyList<string> PerTensorFp8);
