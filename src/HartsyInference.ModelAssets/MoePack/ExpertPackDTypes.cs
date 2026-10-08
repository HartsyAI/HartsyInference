using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>The quant dtypes an expert pack can store, resolved from the names in its manifest.</summary>
internal static class ExpertPackDTypes
{
    /// <summary>Resolves a manifest dtype name to the stored dtype; unknown names are rejected.</summary>
    /// <exception cref="InvalidDataException">The name is not a dtype a pack can hold.</exception>
    public static DType Resolve(string name) => name switch
    {
        "Q8_0" => DType.Q8_0,
        "Q4_K" => DType.Q4_K,
        "Q5_K" => DType.Q5_K,
        "Q6_K" => DType.Q6_K,
        "F32" => DType.F32,
        _ => throw new InvalidDataException($"Expert pack dtype '{name}' is not supported."),
    };
}
