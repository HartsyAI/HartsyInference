using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>The quantized dtypes an expert pack can store, resolved from the names in its manifest. F32 is not one.</summary>
internal static class ExpertPackDTypes
{
    /// <summary>
    /// True when a pack can store <paramref name="dtype"/>: exactly the dtypes <see cref="Resolve"/> reads and the quantizer writes.
    /// </summary>
    public static bool IsPackDType(DType dtype) =>
        dtype == DType.Q8_0 || dtype == DType.Q4_K || dtype == DType.Q5_K || dtype == DType.Q6_K;

    /// <summary>Resolves a manifest dtype name to the stored dtype; unknown names are rejected.</summary>
    /// <exception cref="InvalidDataException">The name is not a dtype a pack can hold.</exception>
    public static DType Resolve(string name) => name switch
    {
        "Q8_0" => DType.Q8_0,
        "Q4_K" => DType.Q4_K,
        "Q5_K" => DType.Q5_K,
        "Q6_K" => DType.Q6_K,
        _ => throw new InvalidDataException($"Expert pack dtype '{name}' is not supported."),
    };
}
