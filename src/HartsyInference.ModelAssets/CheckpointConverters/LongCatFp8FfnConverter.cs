using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>The weights the LongCat reference runs in FP8 (<c>enable_fp8_ffn</c>, quantization.py @ 58eb336): the SwiGLU's two input projections,
/// <c>ffn.w1</c> and <c>ffn.w3</c>, of blocks 6 and 9 to 47, taken from the BF16 weights. Each becomes an FP8 E4M3 weight with one scale per
/// tensor, its largest magnitude over 448.</summary>
/// <remarks>Weights only. The reference runs those layers through Transformer Engine, which also quantizes their activations with delayed
/// scaling; neither is reproduced here, so no parity with the reference's FP8 mode is claimed.</remarks>
public static class LongCatFp8FfnConverter
{
    /// <summary>The reference's <c>DEFAULT_FP8_FFN_BLOCKS</c>.</summary>
    public static IReadOnlyList<int> Blocks { get; } = [6, .. Enumerable.Range(9, 39)];

    /// <summary>The weight keys of <paramref name="blocks"/> (the reference's list when null).</summary>
    public static IEnumerable<string> Keys(IReadOnlyList<int>? blocks = null) =>
        (blocks ?? Blocks).SelectMany(static block => new[] { $"blocks.{block}.ffn.w1.weight", $"blocks.{block}.ffn.w3.weight" });

    /// <summary>Replaces each selected weight in <paramref name="weights"/> with its FP8 form, adding the new tensors to
    /// <paramref name="owned"/>, which the caller disposes; a replaced tensor that was in <paramref name="owned"/> is disposed here. Throws,
    /// before changing anything, unless every selected weight is present and BF16. Returns the number replaced.</summary>
    public static int Convert(IDictionary<string, Tensor> weights, List<Tensor> owned, IReadOnlyList<int>? blocks = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(owned);
        string[] keys = [.. Keys(blocks)];
        foreach (string key in keys)
        {
            if (!weights.TryGetValue(key, out Tensor? weight))
                throw new InvalidDataException($"The FP8 FFN selection needs '{key}', which this checkpoint does not hold.");
            if (weight.DType != DType.BF16)
                throw new NotSupportedException($"The FP8 FFN selection quantizes BF16 weights; '{key}' is {weight.DType}.");
        }
        foreach (string key in keys)
        {
            Tensor source = weights[key];
            using Tensor wide = source.CastTo(DType.F32);
            Tensor packed = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(wide);
            owned.Add(packed);
            weights[key] = packed;
            if (owned.Remove(source))
                source.Dispose();
        }
        return keys.Length;
    }
}
