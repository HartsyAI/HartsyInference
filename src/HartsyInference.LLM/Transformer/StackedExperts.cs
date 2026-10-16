using System.Globalization;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Splits a MoE layer's stacked expert tensors into the per-expert <c>[out, in]</c> weights
/// <see cref="MoeFeedForward.LoadWeights"/> reads. Each expert is a zero-copy row view that roots its source
/// (<see cref="Tensor.SliceRows"/>), so splitting never duplicates the weights and a source the caller owns outlives
/// the views. Rows are contiguous in float and block-quantized layouts alike, so the split is dtype-agnostic.</summary>
public static class StackedExperts
{
    /// <summary>Splits <paramref name="stacked"/> (<c>[E, outDim, inDim]</c>, or already flattened to <c>[E·outDim, inDim]</c>)
    /// into <paramref name="experts"/> views stored under <paramref name="targetFormat"/>, whose <c>{0}</c> is the expert index.</summary>
    public static void Split(IDictionary<string, Tensor> destination, Tensor stacked, string targetFormat, int experts, int outDim, int inDim)
    {
        Tensor flat = Flatten(stacked, experts, outDim, inDim);
        for (int x = 0; x < experts; x++)
        {
            destination[Name(targetFormat, x)] = flat.SliceRows((long)x * outDim, outDim);
        }
    }

    /// <summary>Splits a fused gate+up stack (<c>[E, 2·inter, inDim]</c>, each expert's gate rows then its up rows, as ggml's
    /// <c>gate_up_exps</c> and Hugging Face's <c>experts.gate_up_proj</c> store them) into per-expert gate and up views.</summary>
    public static void SplitGateUp(IDictionary<string, Tensor> destination, Tensor stacked, string gateFormat, string upFormat,
        int experts, int inter, int inDim)
    {
        Tensor flat = Flatten(stacked, experts, 2 * inter, inDim);
        for (int x = 0; x < experts; x++)
        {
            destination[Name(gateFormat, x)] = flat.SliceRows((long)x * 2 * inter, inter);
            destination[Name(upFormat, x)] = flat.SliceRows((long)x * 2 * inter + inter, inter);
        }
    }

    private static Tensor Flatten(Tensor stacked, int experts, int outDim, int inDim)
    {
        ArgumentNullException.ThrowIfNull(stacked);
        if (stacked.Shape.ElementCount != (long)experts * outDim * inDim)
        {
            throw new InvalidDataException(
                $"A stacked expert tensor of shape {stacked.Shape} does not hold {experts} experts of [{outDim}, {inDim}].");
        }
        return stacked.Shape.Rank == 2 && stacked.Shape[0] == (long)experts * outDim
            ? stacked : stacked.Reshape(new TensorShape((long)experts * outDim, inDim));
    }

    private static string Name(string format, int expert) => format.Replace("{0}", expert.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal);
}
