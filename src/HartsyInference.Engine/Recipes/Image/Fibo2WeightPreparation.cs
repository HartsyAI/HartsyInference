using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Prepares a Fibo 2.0 checkpoint's weights for the denoiser and text encoder: BF16 is cast to F16, and the Comfy-Org
/// <c>int8_tensorwise</c> + ConvRot builds keep their packed linears (the per-row scale and ConvRot group ride on
/// <see cref="Tensor.QuantInfo"/>) where every executing backend can read them, and are widened on the host where one cannot.</summary>
/// <remarks><para>The fused tensors the blocks split at load (<c>attention.qkv</c>, <c>to_kv</c>, <c>w13</c>) split correctly
/// while packed: ConvRot rotates the <b>input</b> dimension and the scale is per <b>output</b> row, so a row range of the
/// weight is a self-contained quantized weight whose scale is the matching slice. <c>CheckpointConvertUtils.SplitRows</c> and
/// <c>ZImageBlock.SplitQkv</c> narrow <see cref="QuantWeightInfo"/> that way, so nothing here needs to split anything.</para></remarks>
internal static class Fibo2WeightPreparation
{
    /// <summary>Casts BF16 weights to F16, which the CPU and GPU kernels take directly, as the Flux.2 recipe does. The copies join
    /// <paramref name="owned"/> so they outlive the transformer that borrows them. An <c>int8_tensorwise</c> matrix passes through
    /// untouched with its <see cref="Tensor.QuantInfo"/>; other dtypes pass through as they are.</summary>
    /// <exception cref="NotSupportedException">A tensor is neither dense nor a ComfyUI int8_tensorwise linear with its scale.</exception>
    internal static Dictionary<string, Tensor> HalfPrecisionWeights(IReadOnlyDictionary<string, Tensor> source, List<IDisposable> owned)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(owned);
        Dictionary<string, Tensor> weights = new(source.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in source)
        {
            Tensor tensor = pair.Value;
            if (IsPackedInt8(tensor))
            {
                weights[pair.Key] = tensor;
                continue;
            }
            bool dense = tensor.DType == DType.F32 || tensor.DType == DType.F16 || tensor.DType == DType.BF16;
            if (!dense)
            {
                string detail = tensor.DType == DType.I8
                    ? "an int8 weight with no int8_tensorwise row scale (its '.weight_scale' or '.comfy_quant' companion is missing)"
                    : tensor.DType.ToString();
                throw new NotSupportedException(
                    $"Fibo 2.0 tensor '{pair.Key}' is {detail}. Only dense (F32, F16, BF16) and ComfyUI int8_tensorwise ConvRot "
                    + "checkpoints load; use the bf16 or int8_convrot build.");
            }
            if (tensor.DType == DType.BF16)
            {
                Tensor half = tensor.CastTo(DType.F16);
                owned.Add(half);
                weights[pair.Key] = half;
            }
            else
            {
                weights[pair.Key] = tensor;
            }
        }
        return weights;
    }

    /// <summary>Whether any weight is a packed int8 matrix, which picks the int8 text encoder build.</summary>
    internal static bool HasPackedWeights(IReadOnlyDictionary<string, Tensor> weights) =>
        weights.Values.Any(IsPackedInt8);

    /// <summary>Widens, in place, every packed weight that some backend in <paramref name="backends"/> cannot hold packed (the CPU
    /// backend, and any device without the int8 kernels). The widened copies are owned by <paramref name="owned"/>.</summary>
    /// <param name="wideDType">F16 for the denoiser, matching the BF16-to-F16 cast the dense build gets; BF16 for the encoder.</param>
    internal static void WidenUnsupported(IDictionary<string, Tensor> weights, IEnumerable<IBackend> backends, DType wideDType,
        List<IDisposable> owned)
    {
        QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackends(weights, backends, wideDType);
        owned.Add(prepared);
    }

    private static bool IsPackedInt8(Tensor tensor) =>
        tensor.DType == DType.I8 && tensor.Shape.Rank == 2
        && tensor.QuantInfo is { Format: "int8_tensorwise", RowScale: not null };
}
