using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Turns LongCat-Video's per-row INT8 layout into the engine's <c>int8_tensorwise</c> weights. The release stores each quantized Linear as
/// <c>&lt;name&gt;.weight_int8</c> (I8 <c>[out, in]</c>) plus <c>&lt;name&gt;.weight_scale</c> (F32 <c>[out]</c>), <c>W = int8 · scale[:, None]</c>,
/// with no rotation and no descriptor; everything else (norms, biases, the patch embedding, <c>final_layer.linear</c>) stays dense. The
/// converter renames <c>.weight_int8</c> to <c>.weight</c>, folds the scale onto <see cref="Tensor.QuantInfo"/> as an unrotated
/// <c>int8_tensorwise</c> weight, and drops the
/// companion, so the quantized-weight policy and a backend's resident int8 path take it as any other int8_tensorwise weight.</summary>
/// <remarks>The scale tensors are borrowed from the checkpoint that produced them and live as long as it does. The quantized tensors of the
/// input are annotated in place (their <see cref="Tensor.QuantInfo"/> is set on the tensors passed in), so a caller that shares that dictionary
/// sees the annotation too. In an INT8 layout a scale without its quantized weight is refused.</remarks>
public static class LongCatInt8Converter
{
    /// <summary>Suffix of a quantized weight's key.</summary>
    public const string QuantizedSuffix = ".weight_int8";

    /// <summary>Suffix of its per-row scale.</summary>
    public const string ScaleSuffix = ".weight_scale";

    /// <summary>True when any key ends in <see cref="QuantizedSuffix"/>.</summary>
    public static bool IsInt8Layout(IEnumerable<string> keys) => keys.Any(static key => key.EndsWith(QuantizedSuffix, StringComparison.Ordinal));

    /// <summary>True for the layers the reference runs on float32 input: the time embedder's two linears, and every adaLN modulation linear (block,
    /// audio, final). Under autocast in longcat_video_dit_avatar.py (lines 130-133, 166-168, 499-500) and FinalLayer_FP32 (blocks.py 159-160).</summary>
    public static bool IsFloat32InputLayer(string denseKey) =>
        denseKey.StartsWith("t_embedder.mlp.", StringComparison.Ordinal)
        || denseKey.Contains("adaLN_modulation.1.", StringComparison.Ordinal);

    /// <summary>Returns a dictionary under the dense key names, each quantized weight carrying its scale. Dense tensors pass through
    /// untouched.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        Dictionary<string, Tensor> result = new(source.Count, StringComparer.Ordinal);
        bool int8 = IsInt8Layout(source.Keys);
        foreach ((string key, Tensor tensor) in source)
        {
            if (int8 && key.EndsWith(ScaleSuffix, StringComparison.Ordinal))
            {
                if (!source.ContainsKey(key[..^ScaleSuffix.Length] + QuantizedSuffix))
                    throw new InvalidDataException($"'{key}' has no '{key[..^ScaleSuffix.Length]}{QuantizedSuffix}'; an orphan dequant scale is refused.");
                continue;
            }
            if (!key.EndsWith(QuantizedSuffix, StringComparison.Ordinal))
            {
                result[key] = tensor;
                continue;
            }

            string baseKey = key[..^QuantizedSuffix.Length];
            string denseKey = baseKey + ".weight";
            if (source.ContainsKey(denseKey))
                throw new InvalidDataException($"The checkpoint holds both '{key}' and '{denseKey}'.");
            if (tensor.DType != DType.I8 || tensor.Shape.Rank != 2)
                throw new InvalidDataException($"'{key}' must be a rank-2 I8 weight; got {tensor.DType} {tensor.Shape}.");
            if (!source.TryGetValue(baseKey + ScaleSuffix, out Tensor? scale))
                throw new InvalidDataException($"'{key}' has no '{baseKey}{ScaleSuffix}', so its dequant scale is unrecoverable.");
            if (scale.DType != DType.F32)
                throw new InvalidDataException($"'{baseKey}{ScaleSuffix}' must be F32; got {scale.DType}.");
            if (scale.ElementCount != tensor.Shape[0] && scale.ElementCount != 1)
                throw new InvalidDataException(
                    $"'{baseKey}{ScaleSuffix}' must hold {tensor.Shape[0]} per-row scales (or one per-tensor scale); got {scale.ElementCount}.");

            // The reference's rounding (quantization.py lines 28-29 cast to the input's dtype). A float32-input layer (the time embedder and the adaLN
            // linears, which the reference runs under autocast) keeps the exact float32 product and a float32 GEMM; every other layer is BF16 input.
            tensor.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, ConvRotGroupSize = 0, ReferenceRounding = true };
            if (IsFloat32InputLayer(denseKey))
                tensor.ForceF32Gemm = true;
            result[denseKey] = tensor;
        }
        return result;
    }
}
