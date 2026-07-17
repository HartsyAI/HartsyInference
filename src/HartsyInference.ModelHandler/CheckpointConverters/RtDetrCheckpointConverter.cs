using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Converts a <c>PekingU/rtdetr_r*vd</c> RT-DETR checkpoint into the weight dict the
/// <c>RtDetrModel</c> loads. RT-DETR stores every conv's BatchNorm as separate
/// <c>running_mean/var/weight/bias</c> buffers (three naming conventions —
/// <c>*.normalization.*</c> in the ResNet backbone, <c>*.norm.*</c> in the CCFM conv layers, and the
/// <c>Sequential(conv, bn)</c> index <c>*.1.*</c> in the input projections). This folds each BN into
/// its preceding conv (giving the conv a bias), drops the BN buffers + <c>num_batches_tracked</c>, and
/// passes every other tensor (Linear / LayerNorm / embeddings / heads) through unchanged.</summary>
public static class RtDetrCheckpointConverter
{
    private const float BatchNormEps = 1e-5f;

    /// <summary>Folds BN into conv and returns the converted weight dictionary.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> output = new();
        HashSet<string> consumed = new(StringComparer.Ordinal);

        foreach (string key in all.Keys)
        {
            if (!key.EndsWith(".running_var", StringComparison.Ordinal))
                continue;
            string bnPrefix = key[..^".running_var".Length];
            string convPrefix = ConvPrefixFor(bnPrefix);
            string convWeightKey = $"{convPrefix}.weight";
            if (!all.ContainsKey(convWeightKey))
                throw new InvalidOperationException($"RT-DETR converter: BN at '{bnPrefix}' has no conv weight '{convWeightKey}'.");

            (Tensor foldedWeight, Tensor foldedBias) = FoldBatchNorm(
                all[convWeightKey],
                all[$"{bnPrefix}.weight"],
                all[$"{bnPrefix}.bias"],
                all[$"{bnPrefix}.running_mean"],
                all[$"{bnPrefix}.running_var"]);

            output[convWeightKey] = foldedWeight;
            output[$"{convPrefix}.bias"] = foldedBias;

            consumed.Add(convWeightKey);
            consumed.Add($"{bnPrefix}.weight");
            consumed.Add($"{bnPrefix}.bias");
            consumed.Add($"{bnPrefix}.running_mean");
            consumed.Add($"{bnPrefix}.running_var");
            consumed.Add($"{bnPrefix}.num_batches_tracked");
        }

        foreach ((string key, Tensor tensor) in all)
        {
            if (consumed.Contains(key))
                continue;
            if (key.EndsWith(".num_batches_tracked", StringComparison.Ordinal))
                continue;
            output[key] = tensor;
        }

        return output;
    }

    private static string ConvPrefixFor(string bnPrefix)
    {
        if (bnPrefix.EndsWith(".normalization", StringComparison.Ordinal))
            return $"{bnPrefix[..^".normalization".Length]}.convolution";
        if (bnPrefix.EndsWith(".norm", StringComparison.Ordinal))
            return $"{bnPrefix[..^".norm".Length]}.conv";
        if (bnPrefix.EndsWith(".1", StringComparison.Ordinal))
            return $"{bnPrefix[..^".1".Length]}.0";
        throw new InvalidOperationException($"RT-DETR converter: unrecognized BatchNorm prefix '{bnPrefix}'.");
    }

    private static unsafe (Tensor Weight, Tensor Bias) FoldBatchNorm(Tensor convWeight, Tensor gamma, Tensor beta,
        Tensor runningMean, Tensor runningVar)
    {
        Tensor w = EnsureF32(convWeight);
        Tensor g = EnsureF32(gamma);
        Tensor b = EnsureF32(beta);
        Tensor mean = EnsureF32(runningMean);
        Tensor var = EnsureF32(runningVar);

        int outC = (int)w.Shape[0];
        int perFilter = (int)(w.ElementCount / outC);

        Tensor foldedWeight = new Tensor(w.Shape, DType.F32);
        Tensor foldedBias = new Tensor(new TensorShape(outC), DType.F32);
        ReadOnlySpan<float> ws = w.AsSpan<float>(), gs = g.AsSpan<float>(), bs = b.AsSpan<float>();
        ReadOnlySpan<float> ms = mean.AsSpan<float>(), vs = var.AsSpan<float>();
        Span<float> fw = foldedWeight.AsSpan<float>(), fb = foldedBias.AsSpan<float>();
        for (int o = 0; o < outC; o++)
        {
            float scale = gs[o] / MathF.Sqrt(vs[o] + BatchNormEps);
            fb[o] = bs[o] - ms[o] * scale;
            int baseIdx = o * perFilter;
            for (int i = 0; i < perFilter; i++)
                fw[baseIdx + i] = ws[baseIdx + i] * scale;
        }
        return (foldedWeight, foldedBias);
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
