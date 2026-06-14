using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Quality;

/// <summary>Casts a weight dictionary to a target dtype according to a <see cref="QualityProfile"/>. Norm scales (1D, ending in <c>.norm</c> / <c>.scale</c>) and biases (1D, ending in <c>.bias</c>) are kept at F16/BF16/F32 minimum — never cast to FP8 or quantized formats.</summary>
public static class QualityProfileApplier
{
    /// <summary>Casts every rank-2+ weight tensor in <paramref name="weights"/> to <paramref name="targetDType"/>. Returns the number of tensors cast (rank-1 norms/biases are skipped). Mutates the dictionary in place; disposes the original tensors after replacing them.</summary>
    public static int Apply(IDictionary<string, Tensor> weights, DType targetDType)
    {
        if (targetDType == DType.F32 || targetDType == DType.F16 || targetDType == DType.BF16 || targetDType.IsFp8)
        {
            return ApplyDirectCast(weights, targetDType);
        }
        if (targetDType.IsQuantized)
        {
            Logs.Warning($"QualityProfileApplier: target dtype {targetDType} is quantized. Quantization happens at load time in the GGUF/quantized loader path; this method is a no-op for quantized targets. Pass the matching loader to the pipeline instead.");
            return 0;
        }
        throw new ArgumentException($"Unsupported target dtype: {targetDType}", nameof(targetDType));
    }

    private static int ApplyDirectCast(IDictionary<string, Tensor> weights, DType targetDType)
    {
        int casts = 0;
        foreach (string key in weights.Keys.ToList())
        {
            Tensor src = weights[key];
            if (src.DType == targetDType) continue;
            if (src.DType.IsQuantized) continue;

            DType effectiveTarget = targetDType;
            if (ShouldKeepHigherPrecision(key, src) && targetDType.IsFp8)
            {
                effectiveTarget = (src.DType == DType.BF16) ? DType.F16 : src.DType;
                if (effectiveTarget == src.DType) continue;
            }

            try
            {
                Tensor cast = src.CastTo(effectiveTarget);
                weights[key] = cast;
                src.Dispose();
                casts++;
            }
            catch (Exception ex)
            {
                Logs.Warning($"QualityProfileApplier: failed to cast {key} ({src.DType} → {effectiveTarget}); leaving as-is. {ex.Message}");
            }
        }
        return casts;
    }

    private static bool ShouldKeepHigherPrecision(string key, Tensor src)
    {
        if (src.Shape.Rank == 1) return true;
        string lowered = key.ToLowerInvariant();
        if (lowered.EndsWith(".bias", StringComparison.Ordinal)) return true;
        if (lowered.Contains(".norm", StringComparison.Ordinal)) return true;
        if (lowered.Contains("_norm.", StringComparison.Ordinal)) return true;
        if (lowered.Contains("layernorm", StringComparison.Ordinal)) return true;
        if (lowered.Contains("rmsnorm", StringComparison.Ordinal)) return true;
        if (lowered.Contains("pos_embed", StringComparison.Ordinal)) return true;
        if (lowered.Contains("positional_encoding", StringComparison.Ordinal)) return true;
        return false;
    }
}
