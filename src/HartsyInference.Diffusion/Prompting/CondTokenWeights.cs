using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>SwarmUI's <see cref="PromptWeightingMode.CondScale"/> mechanism: the prompt is encoded at weight 1.0 and
/// each token's conditioning ROW is then multiplied by that token's weight — <c>multiply_cond_by_token_weights</c>
/// (<c>SwarmText.py:256-271</c>). Rows are matched right-aligned, <c>pos = condLen − weights.Length + i</c>, so a
/// pipeline that trims a template prefix off the encoder output still lands each weight on its own token; positions
/// outside the cond are skipped. A prompt-wide uniform weight with nothing left in range instead scales the whole
/// cond and its <c>pooled_output</c>.</summary>
/// <remarks>Every method here returns NEW tensors and never mutates its input. Conditioning is cached across
/// generations in several pipelines under a key made only of token ids, and those ids are identical with and without
/// weighting under this mechanism — scaling in place would hand the next plain request the weighted cond.</remarks>
public static class CondTokenWeights
{
    /// <summary>Applies the whole <c>encode_leaves</c> weighting decision (<c>SwarmText.py:554-583</c>) to an
    /// already-encoded conditioning pair: per-token row scaling when any weight lands inside the cond, otherwise the
    /// uniform whole-cond + pooled fallback, otherwise nothing at all.</summary>
    /// <param name="cond">The conditioning <c>[batch, seq, hidden]</c> (or <c>[seq, hidden]</c>) as encoded at weight 1.</param>
    /// <param name="pooled">The family's pooled vector, scaled only on the uniform fallback; null when there is none.</param>
    /// <returns>Replacements the caller owns and must dispose; a null member means "keep using the original".</returns>
    public static CondScaleResult Apply(IBackend backend, Tensor cond, Tensor? pooled, WeightedTokenSequence sequence)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        Tensor? scaled = ScaleRightAligned(backend, cond, sequence.Weights);
        if (scaled is not null)
        {
            return new CondScaleResult(scaled, null);
        }
        float? uniform = sequence.UniformWeight;
        if (uniform is null || uniform.Value == 1f)
        {
            return default;
        }
        return ScaleUniform(backend, cond, pooled, uniform.Value);
    }

    /// <summary>Multiplies each in-range conditioning row by its token's weight, right-aligned against
    /// <paramref name="weights"/>. Returns null — and touches nothing — when no weight other than 1 lands inside the
    /// cond, which is what keeps an all-ones prompt byte-identical to the plain encode.</summary>
    public static Tensor? ScaleRightAligned(IBackend backend, Tensor cond, ReadOnlySpan<float> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cond);
        if (cond.DType != DType.F32)
        {
            throw new ArgumentException($"CondScale weighting needs an F32 conditioning tensor, got {cond.DType}.", nameof(cond));
        }
        if (cond.Shape.Rank < 2)
        {
            throw new ArgumentException($"Conditioning must be at least rank 2, got rank {cond.Shape.Rank}.", nameof(cond));
        }
        int rows = (int)cond.Shape[cond.Shape.Rank - 2];
        int batch = (int)(cond.ElementCount / (rows * cond.Shape[cond.Shape.Rank - 1]));
        int offset = rows - weights.Length;
        float[] rowScales = new float[checked(batch * rows)];
        Array.Fill(rowScales, 1f);
        bool any = false;
        for (int i = 0; i < weights.Length; i++)
        {
            int pos = offset + i;
            if (weights[i] == 1f || pos < 0 || pos >= rows)
            {
                continue;
            }
            for (int b = 0; b < batch; b++)
            {
                rowScales[(b * rows) + pos] = weights[i];
            }
            any = true;
        }
        if (!any)
        {
            return null;
        }
        using Tensor mask = new Tensor(new TensorShape(rowScales.Length), DType.F32);
        rowScales.CopyTo(mask.AsSpan<float>());
        Tensor result = new Tensor(cond.Shape, DType.F32);
        try
        {
            backend.MaskRows(result, cond, mask);
            // The CUDA path leaves the output as a device activation; the text-encoder stage frees those before the
            // denoise loop runs, so pull it back to the host now rather than handing the caller a reclaimable buffer.
            result.AsReadOnlySpan<float>();
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return result;
    }

    /// <summary>Scales the whole conditioning — and the pooled vector when there is one — by a single weight, the
    /// fallback SwarmUI takes when every leaf carries the same weight and no per-token position survived
    /// (<c>SwarmText.py:578-583</c>). Unconditional: the <c>w == 1</c> short-circuit belongs to <see cref="Apply"/>.</summary>
    public static CondScaleResult ScaleUniform(IBackend backend, Tensor cond, Tensor? pooled, float weight)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cond);
        Tensor scaledCond = new Tensor(cond.Shape, cond.DType);
        Tensor? scaledPooled = null;
        try
        {
            backend.Scale(scaledCond, cond, weight);
            scaledCond.AsReadOnlySpan<float>();
            if (pooled is not null)
            {
                scaledPooled = new Tensor(pooled.Shape, pooled.DType);
                backend.Scale(scaledPooled, pooled, weight);
                scaledPooled.AsReadOnlySpan<float>();
            }
        }
        catch
        {
            scaledPooled?.Dispose();
            scaledCond.Dispose();
            throw;
        }
        return new CondScaleResult(scaledCond, scaledPooled);
    }
}
