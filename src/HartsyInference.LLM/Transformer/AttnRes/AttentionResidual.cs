using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.AttnRes;

/// <summary>Block attention residual (Kimi AttnRes, arXiv 2603.15031), shared by any family that mixes block summaries over
/// depth. A learned pseudo-query attends over the sources: keys are <c>RMSNorm(source)</c>, each logit is the dot of its
/// key with the query, and the output is the softmax-weighted sum of the raw sources.</summary>
public static class AttentionResidual
{
    /// <summary>
    /// Block attention residual mix (Kimi AttnRes) for one token. <paramref name="sources"/> are the completed block
    /// summaries plus the current partial block sum (the embedding is the first source), each of width d. Keys are
    /// <c>RMSNorm(source)</c> with <paramref name="normWeight"/>; the logit of source n is the dot of its normalized key
    /// with <paramref name="query"/> (the layer's learned pseudo-query, <c>proj.weight</c>). The output is the softmax-
    /// weighted sum of the raw sources over the depth axis.
    /// </summary>
    public static void MixReference(IReadOnlyList<float[]> sources, ReadOnlySpan<float> normWeight, ReadOnlySpan<float> query,
        Span<float> output, float eps = 1e-6f)
    {
        if (sources.Count == 0) throw new ArgumentException("At least one source is required.", nameof(sources));
        int d = sources[0].Length;
        if (normWeight.Length != d || query.Length != d || output.Length != d)
            throw new ArgumentException($"normWeight, query and output must have width {d}.");
        Span<float> logits = stackalloc float[sources.Count];
        float max = float.NegativeInfinity;
        for (int n = 0; n < sources.Count; n++)
        {
            ReadOnlySpan<float> s = sources[n];
            if (s.Length != d) throw new ArgumentException("All sources must have the same width.", nameof(sources));
            float ss = 0f;
            for (int j = 0; j < d; j++) ss += s[j] * s[j];
            float inv = 1f / MathF.Sqrt(ss / d + eps);
            float logit = 0f;
            for (int j = 0; j < d; j++) logit += s[j] * inv * normWeight[j] * query[j];
            logits[n] = logit;
            max = MathF.Max(max, logit);
        }
        float total = 0f;
        for (int n = 0; n < sources.Count; n++)
        {
            logits[n] = MathF.Exp(logits[n] - max);
            total += logits[n];
        }
        output.Clear();
        for (int n = 0; n < sources.Count; n++)
        {
            float p = logits[n] / total;
            ReadOnlySpan<float> s = sources[n];
            for (int j = 0; j < d; j++) output[j] += p * s[j];
        }
    }

    /// <summary>The same mix on a backend, composed from shared ops: <c>RmsNorm</c> of the sources, a <c>MatMul</c> with the
    /// query for the logits, <c>SoftmaxRows</c>, then a <c>MatMul</c> of the weights with the raw sources.</summary>
    /// <param name="sources">F32 <c>[n, d]</c>: the block summaries plus the current partial sum, one row each.</param>
    /// <param name="normWeight">F32 <c>[d]</c> RMSNorm weight of the keys.</param>
    /// <param name="query">F32 <c>[d, 1]</c> the layer's learned pseudo-query.</param>
    /// <param name="output">F32 <c>[1, d]</c>.</param>
    public static void Mix(IBackend backend, Tensor sources, Tensor normWeight, Tensor query, Tensor output, float eps = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(backend);
        long d = output.ElementCount;
        if (d < 1 || sources.ElementCount % d != 0)
            throw new ArgumentException($"AttentionResidual.Mix sources {sources.Shape} are not whole rows of width {d}.", nameof(sources));
        int n = checked((int)(sources.ElementCount / d));
        if (normWeight.ElementCount != d || query.ElementCount != d)
            throw new ArgumentException($"AttentionResidual.Mix normWeight and query must hold {d} values.", nameof(query));
        using Tensor normed = new(new TensorShape(n, d), DType.F32);
        backend.RmsNorm(normed, sources, normWeight, eps);
        using Tensor logits = new(new TensorShape(n, 1), DType.F32);
        backend.MatMul(logits, normed, query);
        using Tensor weights = new(new TensorShape(1, n), DType.F32);
        backend.SoftmaxRows(weights, logits, 1, n);
        backend.MatMul(output, weights, sources);
    }
}
