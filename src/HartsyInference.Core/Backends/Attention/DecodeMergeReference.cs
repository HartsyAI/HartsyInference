using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The definition of the speculative-decode merges: folding a step's own rows into an attention result that
/// was computed over the committed prefix alone. Written from Comfy Kitchen's torch definitions at the pinned commit
/// (benchmarks/attention/protocol.json) and checked against their outputs.</summary>
/// <remarks>
/// <para>A verify step decodes several rows at once. Each row first attends the committed prefix of the cache, without a
/// causal rule, which yields its output and the natural-log sum of its exponentiated scores (the LSE). The rows of the
/// step are not in the cache yet, so a second pass adds them: row <c>r</c> also attends the step rows its mask names, and
/// the two softmaxes are joined exactly through the LSE.</para>
/// <para>With <c>m = max(lse, max step score)</c>, the prefix weighs <c>exp(lse − m)</c>, a step row <c>exp(score − m)</c>,
/// and the merged row is the weighted sum of the prefix output and the step values over the sum of the weights. That is
/// the softmax over prefix and step keys together, with the prefix counted once.</para>
/// <para>F32 throughout, plain loops, no approximation: a GPU port is compared with it.</para>
/// </remarks>
public static class DecodeMergeReference
{
    /// <summary>The most rows a step holds: one mask bit per row.</summary>
    public const int MaxRows = 32;

    /// <summary>The causal step mask: row <c>r</c> attends step rows 0 to <c>r</c>, itself included.</summary>
    public static int[] CausalStepMask(int rows)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(rows, MaxRows);
        int[] mask = new int[rows];
        for (int r = 0; r < rows; r++) mask[r] = r == 31 ? -1 : (1 << (r + 1)) - 1;
        return mask;
    }

    /// <summary><see cref="TreeMerge"/> under <see cref="CausalStepMask"/>: every row attends the step rows up to itself.</summary>
    public static void StepMerge(Tensor merged, Tensor prefixOutput, ReadOnlySpan<float> prefixLse, Tensor query, Tensor key, Tensor value,
        float? scale = null) =>
        TreeMerge(merged, prefixOutput, prefixLse, query, key, value, CausalStepMask((int)query.Shape[2]), scale);

    /// <summary>Folds a step's own rows into a prefix-only result.</summary>
    /// <param name="merged">Output, F32 <c>[B, S, H × D]</c>; may be <paramref name="prefixOutput"/> itself.</param>
    /// <param name="prefixOutput">The step's rows attended over the committed prefix only, F32 <c>[B, S, H × D]</c>.</param>
    /// <param name="prefixLse">The natural-log LSE of those rows' scaled scores, <c>[B, H, S]</c>; negative infinity for a
    /// row whose prefix was empty.</param>
    /// <param name="query">The step's queries, F32 <c>[B, H, S, D]</c>.</param>
    /// <param name="key">The step's own keys, F32 <c>[B, Hkv, S, D]</c>.</param>
    /// <param name="value">The step's own values, F32 <c>[B, Hkv, S, D]</c>.</param>
    /// <param name="mask">One entry per row: bit <c>t</c> set means the row attends step row <c>t</c>.</param>
    /// <param name="scale">The score multiplier; null is <c>D^-1/2</c>.</param>
    /// <remarks>A row with no key at all, an empty prefix and an empty mask, is written as zeros.</remarks>
    /// <exception cref="ArgumentException">An operand does not fit the call.</exception>
    public static void TreeMerge(Tensor merged, Tensor prefixOutput, ReadOnlySpan<float> prefixLse, Tensor query, Tensor key, Tensor value,
        ReadOnlySpan<int> mask, float? scale = null)
    {
        Check(merged, prefixOutput, prefixLse, query, key, value, mask, scale);
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], rows = (int)query.Shape[2], dim = (int)query.Shape[3];
        int kvHeads = (int)key.Shape[1], group = heads / kvHeads;
        float alpha = scale ?? 1f / MathF.Sqrt(dim);
        ReadOnlySpan<float> q = query.AsReadOnlySpan<float>(), k = key.AsReadOnlySpan<float>(), v = value.AsReadOnlySpan<float>();
        ReadOnlySpan<float> prefix = prefixOutput.AsReadOnlySpan<float>();
        Span<float> o = merged.AsSpan<float>();
        Span<float> scores = stackalloc float[MaxRows];
        float[] row = new float[dim];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
            {
                int kh = h / group;
                for (int r = 0; r < rows; r++)
                {
                    int qAt = ((b * heads + h) * rows + r) * dim, outAt = ((b * rows + r) * heads + h) * dim;
                    float lse = prefixLse[(b * heads + h) * rows + r];
                    float max = lse;
                    for (int t = 0; t < rows; t++)
                    {
                        if (((mask[r] >> t) & 1) == 0)
                        {
                            scores[t] = float.NegativeInfinity;
                            continue;
                        }
                        int kAt = ((b * kvHeads + kh) * rows + t) * dim;
                        float dot = 0f;
                        for (int d = 0; d < dim; d++) dot += q[qAt + d] * k[kAt + d];
                        scores[t] = dot * alpha;
                        max = MathF.Max(max, scores[t]);
                    }
                    if (float.IsNegativeInfinity(max))
                    {
                        o.Slice(outAt, dim).Clear();
                        continue;
                    }
                    float prefixWeight = MathF.Exp(lse - max), total = prefixWeight;
                    for (int d = 0; d < dim; d++) row[d] = prefix[outAt + d] * prefixWeight;
                    for (int t = 0; t < rows; t++)
                    {
                        if (float.IsNegativeInfinity(scores[t])) continue;
                        float weight = MathF.Exp(scores[t] - max);
                        total += weight;
                        int vAt = ((b * kvHeads + kh) * rows + t) * dim;
                        for (int d = 0; d < dim; d++) row[d] += weight * v[vAt + d];
                    }
                    for (int d = 0; d < dim; d++) o[outAt + d] = row[d] / total;
                }
            }
    }

    private static void Check(Tensor merged, Tensor prefixOutput, ReadOnlySpan<float> prefixLse, Tensor query, Tensor key, Tensor value,
        ReadOnlySpan<int> mask, float? scale)
    {
        if (query.Shape.Rank != 4 || key.Shape.Rank != 4 || key.Shape != value.Shape)
            throw new ArgumentException($"A merge takes Q [B, H, S, D] and K, V [B, Hkv, S, D]; got Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        long batch = query.Shape[0], heads = query.Shape[1], rows = query.Shape[2], dim = query.Shape[3], kvHeads = key.Shape[1];
        if (query.ElementCount == 0 || key.Shape[0] != batch || key.Shape[2] != rows || key.Shape[3] != dim || kvHeads <= 0 || heads % kvHeads != 0)
            throw new ArgumentException($"The step's keys do not match its queries; got Q={query.Shape}, K={key.Shape}.");
        if (rows > MaxRows) throw new ArgumentException($"A step holds at most {MaxRows} rows, one mask bit each; got {rows}.", nameof(query));
        TensorShape output = new(batch, rows, heads * dim);
        if (merged.Shape != output || prefixOutput.Shape != output)
            throw new ArgumentException($"The prefix output and the merged output are [B, S, H × D] = {output}; got {prefixOutput.Shape} and {merged.Shape}.");
        foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[merged, prefixOutput, query, key, value])
            if (tensor.DType != DType.F32) throw new ArgumentException("The merge reference reads and writes F32.");
        if (prefixLse.Length != batch * heads * rows)
            throw new ArgumentException($"The prefix LSE holds {prefixLse.Length} values; [B, H, S] needs {batch * heads * rows}.", nameof(prefixLse));
        if (mask.Length != rows) throw new ArgumentException($"The mask holds {mask.Length} rows; the step has {rows}.", nameof(mask));
        for (int r = 0; r < rows; r++)
            if (rows < 32 && (uint)mask[r] >> (int)rows != 0)
                throw new ArgumentException($"Mask row {r} names a step row past the {rows} the step has.", nameof(mask));
        if (scale is { } s && !float.IsFinite(s)) throw new ArgumentException("The scale must be finite.", nameof(scale));
        foreach (float lse in prefixLse)
            if (float.IsNaN(lse) || float.IsPositiveInfinity(lse))
                throw new ArgumentException($"A prefix LSE is finite or negative infinity; got {lse}.", nameof(prefixLse));
    }
}
