namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Block-diagonal output projection <c>wo_a</c>: each head group projects only its own slice (upstream <c>einsum("bsgd,grd->bsgr")</c>).</summary>
public static class DeepSeekV41GroupedProjection
{
    /// <summary>Computes <c>dest[t,g,r] = sum_d x[t,g,d] * weight[g,r,d]</c>.</summary>
    /// <param name="x">Attention output, <c>[tokens, groups, groupDim]</c>.</param>
    /// <param name="weight">Dequantized <c>wo_a</c>, <c>[groups, rank, groupDim]</c> (the checkpoint's <c>[groups * rank, groupDim]</c> matrix).</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="groups">Head groups (<c>o_groups</c>).</param>
    /// <param name="rank">Per-group low-rank width (<c>o_lora_rank</c>).</param>
    /// <param name="groupDim">Input width per group (<c>heads * head_dim / groups</c>).</param>
    /// <param name="dest">Receives <c>[tokens, groups, rank]</c>, ready to feed <c>wo_b</c> flattened.</param>
    public static void Apply(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, int tokens, int groups, int rank, int groupDim, Span<float> dest)
    {
        if (x.Length != (long)tokens * groups * groupDim) throw new ArgumentException("x must hold tokens x groups x groupDim values.", nameof(x));
        if (weight.Length != (long)groups * rank * groupDim) throw new ArgumentException("weight must hold groups x rank x groupDim values.", nameof(weight));
        if (dest.Length != (long)tokens * groups * rank) throw new ArgumentException("dest must hold tokens x groups x rank values.", nameof(dest));
        for (int t = 0; t < tokens; t++)
            for (int g = 0; g < groups; g++)
            {
                ReadOnlySpan<float> xv = x.Slice((t * groups + g) * groupDim, groupDim);
                Span<float> outRow = dest.Slice((t * groups + g) * rank, rank);
                for (int r = 0; r < rank; r++)
                {
                    ReadOnlySpan<float> w = weight.Slice((g * rank + r) * groupDim, groupDim);
                    float sum = 0f;
                    for (int d = 0; d < groupDim; d++) sum += xv[d] * w[d];
                    outRow[r] = sum;
                }
            }
    }

    /// <summary>The same projection with the weight held as a <see cref="DeepSeekV41Weight"/>, decoded one group (or less) at a time so a stored weight is never widened whole.</summary>
    public static void Apply(ReadOnlySpan<float> x, DeepSeekV41Weight weight, int tokens, int groups, int rank, int groupDim, Span<float> dest)
    {
        if (weight.IsWidened)
        {
            Apply(x, weight.ToF32(), tokens, groups, rank, groupDim, dest);
            return;
        }
        if (x.Length != (long)tokens * groups * groupDim) throw new ArgumentException("x must hold tokens x groups x groupDim values.", nameof(x));
        if (weight.Rows != (long)groups * rank || weight.Cols != groupDim) throw new ArgumentException("weight must be [groups * rank, groupDim].", nameof(weight));
        if (dest.Length != (long)tokens * groups * rank) throw new ArgumentException("dest must hold tokens x groups x rank values.", nameof(dest));
        int window = DeepSeekV41Weight.WindowRows(groupDim, groups * rank);
        float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent(window * groupDim);
        try
        {
            for (int row = 0; row < groups * rank; row += window)
            {
                int count = Math.Min(window, groups * rank - row);
                ReadOnlySpan<float> w = weight.ReadRows(row, count, groupDim, scratch);
                for (int r = 0; r < count; r++)
                {
                    int g = (row + r) / rank, o = (row + r) % rank;
                    ReadOnlySpan<float> wr = w.Slice(r * groupDim, groupDim);
                    for (int t = 0; t < tokens; t++)
                    {
                        ReadOnlySpan<float> xv = x.Slice((t * groups + g) * groupDim, groupDim);
                        float sum = 0f;
                        for (int d = 0; d < groupDim; d++) sum += xv[d] * wr[d];
                        dest[(t * groups + g) * rank + o] = sum;
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }
}
