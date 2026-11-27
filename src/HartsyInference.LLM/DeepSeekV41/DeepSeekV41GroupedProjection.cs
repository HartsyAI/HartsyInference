using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The grouped <c>wo_a</c> projection with the weight held as a <see cref="DeepSeekV41Weight"/>, decoded a window of rows at a time.</summary>
public static class DeepSeekV41GroupedProjection
{
    /// <summary>Decodes one group (or less) of the weight at a time so a stored weight is never widened whole.</summary>
    public static void Apply(ReadOnlySpan<float> x, DeepSeekV41Weight weight, int tokens, int groups, int rank, int groupDim,
        Span<float> dest)
    {
        if (weight.IsWidened)
        {
            GroupedOutputProjection.Apply(x, weight.ToF32(), tokens, groups, rank, groupDim, dest);
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
                GroupedOutputProjection.ApplyRows(x, w, row, count, tokens, groups, rank, groupDim, dest);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }
}
