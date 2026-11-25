using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Which 128-token blocks one query attends to in one KV group: the fixed prefix blocks, the local window ending at the query's own block, and
/// the remote blocks whose best indexer score is highest among those outside both windows.</summary>
internal static class MaxwellBlockRoute
{
    /// <summary>Upper bound on the blocks a route holds.</summary>
    public static int MaxBlocks(MaxwellSparseConfig sparse) => sparse.PrefixBlocks + sparse.RemoteBlocks + sparse.LocalBlocks;

    /// <summary>Writes the blocks for a query at <paramref name="position"/> into <paramref name="blocks"/>, ascending, and returns how many.</summary>
    /// <param name="indexQuery">The group's normed and rotated index query, <c>[headDim]</c>.</param>
    /// <param name="indexKeys">Every index key so far, <c>[positions, headDim]</c>, at least <c>position + 1</c> rows.</param>
    /// <remarks>Scores are compared as raw dot products: the reference's positive scale does not change the order. Equal scores keep the lower block.</remarks>
    public static int Select(MaxwellSparseConfig sparse, ReadOnlySpan<float> indexQuery, ReadOnlySpan<float> indexKeys, int position,
        Span<int> blocks)
    {
        int block = sparse.BlockSize, width = indexQuery.Length, current = position / block;
        int count = 0;
        for (int b = 0; b < sparse.PrefixBlocks && b <= current; b++) blocks[count++] = b;
        int localFirst = Math.Max(current - sparse.LocalBlocks + 1, sparse.PrefixBlocks);
        // Remote candidates start at or after the prefix and end at or before the local window's first block.
        int lastRemote = Math.Max(current - sparse.LocalBlocks + 1, 0) - 1;
        int candidates = lastRemote - sparse.PrefixBlocks + 1;
        if (candidates > 0)
        {
            Span<float> best = candidates <= 512 ? stackalloc float[candidates] : new float[candidates];
            for (int c = 0; c < candidates; c++)
            {
                int first = (sparse.PrefixBlocks + c) * block;
                float max = float.NegativeInfinity;
                for (int j = first; j < first + block; j++) max = MathF.Max(max, HostMath.Dot(indexQuery, indexKeys.Slice(j * width, width)));
                best[c] = max;
            }
            int take = Math.Min(sparse.RemoteBlocks, candidates);
            for (int n = 0; n < take; n++)
            {
                int pick = 0;
                for (int c = 1; c < candidates; c++)
                    if (best[c] > best[pick]) pick = c;
                blocks[count++] = sparse.PrefixBlocks + pick;
                best[pick] = float.NegativeInfinity;
            }
        }
        for (int b = localFirst; b <= current; b++) blocks[count++] = b;
        blocks[..count].Sort();
        return count;
    }
}
