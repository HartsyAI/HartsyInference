using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Turns the router's compact top-k ids into the experts a cache must hold.</summary>
/// <remarks><see cref="IBackend.MoeRoute"/> already writes <c>topkIdx [tokens,k]</c> I32 as its own small tensor, so reading it moves
/// <c>tokens * k * 4</c> bytes (24 per token at k=6) and nothing else; the activation sync fires on that tensor alone.</remarks>
public static class ExpertRouting
{
    /// <summary>Bytes the id readback moves for <paramref name="tokens"/> tokens at <paramref name="topK"/> experts each.</summary>
    public static long CompactBytes(long tokens, int topK) => tokens * topK * sizeof(int);

    /// <summary>The distinct experts routed to, in ascending expert order, read from an I32 id tensor.</summary>
    public static ExpertKey[] DistinctKeys(Tensor topkIdx, int layer, int expertCount)
    {
        ArgumentNullException.ThrowIfNull(topkIdx);
        if (topkIdx.DType != DType.I32)
            throw new ArgumentException($"Routing ids must be I32; got {topkIdx.DType}.", nameof(topkIdx));
        return DistinctKeys(topkIdx.AsSpan<int>(), layer, expertCount);
    }

    /// <summary>The distinct experts among <paramref name="ids"/>, in ascending expert order.</summary>
    public static ExpertKey[] DistinctKeys(ReadOnlySpan<int> ids, int layer, int expertCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertCount);
        bool[] seen = new bool[expertCount];
        int distinct = 0;
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)expertCount)
                throw new ArgumentOutOfRangeException(nameof(ids), id, $"Expert id is outside [0..{expertCount}).");
            if (seen[id]) continue;
            seen[id] = true;
            distinct++;
        }
        ExpertKey[] keys = new ExpertKey[distinct];
        int next = 0;
        for (int expert = 0; expert < expertCount; expert++)
        {
            if (seen[expert]) keys[next++] = new ExpertKey(layer, expert);
        }
        return keys;
    }
}
