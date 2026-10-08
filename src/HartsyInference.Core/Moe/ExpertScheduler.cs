using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Builds a layer's execution plan from the router's top-k ids and the cache's residency. Planning reads residency and
/// never changes cache state or uploads anything. It is allocation-free: the caller owns the scratch and the output list.
/// </summary>
public static class ExpertScheduler
{
    /// <summary>
    /// Plans one layer. Every routed expert appears once in <paramref name="output"/>, in ascending expert order, with the
    /// number of (token, slot) pairs it serves.
    /// </summary>
    /// <param name="cache">The residency-aware cache; only its read-only lookup is used.</param>
    /// <param name="ids">Flattened router ids, <c>tokens × k</c>, each in <c>[0, expertCount)</c>.</param>
    /// <param name="layer">Layer of the experts.</param>
    /// <param name="bank">Bank of the layer.</param>
    /// <param name="expertCount">Routed experts in the layer.</param>
    /// <param name="policy">Placement policy.</param>
    /// <param name="countScratch">At least <paramref name="expertCount"/> entries; overwritten.</param>
    /// <param name="residentScratch">At least the number of distinct routed experts; overwritten.</param>
    /// <param name="keyScratch">At least <paramref name="expertCount"/> entries; overwritten.</param>
    /// <param name="output">Receives one assignment per routed expert. Must hold at least that many entries.</param>
    /// <returns>The number of assignments written.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An id is outside the layer, or a scratch or output buffer is too small.</exception>
    public static int Plan(IResidencyAwareExpertCache cache, ReadOnlySpan<int> ids, int layer, ushort bank, int expertCount,
        IMissExecutionPolicy policy, Span<int> countScratch, Span<bool> residentScratch, Span<ExpertKey> keyScratch,
        Span<ExpertAssignment> output)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertCount);
        if (countScratch.Length < expertCount || keyScratch.Length < expertCount)
            throw new ArgumentOutOfRangeException(nameof(countScratch), "Scratch buffers must hold one entry per routed expert.");

        countScratch[..expertCount].Clear();
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)expertCount) throw new ArgumentOutOfRangeException(nameof(ids), id, $"Expert id is outside [0..{expertCount}).");
            countScratch[id]++;
        }

        int distinct = 0;
        for (int expert = 0; expert < expertCount; expert++)
        {
            if (countScratch[expert] == 0) continue;
            keyScratch[distinct++] = new ExpertKey(layer, expert, bank);
        }
        if (residentScratch.Length < distinct) throw new ArgumentOutOfRangeException(nameof(residentScratch), "Residency scratch is too small.");
        if (output.Length < distinct) throw new ArgumentOutOfRangeException(nameof(output), "The output must hold one entry per routed expert.");
        cache.LookupResident(keyScratch[..distinct], residentScratch[..distinct]);

        for (int i = 0; i < distinct; i++)
        {
            ExpertKey key = keyScratch[i];
            int rows = countScratch[key.Expert];
            output[i] = new ExpertAssignment(key, policy.Place(key, residentScratch[i], rows), rows);
        }
        return distinct;
    }
}
