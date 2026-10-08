using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Builds a layer's execution plan from the router's top-k ids and the cache's residency. Planning reads residency and
/// never changes cache state or uploads anything. It is allocation-free once the caller's buffers have warmed up: the caller
/// owns the scratch, the output, the miss list and the lease, and reuses them from layer to layer.
/// </summary>
public static class ExpertScheduler
{
    /// <summary>
    /// Plans one layer and pins the experts it will run on the GPU. Every routed expert appears once in
    /// <paramref name="output"/>, in ascending expert order, with the number of (token, slot) pairs it serves. The resident
    /// experts are pinned through <paramref name="lease"/>, so the cache cannot evict them before execution; the caller disposes
    /// the lease when the layer has run; the same lease object can then plan the next layer. Misses are never uploaded here.
    /// </summary>
    /// <param name="cache">The residency-aware cache.</param>
    /// <param name="ids">Flattened router ids, <c>tokens × k</c>, each in <c>[0, expertCount)</c>.</param>
    /// <param name="layer">Layer of the experts.</param>
    /// <param name="bank">Bank of the layer.</param>
    /// <param name="expertCount">Routed experts in the layer.</param>
    /// <param name="policy">Placement policy. It may not place a non-resident expert on the GPU.</param>
    /// <param name="countScratch">At least <paramref name="expertCount"/> entries; overwritten.</param>
    /// <param name="residentScratch">At least the number of distinct routed experts; overwritten.</param>
    /// <param name="keyScratch">At least <paramref name="expertCount"/> entries; overwritten.</param>
    /// <param name="output">Receives one assignment per routed expert. Must hold at least that many entries.</param>
    /// <param name="missScratch">Receives the experts that are not resident; cleared first. Its capacity must hold every distinct
    /// routed expert, so the planner never grows it.</param>
    /// <param name="lease">The caller's lease, unbound or released. It receives the pin on the resident experts; dispose it after the
    /// layer runs.</param>
    /// <returns>The number of assignments written.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An id is outside the layer, or a buffer is too small.</exception>
    /// <exception cref="InvalidOperationException">The policy placed a non-resident expert on the GPU, or the lease is still pinned.</exception>
    public static int Plan(IResidencyAwareExpertCache cache, ReadOnlySpan<int> ids, int layer, ushort bank, int expertCount,
        IMissExecutionPolicy policy, Span<int> countScratch, Span<bool> residentScratch, Span<ExpertKey> keyScratch,
        Span<ExpertAssignment> output, List<ExpertKey> missScratch, ExpertLease lease)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentNullException.ThrowIfNull(missScratch);
        ArgumentNullException.ThrowIfNull(lease);
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

        // Pinning is what makes the plan safe: the resident experts cannot be evicted between planning and execution.
        missScratch.Clear();
        if (missScratch.Capacity < distinct)
            throw new ArgumentOutOfRangeException(nameof(missScratch), "The miss list must have capacity for every distinct routed expert.");
        cache.AcquireResident(keyScratch[..distinct], missScratch, lease);
        try
        {
            for (int i = 0; i < distinct; i++)
            {
                ExpertKey key = keyScratch[i];
                bool resident = !missScratch.Contains(key);
                residentScratch[i] = resident;
                int rows = countScratch[key.Expert];
                ExpertPlacement placement = policy.Place(key, resident, rows);
                if (placement == ExpertPlacement.Gpu && !resident)
                    throw new InvalidOperationException($"{key} is not resident; it cannot run on the GPU without an upload.");
                output[i] = new ExpertAssignment(key, placement, rows);
            }
            return distinct;
        }
        catch
        {
            lease.Dispose();
            throw;
        }
    }
}
