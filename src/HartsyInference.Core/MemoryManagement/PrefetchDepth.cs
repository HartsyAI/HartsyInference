namespace HartsyInference.Core.MemoryManagement;

/// <summary>How many blocks a streamed loop may keep in flight ahead of the one it is running.</summary>
/// <remarks>The <c>- 2</c> is the load-bearing part and the reason this is shared: the streaming working set
/// briefly holds <c>prefetchAhead + 2</c> blocks at the moment block <c>N+1</c> becomes resident before block
/// <c>N-1</c> is evicted (see <see cref="BlockStreamingController.BeforeBlockForward"/>), so a depth chosen from the
/// budget alone overcommits by exactly two blocks. Six pipelines had rediscovered that separately.
/// <para>What is NOT shared is which numbers to feed it. A family with heterogeneous blocks budgets on its widest,
/// a family whose block 0 is the widest budgets on that, and a family running two denoisers at once halves the
/// budget and caps the depth — those are real per-architecture facts, so they stay at the call sites where the
/// reasoning is visible rather than being averaged into one wrong default.</para></remarks>
public static class PrefetchDepth
{
    /// <summary>The deepest prefetch <paramref name="availableBytes"/> supports at <paramref name="perBlockBytes"/> per block, clamped to <c>[0, maxDepth]</c>.</summary>
    /// <param name="perBlockBytes">The block size to budget against — the caller's widest, not an average.</param>
    /// <param name="unknownBlockDepth">Depth to use when the block size is unknown; callers differ on whether an unmeasurable block should still overlap one upload.</param>
    public static int Choose(long availableBytes, long perBlockBytes, int maxDepth = 2, int unknownBlockDepth = 0)
    {
        if (maxDepth < 0) throw new ArgumentOutOfRangeException(nameof(maxDepth));
        if (availableBytes <= 0)
        {
            return 0;
        }
        if (perBlockBytes <= 0)
        {
            return unknownBlockDepth;
        }
        return Math.Clamp((int)(availableBytes / perBlockBytes) - 2, 0, maxDepth);
    }
}
