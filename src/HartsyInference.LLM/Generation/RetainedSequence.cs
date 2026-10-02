using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>A retained decode sequence (KV cache plus the token ids already committed to it), kept alive under one
/// key across separate <see cref="TextGenerationPipeline.Generate(GenerationRequest,RetainedSequence,Action{int},CancellationToken)"/>
/// calls so a later call can reuse its longest common token-id prefix instead of prefilling from scratch.</summary>
/// <remarks>Starts empty (<see cref="Cache"/> null, <see cref="TokenIds"/> empty) for a key nothing has been stored
/// under yet — <see cref="RetainedSequenceStore.Checkout"/> hands out exactly such an instance on a miss so every
/// caller has the same object shape to pass to <c>Generate</c>, whether or not anything was actually cached. Not
/// thread-safe on its own: <see cref="RetainedSequenceStore"/>'s checkout/checkin contract is what makes concurrent
/// use safe (a checked-out instance has exactly one owner until it is checked back in).</remarks>
public sealed class RetainedSequence : IDisposable
{
    /// <summary>The retained KV cache, or null when nothing is cached yet (or it was just dropped).</summary>
    public ISequenceState? Cache { get; private set; }

    /// <summary>Every token id committed to <see cref="Cache"/> so far (prompt followed by whatever was generated),
    /// in order. Empty when <see cref="Cache"/> is null.</summary>
    public int[] TokenIds { get; private set; } = [];

    /// <summary>Device bytes <see cref="Cache"/> occupies (its capacity, not just what is committed, though the
    /// pipeline shrinks it toward that) — what <see cref="RetainedSequenceStore"/> charges this entry against its
    /// byte budget.</summary>
    public long Bytes { get; private set; }

    /// <summary>Replaces this entry's contents in place, so a reference a caller already holds stays valid across a
    /// checkout/checkin round trip. Disposes the previous <see cref="Cache"/> first, unless <paramref name="cache"/>
    /// IS that same instance (the common case: the existing cache was reused and truncated, not replaced).</summary>
    public void Update(ISequenceState cache, int[] tokenIds, long bytes)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (Cache is { } old && !ReferenceEquals(old, cache))
        {
            old.Dispose();
        }
        Cache = cache;
        TokenIds = tokenIds;
        Bytes = bytes;
    }

    /// <summary>Detaches this entry's contents WITHOUT disposing <see cref="Cache"/> — for a caller that has
    /// already disposed it (or taken over its disposal) and only needs the fields cleared.</summary>
    public void Clear()
    {
        Cache = null;
        TokenIds = [];
        Bytes = 0;
    }

    /// <summary>Disposes <see cref="Cache"/> (if any) and clears this entry. Idempotent.</summary>
    public void Dispose()
    {
        Cache?.Dispose();
        Clear();
    }
}
