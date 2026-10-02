namespace HartsyInference.Audio.Sampling;

/// <summary>Shared numeric helper for every audio sampler that needs a full descending argsort of a candidate
/// set too large for <see cref="NucleusSampler"/>'s bounded top-k selection (unbounded top-k, or a model's own
/// bespoke CFG-window/top-p pass). Mirrors <c>HartsyInference.LLM.Sampling.SamplerMath.SortDescendingByValue</c>
/// (PR #215, which fixed the equivalent LLM-package anti-pattern) rather than referencing it directly: that type
/// is internal to the LLM package, and more importantly these callers draw over a SORTED sequence (ties and
/// floating-point summation order are part of the observable output here, not just an internal cutoff like
/// TopPStep's), so the fix is reproduced at the same semantic layer it is consumed from instead of reached for
/// across a package boundary.</summary>
internal static class SortHelpers
{
    /// <summary>Sorts <paramref name="values"/>[0, <paramref name="count"/>) into descending order in place and
    /// permutes <paramref name="order"/>[0, <paramref name="count"/>) (caller-populated with each slot's
    /// pre-sort identity — typically <c>0..count-1</c>) to match: after the call, <c>values[rank]</c> is the
    /// rank-th largest value and <c>order[rank]</c> is that value's pre-sort identity. Both buffers are
    /// caller-owned (reused across calls — every call site below keeps a <c>[ThreadStatic]</c> pair sized once
    /// and grown only on demand) so this allocates nothing.</summary>
    /// <remarks>Previously every call site below ran <c>Array.Sort(order, (a, b) =&gt; values[b].CompareTo(values[a]))</c>:
    /// correct, but a fresh array per call plus an O(n log n) virtual delegate dispatch per comparison — the
    /// same anti-pattern PR #215 fixed in the LLM package's sampler at a ~152K-token vocabulary.
    /// <see cref="Array.Sort{TKey, TValue}(TKey[], TValue[], int, int)"/> sorts the primitive <c>float</c> keys
    /// directly (no delegate indirection), but only ascending; negating before and after is exact (sign-bit
    /// flip, no rounding) and turns "ascending by -value" into "descending by value" without a second array or
    /// a reversal pass. Verified to reproduce the EXACT SAME permutation as the old delegate sort — including
    /// exact ties, at every rank, across random/all-equal/low-cardinality/sorted/organ-pipe/adversarial-pivot
    /// inputs from 1 to 102,048 elements (see <c>SortHelpersTests</c>) — so every caller below can keep reading
    /// results exactly where it used to, with only the sort mechanism swapped out. The one place old and new
    /// disagree is NaN placement (the primitive overload pre-passes NaNs to the front; the old delegate sort,
    /// via <c>float.CompareTo</c>, left them at the back): NaN logits mean sampling was already broken upstream,
    /// so neither placement was ever relied upon, and this does not try to make it defined.</remarks>
    public static void SortDescendingByValue(float[] values, int[] order, int count)
    {
        for (int i = 0; i < count; i++)
        {
            values[i] = -values[i];
        }
        Array.Sort(values, order, 0, count);
        for (int i = 0; i < count; i++)
        {
            values[i] = -values[i];
        }
    }
}
