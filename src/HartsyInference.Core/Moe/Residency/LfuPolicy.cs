using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>
/// Perfect least-frequently-used. Every routed access adds one to an expert's count, and counts survive eviction. The
/// victim is the candidate with the lowest count; equal counts go to the least recently used. This never ages, so an expert
/// that was hot long ago keeps its count. <see cref="DecayedLfuHysteresisPolicy"/> addresses that.
/// </summary>
public sealed class LfuPolicy : IAdaptiveResidencyPolicy
{
    private readonly Dictionary<ExpertKey, FrequencyEntry> entries = new();
    private ulong clock;

    /// <inheritdoc/>
    public string Name => "Lfu";

    /// <inheritdoc/>
    public void NoteAccess(ExpertKey key)
    {
        clock++;
        ref FrequencyEntry entry = ref CollectionsMarshal.GetValueRefOrAddDefault(entries, key, out _);
        entry.Count++;
        entry.LastUse = clock;
    }

    /// <inheritdoc/>
    public void NoteInsert(ExpertKey key)
    {
    }

    /// <inheritdoc/>
    public void NoteEvict(ExpertKey key)
    {
    }

    /// <inheritdoc/>
    public ExpertKey ChooseVictim(ReadOnlySpan<ExpertKey> candidates)
    {
        if (candidates.IsEmpty) throw new ArgumentException("No candidates.", nameof(candidates));

        ExpertKey victim = candidates[0];
        FrequencyEntry best = entries.TryGetValue(victim, out FrequencyEntry first) ? first : default;
        for (int i = 1; i < candidates.Length; i++)
        {
            FrequencyEntry entry = entries.TryGetValue(candidates[i], out FrequencyEntry found) ? found : default;
            if (entry.Count < best.Count || (entry.Count == best.Count && entry.LastUse < best.LastUse))
            {
                victim = candidates[i];
                best = entry;
            }
        }

        return victim;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        entries.Clear();
        clock = 0;
    }

    private struct FrequencyEntry
    {
        public ulong Count;
        public ulong LastUse;
    }
}
