using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>
/// Segmented LRU. Residents start on probation. A hit on a probationary resident promotes it to the protected segment;
/// when the protected segment exceeds its capacity, its least recently used member is demoted back to probation.
/// Victims come from probation first (least recently used), and from protected only when probation has no candidate.
/// Recency is a logical access clock, so the order is exact and deterministic.
/// </summary>
public sealed class SegmentedLruPolicy : IAdaptiveResidencyPolicy
{
    private readonly Dictionary<ExpertKey, SegmentEntry> entries = new();
    private readonly int protectedCapacity;
    private ulong clock;
    private int protectedCount;

    /// <summary>Creates the policy. <paramref name="protectedCapacity"/> bounds how many residents may be protected at once.</summary>
    public SegmentedLruPolicy(int protectedCapacity)
    {
        if (protectedCapacity < 0) throw new ArgumentOutOfRangeException(nameof(protectedCapacity));
        this.protectedCapacity = protectedCapacity;
    }

    /// <inheritdoc/>
    public string Name => "SegmentedLru";

    /// <summary>Residents currently in the protected segment.</summary>
    public int ProtectedCount => protectedCount;

    /// <inheritdoc/>
    public void NoteAccess(ExpertKey key)
    {
        clock++;
        ref SegmentEntry entry = ref CollectionsMarshal.GetValueRefOrAddDefault(entries, key, out _);
        entry.LastUse = clock;
        if (!entry.Resident || entry.Protected) return;

        entry.Protected = true;
        protectedCount++;
        if (protectedCount > protectedCapacity) DemoteLeastRecentProtected();
    }

    /// <inheritdoc/>
    public void NoteInsert(ExpertKey key)
    {
        ref SegmentEntry entry = ref CollectionsMarshal.GetValueRefOrAddDefault(entries, key, out _);
        entry.Resident = true;
        if (entry.Protected)
        {
            entry.Protected = false;
            protectedCount--;
        }
    }

    /// <inheritdoc/>
    public void NoteEvict(ExpertKey key)
    {
        ref SegmentEntry entry = ref CollectionsMarshal.GetValueRefOrNullRef(entries, key);
        if (Unsafe.IsNullRef(ref entry)) return;

        if (entry.Protected) protectedCount--;
        entry.Protected = false;
        entry.Resident = false;
    }

    /// <inheritdoc/>
    public ExpertKey ChooseVictim(ReadOnlySpan<ExpertKey> candidates)
    {
        if (candidates.IsEmpty) throw new ArgumentException("No candidates.", nameof(candidates));

        ExpertKey probation = default;
        ExpertKey protectedKey = default;
        ulong probationUse = ulong.MaxValue;
        ulong protectedUse = ulong.MaxValue;
        bool haveProbation = false;
        bool haveProtected = false;

        foreach (ExpertKey candidate in candidates)
        {
            SegmentEntry entry = entries.TryGetValue(candidate, out SegmentEntry found) ? found : default;
            if (entry.Protected)
            {
                if (!haveProtected || entry.LastUse < protectedUse)
                {
                    protectedKey = candidate;
                    protectedUse = entry.LastUse;
                    haveProtected = true;
                }
            }
            else if (!haveProbation || entry.LastUse < probationUse)
            {
                probation = candidate;
                probationUse = entry.LastUse;
                haveProbation = true;
            }
        }

        return haveProbation ? probation : protectedKey;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        entries.Clear();
        clock = 0;
        protectedCount = 0;
    }

    private void DemoteLeastRecentProtected()
    {
        ExpertKey demote = default;
        ulong best = ulong.MaxValue;
        bool found = false;
        foreach (KeyValuePair<ExpertKey, SegmentEntry> pair in entries)
        {
            if (pair.Value.Protected && pair.Value.LastUse < best)
            {
                demote = pair.Key;
                best = pair.Value.LastUse;
                found = true;
            }
        }

        if (!found) return;
        ref SegmentEntry entry = ref CollectionsMarshal.GetValueRefOrNullRef(entries, demote);
        entry.Protected = false;
        protectedCount--;
    }

    private struct SegmentEntry
    {
        public ulong LastUse;
        public bool Resident;
        public bool Protected;
    }
}
