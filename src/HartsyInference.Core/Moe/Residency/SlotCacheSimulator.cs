using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>
/// A standalone slot cache driven by one policy: a fixed number of slots, misses fill free slots first, and a full cache
/// asks the policy for a victim. It keeps a digest of every eviction decision so two replays can be compared exactly.
/// It is a model for measurement only; it does not touch any expert cache.
/// </summary>
public sealed class SlotCacheSimulator
{
    private const ulong FnvOffset = 14695981039346656037UL;
    private const ulong FnvPrime = 1099511628211UL;

    private readonly IAdaptiveResidencyPolicy policy;
    private readonly ExpertKey[] slots;
    private int used;
    private ulong digest = FnvOffset;

    /// <summary>Creates a simulator with <paramref name="slotBudget"/> slots driven by <paramref name="policy"/>.</summary>
    public SlotCacheSimulator(IAdaptiveResidencyPolicy policy, int slotBudget)
    {
        ArgumentNullException.ThrowIfNull(policy);
        if (slotBudget <= 0) throw new ArgumentOutOfRangeException(nameof(slotBudget));
        this.policy = policy;
        slots = new ExpertKey[slotBudget];
    }

    /// <summary>Accesses that found their expert resident.</summary>
    public long Hits { get; private set; }

    /// <summary>Accesses that had to bring the expert in.</summary>
    public long Misses { get; private set; }

    /// <summary>Experts evicted to make room.</summary>
    public long Evictions { get; private set; }

    /// <summary>Bytes brought in by misses.</summary>
    public long BytesMoved { get; private set; }

    /// <summary>FNV-1a digest over the victim sequence, so equal digests mean equal eviction decisions.</summary>
    public ulong Digest => digest;

    /// <summary>The experts currently resident, in slot order.</summary>
    public ReadOnlySpan<ExpertKey> Residents => slots.AsSpan(0, used);

    /// <summary>Routes one access. Returns true on a hit.</summary>
    public bool Access(ExpertKey key, long bytes)
    {
        policy.NoteAccess(key);
        if (IndexOf(key) >= 0)
        {
            Hits++;
            return true;
        }

        Misses++;
        BytesMoved += bytes;
        if (used < slots.Length)
        {
            slots[used++] = key;
            policy.NoteInsert(key);
            return false;
        }

        ExpertKey victim = policy.ChooseVictim(slots.AsSpan(0, used));
        int at = IndexOf(victim);
        if (at < 0) throw new InvalidOperationException($"Policy {policy.Name} chose {victim}, which is not resident.");

        policy.NoteEvict(victim);
        Evictions++;
        Fold(victim);
        slots[at] = key;
        policy.NoteInsert(key);
        return false;
    }

    private int IndexOf(ExpertKey key)
    {
        for (int i = 0; i < used; i++)
        {
            if (slots[i] == key) return i;
        }

        return -1;
    }

    private void Fold(ExpertKey key)
    {
        digest = (digest ^ (ulong)(uint)key.Bank) * FnvPrime;
        digest = (digest ^ (ulong)(uint)key.Layer) * FnvPrime;
        digest = (digest ^ (ulong)(uint)key.Expert) * FnvPrime;
    }
}
