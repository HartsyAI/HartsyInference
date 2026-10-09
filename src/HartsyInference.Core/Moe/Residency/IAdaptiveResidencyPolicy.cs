using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>
/// Chooses which resident expert to evict. Standalone: it never reads or writes an expert cache. The caller reports every
/// routed access, every insert after it lands, and every eviction, and asks for a victim only when a slot is needed.
/// Implementations are deterministic: the same call sequence gives the same victims. Once warmed, the four hot methods
/// allocate nothing.
/// </summary>
public interface IAdaptiveResidencyPolicy
{
    /// <summary>Stable policy name for reports.</summary>
    string Name { get; }

    /// <summary>Called for every routed access, hit or miss, before any eviction that access causes.</summary>
    void NoteAccess(ExpertKey key);

    /// <summary>Called after an expert has been placed into a slot.</summary>
    void NoteInsert(ExpertKey key);

    /// <summary>Called after an expert has left its slot.</summary>
    void NoteEvict(ExpertKey key);

    /// <summary>
    /// Returns one member of <paramref name="candidates"/> to evict. Every candidate is currently resident and the span is
    /// non-empty. Ties go to the earlier candidate.
    /// </summary>
    ExpertKey ChooseVictim(ReadOnlySpan<ExpertKey> candidates);

    /// <summary>Forgets all history.</summary>
    void Reset();
}
