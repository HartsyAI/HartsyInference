using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>
/// Decayed LFU with hysteresis. The clock advances by one per routed access (one step). Every score decays by
/// <c>decayPerStep</c> per step; scores are kept lazily as (value, epoch) so no pass over all experts is needed.
/// An access adds 1 to the expert's decayed score.
/// <para>
/// Hysteresis: the policy remembers its incumbent, the victim it chose last. The incumbent stays the victim unless some
/// other candidate scores below the incumbent by more than <c>hysteresisMargin</c> (a fraction of the incumbent's score).
/// An access to the incumbent clears it, and so does its eviction, so the next choice is made fresh. This stops the victim
/// from flipping between near-equal experts on every miss.
/// </para>
/// </summary>
public sealed class DecayedLfuHysteresisPolicy : IAdaptiveResidencyPolicy
{
    private readonly Dictionary<ExpertKey, DecayEntry> entries = new();
    private readonly double decayPerStep;
    private readonly double hysteresisMargin;
    private ulong clock;
    private bool hasIncumbent;
    private ExpertKey incumbent;

    /// <summary>
    /// Creates the policy. <paramref name="decayPerStep"/> is in (0, 1]; 1 means no decay. <paramref name="hysteresisMargin"/>
    /// is in [0, 1).
    /// </summary>
    public DecayedLfuHysteresisPolicy(double decayPerStep, double hysteresisMargin)
    {
        if (!(decayPerStep > 0.0 && decayPerStep <= 1.0)) throw new ArgumentOutOfRangeException(nameof(decayPerStep));
        if (!(hysteresisMargin >= 0.0 && hysteresisMargin < 1.0)) throw new ArgumentOutOfRangeException(nameof(hysteresisMargin));
        this.decayPerStep = decayPerStep;
        this.hysteresisMargin = hysteresisMargin;
    }

    /// <inheritdoc/>
    public string Name => "DecayedLfuHysteresis";

    /// <inheritdoc/>
    public void NoteAccess(ExpertKey key)
    {
        clock++;
        ref DecayEntry entry = ref CollectionsMarshal.GetValueRefOrAddDefault(entries, key, out _);
        entry.Score = ScoreAt(entry, clock) + 1.0;
        entry.Epoch = clock;
        if (hasIncumbent && incumbent == key) hasIncumbent = false;
    }

    /// <inheritdoc/>
    public void NoteInsert(ExpertKey key)
    {
    }

    /// <inheritdoc/>
    public void NoteEvict(ExpertKey key)
    {
        if (hasIncumbent && incumbent == key) hasIncumbent = false;
    }

    /// <inheritdoc/>
    public ExpertKey ChooseVictim(ReadOnlySpan<ExpertKey> candidates)
    {
        if (candidates.IsEmpty) throw new ArgumentException("No candidates.", nameof(candidates));

        ExpertKey weakest = candidates[0];
        double weakestScore = ScoreOf(weakest);
        double incumbentScore = 0.0;
        bool incumbentPresent = false;
        for (int i = 0; i < candidates.Length; i++)
        {
            ExpertKey candidate = candidates[i];
            double score = ScoreOf(candidate);
            if (i > 0 && score < weakestScore)
            {
                weakest = candidate;
                weakestScore = score;
            }

            if (hasIncumbent && candidate == incumbent)
            {
                incumbentPresent = true;
                incumbentScore = score;
            }
        }

        ExpertKey victim = weakest;
        if (incumbentPresent && !(weakestScore < incumbentScore * (1.0 - hysteresisMargin)))
        {
            victim = incumbent;
        }

        incumbent = victim;
        hasIncumbent = true;
        return victim;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        entries.Clear();
        clock = 0;
        hasIncumbent = false;
    }

    /// <summary>The decayed score of one expert at the current step; zero for an expert never routed.</summary>
    public double ScoreOf(ExpertKey key)
    {
        return entries.TryGetValue(key, out DecayEntry entry) ? ScoreAt(entry, clock) : 0.0;
    }

    private double ScoreAt(DecayEntry entry, ulong now)
    {
        if (entry.Score == 0.0) return 0.0;
        return entry.Score * Math.Pow(decayPerStep, (double)(now - entry.Epoch));
    }

    private struct DecayEntry
    {
        public double Score;
        public ulong Epoch;
    }
}
