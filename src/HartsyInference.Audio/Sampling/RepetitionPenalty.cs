namespace HartsyInference.Audio.Sampling;

/// <summary>HuggingFace-style repetition penalty applied to a logit buffer before sampling. Free-running AR
/// TTS LMs run unconditioned (no reference voice) drift into degenerate loops / trailing babble; penalizing
/// recently-emitted token ids suppresses that tail. Callers pre-shape the logit buffer per
/// <see cref="NucleusSampler"/>'s contract. A penalty of 1.0 is a no-op.</summary>
public static class RepetitionPenalty
{
    /// <summary>Divides the logit of each recently-emitted token by <paramref name="penalty"/> when positive,
    /// multiplies when negative (the HF <c>RepetitionPenaltyLogitsProcessor</c> rule). No-op when
    /// <paramref name="penalty"/> ≤ 1.</summary>
    public static void Apply(Span<float> logits, ReadOnlySpan<int> recent, float penalty)
    {
        if (penalty <= 1f) return;
        for (int i = 0; i < recent.Length; i++)
        {
            int id = recent[i];
            if ((uint)id >= (uint)logits.Length) continue;
            float v = logits[id];
            logits[id] = v > 0 ? v / penalty : v * penalty;
        }
    }
}
