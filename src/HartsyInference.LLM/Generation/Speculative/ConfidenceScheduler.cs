namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Chooses how many drafted tokens to verify, by Algorithm 1 of the DSpark paper (confidence-scheduled verification, section 3.2.2), for one sequence.</summary>
/// <remarks>The head's confidences are logits, so position <c>j</c> survives verification with probability <c>c_j = sigmoid(logit_j)</c>, and its survival
/// <c>a_j</c> is the product of <c>c_1</c> to <c>c_j</c>. The throughput objective for one sequence is <c>Θ(l) = (1 + Σ_{j≤l} a_j) · SPS(1 + l)</c>. Survival is
/// non-increasing, so the positions are admitted in order, and the walk stops at the first position that does not improve Θ. That is the global optimum only
/// when Θ is unimodal along the walk. The paper's section 5.2 global search for jagged SPS curves is not implemented here.
/// Calibration (sequential temperature scaling) is also not applied: the confidences are used as the head emits them.</remarks>
internal sealed class ConfidenceScheduler
{
    private readonly SpsProfile _profile;
    private readonly int _maxDraft;

    /// <param name="profile">The engine's steps per second at each batch size; it must cover <c>1 + maxDraft</c> tokens.</param>
    /// <param name="maxDraft">The most positions a draft can have, the head's block size.</param>
    public ConfidenceScheduler(SpsProfile profile, int maxDraft)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (maxDraft < 1) throw new ArgumentOutOfRangeException(nameof(maxDraft), maxDraft, "The scheduler needs at least one drafted position.");
        if (profile.MaxBatch < 1 + maxDraft)
            throw new ArgumentException($"The profile covers batch sizes up to {profile.MaxBatch}; {1 + maxDraft} are needed.", nameof(profile));
        _profile = profile;
        _maxDraft = maxDraft;
    }

    /// <summary>The most positions a draft can have; the proposer checks it against the head's block size.</summary>
    internal int MaxDraft => _maxDraft;

    /// <summary>How many of the first positions to verify, from the head's confidence logits in draft order, capped at <paramref name="maxTokens"/>.</summary>
    public int Choose(ReadOnlySpan<float> confidenceLogits, int maxTokens)
    {
        int limit = Math.Min(Math.Min(maxTokens, _maxDraft), confidenceLogits.Length);
        if (limit <= 0) return 0;

        double best = _profile.StepsPerSecond(1);   // Θ at l = 0: one token, batch 1
        double tau = 1, survival = 1;
        int chosen = 0;
        for (int j = 1; j <= limit; j++)
        {
            float logit = confidenceLogits[j - 1];
            if (!float.IsFinite(logit)) throw new ArgumentException("Confidence logits must be finite.", nameof(confidenceLogits));
            survival *= Sigmoid(logit);   // a_j
            tau += survival;
            double theta = tau * _profile.StepsPerSecond(1 + j);
            if (theta <= best) break;   // the first position that does not improve Θ ends the walk
            best = theta;
            chosen = j;
        }
        return chosen;
    }

    private static double Sigmoid(float logit) => 1.0 / (1.0 + Math.Exp(-(double)logit));
}
