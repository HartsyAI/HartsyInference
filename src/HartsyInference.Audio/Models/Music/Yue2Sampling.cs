namespace HartsyInference.Audio.Models.Music;

/// <summary>One autoregressive pass's sampler settings. YuE2 samples its two passes very differently — the score
/// planner is near-greedy with a wide penalty window, the semantic pass is hot with a heavy penalty — so a single
/// shared set of knobs cannot express the release protocol.</summary>
public sealed record Yue2Sampling
{
    public float Temperature { get; init; } = 1.0f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 100;
    public float RepetitionPenalty { get; init; } = 1.2f;

    /// <summary>How many recently emitted ids the penalty counts over.</summary>
    public int PenaltyWindow { get; init; } = 50;

    /// <summary>Steps before the phase's end token is allowed, so the pass cannot terminate instantly.</summary>
    public int MinTokens { get; init; } = 200;

    /// <summary>Hard cap on emitted tokens for this pass.</summary>
    public int MaxTokens { get; init; } = 9_000;

    /// <summary>Score-planning defaults from <c>yue2_generation_config.json</c>.</summary>
    public static Yue2Sampling Abc => new()
    {
        Temperature = 0.7f, TopP = 0.9f, TopK = 30, RepetitionPenalty = 1.005f,
        PenaltyWindow = 100, MinTokens = 32, MaxTokens = 4_096,
    };

    /// <summary>Semantic-generation defaults from <c>yue2_generation_config.json</c>.</summary>
    public static Yue2Sampling Semantic => new();

    /// <summary>Throws if the settings are outside what the release protocol accepts.</summary>
    public void Validate()
    {
        if (!float.IsFinite(Temperature) || Temperature is < 0 or > 5)
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature, "Temperature must be in [0, 5].");
        if (!float.IsFinite(TopP) || TopP is <= 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(TopP), TopP, "top_p must be in (0, 1].");
        if (TopK < 1)
            throw new ArgumentOutOfRangeException(nameof(TopK), TopK, "top_k must be at least 1.");
        if (!float.IsFinite(RepetitionPenalty) || RepetitionPenalty <= 0)
            throw new ArgumentOutOfRangeException(nameof(RepetitionPenalty), RepetitionPenalty, "The repetition penalty must be positive and finite.");
        if (PenaltyWindow is < 1 or > 100)
            throw new ArgumentOutOfRangeException(nameof(PenaltyWindow), PenaltyWindow, "The penalty window must be in [1, 100].");
        if (MaxTokens < 1 || MinTokens < 0 || MinTokens > MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(MinTokens), "Require 0 <= min_tokens <= max_tokens and max_tokens >= 1.");
    }
}

/// <summary>YuE2's logit processing, reproducing the released <c>distribution()</c> step for step. The order is
/// load-bearing: the vocabulary mask runs before the repetition penalty (so penalising a masked id keeps it masked),
/// and top-k runs before top-p (so the nucleus is taken over the already-truncated set).</summary>
/// <remarks><para>The reference keeps logits in BF16 when <see cref="Yue2Cot.Off"/> is in play and upcasts to F32
/// otherwise. We compute in F32 throughout — the engine's logits are F32 to begin with and are already not
/// bit-identical to torch's BF16 ones — but the <b>behavioural</b> half of that switch is reproduced: the historical
/// BF16 path keeps the top three sorted entries out of the nucleus cut instead of the top one.</para></remarks>
public static class Yue2LogitProcessor
{
    /// <summary>Rewrites <paramref name="scores"/> in place into the distribution to draw from. After this call a
    /// temperature of zero means "take the argmax"; otherwise softmax over the finite entries.</summary>
    /// <param name="scores">Logits over the phase's <see cref="Yue2Protocol.Window"/>, indexed <c>id - baseId</c>;
    /// overwritten.</param>
    /// <param name="history">Every id emitted in this pass so far, oldest first (absolute ids).</param>
    /// <param name="step">Tokens emitted so far, which gates the end token.</param>
    /// <param name="legacyOff">The historical <see cref="Yue2Cot.Off"/> nucleus rule (keep three, not one).</param>
    /// <param name="baseId">The absolute id that <c>scores[0]</c> stands for.</param>
    public static void Apply(Span<float> scores, Yue2Sampling sampling, ReadOnlySpan<int> history,
        int step, Yue2Phase phase, bool legacyOff, int baseId)
    {
        ArgumentNullException.ThrowIfNull(sampling);
        int end = (phase == Yue2Phase.Abc ? Yue2Protocol.AbcEnd : Yue2Protocol.MusicEnd) - baseId;

        // 1. Hard vocabulary mask: only the phase's own span, plus its end token, may be drawn. The end token sits
        //    outside its phase's span in both phases, so its logit is carried across the mask rather than re-derived.
        float rawEnd = (uint)end < (uint)scores.Length ? scores[end] : float.NegativeInfinity;
        int allowedStart = (phase == Yue2Phase.Abc ? 0 : Yue2Protocol.CodecOffset) - baseId;
        int allowedEnd = (phase == Yue2Phase.Abc ? Yue2Protocol.Eod : Yue2Protocol.CodecOffset + Yue2Protocol.CodecSize) - baseId;
        for (int i = 0; i < scores.Length; i++)
        {
            if (i < allowedStart || i >= allowedEnd) scores[i] = float.NegativeInfinity;
        }
        if ((uint)end < (uint)scores.Length) scores[end] = step < sampling.MinTokens ? float.NegativeInfinity : rawEnd;

        // 2. Windowed repetition penalty: alpha = penalty^count over the last PenaltyWindow ids.
        ApplyWindowPenalty(scores, history, sampling.PenaltyWindow, sampling.RepetitionPenalty, baseId);

        if (sampling.Temperature == 0) return;
        if (sampling.Temperature != 1f)
        {
            float inv = 1f / sampling.Temperature;
            for (int i = 0; i < scores.Length; i++) scores[i] *= inv;
        }

        // 3. top-k: everything strictly below the k-th largest value goes to -inf. Ties at the cut are all kept,
        //    which is what torch.topk's threshold comparison does.
        float threshold = KthLargest(scores, Math.Min(sampling.TopK, scores.Length));
        if (float.IsFinite(threshold))
        {
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] < threshold) scores[i] = float.NegativeInfinity;
            }
        }

        if (sampling.TopP >= 1f) return;
        ApplyNucleus(scores, sampling.TopP, legacyOff ? 3 : 1);
    }

    private static void ApplyWindowPenalty(Span<float> scores, ReadOnlySpan<int> history, int window, float penalty,
        int baseId)
    {
        if (penalty == 1f || history.IsEmpty) return;
        ReadOnlySpan<int> recent = history.Length > window ? history[^window..] : history;

        // Count occurrences, then penalise once per distinct id — penalty^count, not penalty applied count times,
        // which for a non-integer penalty is the same thing but avoids compounding float error.
        Dictionary<int, int> counts = new(recent.Length);
        for (int i = 0; i < recent.Length; i++)
        {
            int id = recent[i] - baseId;
            if ((uint)id >= (uint)scores.Length) continue;
            counts[id] = counts.TryGetValue(id, out int c) ? c + 1 : 1;
        }
        foreach ((int id, int count) in counts)
        {
            float alpha = MathF.Pow(penalty, count);
            float v = scores[id];
            if (!float.IsFinite(v)) continue;
            scores[id] = v < 0 ? v * alpha : v / alpha;
        }
    }

    private static float KthLargest(ReadOnlySpan<float> scores, int k)
    {
        if (k >= scores.Length) return float.NegativeInfinity;

        // A bounded insertion buffer beats a full sort: k is at most a few hundred against a 184k vocabulary.
        Span<float> top = k <= 1024 ? stackalloc float[k] : new float[k];
        int n = 0;
        float min = float.PositiveInfinity;
        for (int i = 0; i < scores.Length; i++)
        {
            float v = scores[i];
            if (float.IsNegativeInfinity(v)) continue;
            if (n < k)
            {
                top[n++] = v;
                if (n == k) { min = float.PositiveInfinity; for (int j = 0; j < k; j++) if (top[j] < min) min = top[j]; }
            }
            else if (v > min)
            {
                int pos = 0;
                for (int j = 1; j < k; j++) if (top[j] < top[pos]) pos = j;
                top[pos] = v;
                min = float.PositiveInfinity; for (int j = 0; j < k; j++) if (top[j] < min) min = top[j];
            }
        }
        return n < k ? float.NegativeInfinity : min;
    }

    private static void ApplyNucleus(Span<float> scores, float topP, int alwaysKeep)
    {
        // Only finite entries can carry probability, so the nucleus is computed over them alone — identical to the
        // reference's full-vocabulary sort, where every masked entry softmaxes to exactly zero.
        List<int> candidates = [];
        List<float> values = [];
        for (int i = 0; i < scores.Length; i++)
        {
            if (!float.IsFinite(scores[i])) continue;
            candidates.Add(i);
            values.Add(-scores[i]);   // negated so an ascending sort gives descending score
        }
        if (candidates.Count <= alwaysKeep) return;

        int[] order = [.. candidates];
        Array.Sort([.. values], order);

        float max = scores[order[0]];
        double total = 0;
        double[] probabilities = new double[order.Length];
        for (int i = 0; i < order.Length; i++)
        {
            probabilities[i] = Math.Exp(scores[order[i]] - max);
            total += probabilities[i];
        }

        double cumulative = 0;
        for (int i = 0; i < order.Length; i++)
        {
            double p = probabilities[i] / total;
            // The reference removes on the EXCLUSIVE prefix sum, so the entry that crosses the threshold survives.
            if (i >= alwaysKeep && cumulative > topP) scores[order[i]] = float.NegativeInfinity;
            cumulative += p;
        }
    }
}
