namespace HartsyInference.Audio.Sampling;

/// <summary>Repetition-Aware Sampling as CosyVoice 2 defines it (<c>cosyvoice/utils/common.py</c> <c>ras_sampling</c>):
/// draw from the top-k ∩ top-p nucleus; if the drawn token already appears often enough in the trailing window of
/// decoded tokens, mask it and redraw from the full, unfiltered distribution.</summary>
/// <remarks>The repeat test counts occurrences anywhere in the window, not a trailing run: with the reference defaults
/// (window 10, <c>tau_r</c> 0.1) a single earlier occurrence triggers the redraw. Temperature is 1 throughout, matching
/// the reference, which samples from <c>softmax(log_softmax(logits))</c>. Stop-token gating is the caller's job and
/// must already be applied to <paramref name="logits"/>; the redraw then honors it too.</remarks>
public static class RasSampling
{
    /// <summary>Draws one token id from <paramref name="logits"/>; <paramref name="logits"/> is modified when the
    /// repetition redraw fires (the repeated token is set to -inf).</summary>
    public static int Draw(Span<float> logits, IReadOnlyList<int> history, int topK, float topP, int window, float tauR,
        ref uint rng, bool useRas = true)
    {
        int chosen = NucleusSampler.Draw(logits, logits.Length, 1f, topK, topP, ref rng);
        if (!useRas || window <= 0) return chosen;

        int from = Math.Max(0, history.Count - window);
        int repeats = 0;
        for (int i = from; i < history.Count; i++)
            if (history[i] == chosen) repeats++;
        if (repeats < window * tauR) return chosen;

        logits[chosen] = float.NegativeInfinity;
        return NucleusSampler.Draw(logits, logits.Length, 1f, 0, 1f, ref rng);
    }
}
