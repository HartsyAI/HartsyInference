using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Port of fish-speech's top-k / top-p draw for the Fish Audio models. Candidates are ranked by logit; the highest-ranked
/// one is always kept, and a lower one only while it ranks under top-k and the inclusive cumulative probability up to it stays
/// within top-p. The two models differ in which distribution that cumulative reads: S2 (<c>logits_to_probs</c>) filters the
/// untempered distribution and applies temperature afterwards, Drama 3 (<c>sample_top_k_top_p</c>) tempers first and filters the
/// tempered one (<paramref name="temperatureFirst"/>).</summary>
public static class FishAudioSampling
{
    /// <summary>Floor on the temperature, as upstream clamps it.</summary>
    public const float MinTemperature = 1e-5f;

    private const int InsertionLimit = 256;

    /// <summary>Draws a vocab id. <paramref name="allowed"/> restricts the candidates (the logit-bias constraint on the
    /// slow head: semantic ids + <c>&lt;|im_end|&gt;</c>); null = the whole vocabulary.</summary>
    public static int Sample(ReadOnlySpan<float> logits, float temperature, float topP, int topK, ref uint rng,
        ReadOnlySpan<int> allowed = default)
    {
        Scratch scratch = new(allowed.IsEmpty ? logits.Length : allowed.Length);
        int pick = Sample(logits, temperature, topP, topK, ref rng, scratch, temperatureFirst: false, allowed);
        return allowed.IsEmpty ? pick : allowed[pick];
    }

    /// <summary>Draws a candidate without allocating: returns its index into <paramref name="allowed"/> (or into
    /// <paramref name="logits"/> when no restriction is given). <paramref name="scratch"/> must hold every candidate.</summary>
    public static int Sample(ReadOnlySpan<float> logits, float temperature, float topP, int topK, ref uint rng, Scratch scratch,
        bool temperatureFirst, ReadOnlySpan<int> allowed = default)
    {
        ArgumentNullException.ThrowIfNull(scratch);
        int n = allowed.IsEmpty ? logits.Length : allowed.Length;
        if (n == 0) throw new ArgumentException("There is nothing to sample from.", nameof(logits));
        if (scratch.Capacity < n) throw new ArgumentException($"The scratch holds {scratch.Capacity} candidates, not {n}.", nameof(scratch));
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK), "top-k must be at least 1.");
        float t = MathF.Max(temperature, MinTemperature);

        // Rank the top min(topK, n) candidates; only they can survive the filter.
        int keep = Math.Min(topK, n);
        Span<int> ranked = scratch.Order.AsSpan(0, keep);
        float max = SelectTop(logits, allowed, n, ranked, scratch.Values.AsSpan(0, keep), scratch);

        // The cumulative is over the full (filter) distribution, so its normalizer spans every candidate.
        float filterScale = temperatureFirst ? 1f / t : 1f;
        double normalizer = 0;
        for (int i = 0; i < n; i++)
        {
            normalizer += Math.Exp((Logit(logits, allowed, i) - max) * filterScale);
        }
        double cumulative = 0;
        int kept = 0;
        for (; kept < keep; kept++)
        {
            cumulative += Math.Exp((scratch.Values[kept] - max) * filterScale) / normalizer;
            if (kept > 0 && cumulative > topP) break;
        }

        double total = 0;
        for (int i = 0; i < kept; i++)
        {
            total += Math.Exp((scratch.Values[i] - max) / t);
        }
        double r = DeterministicRng.NextUniform(ref rng) * total, acc = 0;
        for (int i = 0; i < kept; i++)
        {
            acc += Math.Exp((scratch.Values[i] - max) / t);
            if (r < acc) return ranked[i];
        }
        return ranked[0];
    }

    // Fills ranked/values with the k best candidates, best first (ties keep the lower index first), and returns the best logit.
    // An insertion pass costs n·k, so a wide top-k ranks everything with one sort instead.
    private static float SelectTop(ReadOnlySpan<float> logits, ReadOnlySpan<int> allowed, int n, Span<int> ranked, Span<float> values,
        Scratch scratch)
    {
        int k = ranked.Length, filled = 0;
        if (k > InsertionLimit)
        {
            float[] keys = scratch.Values;
            int[] order = scratch.Order;
            for (int i = 0; i < n; i++)
            {
                keys[i] = -Logit(logits, allowed, i);
                order[i] = i;
            }
            Array.Sort(keys, order, 0, n);
            for (int i = 0; i < k; i++) keys[i] = -keys[i];
            return keys[0];
        }
        for (int i = 0; i < n; i++)
        {
            float v = Logit(logits, allowed, i);
            if (filled == k && v <= values[k - 1]) continue;
            int at = filled < k ? filled : k - 1;
            while (at > 0 && values[at - 1] < v)
            {
                values[at] = values[at - 1];
                ranked[at] = ranked[at - 1];
                at--;
            }
            values[at] = v;
            ranked[at] = i;
            if (filled < k) filled++;
        }
        return values[0];
    }

    private static float Logit(ReadOnlySpan<float> logits, ReadOnlySpan<int> allowed, int i) => allowed.IsEmpty ? logits[i] : logits[allowed[i]];

    /// <summary>Reusable ranking buffers for <see cref="Sample(ReadOnlySpan{float}, float, float, int, ref uint, Scratch, bool, ReadOnlySpan{int})"/>.</summary>
    /// <param name="capacity">Most candidates one draw ranks.</param>
    public sealed class Scratch(int capacity)
    {
        /// <summary>Most candidates one draw ranks.</summary>
        public int Capacity { get; } = capacity > 0 ? capacity : throw new ArgumentOutOfRangeException(nameof(capacity));

        internal int[] Order { get; } = new int[capacity];

        internal float[] Values { get; } = new float[capacity];
    }
}
