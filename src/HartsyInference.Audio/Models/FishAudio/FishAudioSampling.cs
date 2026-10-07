using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Port of fish-speech's <c>logits_to_probs</c> + multinomial draw. Unlike the shared nucleus sampler, the
/// top-p / top-k filter runs on the UNtempered distribution (and drops the token whose inclusive cumulative
/// probability first exceeds top-p), and temperature is applied afterwards.</summary>
public static class FishAudioSampling
{
    /// <summary>Draws a vocab id. <paramref name="allowed"/> restricts the candidates (the logit-bias constraint on the
    /// slow head: semantic ids + <c>&lt;|im_end|&gt;</c>); null = the whole vocabulary.</summary>
    public static int Sample(ReadOnlySpan<float> logits, float temperature, float topP, int topK, ref uint rng,
        ReadOnlySpan<int> allowed = default)
    {
        int n = allowed.IsEmpty ? logits.Length : allowed.Length;
        int[] order = new int[n];
        float[] copy = new float[logits.Length];
        logits.CopyTo(copy);
        for (int i = 0; i < n; i++) order[i] = allowed.IsEmpty ? i : allowed[i];
        Array.Sort(order, (a, b) => copy[b].CompareTo(copy[a]));

        // Cumulative probability over the untempered, descending-sorted logits.
        double max = copy[order[0]], sum = 0;
        double[] e = new double[n];
        for (int i = 0; i < n; i++) { e[i] = Math.Exp(copy[order[i]] - max); sum += e[i]; }

        bool[] keep = new bool[n];
        double cum = 0;
        for (int i = 0; i < n; i++)
        {
            cum += e[i] / sum;
            keep[i] = i == 0 || (cum <= topP && i < topK);
        }

        float t = MathF.Max(temperature, 1e-5f);
        double m2 = copy[order[0]] / t, z = 0;
        double[] w = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (!keep[i]) continue;
            w[i] = Math.Exp(copy[order[i]] / t - m2);
            z += w[i];
        }
        double r = DeterministicRng.NextUniform(ref rng) * z, acc = 0;
        for (int i = 0; i < n; i++)
        {
            if (!keep[i]) continue;
            acc += w[i];
            if (r < acc) return order[i];
        }
        return order[0];
    }
}
