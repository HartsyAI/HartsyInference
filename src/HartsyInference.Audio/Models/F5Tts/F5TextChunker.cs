using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>F5-TTS long-text chunking: upstream <c>chunk_text</c>, the <c>max_chars</c> rule of <c>infer_process</c>
/// and the linear cross-fade that joins the per-chunk waves.</summary>
public static partial class F5TextChunker
{
    /// <summary>Cross-fade length between consecutive chunk waves, in seconds.</summary>
    public const double CrossFadeSeconds = 0.15;

    private const double MaxCharsWindowSeconds = 22.0;

    /// <summary>Upstream <c>max_chars</c>: the reference's bytes per second, scaled by the room left in 22 s.</summary>
    /// <param name="refBytes">UTF-8 bytes of the normalized reference text, before the extra space.</param>
    /// <param name="refSeconds">Reference length in seconds; must be positive.</param>
    public static int MaxChars(int refBytes, double refSeconds, double speed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(refSeconds);
        return (int)(refBytes / refSeconds * (MaxCharsWindowSeconds - refSeconds) * speed);
    }

    /// <summary>Splits <paramref name="text"/> at sentence punctuation into chunks of at most
    /// <paramref name="maxChars"/> UTF-8 bytes; a sentence longer than the limit is kept whole.</summary>
    public static IReadOnlyList<string> Chunk(string text, int maxChars)
    {
        List<string> chunks = new();
        StringBuilder current = new();
        int currentBytes = 0;
        foreach (string sentence in SentenceBoundary().Split(text))
        {
            if (sentence.Length == 0) continue;
            // The split consumes the whitespace after ASCII punctuation; one space is restored, as upstream does.
            string piece = sentence[^1] < 0x80 ? sentence + " " : sentence;
            if (currentBytes + Encoding.UTF8.GetByteCount(sentence) <= maxChars)
            {
                current.Append(piece);
                currentBytes += Encoding.UTF8.GetByteCount(piece);
                continue;
            }
            if (current.Length > 0) chunks.Add(current.ToString().Trim());
            current.Clear().Append(piece);
            currentBytes = Encoding.UTF8.GetByteCount(piece);
        }
        if (current.Length > 0) chunks.Add(current.ToString().Trim());
        return chunks;
    }

    /// <summary>Samples in one cross-fade at <paramref name="sampleRate"/>.</summary>
    public static int CrossFadeSamples(int sampleRate) => (int)(CrossFadeSeconds * sampleRate);

    /// <summary>Joins the waves in order; each join overlaps <paramref name="crossFadeSamples"/> with a linear fade,
    /// clamped to the shorter wave, and a join with no overlap concatenates.</summary>
    public static float[] CrossFade(IReadOnlyList<float[]> waves, int crossFadeSamples)
    {
        if (waves.Count == 0) throw new ArgumentException("At least one wave is required.", nameof(waves));
        float[] joined = waves[0];
        for (int i = 1; i < waves.Count; i++) joined = Join(joined, waves[i], crossFadeSamples);
        return joined;
    }

    private static float[] Join(float[] prev, float[] next, int crossFadeSamples)
    {
        int n = Math.Min(crossFadeSamples, Math.Min(prev.Length, next.Length));
        if (n <= 0)
        {
            float[] concatenated = new float[prev.Length + next.Length];
            prev.CopyTo(concatenated, 0);
            next.CopyTo(concatenated, prev.Length);
            return concatenated;
        }
        int head = prev.Length - n;
        float[] outp = new float[prev.Length + next.Length - n];
        Array.Copy(prev, 0, outp, 0, head);
        for (int k = 0; k < n; k++)
        {
            double fadeIn = n == 1 ? 0.0 : k / (double)(n - 1);
            outp[head + k] = (float)(prev[head + k] * (1.0 - fadeIn) + next[k] * fadeIn);
        }
        Array.Copy(next, n, outp, head + n, next.Length - n);
        return outp;
    }

    /// <summary>Whitespace after ASCII punctuation, or directly after full-width punctuation.</summary>
    [GeneratedRegex(@"(?<=[;:,.!?])\s+|(?<=[；：，。！？])")]
    private static partial Regex SentenceBoundary();
}
