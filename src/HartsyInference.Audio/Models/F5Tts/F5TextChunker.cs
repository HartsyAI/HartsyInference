using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Dsp;

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
    public static float[] CrossFade(IReadOnlyList<float[]> waves, int crossFadeSamples) => WaveCrossFade.Join(waves, crossFadeSamples);

    /// <summary>Whitespace after ASCII punctuation, or directly after full-width punctuation.</summary>
    [GeneratedRegex(@"(?<=[;:,.!?])\s+|(?<=[；：，。！？])")]
    private static partial Regex SentenceBoundary();
}
