using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Audio.Preprocessing;

namespace HartsyInference.Audio.Tests;

/// <summary>The comparisons the TTS regression benches share: a PCM digest, sample-wise and log-spectral
/// correlation against a reference waveform, and content-word recall of a transcript. Kept in one place so a
/// baseline run and a candidate run always score the same way.</summary>
internal static class AudioParityMetrics
{
    private static readonly HashSet<string> StopWords = new(StringComparer.Ordinal)
    {
        "a", "an", "and", "the", "of", "for", "to", "so", "my", "your", "you", "what", "can", "do", "not", "is", "it", "in", "on",
        "i", "at", "are", "will", "with", "have", "when", "they",
    };

    private static readonly string[] Ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] Tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    public static string Sha256Hex(float[] wave) =>
        Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes<float>(wave))).ToLowerInvariant();

    /// <summary>Max-abs difference and Pearson correlation over the common prefix (double accumulation).</summary>
    public static (double MaxAbs, double Corr) Compare(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(a[i] - b[i]));
        }
        return (maxAbs, Pearson(a, b, n));
    }

    /// <summary>Pearson correlation of the log-magnitude STFTs (n_fft 1024, hop 256, Hann) over the common prefix.
    /// An NSF source integrates F0 into a phase, so a sub-cent F0 change drifts the waveform's phase over seconds
    /// and sinks the sample-wise correlation while the audio is perceptually unchanged; the spectral envelope is
    /// the phase-invariant comparison that separates a real content change from that drift.</summary>
    public static double LogSpectralCorrelation(float[] a, float[] b)
    {
        const int nFft = 1024, hop = 256, bins = nFft / 2 + 1;
        int n = Math.Min(a.Length, b.Length);
        int frames = n >= nFft ? 1 + (n - nFft) / hop : 0;
        if (frames == 0) return 0;
        float[] window = HannWindow.Get(nFft);
        float[] la = new float[frames * bins], lb = new float[frames * bins];
        float[] frame = new float[nFft], re = new float[bins], im = new float[bins];
        for (int f = 0; f < frames; f++)
        {
            foreach ((float[] src, float[] dst) in new[] { (a, la), (b, lb) })
            {
                for (int k = 0; k < nFft; k++) frame[k] = src[f * hop + k] * window[k];
                Fft.RealTransform(frame, re, im, nFft);
                for (int k = 0; k < bins; k++) dst[f * bins + k] = MathF.Log(1e-5f + MathF.Sqrt(re[k] * re[k] + im[k] * im[k]));
            }
        }
        return Pearson(la, lb, la.Length);
    }

    public static double Pearson(float[] a, float[] b, int n)
    {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < n; i++)
        {
            sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
        }
        double cov = sab - sa * sb / n;
        double va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return va > 0 && vb > 0 ? cov / Math.Sqrt(va * vb) : 0;
    }

    /// <summary>Content words of <paramref name="reference"/> (stop words removed) present in <paramref name="hypothesis"/>.</summary>
    public static double ContentWordRecall(string reference, string hypothesis)
    {
        HashSet<string> hyp = new HashSet<string>(Words(hypothesis), StringComparer.Ordinal);
        List<string> content = Words(reference).Where(w => !StopWords.Contains(w)).Distinct().ToList();
        return content.Count == 0 ? 0 : content.Count(hyp.Contains) / (double)content.Count;
    }

    /// <summary>Lower-cased, punctuation-stripped words with small integers spelled out — Whisper writes "at 3" for
    /// "at three", and that is a correct transcription, not a missed word.</summary>
    public static IEnumerable<string> Words(string text)
    {
        StringBuilder sb = new StringBuilder(text.Length);
        foreach (char c in text.ToLowerInvariant())
        {
            sb.Append(char.IsLetterOrDigit(c) || c == '\'' ? c : ' ');
        }
        return sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(SpellSmallNumber);
    }

    public static string Ms(double seconds) => (seconds * 1000).ToString("F1", CultureInfo.InvariantCulture);

    public static string Cell(string text) => text.Trim().Replace("|", "\\|").Replace('\n', ' ');

    private static string SpellSmallNumber(string token)
    {
        if (!int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture, out int value) || value > 99)
        {
            return token;
        }
        if (value < 20)
        {
            return Ones[value];
        }
        return value % 10 == 0 ? Tens[value / 10] : $"{Tens[value / 10]}{Ones[value % 10]}";
    }
}
