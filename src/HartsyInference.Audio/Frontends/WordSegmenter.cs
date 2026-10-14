using System.Globalization;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>The Python <c>wordsegment</c> 1.3 segmenter: splits run-together text into words ("stateoftheart" →
/// state, of, the, art) by maximizing unigram/bigram probability over Norvig's Google web corpus counts, with the
/// library's scoring, 24-letter word limit, 250-character chunking and tie-breaking.</summary>
internal sealed class WordSegmenter
{
    private const double Total = 1024908267229.0;
    private const int Limit = 24;
    private const int ChunkSize = 250;

    private static readonly Lazy<WordSegmenter> Shared = new(Load);

    private readonly Dictionary<string, double> _unigrams;
    private readonly Dictionary<string, double> _bigrams;

    private WordSegmenter(Dictionary<string, double> unigrams, Dictionary<string, double> bigrams)
    {
        _unigrams = unigrams;
        _bigrams = bigrams;
    }

    /// <summary>The segmenter over the embedded corpus counts (loaded on first use).</summary>
    public static WordSegmenter English => Shared.Value;

    /// <summary><c>wordsegment.segment(text)</c>: lower-cases, drops everything but a-z and 0-9, then segments.</summary>
    public List<string> Segment(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string clean = Clean(text);
        List<string> result = [];
        Dictionary<(string, string), (double Score, List<string> Words)> memo = new();
        string prefix = "";
        for (int offset = 0; offset < clean.Length; offset += ChunkSize)
        {
            string chunk = clean.Substring(offset, Math.Min(ChunkSize, clean.Length - offset));
            List<string> chunkWords = Search(prefix + chunk, "<s>", memo).Words;
            int keep = Math.Max(0, chunkWords.Count - 5);
            prefix = string.Concat(chunkWords.GetRange(keep, chunkWords.Count - keep));
            result.AddRange(chunkWords.GetRange(0, keep));
        }
        result.AddRange(Search(prefix, "<s>", memo).Words);
        return result;
    }

    private (double Score, List<string> Words) Search(string text, string previous,
        Dictionary<(string, string), (double, List<string>)> memo)
    {
        if (text.Length == 0) return (0.0, []);
        (double Score, List<string> Words)? best = null;
        int max = Math.Min(text.Length, Limit);
        for (int pos = 1; pos <= max; pos++)
        {
            string prefix = text[..pos], suffix = text[pos..];
            double prefixScore = Math.Log10(Score(prefix, previous));
            if (!memo.TryGetValue((suffix, prefix), out (double, List<string>) rest))
            {
                rest = Search(suffix, prefix, memo);
                memo[(suffix, prefix)] = rest;
            }
            (double restScore, List<string> restWords) = rest;
            List<string> words = new(restWords.Count + 1) { prefix };
            words.AddRange(restWords);
            (double, List<string>) candidate = (prefixScore + restScore, words);
            if (best is null || IsGreater(candidate, best.Value)) best = candidate;
        }
        return best!.Value;
    }

    // Python's max over (score, words) tuples: score first, then the word lists compared lexicographically.
    private static bool IsGreater((double Score, List<string> Words) a, (double Score, List<string> Words) b)
    {
        if (a.Score != b.Score) return a.Score > b.Score;
        int n = Math.Min(a.Words.Count, b.Words.Count);
        for (int i = 0; i < n; i++)
        {
            int c = string.CompareOrdinal(a.Words[i], b.Words[i]);
            if (c != 0) return c > 0;
        }
        return a.Words.Count > b.Words.Count;
    }

    private double Score(string word, string? previous)
    {
        if (previous is null)
        {
            if (_unigrams.TryGetValue(word, out double count)) return count / Total;
            return 10.0 / (Total * PowerOfTen(word.Length));
        }
        if (_bigrams.TryGetValue(previous + " " + word, out double pair) && _unigrams.ContainsKey(previous))
            return pair / Total / Score(previous, null);
        return Score(word, null);
    }

    // Python computes 10 ** n as an exact integer and rounds it to double once; parsing "1eN" rounds the same way.
    private static double PowerOfTen(int n) => double.Parse("1e" + n.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);

    private static string Clean(string text)
    {
        StringBuilder sb = new(text.Length);
        foreach (char ch in text.ToLowerInvariant())
            if (ch is >= 'a' and <= 'z' or >= '0' and <= '9') sb.Append(ch);
        return sb.ToString();
    }

    private static WordSegmenter Load() => new(ReadCounts("wordsegment_unigrams.txt.gz"), ReadCounts("wordsegment_bigrams.txt.gz"));

    private static Dictionary<string, double> ReadCounts(string resource)
    {
        Dictionary<string, double> counts = new(StringComparer.Ordinal);
        foreach (string line in EmbeddedTextResource.ReadLines(resource))
        {
            int tab = line.LastIndexOf('\t');
            counts[line[..tab]] = double.Parse(line.AsSpan(tab + 1), NumberStyles.Float, CultureInfo.InvariantCulture);
        }
        return counts;
    }
}
