using System.Text.Json;
using System.Text.RegularExpressions;

namespace SharpInference.Tokenizers;

/// <summary>ACE-Step's lyric tokenizer — the XTTS-v2 <c>VoiceBpeTokenizer</c> (vocab ~6693) plus the
/// <c>tokenize_lyrics</c> line protocol: stream starts with token 261, every line is BPE-encoded with an XTTS-style
/// language prefix (<c>[en]…</c>), spaces become the <c>[SPACE]</c> special token, structure tags like
/// <c>[verse]</c>/<c>[chorus]</c> are byte-encoded as English, and every line (and each blank line) is terminated by
/// token 2. Loads a <c>vocab.json</c> + <c>merges.txt</c> pair (export once from the HF <c>tokenizer.json</c> in the
/// checkpoint repo — see the converter docs). Language auto-detection is a per-line Unicode-script heuristic;
/// pass explicit codes for mixed-script lines. CJK G2P preprocessing is the caller's responsibility (matches the
/// official pipeline's external phonemizer). Token-stream equality vs Python is validation-gated.</summary>
public sealed partial class AceStepLyricTokenizer
{
    /// <summary>Stream-opening token.</summary>
    public const int StartToken = 261;

    /// <summary>End-of-line / blank-line separator token.</summary>
    public const int LineSeparatorToken = 2;

    [GeneratedRegex(@"^\s*\[[^\]]+\]\s*$")]
    private static partial Regex StructureTagPattern();

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _mergeRanks;
    private readonly List<string> _specialTokens;

    public AceStepLyricTokenizer(string vocabJsonPath, string mergesPath)
    {
        using FileStream fs = File.OpenRead(vocabJsonPath);
        _vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(fs)
            ?? throw new InvalidDataException($"empty vocab at {vocabJsonPath}");
        _mergeRanks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (string line in File.ReadLines(mergesPath))
        {
            if (line.Length == 0 || line.StartsWith('#')) continue;
            int space = line.IndexOf(' ');
            if (space <= 0) continue;
            _mergeRanks[(line[..space], line[(space + 1)..])] = rank++;
        }
        // Specials = bracketed vocab entries ([SPACE], [en], [zh], …), longest-first for greedy matching.
        _specialTokens = [.. _vocab.Keys.Where(k => k.Length > 2 && k[0] == '[' && k[^1] == ']')
            .OrderByDescending(k => k.Length)];
    }

    /// <summary>Vocabulary size.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>The full ACE-Step lyric protocol: <c>[261]</c> then per line tokens + <c>[2]</c> (bare 2 for blank
    /// lines). <paramref name="languageOverride"/> forces one code for all lines (otherwise per-line heuristic).</summary>
    public int[] TokenizeLyrics(string lyrics, string? languageOverride = null)
    {
        List<int> ids = [StartToken];
        foreach (string rawLine in lyrics.Split('\n'))
        {
            string line = rawLine.TrimEnd('\r');
            if (string.IsNullOrWhiteSpace(line))
            {
                ids.Add(LineSeparatorToken);
                continue;
            }
            string lang = StructureTagPattern().IsMatch(line) ? "en" : languageOverride ?? DetectLanguage(line);
            ids.AddRange(EncodeLine(line, lang));
            ids.Add(LineSeparatorToken);
        }
        return [.. ids];
    }

    /// <summary>Encodes one line XTTS-style: lowercase, <c>[lang]</c> prefix, spaces → <c>[SPACE]</c>, then BPE.</summary>
    public int[] EncodeLine(string text, string lang)
    {
        string prepared = $"[{lang}]{text.Trim().ToLowerInvariant()}".Replace(" ", "[SPACE]");
        List<int> ids = [];
        foreach (string segment in SplitSpecials(prepared))
        {
            if (_vocab.TryGetValue(segment, out int special) && segment[0] == '[')
            {
                ids.Add(special);
                continue;
            }
            ids.AddRange(Bpe(segment));
        }
        return [.. ids];
    }

    /// <summary>Splits text into special-token and plain runs (greedy, longest-first specials).</summary>
    private IEnumerable<string> SplitSpecials(string text)
    {
        int pos = 0;
        int runStart = 0;
        while (pos < text.Length)
        {
            string? match = null;
            if (text[pos] == '[')
                foreach (string sp in _specialTokens)
                    if (pos + sp.Length <= text.Length && string.CompareOrdinal(text, pos, sp, 0, sp.Length) == 0)
                    {
                        match = sp;
                        break;
                    }
            if (match is not null)
            {
                if (pos > runStart) yield return text[runStart..pos];
                yield return match;
                pos += match.Length;
                runStart = pos;
            }
            else
            {
                pos++;
            }
        }
        if (runStart < text.Length) yield return text[runStart..];
    }

    /// <summary>Character-level BPE with ranked merges; unknown symbols are skipped.</summary>
    private IEnumerable<int> Bpe(string segment)
    {
        List<string> parts = [.. segment.Select(ch => ch.ToString())];
        while (parts.Count > 1)
        {
            int bestRank = int.MaxValue, bestIdx = -1;
            for (int i = 0; i < parts.Count - 1; i++)
                if (_mergeRanks.TryGetValue((parts[i], parts[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestIdx = i;
                }
            if (bestIdx < 0) break;
            parts[bestIdx] += parts[bestIdx + 1];
            parts.RemoveAt(bestIdx + 1);
        }
        foreach (string p in parts)
            if (_vocab.TryGetValue(p, out int id))
                yield return id;
    }

    /// <summary>Per-line script heuristic over ACE-Step's 17 supported languages (the official pipeline uses a
    /// statistical detector; this covers the script-distinct cases and defaults to English for Latin text).</summary>
    public static string DetectLanguage(string line)
    {
        foreach (char ch in line)
        {
            if (ch >= 0x3040 && ch <= 0x30FF) return "ja";          // hiragana/katakana
            if (ch >= 0xAC00 && ch <= 0xD7AF) return "ko";          // hangul
            if (ch >= 0x4E00 && ch <= 0x9FFF) return "zh";          // han (ja kanji-only lines will misdetect)
            if (ch >= 0x0400 && ch <= 0x04FF) return "ru";          // cyrillic
            if (ch >= 0x0600 && ch <= 0x06FF) return "ar";          // arabic
            if (ch >= 0x0900 && ch <= 0x097F) return "hi";          // devanagari
        }
        return "en";
    }
}
