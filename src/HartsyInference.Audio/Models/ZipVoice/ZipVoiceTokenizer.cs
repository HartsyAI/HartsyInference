using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Phonemizer;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>Ports ZipVoice's <c>EmiliaTokenizer</c> for English text: <c>map_punctuations</c>, segmentation into text and
/// <c>[tag]</c> spans, the <see cref="ZipVoiceEnglishNormalizer"/> rules, and piper_phonemize's clause-by-clause espeak
/// phonemization with its marks, split into single Unicode code points and looked up in the checkpoint's
/// <c>tokens.txt</c>, where unknown symbols are skipped. Chinese and pinyin need jieba and pypinyin and are not ported.</summary>
public sealed partial class ZipVoiceTokenizer
{
    /// <summary>The espeak voice upstream phonemizes with; the British <c>en</c> voice yields phonemes the model never saw.</summary>
    public const string EspeakLanguage = "en-us";

    private readonly Dictionary<string, int> _token2id;

    public int PadId { get; }
    public int VocabSize => _token2id.Count;

    private ZipVoiceTokenizer(Dictionary<string, int> token2id)
    {
        _token2id = token2id;
        PadId = token2id["_"];
    }

    /// <summary>Loads <c>tokens.txt</c> (<c>{token}\t{id}</c> per line, as shipped with the real
    /// <c>k2-fsa/ZipVoice</c> checkpoint).</summary>
    public static ZipVoiceTokenizer FromFile(string tokensPath)
    {
        Dictionary<string, int> map = new();
        foreach (string line in File.ReadLines(tokensPath))
        {
            if (line.Length == 0) continue;
            int tab = line.LastIndexOf('\t');
            if (tab < 0) continue;
            string token = line[..tab];
            if (!int.TryParse(line[(tab + 1)..], out int id)) continue;
            map[token] = id;
        }
        return new ZipVoiceTokenizer(map);
    }

    /// <summary>Upstream <c>map_punctuations</c>: full-width marks to ASCII, curly quotes to straight, ellipses to <c>…</c>.</summary>
    public static string NormalizePunctuation(string text) => text
        .Replace('，', ',').Replace('。', '.').Replace('！', '!').Replace('？', '?')
        .Replace('；', ';').Replace('：', ':').Replace('、', ',')
        .Replace('‘', '\'').Replace('’', '\'').Replace('“', '"').Replace('”', '"')
        .Replace("⋯", "…").Replace("···", "…").Replace("・・・", "…").Replace("...", "…");

    /// <summary>Upstream <c>add_punctuation</c>: trims the text and ends it with a period when it ends with no sentence mark.</summary>
    public static string AddPunctuation(string text)
    {
        string trimmed = text.Trim();
        if (trimmed.Length == 0) throw new ArgumentException("ZipVoice text is empty.", nameof(text));
        return ZipVoiceTextChunker.SentencePunctuation.Contains(trimmed[^1]) ? trimmed : trimmed + ".";
    }

    /// <summary>Text to token symbols, before the vocabulary lookup: what upstream's <c>texts_to_tokens</c> returns.</summary>
    public IReadOnlyList<string> Tokenize(string text, IPhonemizer phonemizer, string language = EspeakLanguage)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(phonemizer);
        if (ContainsHan(text))
            throw new NotSupportedException("ZipVoice's Chinese front end needs jieba and pypinyin; use English text.");
        List<string> tokens = new();
        // Upstream's segmenter drops newlines, which glues the words either side; here they read as spaces.
        foreach ((string segment, SegmentKind kind) in Segment(NormalizePunctuation(text.Replace('\n', ' '))))
        {
            switch (kind)
            {
                case SegmentKind.English:
                    tokens.AddRange(Symbols(phonemizer, ZipVoiceEnglishNormalizer.Normalize(segment), language));
                    break;
                case SegmentKind.Tag:
                    tokens.Add(segment);
                    break;
            }
        }
        return tokens;
    }

    /// <summary>Token symbols to ids; a symbol missing from the vocabulary is skipped, as upstream does.</summary>
    public int[] ToIds(IEnumerable<string> tokens)
    {
        List<int> ids = new();
        foreach (string token in tokens)
        {
            if (_token2id.TryGetValue(token, out int id)) ids.Add(id);
        }
        return [.. ids];
    }

    /// <summary>Text to token ids: <see cref="Tokenize"/> then <see cref="ToIds"/>.</summary>
    public int[] Encode(string text, IPhonemizer phonemizer, string language = EspeakLanguage) =>
        ToIds(Tokenize(text, phonemizer, language));

    /// <summary>The published overload for the bundled espeak port; same as the <see cref="IPhonemizer"/> one.</summary>
    public int[] Encode(string text, EspeakPhonemizer phonemizer, string language = EspeakLanguage) =>
        Encode(text, (IPhonemizer)phonemizer, language);

    // piper_phonemize: each clause is decomposed (NFD), a mark follows it, and `, ; :` add a space; clauses join with nothing.
    // NFD does nothing under invariant globalization (SwarmUI's), which only matters for precomposed letters en-us never emits.
    private static List<string> Symbols(IPhonemizer phonemizer, string text, string language)
    {
        StringBuilder phonemes = new();
        foreach (EspeakClause clause in EspeakClauseReader.Read(text))
        {
            if (string.IsNullOrWhiteSpace(clause.Text)) continue;
            string ipa = phonemizer.PhonemizeToIpa(clause.Text, language).Trim();
            if (ipa.Length == 0) continue;
            phonemes.Append(ipa.Normalize(NormalizationForm.FormD));
            if (clause.Mark == '\0') continue;
            phonemes.Append(clause.Mark);
            if (clause.Mark is ',' or ';' or ':') phonemes.Append(' ');
        }
        List<string> symbols = new();
        foreach (Rune rune in phonemes.ToString().EnumerateRunes()) symbols.Add(rune.ToString());
        return symbols;
    }

    // Upstream get_segment + split_segments: ASCII letters make English runs that digits, spaces and punctuation join;
    // a <...> span is pinyin (skipped, it needs pypinyin) and a [...] span is a single tag token.
    private static List<(string Text, SegmentKind Kind)> Segment(string text)
    {
        List<(string, SegmentKind)> runs = new();
        StringBuilder run = new();
        SegmentKind kind = SegmentKind.Other;
        bool first = true;
        foreach (Match part in Parts().Matches(text))
        {
            SegmentKind type = TypeOf(part.Value);
            if (first || kind == SegmentKind.Other)
            {
                run.Append(part.Value);
                kind = type;
            }
            else if (type == kind || type == SegmentKind.Other)
            {
                run.Append(part.Value);
            }
            else
            {
                runs.Add((run.ToString(), kind));
                run.Clear().Append(part.Value);
                kind = type;
            }
            first = false;
        }
        runs.Add((run.ToString(), kind));

        List<(string Text, SegmentKind Kind)> segments = new();
        foreach ((string runText, SegmentKind runKind) in runs)
        {
            foreach (string piece in Spans().Split(runText))
            {
                if (piece.Length == 0) continue;
                if (piece[0] == '<' && piece[^1] == '>') Logs.Warning($"[ZipVoice] Skipped pinyin span '{piece}'.");
                else if (piece[0] == '[' && piece[^1] == ']') segments.Add((piece, SegmentKind.Tag));
                else if (runKind != SegmentKind.Pinyin) segments.Add((piece, runKind));
            }
        }
        return segments;
    }

    private static SegmentKind TypeOf(string part)
    {
        if (part[0] == '<' && part[^1] == '>') return SegmentKind.Pinyin;
        return part.Length == 1 && part[0] is ((>= 'A' and <= 'Z') or (>= 'a' and <= 'z')) ? SegmentKind.English : SegmentKind.Other;
    }

    // Upstream's is_chinese stops at U+9FA5; every Han block is refused here so none is phonemized as English.
    private static bool ContainsHan(string text)
    {
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (rune.Value is (>= 0x3400 and <= 0x4DBF) or (>= 0x4E00 and <= 0x9FFF) or (>= 0xF900 and <= 0xFAFF)
                or (>= 0x20000 and <= 0x323AF)) return true;
        }
        return false;
    }

    // A single character, or a <...> / [...] span read as one part.
    [GeneratedRegex(@"[<\[].*?[>\]]|.")]
    private static partial Regex Parts();

    [GeneratedRegex(@"([<\[].*?[>\]])")]
    private static partial Regex Spans();

    private enum SegmentKind
    {
        English,
        Pinyin,
        Other,
        Tag,
    }
}
