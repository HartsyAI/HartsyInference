using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>GPT-SoVITS's text preparation around the G2P (upstream <c>TTS_infer_pack/TextPreprocessor.py</c> and
/// <c>text_segmentation_method.py</c>): the target text is cut into segments that are synthesized one at a time,
/// each with the reference transcript's phones in front, and joined with a short silence. Cutting, the short-text
/// "." prefixes, merging of tiny pieces, the final-punctuation rule and the under-six-phone retry all change the
/// phones the model sees, so they follow upstream exactly.</summary>
public static partial class GptSoVitsTextSegmenter
{
    /// <summary>Upstream's sentence-ending marks (<c>splits</c>).</summary>
    public static readonly IReadOnlySet<char> Splits =
        new HashSet<char> { '，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…' };

    private static readonly HashSet<char> CutPunctuation = ['!', '?', '…', ',', '.', '-', ' '];
    private static readonly HashSet<char> Cut5Marks = [',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '：', '…'];

    /// <summary>Upstream <c>TextPreprocessor.preprocess(text, "en", method)</c>: the English segments to synthesize, in
    /// order. Segments whose normalized text is empty are dropped, as upstream does.</summary>
    public static List<GptSoVitsTextSegment> PrepareEnglish(string text, GptSoVitsCutMethod method = GptSoVitsCutMethod.Cut1)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<GptSoVitsTextSegment> segments = [];
        foreach (string piece in PreSegment(ReplaceConsecutivePunctuation(text), "en", method))
        {
            GptSoVitsTextSegment segment = EnglishPhones(piece);
            if (segment.NormalizedText.Length > 0) segments.Add(segment);
        }
        return segments;
    }

    /// <summary>The reference transcript's phones as upstream <c>TTS.run</c> prepares them: final punctuation added
    /// when missing, then the same phone pass as a segment (no cutting).</summary>
    public static GptSoVitsTextSegment PrepareEnglishPrompt(string promptText)
    {
        ArgumentNullException.ThrowIfNull(promptText);
        string prompt = promptText.Trim('\n');
        if (prompt.Length == 0) throw new ArgumentException("The reference transcript is empty.", nameof(promptText));
        if (!Splits.Contains(prompt[^1])) prompt += ".";
        return EnglishPhones(prompt);
    }

    /// <summary>Upstream <c>TextPreprocessor.pre_seg_text</c>: the cut pieces of <paramref name="text"/>, each ending in
    /// a sentence mark. <paramref name="language"/> picks the mark added ("." for "en", "。" otherwise).</summary>
    public static List<string> PreSegment(string text, string language, GptSoVitsCutMethod method)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(language);
        string mark = language == "en" ? "." : "。";
        text = text.Trim('\n');
        if (text.Length == 0) return [];
        if (!Splits.Contains(text[0]) && FirstSentence(text).Length < 4) text = mark + text;
        text = Cut(text, method);
        while (text.Contains("\n\n", StringComparison.Ordinal)) text = text.Replace("\n\n", "\n", StringComparison.Ordinal);

        List<string> pieces = [];
        bool anyText = false;
        foreach (string piece in text.Split('\n'))
        {
            if (piece is not ("" or " " or "\n")) anyText = true;
            if (piece is not ("" or " ")) pieces.Add(piece);
        }
        if (!anyText) throw new ArgumentException("The text has nothing to speak.", nameof(text));
        pieces = MergeShort(pieces, 5);

        List<string> result = [];
        foreach (string p in pieces)
        {
            string piece = p;
            if (piece.Trim().Length == 0 || NonWordRegex().Replace(piece, "").Length == 0) continue;
            if (!Splits.Contains(piece[^1])) piece += mark;
            if (piece.Length > 510) result.AddRange(SplitBig(piece, 510));
            else result.Add(piece);
        }
        return result;
    }

    /// <summary>Upstream cut method <paramref name="method"/> over <paramref name="text"/>: pieces joined by newlines.</summary>
    public static string Cut(string text, GptSoVitsCutMethod method)
    {
        ArgumentNullException.ThrowIfNull(text);
        return method switch
        {
            GptSoVitsCutMethod.Cut0 => IsAll(text, CutPunctuation) ? "\n" : text,
            GptSoVitsCutMethod.Cut1 => Cut1(text),
            GptSoVitsCutMethod.Cut2 => Cut2(text),
            GptSoVitsCutMethod.Cut3 => Cut3(text),
            GptSoVitsCutMethod.Cut4 => Cut4(text),
            GptSoVitsCutMethod.Cut5 => Cut5(text),
            _ => throw new ArgumentOutOfRangeException(nameof(method), method, "Unknown GPT-SoVITS cut method."),
        };
    }

    /// <summary>Upstream <c>TextPreprocessor.replace_consecutive_punctuation</c>: a run of <c>!?…,.-</c> keeps its
    /// first mark.</summary>
    public static string ReplaceConsecutivePunctuation(string text) => PunctuationRunRegex().Replace(text, "$1");

    // TextPreprocessor.get_phones_and_bert for "en", without the all-zero BERT features.
    private static GptSoVitsTextSegment EnglishPhones(string text)
    {
        text = MultiSpaceRegex().Replace(text, " ");
        string norm = GptSoVitsEnglishG2P.Normalize(text);
        List<string> phones = GptSoVitsFrontend.CleanText(text, "en").Phones;
        if (phones.Count < 6)
        {
            text = "." + text;
            norm = GptSoVitsEnglishG2P.Normalize(text);
            phones = GptSoVitsFrontend.CleanText(text, "en").Phones;
        }
        return new GptSoVitsTextSegment(text, norm, phones);
    }

    private static string FirstSentence(string text)
    {
        int end = 0;
        while (end < text.Length && !Splits.Contains(text[end])) end++;
        return text[..end].Trim();
    }

    private static List<string> MergeShort(List<string> texts, int threshold)
    {
        if (texts.Count < 2) return texts;
        List<string> result = [];
        StringBuilder current = new();
        foreach (string t in texts)
        {
            current.Append(t);
            if (current.Length >= threshold)
            {
                result.Add(current.ToString());
                current.Clear();
            }
        }
        if (current.Length > 0)
        {
            if (result.Count == 0) result.Add(current.ToString());
            else result[^1] += current.ToString();
        }
        return result;
    }

    private static List<string> SplitBig(string text, int maxLength)
    {
        List<string> result = [];
        string current = "";
        foreach (string piece in SplitsCaptureRegex().Split(text))
        {
            if (current.Length + piece.Length > maxLength)
            {
                result.Add(current);
                current = piece;
            }
            else current += piece;
        }
        if (current.Length > 0) result.Add(current);
        return result;
    }

    // text_segmentation_method.split: sentences ending at each mark, a "。" appended when the text has no final mark.
    private static List<string> SplitSentences(string text)
    {
        text = text.Replace("……", "。", StringComparison.Ordinal).Replace("——", "，", StringComparison.Ordinal);
        if (!Splits.Contains(text[^1])) text += "。";
        List<string> sentences = [];
        int tail = 0;
        for (int head = 0; head < text.Length; head++)
        {
            if (!Splits.Contains(text[head])) continue;
            sentences.Add(text[tail..(head + 1)]);
            tail = head + 1;
        }
        return sentences;
    }

    private static string Cut1(string text)
    {
        text = text.Trim('\n');
        List<string> sentences = SplitSentences(text);
        List<string> opts = [];
        if (sentences.Count > 4)
        {
            // range(0, n, 4) with the last start replaced by the end: the final group absorbs the remainder.
            int groups = (sentences.Count + 3) / 4 - 1;
            for (int g = 0; g < groups; g++)
            {
                int start = g * 4, end = g == groups - 1 ? sentences.Count : start + 4;
                opts.Add(string.Concat(sentences.GetRange(start, end - start)));
            }
        }
        else opts.Add(text);
        return JoinSpoken(opts, CutPunctuation);
    }

    private static string Cut2(string text)
    {
        text = text.Trim('\n');
        List<string> sentences = SplitSentences(text);
        if (sentences.Count < 2) return text;
        List<string> opts = [];
        int sum = 0;
        StringBuilder tmp = new();
        foreach (string s in sentences)
        {
            sum += s.Length;
            tmp.Append(s);
            if (sum > 50)
            {
                sum = 0;
                opts.Add(tmp.ToString());
                tmp.Clear();
            }
        }
        if (tmp.Length > 0) opts.Add(tmp.ToString());
        if (opts.Count > 1 && opts[^1].Length < 50)
        {
            opts[^2] += opts[^1];
            opts.RemoveAt(opts.Count - 1);
        }
        return JoinSpoken(opts, CutPunctuation);
    }

    private static string Cut3(string text)
    {
        text = text.Trim('\n');
        return JoinSpoken([.. text.Trim('。').Split('。')], CutPunctuation);
    }

    private static string Cut4(string text)
    {
        text = text.Trim('\n');
        return JoinSpoken([.. EnglishPeriodRegex().Split(text.Trim('.'))], CutPunctuation);
    }

    private static string Cut5(string text)
    {
        text = text.Trim('\n');
        List<string> merged = [];
        StringBuilder items = new();
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            items.Append(ch);
            if (!Cut5Marks.Contains(ch)) continue;
            bool decimalPoint = ch == '.' && i > 0 && i < text.Length - 1 && char.IsDigit(text[i - 1]) && char.IsDigit(text[i + 1]);
            if (decimalPoint) continue;
            merged.Add(items.ToString());
            items.Clear();
        }
        if (items.Length > 0) merged.Add(items.ToString());
        return JoinSpoken(merged, Cut5Marks);
    }

    // Drops pieces made only of the given marks (Python set(item).issubset(marks); an empty piece is dropped too).
    private static string JoinSpoken(List<string> pieces, HashSet<char> marks)
    {
        List<string> kept = [];
        foreach (string p in pieces)
            if (!IsAll(p, marks)) kept.Add(p);
        return string.Join("\n", kept);
    }

    private static bool IsAll(string text, HashSet<char> marks)
    {
        foreach (char ch in text)
            if (!marks.Contains(ch)) return false;
        return true;
    }

    [GeneratedRegex(@"([!?…,.\-])([!?…,.\-])+")]
    private static partial Regex PunctuationRunRegex();

    [GeneratedRegex(@" {2,}")]
    private static partial Regex MultiSpaceRegex();

    [GeneratedRegex(@"\W+")]
    private static partial Regex NonWordRegex();

    [GeneratedRegex(@"([，。？！,.?!~:：—…])")]
    private static partial Regex SplitsCaptureRegex();

    [GeneratedRegex(@"(?<!\d)\.(?!\d)")]
    private static partial Regex EnglishPeriodRegex();
}
