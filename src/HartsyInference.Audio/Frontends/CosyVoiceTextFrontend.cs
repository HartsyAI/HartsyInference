using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>CosyVoice 2's text frontend (<c>cosyvoice/cli/frontend.py</c> <c>text_normalize</c> with the <c>wetext</c>
/// backend, and <c>cosyvoice/utils/frontend_utils.py</c>): text normalization, digit spell-out and the sentence split
/// that decides how many separate LM runs a request takes.</summary>
/// <remarks>The reference synthesizes each returned segment with its own LM/flow/vocoder pass and the same voice prompt,
/// and normalizes the prompt transcript with <c>split=False</c>. The text-normalization FSTs themselves are injected
/// (<paramref name="englishNormalizer"/>/<paramref name="chineseNormalizer"/>): without them the frontend behaves as the
/// reference does when no frontend tool is installed (spell-out and splitting only).</remarks>
public sealed partial class CosyVoiceTextFrontend(
    Func<string, int> countTokens,
    Func<string, string>? englishNormalizer = null,
    Func<string, string>? chineseNormalizer = null)
{
    /// <summary>The literals CosyVoice 2's tokenizer adds on top of Qwen (<c>cosyvoice/tokenizer/tokenizer.py</c>
    /// <c>CosyVoice2Tokenizer</c>) with the ids the reference assigns them on the CosyVoice-BlankEN vocabulary.</summary>
    public static readonly IReadOnlyDictionary<string, int> SpecialTokens = new Dictionary<string, int>
    {
        ["<|endoftext|>"] = 151643, ["<|im_start|>"] = 151644, ["<|im_end|>"] = 151645, ["<|endofprompt|>"] = 151646,
        ["[breath]"] = 151647, ["<strong>"] = 151648, ["</strong>"] = 151649, ["[noise]"] = 151650,
        ["[laughter]"] = 151651, ["[cough]"] = 151652, ["[clucking]"] = 151653, ["[accent]"] = 151654,
        ["[quick_breath]"] = 151655, ["<laughter>"] = 151656, ["</laughter>"] = 151657, ["[hissing]"] = 151658,
        ["[sigh]"] = 151659, ["[vocalized-noise]"] = 151660, ["[lipsmack]"] = 151661, ["[mn]"] = 151662,
    };

    private const int TokenMaxN = 80;
    private const int TokenMinN = 60;
    private const int MergeLen = 20;

    private static readonly char[] EnglishSplit = ['.', '?', '!', ';', ':'];
    private static readonly char[] ChineseSplit = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';'];

    private readonly Func<string, int> _countTokens = countTokens ?? throw new ArgumentNullException(nameof(countTokens));

    [GeneratedRegex(@"[一-鿿]+")]
    private static partial Regex ChineseChars();

    [GeneratedRegex(@"[，,、]+$")]
    private static partial Regex TrailingChineseCommas();

    [GeneratedRegex(@"^[\p{P}\p{S}]*$")]
    private static partial Regex OnlyPunctuation();

    /// <summary>The synthesis segments for <paramref name="text"/> (reference <c>text_normalize(text, split=True)</c>).
    /// Segments that are only punctuation are dropped.</summary>
    public List<string> Segments(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (IsSsml(text) || text.Length == 0) return [text];
        (string normalized, bool chinese) = NormalizeCore(text);
        List<string> segments = SplitParagraph(normalized, chinese);
        segments.RemoveAll(static s => OnlyPunctuation().IsMatch(s));
        return segments;
    }

    /// <summary>The normalized text without splitting (reference <c>text_normalize(text, split=False)</c>), used for the
    /// prompt transcript.</summary>
    public string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (IsSsml(text) || text.Length == 0) return text;
        return NormalizeCore(text).Text;
    }

    private static bool IsSsml(string text) => text.Contains("<|", StringComparison.Ordinal) && text.Contains("|>", StringComparison.Ordinal);

    private (string Text, bool Chinese) NormalizeCore(string text)
    {
        text = text.Trim();
        if (ChineseChars().IsMatch(text))
        {
            if (chineseNormalizer is not null) text = chineseNormalizer(text);
            text = text.Replace("\n", "", StringComparison.Ordinal);
            text = ReplaceBlank(text);
            text = text.Replace("²", "平方", StringComparison.Ordinal).Replace("³", "立方", StringComparison.Ordinal);
            text = text.Replace(".", "。", StringComparison.Ordinal);
            text = text.Replace(" - ", "，", StringComparison.Ordinal);
            text = RemoveBracket(text);
            text = TrailingChineseCommas().Replace(text, "。");
            return (text, true);
        }
        if (englishNormalizer is not null) text = englishNormalizer(text);
        return (SpellOutDigitRuns(text), false);
    }

    /// <summary>Reference <c>spell_out_number</c>: every run of digits becomes inflect's <c>number_to_words</c> reading,
    /// everything else is kept. Python's <c>str.isdigit</c> also accepts non-ASCII decimal digits, which <c>int()</c>
    /// reads by value, so those runs are mapped to ASCII first.</summary>
    internal static string SpellOutDigitRuns(string text)
    {
        StringBuilder sb = new(text.Length);
        int start = -1;
        for (int i = 0; i <= text.Length; i++)
        {
            if (i < text.Length && char.IsDigit(text[i]))
            {
                if (start < 0) start = i;
                continue;
            }
            if (start >= 0)
            {
                StringBuilder digits = new(i - start);
                for (int j = start; j < i; j++) digits.Append((char)('0' + (int)char.GetNumericValue(text[j])));
                sb.Append(InflectNumberWords.NumberToWords(digits.ToString()));
                start = -1;
            }
            if (i < text.Length) sb.Append(text[i]);
        }
        return sb.ToString();
    }

    /// <summary>Reference <c>split_paragraph</c> with <c>token_max_n=80, token_min_n=60, merge_len=20,
    /// comma_split=False</c>: split after sentence punctuation (keeping a following closing quote), greedily pack
    /// sentences up to the token budget, and merge a short tail into the previous segment.</summary>
    internal List<string> SplitParagraph(string text, bool chinese)
    {
        char[] marks = chinese ? ChineseSplit : EnglishSplit;
        if (text.Length == 0) return [];
        if (Array.IndexOf(marks, text[^1]) < 0) text += chinese ? "。" : ".";

        List<string> utts = [];
        int st = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (Array.IndexOf(marks, text[i]) < 0) continue;
            if (i - st > 0) utts.Add(text[st..i] + text[i]);
            if (i + 1 < text.Length && (text[i + 1] == '"' || text[i + 1] == '”'))
            {
                // Mirrors the reference exactly, including its pop of the last utterance even when this mark added none.
                if (utts.Count > 0)
                {
                    string tmp = utts[^1];
                    utts.RemoveAt(utts.Count - 1);
                    utts.Add(tmp + text[i + 1]);
                }
                st = i + 2;
            }
            else
            {
                st = i + 1;
            }
        }

        List<string> final = [];
        string cur = "";
        foreach (string utt in utts)
        {
            if (Length(cur + utt, chinese) > TokenMaxN && Length(cur, chinese) > TokenMinN)
            {
                final.Add(cur);
                cur = "";
            }
            cur += utt;
        }
        if (cur.Length > 0)
        {
            if (Length(cur, chinese) < MergeLen && final.Count != 0) final[^1] += cur;
            else final.Add(cur);
        }
        return final;
    }

    // Python's len counts code points, not UTF-16 units; they differ for astral characters (CJK Ext-B, emoji).
    private int Length(string s, bool chinese) => chinese ? CodePoints(s) : _countTokens(s);

    private static int CodePoints(string s)
    {
        int n = 0;
        foreach (Rune _ in s.EnumerateRunes()) n++;
        return n;
    }

    // Reference replace_blank: drop a space unless both neighbours are non-space ASCII.
    private static string ReplaceBlank(string text)
    {
        StringBuilder sb = new(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c != ' ')
            {
                sb.Append(c);
                continue;
            }
            // Python's text[i + 1] / text[i - 1]: i - 1 wraps to the last char at i == 0, and i + 1 past the end raises
            // IndexError (the Chinese normalizer can leave a trailing space). Drop that space instead of failing.
            if (i + 1 >= text.Length) continue;
            char next = text[i + 1];
            char prev = text[i == 0 ? text.Length - 1 : i - 1];
            if (next < 128 && next != ' ' && prev < 128 && prev != ' ') sb.Append(c);
        }
        return sb.ToString();
    }

    private static string RemoveBracket(string text) =>
        text.Replace("（", "", StringComparison.Ordinal).Replace("）", "", StringComparison.Ordinal)
            .Replace("【", "", StringComparison.Ordinal).Replace("】", "", StringComparison.Ordinal)
            .Replace("`", "", StringComparison.Ordinal)
            .Replace("——", " ", StringComparison.Ordinal);
}
