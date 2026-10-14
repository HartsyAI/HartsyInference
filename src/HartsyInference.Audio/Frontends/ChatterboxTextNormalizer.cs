using System.Globalization;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Chatterbox's text cleanup before tokenization, a port of upstream <c>punc_norm</c>
/// (resemble-ai/chatterbox <c>src/chatterbox/tts.py</c>).</summary>
/// <remarks>Steps run in upstream order: empty text becomes a fixed prompt; a lowercase first character is
/// uppercased (before any whitespace is touched, so leading whitespace leaves the text uncapitalized); whitespace runs
/// collapse to one space as Python's <c>str.split()</c> does; LLM punctuation is rewritten; trailing spaces go; and a
/// full stop is appended unless the text already ends in <c>. ! ? - ,</c>. "Lowercase" means any character with a
/// distinct uppercase that is not titlecase, as Python's <c>islower()</c> sees it, and uppercasing uses Python's full
/// case mapping (<c>ß</c> becomes <c>SS</c>).</remarks>
public static class ChatterboxTextNormalizer
{
    /// <summary>What upstream speaks for empty input.</summary>
    public const string EmptyTextPrompt = "You need to add some text for me to talk.";

    // Python str.upper() results .NET invariant casing lacks: SpecialCasing expansions, and dotless i (no invariant mapping).
    private static readonly Dictionary<int, string> PythonUpperOverrides = new()
    {
        [0x0131] = "I", [0x00DF] = "SS", [0x0149] = "ʼN", [0x01F0] = "J̌", [0x0390] = "Ϊ́",
        [0x03B0] = "Ϋ́", [0x0587] = "ԵՒ", [0x1E96] = "H̱", [0x1E97] = "T̈",
        [0x1E98] = "W̊", [0x1E99] = "Y̊", [0x1E9A] = "Aʾ", [0x1F50] = "Υ̓",
        [0x1F52] = "Υ̓̀", [0x1F54] = "Υ̓́", [0x1F56] = "Υ̓͂",
        [0x1F80] = "ἈΙ", [0x1F81] = "ἉΙ", [0x1F82] = "ἊΙ", [0x1F83] = "ἋΙ",
        [0x1F84] = "ἌΙ", [0x1F85] = "ἍΙ", [0x1F86] = "ἎΙ", [0x1F87] = "ἏΙ",
        [0x1F90] = "ἨΙ", [0x1F91] = "ἩΙ", [0x1F92] = "ἪΙ", [0x1F93] = "ἫΙ",
        [0x1F94] = "ἬΙ", [0x1F95] = "ἭΙ", [0x1F96] = "ἮΙ", [0x1F97] = "ἯΙ",
        [0x1FA0] = "ὨΙ", [0x1FA1] = "ὩΙ", [0x1FA2] = "ὪΙ", [0x1FA3] = "ὫΙ",
        [0x1FA4] = "ὬΙ", [0x1FA5] = "ὭΙ", [0x1FA6] = "ὮΙ", [0x1FA7] = "ὯΙ",
        [0x1FB2] = "ᾺΙ", [0x1FB3] = "ΑΙ", [0x1FB4] = "ΆΙ", [0x1FB6] = "Α͂",
        [0x1FB7] = "Α͂Ι", [0x1FC2] = "ῊΙ", [0x1FC3] = "ΗΙ",
        [0x1FC4] = "ΉΙ", [0x1FC6] = "Η͂", [0x1FC7] = "Η͂Ι",
        [0x1FD2] = "Ϊ̀", [0x1FD3] = "Ϊ́", [0x1FD6] = "Ι͂",
        [0x1FD7] = "Ϊ͂", [0x1FE2] = "Ϋ̀", [0x1FE3] = "Ϋ́",
        [0x1FE4] = "Ρ̓", [0x1FE6] = "Υ͂", [0x1FE7] = "Ϋ͂",
        [0x1FF2] = "ῺΙ", [0x1FF3] = "ΩΙ", [0x1FF4] = "ΏΙ", [0x1FF6] = "Ω͂",
        [0x1FF7] = "Ω͂Ι", [0xFB00] = "FF", [0xFB01] = "FI", [0xFB02] = "FL", [0xFB03] = "FFI",
        [0xFB04] = "FFL", [0xFB05] = "ST", [0xFB06] = "ST", [0xFB13] = "ՄՆ", [0xFB14] = "ՄԵ",
        [0xFB15] = "ՄԻ", [0xFB16] = "ՎՆ", [0xFB17] = "ՄԽ",
    };

    private static readonly (string Old, string New)[] PunctuationReplacements =
    [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];

    /// <summary>Returns <paramref name="text"/> as upstream <c>punc_norm</c> would.</summary>
    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0)
        {
            return EmptyTextPrompt;
        }
        string s = CollapseWhitespace(CapitalizeFirst(text));
        foreach ((string oldValue, string newValue) in PunctuationReplacements)
        {
            s = s.Replace(oldValue, newValue, StringComparison.Ordinal);
        }
        s = s.TrimEnd(' ');
        if (s.Length == 0 || s[^1] is not ('.' or '!' or '?' or '-' or ','))
        {
            s += ".";
        }
        return s;
    }

    internal static string CapitalizeFirst(string text)
    {
        if (!Rune.TryGetRuneAt(text, 0, out Rune first))
        {
            return text;   // a lone surrogate has no case
        }
        if (!PythonUpperOverrides.TryGetValue(first.Value, out string? upper))
        {
            // Python's islower() also covers Other_Lowercase symbols such as U+24D0; titlecase letters are not lowercase.
            Rune mapped = Rune.ToUpperInvariant(first);
            if (mapped == first || Rune.GetUnicodeCategory(first) == UnicodeCategory.TitlecaseLetter)
            {
                return text;
            }
            upper = mapped.ToString();
        }
        return string.Concat(upper, text.AsSpan(first.Utf16SequenceLength));
    }

    // Python's str.split() also splits on the ASCII file/group/record/unit separators, which .NET does not call space.
    private static bool IsPythonWhitespace(char c) => char.IsWhiteSpace(c) || c is >= '\x1C' and <= '\x1F';

    private static string CollapseWhitespace(string text)
    {
        StringBuilder sb = new(text.Length);
        bool pendingSpace = false;
        foreach (char c in text)
        {
            if (IsPythonWhitespace(c))
            {
                pendingSpace = sb.Length > 0;
                continue;
            }
            if (pendingSpace)
            {
                sb.Append(' ');
                pendingSpace = false;
            }
            sb.Append(c);
        }
        return sb.ToString();
    }
}
