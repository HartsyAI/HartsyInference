using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Chatterbox's text cleanup before tokenization, a port of upstream <c>punc_norm</c>
/// (resemble-ai/chatterbox <c>src/chatterbox/tts.py</c>).</summary>
/// <remarks>Steps run in upstream order: empty text becomes a fixed prompt; a lowercase first character is
/// uppercased (before any whitespace is touched, so leading whitespace leaves the text uncapitalized); whitespace runs
/// collapse to one space as Python's <c>str.split()</c> does; LLM punctuation is rewritten; trailing spaces go; and a
/// full stop is appended unless the text already ends in <c>. ! ? - ,</c>. Uppercasing follows Python's full case
/// mapping for <c>ß</c>, <c>ŉ</c> and the Latin ligatures; other multi-character expansions fall back to the simple
/// mapping.</remarks>
public static class ChatterboxTextNormalizer
{
    /// <summary>What upstream speaks for empty input.</summary>
    public const string EmptyTextPrompt = "You need to add some text for me to talk.";

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

    private static string CapitalizeFirst(string text)
    {
        Rune first = Rune.GetRuneAt(text, 0);
        if (!Rune.IsLower(first))
        {
            return text;
        }
        string upper = first.Value switch
        {
            0x00DF => "SS",
            0x0149 => "ʼN",
            0xFB00 => "FF",
            0xFB01 => "FI",
            0xFB02 => "FL",
            0xFB03 => "FFI",
            0xFB04 => "FFL",
            0xFB05 or 0xFB06 => "ST",
            _ => Rune.ToUpperInvariant(first).ToString(),
        };
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
