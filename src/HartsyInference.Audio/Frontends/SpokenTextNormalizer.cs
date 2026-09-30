using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Reduces a language model's reply to the words a voice should say.</summary>
/// <remarks>A chat model writes for a screen: it thinks aloud inside <c>&lt;think&gt;</c> blocks, bolds, bullets,
/// links and decorates with emoji, none of which a listener should hear read out. This strips the thinking, the
/// markdown syntax (keeping the words it wrapped, and a link's text over its URL), any emoji or character outside
/// the Basic Multilingual Plane, and collapses the whitespace that leaves behind. Digits are left exactly as
/// written: expanding numbers to words is the G2P's business, not this class's.</remarks>
public static partial class SpokenTextNormalizer
{
    /// <summary>The speakable words of <paramref name="text"/>, or an empty string when nothing is left.</summary>
    public static string ToSpeakable(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return "";
        }
        string s = ThinkBlock().Replace(text, " ");
        s = FencedCode().Replace(s, " ");
        s = Image().Replace(s, "$1");
        s = Link().Replace(s, "$1");
        s = Heading().Replace(s, "");
        s = ListMarker().Replace(s, "");
        s = BlockQuote().Replace(s, "");
        s = InlineCode().Replace(s, "$1");
        s = StrongAsterisk().Replace(s, "$1");
        s = StrongUnderscore().Replace(s, "$1");
        s = Strike().Replace(s, "$1");
        s = EmphasisAsterisk().Replace(s, "$1");
        s = EmphasisUnderscore().Replace(s, "$1");
        s = DropSymbols(s);
        s = Whitespace().Replace(s, " ");
        return s.Trim();
    }

    /// <summary>Removes emoji, their joiners and selectors, and every character outside the BMP.</summary>
    private static string DropSymbols(string s)
    {
        StringBuilder? sb = null;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            bool drop = char.IsSurrogate(c) || IsEmojiOrSymbol(c);
            if (drop)
            {
                sb ??= new StringBuilder(s.Length).Append(s, 0, i);
                continue;
            }
            sb?.Append(c);
        }
        return sb?.ToString() ?? s;
    }

    /// <summary>The BMP blocks emoji draw from: miscellaneous symbols and dingbats, the arrows/symbols block that holds
    /// ⭐ and ⬆, the variation selector and zero-width joiner that compose them, and the combining keycap.</summary>
    private static bool IsEmojiOrSymbol(char c) =>
        c is (>= '☀' and <= '➿') or (>= '⬀' and <= '⯿') or '️' or '‍' or '⃣';

    [GeneratedRegex(@"<think>.*?(?:</think>|\z)", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ThinkBlock();

    [GeneratedRegex(@"```.*?(?:```|\z)", RegexOptions.Singleline)]
    private static partial Regex FencedCode();

    [GeneratedRegex(@"!\[([^\]]*)\]\([^)]*\)")]
    private static partial Regex Image();

    [GeneratedRegex(@"\[([^\]]+)\]\([^)]*\)")]
    private static partial Regex Link();

    [GeneratedRegex(@"^[ \t]{0,3}#{1,6}[ \t]+", RegexOptions.Multiline)]
    private static partial Regex Heading();

    [GeneratedRegex(@"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", RegexOptions.Multiline)]
    private static partial Regex ListMarker();

    [GeneratedRegex(@"^[ \t]*>[ \t]?", RegexOptions.Multiline)]
    private static partial Regex BlockQuote();

    [GeneratedRegex(@"`([^`]*)`")]
    private static partial Regex InlineCode();

    [GeneratedRegex(@"\*\*(.+?)\*\*", RegexOptions.Singleline)]
    private static partial Regex StrongAsterisk();

    [GeneratedRegex(@"__(.+?)__", RegexOptions.Singleline)]
    private static partial Regex StrongUnderscore();

    [GeneratedRegex(@"~~(.+?)~~", RegexOptions.Singleline)]
    private static partial Regex Strike();

    [GeneratedRegex(@"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")]
    private static partial Regex EmphasisAsterisk();

    [GeneratedRegex(@"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")]
    private static partial Regex EmphasisUnderscore();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();
}
