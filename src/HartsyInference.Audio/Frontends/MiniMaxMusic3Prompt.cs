using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Frontends;

/// <summary>Assembles MiniMax Music 3's special-token prompt from a music description and lyrics, and tokenizes it
/// into the conditional/unconditional id pair the autoregressive stage consumes.
///
/// <para>The assembled string is part of the checkpoint contract — whitespace-level changes to it change the
/// generated audio — so the caption cleaner and lyric normalizer below are literal ports of the reference
/// <c>_clean_caption</c>/<c>_normalize_lyrics</c>, including the regexes and their evaluation order.</para></summary>
public static partial class MiniMaxMusic3Prompt
{
    /// <summary>Longest assembled prompt the checkpoint accepts, in tokens.</summary>
    public const int MaxPromptTokens = 5000;

    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";
    private const string CaptionStart = "<|caption_start|>";
    private const string CaptionEnd = "<|caption_end|>";
    private const string LyricsStart = "<|lyrics_start|>";
    private const string LyricsEnd = "<|lyrics_end|>";
    private const string AudioStart = "<|audio_start|>";
    private const string AudioCfg = "<|audio_cfg|>";

    // Python's str.splitlines() breaks on more than \r\n\r; the caption cleaner joins the result back with "\n",
    // so every one of these is normalized away. "\r\n" must stay first — Split tries separators in array order.
    private static readonly string[] _pythonLineBreaks =
    [
        "\r\n", "\n", "\r", "\v", "\f", "\u001c", "\u001d", "\u001e", "\u0085", "\u2028", "\u2029",
    ];

    [GeneratedRegex(@"<\|([^|]*)\|>")]
    private static partial Regex SpecialTagRegex();

    [GeneratedRegex(@"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")]
    private static partial Regex LeadingTagsRegex();

    [GeneratedRegex(@"^\s{0,3}#{1,6}\s+")]
    private static partial Regex MarkdownHeadingRegex();

    [GeneratedRegex(@"^\s*[*+-]\s+")]
    private static partial Regex MarkdownBulletRegex();

    [GeneratedRegex(@"^\s*\*\s+")]
    private static partial Regex MarkdownStarBulletRegex();

    [GeneratedRegex(@"\*\*([^*]+)\*\*")]
    private static partial Regex MarkdownBoldRegex();

    [GeneratedRegex(@"(?<!\*)\*([^*\n]+)\*(?!\*)")]
    private static partial Regex MarkdownItalicRegex();

    [GeneratedRegex(@"^\s*[-*_]{3,}\s*$", RegexOptions.Multiline)]
    private static partial Regex HorizontalRuleRegex();

    [GeneratedRegex(@"\n{2,}")]
    private static partial Regex BlankLineRunRegex();

    [GeneratedRegex(@"\[([^\]]+)\]")]
    private static partial Regex BracketTagRegex();

    /// <summary>Builds the assembled prompt string for <paramref name="caption"/> (the music description) and
    /// <paramref name="lyrics"/>.</summary>
    public static string Build(string caption, string lyrics)
    {
        ArgumentNullException.ThrowIfNull(caption);
        ArgumentNullException.ThrowIfNull(lyrics);
        return $"{ImStart}{CaptionStart}{CleanCaption(caption)}{CaptionEnd}"
            + $"{LyricsStart}{NormalizeLyrics(lyrics)}{LyricsEnd}{ImEnd}{AudioStart}";
    }

    /// <summary>Tokenizes the assembled prompt into the conditional ids and their classifier-free counterpart
    /// (every token except the first and the two trailing structure tokens replaced by <c>&lt;|audio_cfg|&gt;</c>).</summary>
    public static (int[] Conditional, int[] Unconditional) Tokenize(GgufTokenizer tokenizer, string caption, string lyrics)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        int[] conditional = tokenizer.Encode(Build(caption, lyrics), addSpecial: true);
        if (conditional.Length > MaxPromptTokens)
        {
            throw new ArgumentException(
                $"The assembled MiniMax Music 3 prompt has {conditional.Length} tokens; the maximum is {MaxPromptTokens}.",
                nameof(caption));
        }
        int cfgId = tokenizer.SpecialId(AudioCfg)
            ?? throw new InvalidOperationException($"The tokenizer has no '{AudioCfg}' token — it is not a MiniMax Music 3 tokenizer.");
        int[] unconditional = (int[])conditional.Clone();
        for (int i = 1; i < unconditional.Length - 2; i++)
        {
            unconditional[i] = cfgId;
        }
        return (conditional, unconditional);
    }

    /// <summary>Rewrites <c>&lt;|key value|&gt;</c> metadata tags to <c>"key is value"</c> and strips the markdown
    /// forms the model's input contract accepts.</summary>
    public static string CleanCaption(string caption)
    {
        ArgumentNullException.ThrowIfNull(caption);
        string text = SpecialTagRegex().Replace(caption, static match =>
        {
            string inner = match.Groups[1].Value.Trim();
            int split = inner.AsSpan().IndexOfAny(" \t\n\r\v\f");
            if (split < 0)
            {
                return inner;
            }
            // Python's str.split(None, 1) collapses the whitespace run at the split point.
            int valueStart = split;
            while (valueStart < inner.Length && char.IsWhiteSpace(inner[valueStart]))
            {
                valueStart++;
            }
            return valueStart >= inner.Length ? inner[..split] : $"{inner[..split]} is {inner[valueStart..]}";
        });

        string[] lines = text.Split(_pythonLineBreaks, StringSplitOptions.None);
        // Python's splitlines treats a break as a line TERMINATOR, so a trailing one yields no extra empty line.
        int lineCount = lines.Length > 1 && lines[^1].Length == 0 ? lines.Length - 1 : lines.Length;
        StringBuilder joined = new StringBuilder(text.Length);
        for (int i = 0; i < lineCount; i++)
        {
            string line = MarkdownHeadingRegex().Replace(lines[i], "", 1);
            line = MarkdownBulletRegex().Replace(line, "", 1);
            line = MarkdownStarBulletRegex().Replace(line, "", 1);
            while (line.Contains("**", StringComparison.Ordinal))
            {
                string updated = MarkdownBoldRegex().Replace(line, "$1");
                if (string.Equals(updated, line, StringComparison.Ordinal))
                {
                    break;
                }
                line = updated;
            }
            line = MarkdownItalicRegex().Replace(line, "$1");
            if (i > 0)
            {
                joined.Append('\n');
            }
            joined.Append(line.TrimEnd());
        }

        text = HorizontalRuleRegex().Replace(joined.ToString(), "");
        text = text.Replace("• ", "", StringComparison.Ordinal).Replace("    ", "", StringComparison.Ordinal);
        return BlankLineRunRegex().Replace(text, "\n");
    }

    /// <summary>Keeps only the consecutive structural tags (e.g. <c>[verse]</c>) at the start of a line — text sharing
    /// a line with a leading tag is dropped by the checkpoint's input contract — then lowercases every tag.</summary>
    public static string NormalizeLyrics(string lyrics)
    {
        ArgumentNullException.ThrowIfNull(lyrics);
        // Deliberately Split('\n') and not the splitlines() set above: the reference uses str.split here.
        string[] lines = lyrics.Split('\n');
        StringBuilder joined = new StringBuilder(lyrics.Length + 8);
        for (int i = 0; i < lines.Length; i++)
        {
            Match match = LeadingTagsRegex().Match(lines[i]);
            if (i > 0)
            {
                joined.Append('\n');
            }
            joined.Append(match.Success ? match.Groups[1].Value.Trim() : lines[i]);
        }

        string text = joined.ToString()
            .Replace("] ", "]\n", StringComparison.Ordinal)
            .Replace(" [", "\n[", StringComparison.Ordinal)
            .Replace(" ^ ", "\n", StringComparison.Ordinal);
        text = BracketTagRegex().Replace(text, static match => $"[{match.Groups[1].Value.ToLowerInvariant()}]");
        return $"[start]\n{text}";
    }
}
