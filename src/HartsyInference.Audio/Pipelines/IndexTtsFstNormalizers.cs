using System.Text;
using HartsyInference.Audio.Frontends.WeText;

namespace HartsyInference.Audio.Pipelines;

/// <summary>The spoken-form text normalizers the IndexTTS frontend calls where upstream calls wetext's
/// <c>Normalizer(lang="en", operator="tn").normalize</c> and <c>Normalizer(remove_erhua=False, lang="zh",
/// operator="tn").normalize</c>. Each delegate receives the text exactly as upstream passes it (protection
/// placeholders such as <c>GPT&lt;H&gt;5</c>, <c>&lt;pinyin_a&gt;</c>, <c>&lt;n_a&gt;</c> and
/// <c>PRONPLACEHOLDERaPRONPLACEHOLDER</c> included), must return the FST output verbatim and may throw to signal
/// failure, which the frontend handles the way upstream does.</summary>
public sealed record IndexTtsFstNormalizers
{
    /// <summary>English text normalization (wetext <c>en</c> <c>tn</c>).</summary>
    public required Func<string, string> English { get; init; }

    /// <summary>Chinese text normalization (wetext <c>zh</c> <c>tn</c>, erhua kept).</summary>
    public required Func<string, string> Chinese { get; init; }

    /// <summary>Stand-in until the WeText runtime is wired in: English goes through <see cref="EnglishSpokenForm"/>
    /// (an approximation of the FST that covers numbers, money, dates and titles), Chinese is left unchanged.</summary>
    public static IndexTtsFstNormalizers Interim { get; } = new()
    {
        English = EnglishSpokenForm.Expand,
        Chinese = static text => text,
    };

    /// <summary>wetext 0.1.8 <c>Normalizer(lang=..., operator="tn").normalize</c> (what IndexTTS's <c>wetext&gt;=0.0.9</c>
    /// resolves to) over <paramref name="weText"/>'s grammars: the text is stripped, English always runs tagger,
    /// reorder and verbalizer, Chinese only when it holds a decimal digit. <paramref name="weText"/> must hold both
    /// grammars, Chinese with erhua kept.</summary>
    public static IndexTtsFstNormalizers FromWeText(WeTextNormalizer weText)
    {
        ArgumentNullException.ThrowIfNull(weText);
        return new IndexTtsFstNormalizers
        {
            English = text => RunWeText(weText, text, WeTextLanguage.English),
            Chinese = text => RunWeText(weText, text, WeTextLanguage.Chinese),
        };
    }

    private static string RunWeText(WeTextNormalizer weText, string text, WeTextLanguage language)
    {
        string prepared = text.Trim();
        if (prepared.Length == 0) return prepared;
        if (language == WeTextLanguage.Chinese && !ContainsDecimalDigit(prepared)) return prepared;
        string tagged = weText.Tag(prepared, language).Trim();
        if (tagged.Length == 0) throw new FormatException("wetext tagger accepts no path for the text");
        return weText.Verbalize(tagged, language, wetext018: true).Trim();
    }

    // Python's re \d on str: any Unicode decimal digit.
    private static bool ContainsDecimalDigit(string text)
    {
        foreach (Rune rune in text.EnumerateRunes())
            if (Rune.IsDigit(rune)) return true;
        return false;
    }
}
