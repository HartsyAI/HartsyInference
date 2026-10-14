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

    /// <summary>Fallback for when the WeText grammars cannot be loaded (offline, no cached grammars): English goes
    /// through <see cref="EnglishSpokenForm"/>, an approximation of the FST covering numbers, money, dates and titles;
    /// Chinese is left unchanged. Not upstream parity; prefer <see cref="FromWeText"/>.</summary>
    public static IndexTtsFstNormalizers Interim { get; } = new()
    {
        English = EnglishSpokenForm.Expand,
        Chinese = static text => text,
    };

    /// <summary>Upstream's <c>Normalizer(lang="en")</c> / <c>Normalizer(lang="zh", remove_erhua=False)</c> over
    /// <paramref name="weText"/>, which must reproduce wetext 0.1.8 (<see cref="WeTextVersion.V0_1_8"/>, what IndexTTS's
    /// <c>wetext&gt;=0.0.9</c> resolves to) and hold both grammars, Chinese with erhua kept.</summary>
    public static IndexTtsFstNormalizers FromWeText(WeTextNormalizer weText)
    {
        ArgumentNullException.ThrowIfNull(weText);
        if (weText.Version != WeTextVersion.V0_1_8)
            throw new ArgumentException("IndexTTS runs wetext 0.1.8; load the normalizer with WeTextVersion.V0_1_8.", nameof(weText));
        if (weText.Language != WeTextLanguage.Auto)
            throw new ArgumentException("IndexTTS needs both the English and the Chinese grammars.", nameof(weText));
        return new IndexTtsFstNormalizers
        {
            English = text => weText.Normalize(text, WeTextLanguage.English),
            Chinese = text => weText.Normalize(text, WeTextLanguage.Chinese),
        };
    }
}
