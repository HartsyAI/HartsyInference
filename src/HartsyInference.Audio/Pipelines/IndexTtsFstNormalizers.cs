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
}
