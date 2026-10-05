using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>misaki's <c>EspeakFallback</c> rewrite of en-us espeak IPA into misaki's symbol set (one token per
/// diphthong and affricate, r-colouring spelled out, no length marks). Accepts output with or without espeak's
/// <c>^</c> tie, which the port does not emit.</summary>
internal static partial class EspeakToMisaki
{
    // Longest first, as misaki sorts its E2M table.
    private static readonly (string From, string To)[] Map =
    [
        ("ʔˌn̩", "ʔn"), ("ʔn̩", "ʔn"),
        ("a^ɪ", "I"), ("a^ʊ", "W"), ("d^ʒ", "ʤ"), ("e^ɪ", "A"), ("t^ʃ", "ʧ"), ("ɔ^ɪ", "Y"), ("ə^l", "ᵊl"), ("o^ʊ", "O"),
        ("aɪ", "I"), ("aʊ", "W"), ("dʒ", "ʤ"), ("eɪ", "A"), ("tʃ", "ʧ"), ("ɔɪ", "Y"), ("oʊ", "O"),
        ("ʲo", "jo"), ("ʲə", "jə"), ("ʲ", ""), ("ɚ", "əɹ"), ("e", "A"), ("r", "ɹ"), ("x", "k"), ("ç", "k"), ("ɐ", "ə"),
        ("ɬ", "l"), ("̃", ""),
    ];

    /// <summary>The misaki spelling of <paramref name="espeakIpa"/>.</summary>
    public static string Convert(string espeakIpa)
    {
        string ps = espeakIpa.Trim();
        foreach ((string from, string to) in Map) ps = ps.Replace(from, to, StringComparison.Ordinal);
        // Without the tie, a syllabic l is a schwa + l that no vowel follows.
        ps = SyllabicLRegex().Replace(ps, "ᵊl");
        ps = SyllabicMarkRegex().Replace(ps, "ᵊ$1").Replace("̩", "");
        ps = ps.Replace("ɜːɹ", "ɜɹ").Replace("ɜː", "ɜɹ").Replace("ɪə", "iə").Replace("ː", "").Replace('o', 'ɔ');
        return ps.Replace("^", "");
    }

    [GeneratedRegex(@"əl(?![AIOWYaiuæɑɔəɛɜɪʊʌᵻ])")]
    private static partial Regex SyllabicLRegex();

    [GeneratedRegex(@"(\S)̩")]
    private static partial Regex SyllabicMarkRegex();
}
