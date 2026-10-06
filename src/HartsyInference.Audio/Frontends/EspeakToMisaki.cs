using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>misaki's <c>EspeakFallback</c> rewrite of English espeak IPA (read with the <c>^</c> tie between the
/// letters of one phoneme) into misaki's symbol set: one token per diphthong and affricate, r-colouring spelled out,
/// and for American English no length marks; British keeps its lengths and its own diphthongs.</summary>
internal static partial class EspeakToMisaki
{
    // misaki's E2M, sorted longest first (a stable sort, so equal lengths keep the table's order).
    private static readonly (string From, string To)[] Map =
    [
        ("ʔˌn̩", "ʔn"), ("ʔn̩", "ʔn"),
        ("a^ɪ", "I"), ("a^ʊ", "W"), ("d^ʒ", "ʤ"), ("e^ɪ", "A"), ("t^ʃ", "ʧ"), ("ɔ^ɪ", "Y"), ("ə^l", "ᵊl"),
        ("ʲo", "jo"), ("ʲə", "jə"),
        ("e", "A"), ("ʲ", ""), ("ɚ", "əɹ"), ("r", "ɹ"), ("x", "k"), ("ç", "k"), ("ɐ", "ə"), ("ɬ", "l"), ("̃", ""),
    ];

    /// <summary>The misaki spelling of tied espeak IPA <paramref name="espeakIpa"/> (en-us, or en-gb with
    /// <paramref name="british"/>).</summary>
    public static string Convert(string espeakIpa, bool british = false)
    {
        string ps = espeakIpa.Trim();
        foreach ((string from, string to) in Map) ps = ps.Replace(from, to, StringComparison.Ordinal);
        ps = SyllabicMarkRegex().Replace(ps, "ᵊ$1").Replace("̩", "");
        if (british)
        {
            ps = ps.Replace("e^ə", "ɛː").Replace("iə", "ɪə").Replace("ə^ʊ", "Q");
        }
        else
        {
            ps = ps.Replace("o^ʊ", "O").Replace("ɜːɹ", "ɜɹ").Replace("ɜː", "ɜɹ").Replace("ɪə", "iə").Replace("ː", "");
        }
        ps = ps.Replace('o', 'ɔ'); // for espeak < 1.52
        return ps.Replace("^", "");
    }

    [GeneratedRegex(@"(\S)̩")]
    private static partial Regex SyllabicMarkRegex();
}
