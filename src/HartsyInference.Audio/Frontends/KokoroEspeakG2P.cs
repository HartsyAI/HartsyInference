using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Audio.Frontends;

/// <summary>Kokoro's front-end for the languages it phonemizes with espeak-ng (Spanish, French, Hindi, Italian,
/// Brazilian Portuguese): a port of misaki's <c>EspeakG2P</c> over the pure-C# espeak. Like phonemizer, which misaki
/// calls, punctuation runs are kept as written and the text between them is phonemized clause by clause; espeak's
/// tied multi-letter phonemes then take misaki's single-symbol spellings (<c>t^s</c> → ʦ, <c>d^ʒ</c> → ʤ).</summary>
public sealed partial class KokoroEspeakG2P
{
    // misaki EspeakG2P.e2m (version 1), applied in its sorted order.
    private static readonly (string From, string To)[] E2M =
    [
        ("a^ɪ", "I"), ("a^ʊ", "W"), ("d^z", "ʣ"), ("d^ʒ", "ʤ"), ("e^ɪ", "A"), ("o^ʊ", "O"), ("s^s", "S"),
        ("t^s", "ʦ"), ("t^ʃ", "ʧ"), ("ɔ^ɪ", "Y"), ("ə^ʊ", "Q"),
    ];

    private readonly EspeakPhonemizer _espeak;

    /// <summary>Wraps an espeak phonemizer built for the Kokoro language (<c>es</c>, <c>fr-fr</c>, <c>hi</c>,
    /// <c>it</c> or <c>pt-br</c>).</summary>
    public KokoroEspeakG2P(EspeakPhonemizer espeak)
    {
        ArgumentNullException.ThrowIfNull(espeak);
        _espeak = espeak;
    }

    /// <summary>The phoneme string Kokoro takes for <paramref name="text"/>.</summary>
    public string ToIpa(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        // misaki: angle quotes become curly quotes, then parentheses ride through espeak as angle quotes.
        text = text.Replace('«', '“').Replace('»', '”').Replace('(', '«').Replace(')', '»');
        StringBuilder sb = new(text.Length * 2);
        int last = 0;
        foreach (Match mark in MarksRegex().Matches(text))
        {
            AppendClause(sb, text[last..mark.Index]);
            sb.Append(mark.Value);
            last = mark.Index + mark.Length;
        }
        AppendClause(sb, text[last..]);
        string ps = sb.ToString().Trim();
        foreach ((string from, string to) in E2M) ps = ps.Replace(from, to, StringComparison.Ordinal);
        return ps.Replace("^", "").Replace("-", "").Replace('«', '(').Replace('»', ')');
    }

    private void AppendClause(StringBuilder sb, string clause)
    {
        if (string.IsNullOrWhiteSpace(clause)) return;
        sb.Append(_espeak.PhonemizeTied(clause.Trim(), '^'));
    }

    // phonemizer's default marks, each run with the whitespace around it.
    [GeneratedRegex(@"(\s*[;:,.!?¡¿—…""«»“”(){}\[\]]+\s*)+")]
    private static partial Regex MarksRegex();
}
