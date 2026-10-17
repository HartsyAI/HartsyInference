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
    private readonly EspeakPhonemizer? _latin;
    private readonly Func<string, bool>? _switchesToLatin;

    /// <summary>Wraps an espeak phonemizer built for the Kokoro language (<c>es</c>, <c>fr-fr</c>, <c>hi</c>,
    /// <c>it</c> or <c>pt-br</c>). <paramref name="latin"/> is the English phonemizer that reads the words espeak-ng
    /// switches to English; without it those words phonemize to nothing (Hindi) or to a blank (French), and misaki's
    /// <c>remove-flags</c> drops the <c>(en)</c> spans that espeak marks them with. <paramref name="switchesToLatin"/>
    /// says which words switch. Hindi's script has no Latin letter rules, so every Latin-script word switches there
    /// (the default); French switches the words espeak's dictionary or rules mark, as
    /// <see cref="EspeakPhonemizer.SwitchesToEnglish"/> reports.</summary>
    public KokoroEspeakG2P(EspeakPhonemizer espeak, EspeakPhonemizer? latin = null, Func<string, bool>? switchesToLatin = null)
    {
        ArgumentNullException.ThrowIfNull(espeak);
        _espeak = espeak;
        _latin = latin;
        _switchesToLatin = switchesToLatin;
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
        string[] words = clause.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        // The predicate runs once per word; the runs below reuse these flags.
        bool[] latin = new bool[words.Length];
        bool anyLatin = false;
        if (_latin is not null)
        {
            for (int i = 0; i < words.Length; i++)
            {
                latin[i] = IsLatin(words[i]);
                anyLatin |= latin[i];
            }
        }
        // A clause with no word that switches goes to the phonemizer whole, exactly as before.
        if (!anyLatin)
        {
            sb.Append(_espeak.PhonemizeTied(clause.Trim(), '^'));
            return;
        }
        // Each run of words that switch, and each run that does not, goes to its own phonemizer, so the switched
        // words read as English. A run the phonemizer leaves empty adds nothing, and the separator is only written
        // when the output does not already end in one, so runs never meet with two spaces.
        for (int i = 0; i < words.Length;)
        {
            bool run = latin[i];
            int end = i;
            while (end < words.Length && latin[end] == run) end++;
            if (sb.Length > 0 && sb[^1] != ' ') sb.Append(' ');
            EspeakPhonemizer phonemizer = run ? _latin! : _espeak;
            sb.Append(phonemizer.PhonemizeTied(string.Join(' ', words, i, end - i), '^'));
            i = end;
        }
    }

    // Whether espeak-ng reads the word in English: the predicate given, else any ASCII letter in it. The default is for
    // Hindi, whose script has no Latin letter rules, so a token with a Latin letter in it is read in English as a whole.
    // espeak-ng instead splits a mixed token by script: the Latin part goes to English and the Devanagari part stays
    // Hindi ("abcदेव" -> English "abc" + Hindi "देव"; "9am" -> Hindi "9" + English "am"). That split is not done here,
    // so a mixed token goes to English whole: its Devanagari part is dropped, and a digit is read as English ("9am" -> "nine am").
    private bool IsLatin(string word) => _switchesToLatin?.Invoke(word) ?? word.Any(char.IsAsciiLetter);

    // phonemizer's default marks, each run with the whitespace around it.
    [GeneratedRegex(@"(\s*[;:,.!?¡¿—…""«»“”(){}\[\]]+\s*)+")]
    private static partial Regex MarksRegex();
}
