using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Phonemizer.MeCab;

namespace HartsyInference.Audio.Frontends;

/// <summary>Kokoro's Japanese front-end: a port of misaki's <c>JAG2P</c> in its default <c>cutlet</c> mode. Text is
/// normalized (NFKC, full-width Latin to half-width, digits read by <see cref="JapaneseNumberKana"/>), segmented by
/// the pure-C# <see cref="MeCabTokenizer"/> over full UniDic exactly as fugashi does, re-grouped by misaki's
/// <c>ja_words</c> list, and each word's katakana pronunciation is mapped to Kokoro's phoneme symbols.</summary>
public sealed partial class KokoroJapaneseG2P : IDisposable
{
    /// <summary>File name of misaki's word list inside a dictionary directory built by the engine installer.</summary>
    public const string WordsFileName = "ja_words.txt";

    private const int PronField = 9;
    private const int KanaField = 20;
    private const int SymbolType = 3;
    private const int HiraganaType = 6;
    private const int KatakanaType = 7;

    // cutlet's Katakana_Phonetic_Extensions: small katakana ㇰ..ㇿ to their full-size forms.
    private const string PhoneticExtensions = "クシストヌハヒフヘホムラリルレロ";

    private readonly MeCabTokenizer _tagger;
    private readonly HashSet<string> _words;
    private readonly int _longestWord;

    /// <summary>Opens the UniDic MeCab dictionary in <paramref name="dictionaryDirectory"/> and reads misaki's word
    /// list from <paramref name="wordsPath"/> (default: <see cref="WordsFileName"/> inside the directory).</summary>
    /// <exception cref="FileNotFoundException">A dictionary file or the word list is missing.</exception>
    public KokoroJapaneseG2P(string dictionaryDirectory, string? wordsPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(dictionaryDirectory);
        wordsPath ??= Path.Combine(dictionaryDirectory, WordsFileName);
        if (!File.Exists(wordsPath)) throw new FileNotFoundException("misaki's ja_words.txt is missing.", wordsPath);
        _words = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(wordsPath, Encoding.UTF8))
        {
            string word = line.Trim();
            _words.Add(word);
            _longestWord = Math.Max(_longestWord, word.Length);
        }
        _tagger = MeCabTokenizer.Open(dictionaryDirectory);
    }

    /// <summary>The phoneme string Kokoro takes for <paramref name="text"/> (misaki <c>JAG2P()(text)[0]</c>).</summary>
    public string ToIpa(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0) return "";
        text = Normalize(text);
        List<Word> words = [];
        foreach (MeCabToken token in _tagger.Tokenize(text))
        {
            // Python `or`: a missing or empty field falls through to the next.
            string reading = NonEmpty(token.Field(PronField)) ?? NonEmpty(token.Field(KanaField)) ?? token.Surface;
            int type = token.CharType == KatakanaType || !token.IsUnknown ? HiraganaType : token.CharType;
            words.Add(new Word(token.Surface, KataToHira(reading), type));
        }
        string output = Romanize(Group(words));
        string ps = CollapseSpaces(output);
        ps = ps.Replace('(', '«').Replace(')', '»');
        return GlottalSpaceRegex().Replace(ps, "");
    }

    /// <summary>Unmaps the dictionary.</summary>
    public void Dispose() => _tagger.Dispose();

    /// <summary>cutlet <c>_normalize_text</c>: wave dash before a digit reads から, small phonetic-extension katakana
    /// grow, NFKC, then mojimoji's full-to-half width pass, and every digit run becomes a space plus its kana.</summary>
    internal static string Normalize(string text)
    {
        text = WaveDashRegex().Replace(text, "から");
        StringBuilder sb = new(text.Length);
        foreach (char c in text)
            sb.Append(c is >= 'ㇰ' and <= 'ㇿ' ? PhoneticExtensions[c - 'ㇰ'] : c);
        text = sb.ToString().Normalize(NormalizationForm.FormKC);
        sb.Clear();
        foreach (char c in text) sb.Append(ZenToHan(c));
        text = sb.ToString();
        sb.Clear();
        int i = 0;
        while (i < text.Length)
        {
            if (!char.IsDigit(text[i]))
            {
                sb.Append(text[i++]);
                continue;
            }
            int start = i;
            while (i < text.Length && char.IsDigit(text[i])) i++;
            StringBuilder digits = new(i - start);
            for (int k = start; k < i; k++) digits.Append((char)('0' + (int)char.GetNumericValue(text[k])));
            sb.Append(' ').Append(JapaneseNumberKana.Convert(digits.ToString()));
        }
        return sb.ToString();
    }

    /// <summary>jaconv <c>kata2hira</c>: full-width katakana ァ..ヶ, ヽ and ヾ to hiragana; everything else unchanged.</summary>
    internal static string KataToHira(string text)
    {
        StringBuilder? sb = null;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            char h = c switch
            {
                >= 'ァ' and <= 'ヶ' => (char)(c - 0x60),
                'ヽ' => 'ゝ',
                'ヾ' => 'ゞ',
                _ => c,
            };
            if (h != c && sb is null) sb = new StringBuilder(text, 0, i, text.Length);
            sb?.Append(h);
        }
        return sb?.ToString() ?? text;
    }

    // mojimoji zen_to_han(kana=False): full-width ASCII and the ideographic space; after NFKC only the curly quotes
    // ‘ ’ ” are still affected (“ is not in mojimoji's table).
    private static char ZenToHan(char c) => c switch
    {
        >= '！' and <= '～' => (char)(c - 0xFEE0),
        '　' => ' ',
        '‘' => '`',
        '’' => '\'',
        '”' => '"',
        _ => c,
    };

    // cutlet _romaji_tokens, first half: within a run of one character type, join the longest prefix found in
    // misaki's word list into one word.
    private List<Word> Group(List<Word> words)
    {
        List<Word> grouped = new(words.Count);
        int i = 0;
        while (i < words.Count)
        {
            int z = i + 1;
            while (z < words.Count && words[z].CharType == words[i].CharType) z++;
            int match = -1;
            for (int j = z; j > i; j--)
            {
                int length = 0;
                for (int k = i; k < j; k++) length += words[k].Surface.Length;
                if (length > _longestWord) continue;
                if (_words.Contains(Concat(words, i, j, surface: true)))
                {
                    match = j;
                    break;
                }
            }
            if (match < 0)
            {
                grouped.Add(words[i]);
                i++;
                continue;
            }
            grouped.Add(new Word(Concat(words, i, match, surface: true), Concat(words, i, match, surface: false),
                words[i].CharType));
            i = match;
        }
        return grouped;
    }

    // cutlet _romaji_tokens, second half: per-word symbols and cutlet's spacing around punctuation.
    private static string Romanize(List<Word> words)
    {
        List<(string Text, bool Space)> output = new(words.Count);
        foreach (Word word in words)
        {
            string roma = RomanizeWord(word);
            bool space;
            // Python `in` on strings is a substring test, so an empty symbol string matches too.
            if (PyIn(word.Surface, "「『«") || PyIn(roma, "(["))
            {
                if (output.Count > 0) output[^1] = (output[^1].Text, true);
                space = false;
            }
            else if (PyIn(word.Surface, "」』»") || PyIn(roma, "]).,?!:"))
            {
                if (output.Count > 0) output[^1] = (output[^1].Text, false);
                space = true;
            }
            else
            {
                space = roma != " ";
            }
            output.Add((roma, space));
        }
        StringBuilder sb = new();
        foreach ((string text, bool space) in output)
        {
            sb.Append(text.Replace("っ", "", StringComparison.Ordinal));
            if (space) sb.Append(' ');
        }
        return sb.ToString();
    }

    // cutlet _romaji_word.
    private static string RomanizeWord(Word word)
    {
        string surface = word.Surface;
        if (IsAscii(surface)) return surface;
        if (word.CharType == SymbolType)
        {
            StringBuilder sb = new();
            foreach (Rune r in surface.EnumerateRunes())
            {
                string c = r.ToString();
                sb.Append(JapaneseKanaIpa.Table.TryGetValue(c, out string? mapped) ? mapped : c);
            }
            return sb.ToString();
        }
        return word.CharType != HiraganaType ? "" : JapaneseKanaIpa.MapReading(word.Hira);
    }

    private static string Concat(List<Word> words, int from, int to, bool surface)
    {
        StringBuilder sb = new();
        for (int k = from; k < to; k++) sb.Append(surface ? words[k].Surface : words[k].Hira);
        return sb.ToString();
    }

    private static string? NonEmpty(string? s) => string.IsNullOrEmpty(s) ? null : s;

    private static bool PyIn(string needle, string haystack) => haystack.Contains(needle, StringComparison.Ordinal);

    private static bool IsAscii(string s)
    {
        foreach (char c in s)
        {
            if (c > 0x7F) return false;
        }
        return true;
    }

    // Python str.strip() then re.sub(r'\s+', ' ', ...), with Python's notion of whitespace.
    private static string CollapseSpaces(string text)
    {
        StringBuilder sb = new(text.Length);
        bool pending = false;
        foreach (char c in text)
        {
            if (IsPythonSpace(c))
            {
                pending = sb.Length > 0;
                continue;
            }
            if (pending) sb.Append(' ');
            pending = false;
            sb.Append(c);
        }
        return sb.ToString();
    }

    private static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || c is >= '\u001C' and <= '\u001F';

    [GeneratedRegex(@"[〜～](?=\d)")]
    private static partial Regex WaveDashRegex();

    [GeneratedRegex(@"(?<![!"",.:;?»—…”]) (?=ʔ)|(?<=ʔ) (?![""«“])")]
    private static partial Regex GlottalSpaceRegex();

    private readonly record struct Word(string Surface, string Hira, int CharType);
}
