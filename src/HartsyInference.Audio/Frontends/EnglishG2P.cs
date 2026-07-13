using System.Globalization;
using System.IO;
using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>English grapheme-to-phoneme front-end producing IPA for the phoneme-in TTS pipelines (Kokoro,
/// StyleTTS2). Known words come from a CMUdict (ARPAbet → IPA via <see cref="ArpabetToIpa"/>); out-of-vocab
/// words fall back to a compact letter-to-sound rule set (approximate). Text is normalized first (numbers →
/// words, punctuation stripped). The reference Kokoro pipeline uses misaki/espeak; this is the pure-C#
/// stand-in — intelligible, not bit-identical, prosody/quality slightly off for OOV words.</summary>
public sealed class EnglishG2P
{
    private readonly Dictionary<string, string[]> _dict;

    /// <summary>Builds from a CMUdict stream (lines <c>"word  PH PH ..."</c>; <c>;;;</c> comments and
    /// <c>word(2)</c> variants handled — first pronunciation wins). The asset is public-domain
    /// <c>cmudict.dict</c> from CMU Sphinx; the caller supplies it (download or place on disk).</summary>
    public EnglishG2P(Stream cmudict)
    {
        ArgumentNullException.ThrowIfNull(cmudict);
        _dict = LoadCmudict(cmudict);
    }

    /// <summary>Builds from a CMUdict file path.</summary>
    public EnglishG2P(string cmudictPath)
    {
        if (!File.Exists(cmudictPath))
        {
            throw new FileNotFoundException($"CMUdict not found: {cmudictPath}", cmudictPath);
        }
        using FileStream fs = File.OpenRead(cmudictPath);
        _dict = LoadCmudict(fs);
    }

    /// <summary>Number of pronunciation entries loaded.</summary>
    public int WordCount => _dict.Count;

    /// <summary>Converts free text to a space-separated IPA phoneme string for Kokoro-style pipelines.</summary>
    public string ToIpa(string text)
    {
        StringBuilder sb = new();
        foreach (string word in Tokenize(Normalize(text)))
        {
            // Sentence punctuation the model uses for pauses/prosody (Kokoro's vocab has . , ? ! ; :) is
            // carried through and ATTACHED to the preceding phonemes (misaki writes "dˈɔɡ."), with no space,
            // so it does not become its own "word" the CMU lookup would fail on. Dropping it made adjacent
            // sentences run together (e.g. "cow. The boy" → one slurred blob).
            if (word.Length == 1 && word[0] is '.' or ',' or '?' or '!' or ';' or ':')
            {
                sb.Append(word[0]);
                continue;
            }
            if (sb.Length > 0)
            {
                sb.Append(' ');
            }
            sb.Append(_dict.TryGetValue(word, out string[]? phones) ? ArpabetToIpa.ConvertWord(phones) : LetterToSound(word));
        }
        return sb.ToString();
    }

    private static Dictionary<string, string[]> LoadCmudict(Stream stream)
    {
        Dictionary<string, string[]> dict = new(StringComparer.Ordinal);
        using StreamReader reader = new(stream);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0 || line.StartsWith(";;;", StringComparison.Ordinal))
            {
                continue;
            }
            string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                continue;
            }
            string word = parts[0];
            int paren = word.IndexOf('(');
            if (paren >= 0)
            {
                word = word[..paren]; // drop the (2) variant marker
            }
            word = word.ToLowerInvariant();
            if (!dict.ContainsKey(word)) // first pronunciation wins
            {
                dict[word] = parts[1..];
            }
        }
        return dict;
    }

    /// <summary>Lowercase, expand integer runs to words, and reduce to letter/apostrophe tokens.</summary>
    private static string Normalize(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return "";
        }
        StringBuilder sb = new(text.Length);
        for (int i = 0; i < text.Length;)
        {
            char c = text[i];
            if (char.IsDigit(c))
            {
                int j = i;
                while (j < text.Length && char.IsDigit(text[j]))
                {
                    j++;
                }
                sb.Append(' ').Append(NumberToWords(text[i..j])).Append(' ');
                i = j;
            }
            else if (char.IsLetter(c) || c == '\'')
            {
                sb.Append(char.ToLowerInvariant(c));
                i++;
            }
            else if (c is '.' or ',' or '?' or '!' or ';' or ':')
            {
                // Preserve sentence punctuation as its own token (the model has these + uses them for pauses).
                sb.Append(' ').Append(c).Append(' ');
                i++;
            }
            else
            {
                sb.Append(' ');
                i++;
            }
        }
        return sb.ToString();
    }

    private static IEnumerable<string> Tokenize(string normalized)
        => normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    private static readonly string[] Ones =
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    private static readonly string[] Tens =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    /// <summary>Expands a non-negative integer string to English words. Falls back to digit-by-digit for
    /// values beyond the billions range (or anything that doesn't parse).</summary>
    private static string NumberToWords(string digits)
    {
        if (!long.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out long n))
        {
            return string.Join(' ', digits.Select(d => Ones[d - '0']));
        }
        return IntToWords(n);
    }

    private static string IntToWords(long n)
    {
        if (n < 0)
        {
            return "minus " + IntToWords(-n);
        }
        if (n < 20)
        {
            return Ones[n];
        }
        if (n < 100)
        {
            return Tens[n / 10] + (n % 10 != 0 ? " " + Ones[n % 10] : "");
        }
        if (n < 1_000)
        {
            return Ones[n / 100] + " hundred" + (n % 100 != 0 ? " " + IntToWords(n % 100) : "");
        }
        (long unit, string name) = n switch
        {
            < 1_000_000 => (1_000L, "thousand"),
            < 1_000_000_000 => (1_000_000L, "million"),
            _ => (1_000_000_000L, "billion"),
        };
        return IntToWords(n / unit) + " " + name + (n % unit != 0 ? " " + IntToWords(n % unit) : "");
    }

    private static readonly (string Gr, string Ipa)[] Digraphs =
    [
        ("tch", "tʃ"), ("sch", "sk"),
        ("ch", "tʃ"), ("sh", "ʃ"), ("th", "θ"), ("ph", "f"), ("wh", "w"), ("ck", "k"), ("ng", "ŋ"), ("qu", "kw"),
        ("ee", "i"), ("ea", "i"), ("oo", "u"), ("oa", "oʊ"), ("ai", "eɪ"), ("ay", "eɪ"), ("ou", "aʊ"),
        ("ow", "aʊ"), ("oy", "ɔɪ"), ("oi", "ɔɪ"), ("ar", "ɑɹ"), ("or", "ɔɹ"), ("er", "ɚ"), ("ir", "ɚ"), ("ur", "ɚ"),
    ];

    private static readonly Dictionary<char, string> Letters = new()
    {
        ['a'] = "æ", ['b'] = "b", ['c'] = "k", ['d'] = "d", ['e'] = "ɛ", ['f'] = "f", ['g'] = "ɡ", ['h'] = "h",
        ['i'] = "ɪ", ['j'] = "dʒ", ['k'] = "k", ['l'] = "l", ['m'] = "m", ['n'] = "n", ['o'] = "ɑ", ['p'] = "p",
        ['q'] = "k", ['r'] = "ɹ", ['s'] = "s", ['t'] = "t", ['u'] = "ʌ", ['v'] = "v", ['w'] = "w", ['x'] = "ks",
        ['y'] = "j", ['z'] = "z",
    };

    /// <summary>Compact letter-to-sound fallback for out-of-vocabulary words (digraphs first, then single
    /// letters). Approximate — most real words should come from CMUdict.</summary>
    public static string LetterToSound(string word)
    {
        StringBuilder sb = new();
        int i = 0;
        while (i < word.Length)
        {
            bool matched = false;
            foreach ((string gr, string ipa) in Digraphs)
            {
                if (i + gr.Length <= word.Length && word.AsSpan(i, gr.Length).SequenceEqual(gr))
                {
                    sb.Append(ipa);
                    i += gr.Length;
                    matched = true;
                    break;
                }
            }
            if (matched)
            {
                continue;
            }
            if (Letters.TryGetValue(word[i], out string? letter))
            {
                sb.Append(letter);
            }
            i++; // skip apostrophes / unknown
        }
        return sb.ToString();
    }
}
