using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.MeloTts;

/// <summary>MeloTTS English text front-end: turns text into the model's <c>(phoneme ids, tone ids, language ids,
/// word2ph)</c>, mirroring <c>melo/text/english.py::g2p</c>. The orthographic text is WordPiece-tokenized
/// (bert-base-uncased) and the sub-tokens are grouped into words; each word is looked up in CMUdict
/// (ARPABET, stress digit → tone) with a letter fallback for OOV; phones are distributed across the word's
/// sub-tokens to build <c>word2ph</c> (the BERT-token → phoneme expansion). The result is padded with the blank
/// symbol, the tone offset and language id are applied, and (with <c>add_blank</c>) a blank is interleaved
/// between every phoneme.</summary>
public sealed class MeloTtsEnglishFrontend
{
    private const int EnLangId = 2;
    private const int EnToneStart = 7;

    private readonly Dictionary<string, int> _symbolToId;
    private readonly Dictionary<string, (string[] Phones, int[] Tones)> _dict;
    private readonly BertWordPieceTokenizer _tokenizer;

    public MeloTtsEnglishFrontend(BertWordPieceTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _symbolToId = new Dictionary<string, int>(Symbols.Length);
        for (int i = 0; i < Symbols.Length; i++) _symbolToId[Symbols[i]] = i;
        _dict = LoadCmudict();
    }

    /// <summary>Number of BERT-token slots after add_blank; the BERT feature provider must produce a feature per
    /// orthographic BERT token (incl. [CLS]/[SEP]) and expand it by <see cref="Word2Ph"/>.</summary>
    public sealed record Result(int[] PhoneIds, int[] ToneIds, int[] LangIds, int[] Word2Ph, string NormalizedText);

    /// <summary>Runs the g2p and packs ids. <paramref name="addBlank"/> interleaves the blank symbol between every
    /// phoneme (always true for the public checkpoints). <see cref="Result.NormalizedText"/> is the number-expanded
    /// text the g2p and <c>word2ph</c> were built from — the BERT provider must tokenize this same string so its
    /// token count matches <c>word2ph</c>.</summary>
    public Result Process(string text, bool addBlank = true)
    {
        string norm = TextNormalize(text);
        (List<string> phones, List<int> tones, List<int> word2ph) = G2p(norm);

        int[] phoneIds = phones.Select(p => _symbolToId.TryGetValue(p, out int id) ? id : _symbolToId["UNK"]).ToArray();
        int[] toneIds = tones.Select(t => t + EnToneStart).ToArray();
        int[] langIds = phones.Select(_ => EnLangId).ToArray();
        int[] w2p = word2ph.ToArray();

        if (addBlank)
        {
            phoneIds = Intersperse(phoneIds, 0);
            toneIds = Intersperse(toneIds, 0);
            langIds = Intersperse(langIds, 0);
            for (int i = 0; i < w2p.Length; i++) w2p[i] *= 2;
            w2p[0] += 1;
        }
        return new Result(phoneIds, toneIds, langIds, w2p, norm);
    }

    private (List<string> Phones, List<int> Tones, List<int> Word2Ph) G2p(string text)
    {
        // Group WordPiece sub-tokens into words (continuation pieces start with '#').
        List<List<string>> groups = new();
        foreach (string tok in _tokenizer.Tokens(text))
        {
            if (!tok.StartsWith('#')) groups.Add([tok]);
            else if (groups.Count > 0) groups[^1].Add(tok.Replace("#", ""));
        }

        List<string> phones = new();
        List<int> tones = new();
        List<int> word2ph = new();
        foreach (List<string> group in groups)
        {
            string w = string.Concat(group);
            int wordLen = group.Count;
            int phoneLen;
            if (_dict.TryGetValue(w.ToUpperInvariant(), out (string[] Phones, int[] Tones) entry))
            {
                phones.AddRange(entry.Phones);
                tones.AddRange(entry.Tones);
                phoneLen = entry.Phones.Length;
            }
            else if (_symbolToId.ContainsKey(w))
            {
                // punctuation / symbol token passes straight through (e.g. ".", ",", "?").
                phones.Add(w); tones.Add(0); phoneLen = 1;
            }
            else
            {
                // OOV fallback: spell letter phonemes (approximate; CMUdict covers the common case).
                string[] fb = LetterFallback(w);
                phones.AddRange(fb);
                tones.AddRange(Enumerable.Repeat(0, fb.Length));
                phoneLen = fb.Length;
            }
            word2ph.AddRange(DistributePhone(phoneLen, wordLen));
        }

        for (int i = 0; i < phones.Count; i++) phones[i] = PostReplace(phones[i]);

        // pad_start_end with the blank symbol "_".
        phones.Insert(0, "_"); phones.Add("_");
        tones.Insert(0, 0); tones.Add(0);
        word2ph.Insert(0, 1); word2ph.Add(1);
        return (phones, tones, word2ph);
    }

    // distribute_phone: spread n_phone phonemes across n_word sub-tokens as evenly as possible.
    private static int[] DistributePhone(int nPhone, int nWord)
    {
        int[] ppw = new int[Math.Max(nWord, 1)];
        for (int i = 0; i < nPhone; i++)
        {
            int min = int.MaxValue, idx = 0;
            for (int j = 0; j < ppw.Length; j++) if (ppw[j] < min) { min = ppw[j]; idx = j; }
            ppw[idx]++;
        }
        return ppw;
    }

    private static int[] Intersperse(int[] seq, int item)
    {
        int[] outArr = new int[seq.Length * 2 + 1];
        for (int i = 0; i < outArr.Length; i++) outArr[i] = item;
        for (int i = 0; i < seq.Length; i++) outArr[i * 2 + 1] = seq[i];
        return outArr;
    }

    private string PostReplace(string ph)
    {
        ph = ph switch { "\n" => ".", "·" or "、" or "，" or "：" or "；" => ",", "。" => ".", "！" => "!", "？" => "?", "..." => "…", "v" => "V", _ => ph };
        return _symbolToId.ContainsKey(ph) ? ph : "UNK";
    }

    // refine_ph: trailing stress digit -> tone (digit+1), lowercase.
    private static (string Ph, int Tone) RefinePh(string phn)
    {
        int tone = 0;
        if (phn.Length > 0 && char.IsDigit(phn[^1]))
        {
            tone = (phn[^1] - '0') + 1;
            phn = phn[..^1];
        }
        return (phn.ToLowerInvariant(), tone);
    }

    private Dictionary<string, (string[], int[])> LoadCmudict()
    {
        Dictionary<string, (string[], int[])> dict = new(StringComparer.Ordinal);
        using Stream s = OpenResource("cmudict.rep");
        using StreamReader r = new(s);
        int lineIndex = 0;
        string? line;
        while ((line = r.ReadLine()) is not null)
        {
            lineIndex++;
            if (lineIndex < 49) continue;     // skip the licence header (melo start_line = 49)
            line = line.Trim();
            int sep = line.IndexOf("  ", StringComparison.Ordinal);
            if (sep <= 0) continue;
            string word = line[..sep];
            string body = line[(sep + 2)..];
            List<string> phones = new();
            List<int> tones = new();
            foreach (string syllable in body.Split(" - "))
                foreach (string p in syllable.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    (string ph, int tone) = RefinePh(p);
                    phones.Add(ph); tones.Add(tone);
                }
            dict[word] = (phones.ToArray(), tones.ToArray());
        }
        return dict;
    }

    private static Stream OpenResource(string name)
    {
        Assembly asm = typeof(MeloTtsEnglishFrontend).Assembly;
        string? full = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(name, StringComparison.Ordinal));
        return full is null ? throw new FileNotFoundException($"Embedded resource '{name}' not found.")
            : asm.GetManifestResourceStream(full)!;
    }

    // Very small OOV fallback (single-letter ARPABET-ish). CMUdict covers the common words; this just avoids a crash.
    private static string[] LetterFallback(string w) =>
        w.Where(char.IsLetter).Select(c => char.ToLowerInvariant(c).ToString()).ToArray();

    private static string TextNormalize(string text) => NormalizeNumbers(text).ToLowerInvariant();

    // ── Number normalization ─────────────────────────────────────────────────────────────────────────────
    // Mirrors melo/text/english.py::normalize_numbers (the Tacotron numbers.py normalizer + inflect year
    // handling). Digits are otherwise OOV for CMUdict and get dropped, so "In 2026 I paid $42.50" must become
    // "in twenty twenty-six i paid forty-two dollars, fifty cents" before phonemization.
    private static readonly Regex CommaNumberRe = new(@"([0-9][0-9\,]+[0-9])", RegexOptions.Compiled);
    private static readonly Regex PoundsRe = new(@"£([0-9\,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex DollarsRe = new(@"\$([0-9\.\,]*[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex DecimalRe = new(@"([0-9]+\.[0-9]+)", RegexOptions.Compiled);
    private static readonly Regex OrdinalRe = new(@"[0-9]+(st|nd|rd|th)", RegexOptions.Compiled);
    private static readonly Regex NumberRe = new(@"[0-9]+", RegexOptions.Compiled);

    private static string NormalizeNumbers(string text)
    {
        text = CommaNumberRe.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = PoundsRe.Replace(text, m => m.Groups[1].Value + " pounds");
        text = DollarsRe.Replace(text, ExpandDollars);
        text = DecimalRe.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = OrdinalRe.Replace(text, m => TryLong(m.Value[..^2], out long o) ? Ordinal(o) : m.Value);
        text = NumberRe.Replace(text, m => TryLong(m.Value, out long n) ? ExpandNumber(n) : DigitByDigit(m.Value));
        return text;
    }

    private static bool TryLong(string s, out long v) =>
        long.TryParse(s, NumberStyles.None, CultureInfo.InvariantCulture, out v);

    private static string DigitByDigit(string digits) =>
        string.Join(' ', digits.Where(char.IsDigit).Select(d => OnesWords[d - '0']));

    private static string ExpandDollars(Match m)
    {
        string[] parts = m.Groups[1].Value.Split('.');
        long dollars = parts.Length > 0 && parts[0].Length > 0 && TryLong(parts[0], out long d) ? d : 0;
        long cents = parts.Length > 1 && parts[1].Length > 0 && TryLong(parts[1], out long c) ? c : 0;
        string Dollars() => $"{dollars} {(dollars == 1 ? "dollar" : "dollars")}";
        string Cents() => $"{cents} {(cents == 1 ? "cent" : "cents")}";
        if (dollars != 0 && cents != 0) return $"{Dollars()}, {Cents()}";
        if (dollars != 0) return Dollars();
        if (cents != 0) return Cents();
        return "zero dollars";
    }

    // inflect year handling: 1000–2999 read as a pair of two-digit groups ("twenty twenty-six"), with the
    // "two thousand N" and "N hundred" special cases.
    private static string ExpandNumber(long num)
    {
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return "two thousand";
            if (num > 2000 && num < 2010) return "two thousand " + Cardinal(num % 100);
            if (num % 100 == 0) return Cardinal(num / 100) + " hundred";
            long lo = num % 100;
            return Cardinal(num / 100) + " " + (lo < 10 ? "oh " + Cardinal(lo) : Cardinal(lo));
        }
        return Cardinal(num);
    }

    private static readonly string[] OnesWords =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];
    private static readonly string[] TensWords =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    private static string Cardinal(long n)
    {
        if (n < 0) return "minus " + Cardinal(-n);
        if (n < 20) return OnesWords[(int)n];
        if (n < 100) return TensWords[(int)(n / 10)] + (n % 10 != 0 ? "-" + OnesWords[(int)(n % 10)] : "");
        if (n < 1000) return OnesWords[(int)(n / 100)] + " hundred" + (n % 100 != 0 ? " " + Cardinal(n % 100) : "");
        if (n < 1_000_000) return Cardinal(n / 1000) + " thousand" + (n % 1000 != 0 ? " " + Cardinal(n % 1000) : "");
        if (n < 1_000_000_000) return Cardinal(n / 1_000_000) + " million" + (n % 1_000_000 != 0 ? " " + Cardinal(n % 1_000_000) : "");
        return Cardinal(n / 1_000_000_000) + " billion" + (n % 1_000_000_000 != 0 ? " " + Cardinal(n % 1_000_000_000) : "");
    }

    private static string Ordinal(long n)
    {
        string words = Cardinal(n);
        int sep = Math.Max(words.LastIndexOf(' '), words.LastIndexOf('-'));
        string head = sep < 0 ? "" : words[..(sep + 1)];
        string last = sep < 0 ? words : words[(sep + 1)..];
        return head + OrdinalizeWord(last);
    }

    private static string OrdinalizeWord(string w) => w switch
    {
        "one" => "first", "two" => "second", "three" => "third", "five" => "fifth",
        "eight" => "eighth", "nine" => "ninth", "twelve" => "twelfth",
        _ => w.EndsWith('y') ? w[..^1] + "ieth" : w + "th",
    };

    // The shared MeloTTS symbol table (config.symbols, 219 entries) — phoneme ids index directly into it.
    private static readonly string[] Symbols =
    [
        "_", "\"", "(", ")", "*", "/", ":", "AA", "E", "EE", "En", "N", "OO", "Q", "V", "[", "\\", "]", "^", "a",
        "a:", "aa", "ae", "ah", "ai", "an", "ang", "ao", "aw", "ay", "b", "by", "c", "ch", "d", "dh", "dy", "e", "e:",
        "eh", "ei", "en", "eng", "er", "ey", "f", "g", "gy", "h", "hh", "hy", "i", "i0", "i:", "ia", "ian", "iang",
        "iao", "ie", "ih", "in", "ing", "iong", "ir", "iu", "iy", "j", "jh", "k", "ky", "l", "m", "my", "n", "ng",
        "ny", "o", "o:", "ong", "ou", "ow", "oy", "p", "py", "q", "r", "ry", "s", "sh", "t", "th", "ts", "ty", "u",
        "u:", "ua", "uai", "uan", "uang", "uh", "ui", "un", "uo", "uw", "v", "van", "ve", "vn", "w", "x", "y", "z",
        "zh", "zy", "~", "æ", "ç", "ð", "ø", "ŋ", "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ɕ", "ə", "ɛ", "ɜ", "ɡ", "ɣ", "ɥ", "ɦ",
        "ɪ", "ɫ", "ɬ", "ɭ", "ɯ", "ɲ", "ɵ", "ɸ", "ɹ", "ɾ", "ʁ", "ʃ", "ʊ", "ʌ", "ʎ", "ʏ", "ʑ", "ʒ", "ʝ", "ʲ", "ˈ",
        "ˌ", "ː", "̃", "̩", "β", "θ", "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄄ", "ᄅ", "ᄆ", "ᄇ", "ᄈ", "ᄉ", "ᄊ", "ᄋ", "ᄌ", "ᄍ", "ᄎ",
        "ᄏ", "ᄐ", "ᄑ", "ᄒ", "ᅡ", "ᅢ", "ᅣ", "ᅤ", "ᅥ", "ᅦ", "ᅧ", "ᅨ", "ᅩ", "ᅪ", "ᅫ", "ᅬ", "ᅭ", "ᅮ", "ᅯ", "ᅰ",
        "ᅱ", "ᅲ", "ᅳ", "ᅴ", "ᅵ", "ᆨ", "ᆫ", "ᆮ", "ᆯ", "ᆷ", "ᆸ", "ᆼ", "ㄸ", "!", "?", "…", ",", ".", "'", "-", "¿",
        "¡", "SP", "UNK",
    ];
}
