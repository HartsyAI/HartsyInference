using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of misaki's American-English <c>Lexicon</c> (hexgrad/misaki <c>en.py</c>, Apache-2.0): the gold and
/// silver pronunciation dictionaries Kokoro-82M was trained against, plus misaki's stress rules, acronym spelling,
/// -s/-ed/-ing morphology and number reading. There is no part-of-speech tagger, so a per-tag entry reads its
/// <c>DEFAULT</c>, and the few tag-dependent special cases take the reading the tag almost always has.</summary>
public sealed partial class MisakiLexicon
{
    internal const char PrimaryStress = 'ˈ';
    internal const char SecondaryStress = 'ˌ';
    internal const string Vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
    internal const string Consonants = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
    private const string Diphthongs = "AIOQWYʤʧ";
    private const string UsTaus = "AIOWYiuæɑəɛɪɹʊʌ";
    private static readonly string[] Ordinals = ["st", "nd", "rd", "th"];
    private static readonly Dictionary<string, string> Symbols = new(StringComparer.Ordinal)
    {
        ["%"] = "percent", ["&"] = "and", ["+"] = "plus", ["@"] = "at",
    };
    private static readonly Dictionary<char, (string Unit, string Fraction)> Currencies = new()
    {
        ['$'] = ("dollar", "cent"), ['£'] = ("pound", "pence"), ['€'] = ("euro", "cent"),
    };

    private readonly Dictionary<string, Entry> _golds;
    private readonly Dictionary<string, Entry> _silvers;

    private MisakiLexicon(Dictionary<string, Entry> golds, Dictionary<string, Entry> silvers, bool british = false)
    {
        _golds = golds;
        _silvers = silvers;
        British = british;
    }

    /// <summary>misaki's British mode (<c>gb_gold.json</c>/<c>gb_silver.json</c>): the suffixes -s, -ed and -ing take
    /// an <c>ɪ</c> where American takes <c>ᵻ</c>, and there is no flap.</summary>
    public bool British { get; }

    /// <summary>A lexicon with no entries: every word goes to the front-end's fallbacks.</summary>
    public static MisakiLexicon Empty { get; } = new(new Dictionary<string, Entry>(StringComparer.Ordinal),
        new Dictionary<string, Entry>(StringComparer.Ordinal));

    /// <summary>Entries in the gold dictionary, after misaki's case-variant growth.</summary>
    public int GoldCount => _golds.Count;

    /// <summary>Loads <c>us_gold.json</c> and <c>us_silver.json</c> (or the <c>gb_</c> pair with
    /// <paramref name="british"/>).</summary>
    public static MisakiLexicon FromFiles(string goldPath, string silverPath, bool british = false)
    {
        using FileStream gold = File.OpenRead(goldPath);
        using FileStream silver = File.OpenRead(silverPath);
        return FromStreams(gold, silver, british);
    }

    /// <summary>Loads the two dictionaries from JSON streams of <c>{word: phonemes | {tag: phonemes | null}}</c>.</summary>
    public static MisakiLexicon FromStreams(Stream gold, Stream silver, bool british = false)
    {
        ArgumentNullException.ThrowIfNull(gold);
        ArgumentNullException.ThrowIfNull(silver);
        return new MisakiLexicon(Grow(Parse(gold)), Grow(Parse(silver)), british);
    }

    /// <summary>Phonemizes one token, or returns null when the lexicon has no reading for it.</summary>
    /// <param name="text">The token's text.</param>
    /// <param name="ctx">The reading context of the token to its right.</param>
    /// <param name="currency">Currency symbol that preceded a number token.</param>
    /// <param name="isHead">Whether the token starts its whitespace-delimited word.</param>
    /// <param name="nextIsWordTo">Whether the next word is "to" (selects "used to").</param>
    /// <param name="inMixedCase">Whether the surrounding text has lowercase letters, so an all-caps word in it reads
    /// as an acronym (what misaki learns from the tagger's NNP) rather than shouting.</param>
    /// <param name="atClauseStart">Whether the token opens its clause (selects the determiner "that").</param>
    internal string? Phonemize(string text, TokenContext ctx, char? currency, bool isHead, bool nextIsWordTo,
        bool inMixedCase, bool atClauseStart)
    {
        string word = text.Replace('‘', '\'').Replace('’', '\'').Normalize(NormalizationForm.FormKC);
        double? stress = word == word.ToLowerInvariant() ? null : word == word.ToUpperInvariant() ? 2 : 0.5;
        string? ps = GetWord(word, stress, ctx, nextIsWordTo, inMixedCase && LooksLikeAcronym(word), atClauseStart);
        if (ps is not null) return AppendCurrency(ps, currency);
        if (IsNumber(word, isHead)) return GetNumber(word, currency, isHead);
        return null;
    }

    /// <summary>The word spelled letter by letter, as misaki reads an acronym; null if a letter has no reading.</summary>
    internal string? Spell(string word) => GetNnp(word);

    internal static int StressWeight(string? ps)
    {
        if (string.IsNullOrEmpty(ps)) return 0;
        int weight = 0;
        foreach (char c in ps) weight += Diphthongs.Contains(c) ? 2 : 1;
        return weight;
    }

    /// <summary>misaki <c>apply_stress</c>: demotes, strips, or adds stress marks for a requested stress level.</summary>
    internal static string? ApplyStress(string? ps, double? stress)
    {
        if (ps is null || stress is not double s) return ps;
        bool hasPrimary = ps.Contains(PrimaryStress), hasSecondary = ps.Contains(SecondaryStress);
        if (s < -1) return ps.Replace(PrimaryStress.ToString(), "").Replace(SecondaryStress.ToString(), "");
        if (s == -1 || ((s == 0 || s == -0.5) && hasPrimary))
            return ps.Replace(SecondaryStress.ToString(), "").Replace(PrimaryStress, SecondaryStress);
        if ((s == 0 || s == 0.5 || s == 1) && !hasPrimary && !hasSecondary)
            return HasVowel(ps) ? Restress(SecondaryStress + ps) : ps;
        if (s >= 1 && !hasPrimary && hasSecondary) return ps.Replace(SecondaryStress, PrimaryStress);
        if (s > 1 && !hasPrimary && !hasSecondary) return HasVowel(ps) ? Restress(PrimaryStress + ps) : ps;
        return ps;
    }

    private static bool HasVowel(string ps) => ps.AsSpan().IndexOfAny(Vowels) >= 0;

    /// <summary>Moves every stress mark to just before the next vowel.</summary>
    private static string Restress(string ps)
    {
        List<(double Key, char C)> items = new(ps.Length);
        for (int i = 0; i < ps.Length; i++)
        {
            double key = i;
            if (ps[i] is PrimaryStress or SecondaryStress)
            {
                int j = ps.AsSpan(i).IndexOfAny(Vowels);
                if (j >= 0) key = i + j - 0.5;
            }
            items.Add((key, ps[i]));
        }
        StringBuilder sb = new(ps.Length);
        foreach ((double _, char c) in items.OrderBy(static x => x.Key)) sb.Append(c);
        return sb.ToString();
    }

    private string? GetNnp(string word)
    {
        StringBuilder sb = new();
        foreach (char c in word)
        {
            if (!char.IsLetter(c)) continue;
            string? letter = _golds.TryGetValue(char.ToUpperInvariant(c).ToString(), out Entry? e) ? e.Default : null;
            if (letter is null) return null;
            sb.Append(letter);
        }
        string ps = ApplyStress(sb.ToString(), 0)!;
        int last = ps.LastIndexOf(SecondaryStress);
        return last < 0 ? ps : ps[..last] + PrimaryStress + ps[(last + 1)..];
    }

    /// <summary>Stands in for spaCy's NNP tag on an all-caps word: spelled out unless it is a dictionary word an
    /// all-caps writer means as emphasis — a function word ("IF", "COULD") or a longer common word ("PLENTY").</summary>
    private bool LooksLikeAcronym(string word)
    {
        if (word.Length < 2 || !word.Any(char.IsLetter) || word != word.ToUpperInvariant()) return false;
        string lower = word.ToLowerInvariant();
        if (lower == "us") return true;
        if (_golds.ContainsKey(lower)) return false;
        return !_silvers.ContainsKey(lower) || lower.Length <= 4;
    }

    private string? GetSpecialCase(string word, double? stress, TokenContext ctx, bool nextIsWordTo, bool atClauseStart)
    {
        if (Symbols.TryGetValue(word, out string? symbol)) return Lookup(symbol, null, ctx);
        string trimmed = word.Trim('.');
        if (trimmed.Contains('.') && IsAlpha(word.Replace(".", ""))
            && word.Split('.').Max(static p => p.Length) < 3)
        {
            return GetNnp(word);
        }
        switch (word)
        {
            // Without a tagger "a" is read as the determiner unless nothing follows it ("plan A.").
            case "a" or "A": return ctx.FutureVowel is null ? "ˈA" : "ɐ";
            case "AM": return GetNnp(word);
            case "am" or "Am":
                return ctx.FutureVowel is null || word != "am" || stress > 0 ? Gold("am") : "ɐm";
            case "an" or "An" or "AN": return "ɐn";
            case "I": return SecondaryStress + "I";
            case "to" or "To" or "TO":
                return ctx.FutureVowel switch { null => Gold("to"), false => "tə", true => "tʊ" };
            case "in" or "In" or "IN": return (ctx.FutureVowel is null ? PrimaryStress.ToString() : "") + "ɪn";
            case "the" or "The" or "THE": return ctx.FutureVowel == true ? "ði" : "ðə";
            // misaki's tagger marks the pronoun "that" (DT, stressed). Without one, only a "that" closing a clause it
            // did not open is read as the pronoun ("do that!"): measured against spaCy, everywhere else the
            // conjunction/relative reading is the more common one.
            case "that" or "That" or "THAT":
                return ctx.FutureVowel is null && !atClauseStart ? Gold("that", "DT") : Gold("that");
            case "used" or "Used" or "USED":
                return nextIsWordTo ? Gold("used", "VBD") : Gold("used");
        }
        return VersusRegex().IsMatch(word) ? Lookup("versus", null, ctx) : null;
    }

    /// <summary>A gold reading (a tagged one when <paramref name="tag"/> is given), or null when the lexicon lacks it,
    /// so a special case over a partial lexicon falls through to the ordinary lookup instead of throwing.</summary>
    private string? Gold(string word, string? tag = null)
    {
        if (!_golds.TryGetValue(word, out Entry? entry)) return null;
        if (tag is null) return entry.Default;
        return entry.ByTag is not null && entry.ByTag.TryGetValue(tag, out string? tagged) ? tagged : entry.Default;
    }

    private bool IsKnown(string word)
    {
        if (_golds.ContainsKey(word) || Symbols.ContainsKey(word) || _silvers.ContainsKey(word)) return true;
        if (!IsAlpha(word) || !word.All(IsLexiconChar)) return false;
        if (word.Length == 1) return true;
        if (word == word.ToUpperInvariant() && _golds.ContainsKey(word.ToLowerInvariant())) return true;
        return word[1..] == word[1..].ToUpperInvariant();
    }

    /// <summary>Python <c>str.isalpha</c>: non-empty and all letters.</summary>
    private static bool IsAlpha(string s) => s.Length > 0 && s.All(char.IsLetter);

    private static bool IsLexiconChar(char c) => c is '\'' or '-' or (>= 'A' and <= 'Z') or (>= 'a' and <= 'z');

    private string? Lookup(string word, double? stress, TokenContext? ctx, bool nnp = false)
    {
        bool isNnp = false;
        if (word == word.ToUpperInvariant() && !_golds.ContainsKey(word))
        {
            word = word.ToLowerInvariant();
            isNnp = nnp;
        }
        Entry? entry = _golds.GetValueOrDefault(word);
        if (entry is null && !isNnp) entry = _silvers.GetValueOrDefault(word);
        string? ps = entry?.Read(ctx is not null && ctx.FutureVowel is null);
        if (ps is null || (isNnp && !ps.Contains(PrimaryStress)))
        {
            string? spelled = GetNnp(word);
            if (spelled is not null) return spelled;
        }
        return ApplyStress(ps, stress);
    }

    private string? S(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char last = stem[^1];
        if ("ptkfθ".Contains(last)) return stem + "s";
        if ("szʃʒʧʤ".Contains(last)) return stem + (British ? "ɪz" : "ᵻz");
        return stem + "z";
    }

    private string? StemS(string word, double? stress, TokenContext? ctx)
    {
        if (word.Length < 3 || !word.EndsWith('s')) return null;
        string stem;
        if (!word.EndsWith("ss", StringComparison.Ordinal) && IsKnown(word[..^1])) stem = word[..^1];
        else if ((word.EndsWith("'s", StringComparison.Ordinal) || (word.Length > 4 && word.EndsWith("es", StringComparison.Ordinal)
            && !word.EndsWith("ies", StringComparison.Ordinal))) && IsKnown(word[..^2])) stem = word[..^2];
        else if (word.Length > 4 && word.EndsWith("ies", StringComparison.Ordinal) && IsKnown(word[..^3] + "y")) stem = word[..^3] + "y";
        else return null;
        return S(Lookup(stem, stress, ctx));
    }

    private string? Ed(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char last = stem[^1];
        if ("pkfθʃsʧ".Contains(last)) return stem + "t";
        if (last == 'd') return stem + (British ? "ɪd" : "ᵻd");
        if (last != 't') return stem + "d";
        if (British || stem.Length < 2) return stem + "ɪd";
        if (UsTaus.Contains(stem[^2])) return stem[..^1] + "ɾᵻd";
        return stem + "ᵻd";
    }

    private string? StemEd(string word, double? stress, TokenContext? ctx)
    {
        if (word.Length < 4 || !word.EndsWith('d')) return null;
        string stem;
        if (!word.EndsWith("dd", StringComparison.Ordinal) && IsKnown(word[..^1])) stem = word[..^1];
        else if (word.Length > 4 && word.EndsWith("ed", StringComparison.Ordinal) && !word.EndsWith("eed", StringComparison.Ordinal)
            && IsKnown(word[..^2])) stem = word[..^2];
        else return null;
        return Ed(Lookup(stem, stress, ctx));
    }

    private string? Ing(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        if (British)
        {
            if (stem[^1] is 'ə' or 'ː') return null;
        }
        else if (stem.Length > 1 && stem[^1] == 't' && UsTaus.Contains(stem[^2])) return stem[..^1] + "ɾɪŋ";
        return stem + "ɪŋ";
    }

    private string? StemIng(string word, double? stress, TokenContext? ctx)
    {
        if (word.Length < 5 || !word.EndsWith("ing", StringComparison.Ordinal)) return null;
        string stem;
        if (word.Length > 5 && IsKnown(word[..^3])) stem = word[..^3];
        else if (IsKnown(word[..^3] + "e")) stem = word[..^3] + "e";
        else if (word.Length > 5 && DoubledIngRegex().IsMatch(word) && IsKnown(word[..^4])) stem = word[..^4];
        else return null;
        return Ing(Lookup(stem, stress, ctx));
    }

    private string? GetWord(string word, double? stress, TokenContext ctx, bool nextIsWordTo, bool nnp, bool atClauseStart)
    {
        string? special = GetSpecialCase(word, stress, ctx, nextIsWordTo, atClauseStart);
        if (special is not null) return special;
        string wl = word.ToLowerInvariant();
        if (word.Length > 1 && IsAlpha(word.Replace("'", "")) && word != wl && (!nnp || word.Length > 7)
            && !_golds.ContainsKey(word) && !_silvers.ContainsKey(word)
            && (word == word.ToUpperInvariant() || word[1..] == word[1..].ToLowerInvariant())
            && (_golds.ContainsKey(wl) || _silvers.ContainsKey(wl)
                || StemS(wl, stress, ctx) is not null || StemEd(wl, stress, ctx) is not null || StemIng(wl, stress, ctx) is not null))
        {
            word = wl;
        }
        if (IsKnown(word)) return Lookup(word, stress, ctx, nnp);
        if (word.EndsWith("s'", StringComparison.Ordinal) && IsKnown(word[..^2] + "'s")) return Lookup(word[..^2] + "'s", stress, ctx);
        if (word.EndsWith('\'') && IsKnown(word[..^1])) return Lookup(word[..^1], stress, ctx);
        return StemS(word, stress, ctx) ?? StemEd(word, stress, ctx) ?? StemIng(word, stress ?? 0.5, ctx);
    }

    private static bool IsDigits(string s) => s.Length > 0 && s.All(char.IsAsciiDigit);

    internal static bool IsNumber(string word, bool isHead)
    {
        if (!word.Any(char.IsAsciiDigit)) return false;
        foreach (string suffix in (string[])["ing", "'d", "ed", "'s", .. Ordinals, "s"])
        {
            if (word.EndsWith(suffix, StringComparison.Ordinal))
            {
                word = word[..^suffix.Length];
                break;
            }
        }
        for (int i = 0; i < word.Length; i++)
        {
            char c = word[i];
            if (!char.IsAsciiDigit(c) && c is not (',' or '.') && !(isHead && i == 0 && c == '-')) return false;
        }
        return true;
    }

    private static bool IsCurrency(string word)
    {
        int dots = word.Count(static c => c == '.');
        if (dots == 0) return true;
        if (dots > 1) return false;
        string cents = word.Split('.')[1];
        return cents.Length < 3 || cents.All(static c => c == '0');
    }

    private string? GetNumber(string word, char? currency, bool isHead)
    {
        Match suffixMatch = NumberSuffixRegex().Match(word);
        string? suffix = suffixMatch.Success ? suffixMatch.Value : null;
        if (suffix is not null) word = word[..^suffix.Length];
        List<string?> result = [];
        if (word.StartsWith('-'))
        {
            result.Add(Lookup("minus", null, null));
            word = word[1..];
        }
        void ExtendWords(string words)
        {
            foreach (string w in NonLetterRegex().Split(words))
            {
                if (w.Length == 0 || w == "and") continue;
                result.Add(Lookup(w, w == "point" ? -2 : null, null));
            }
        }
        // A run too long for a long is read digit by digit, as misaki reads an over-long non-head number.
        void ExtendNum(string digits)
        {
            if (long.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out long n))
                ExtendWords(EnglishNumberWords.Cardinal(n));
            else
                foreach (char d in digits) ExtendWords(EnglishNumberWords.Cardinal(d - '0'));
        }
        bool isCurrencySymbol = currency is char cs && Currencies.ContainsKey(cs);
        if (IsDigits(word) && suffix is not null && Ordinals.Contains(suffix))
        {
            if (long.TryParse(word, NumberStyles.None, CultureInfo.InvariantCulture, out long ordinal))
                ExtendWords(EnglishNumberWords.Ordinal(ordinal));
            else
                ExtendNum(word);
        }
        else if (result.Count == 0 && word.Length == 4 && !isCurrencySymbol && IsDigits(word))
        {
            ExtendWords(EnglishNumberWords.Year(long.Parse(word, CultureInfo.InvariantCulture)));
        }
        else if (!isHead && !word.Contains('.'))
        {
            string num = word.Replace(",", "");
            if (num.Length == 0) return null;
            if (num[0] == '0' || num.Length > 3)
            {
                foreach (char d in num) ExtendNum(d.ToString());
            }
            else if (num.Length == 3 && !num.EndsWith("00", StringComparison.Ordinal))
            {
                ExtendNum(num[..1]);
                if (num[1] == '0')
                {
                    result.Add(Lookup("O", -2, null));
                    ExtendNum(num[2..]);
                }
                else
                {
                    ExtendNum(num[1..]);
                }
            }
            else
            {
                ExtendNum(num);
            }
        }
        else if (word.Count(static c => c == '.') > 1 || !isHead)
        {
            foreach (string num in word.Replace(",", "").Split('.'))
            {
                if (num.Length == 0) continue;
                if (num[0] == '0' || (num.Length != 2 && num[1..].Any(static n => n != '0')))
                {
                    foreach (char d in num) ExtendNum(d.ToString());
                }
                else
                {
                    ExtendNum(num);
                }
            }
        }
        else if (isCurrencySymbol && IsCurrency(word))
        {
            (string unit, string fraction) = Currencies[currency!.Value];
            string[] parts = word.Replace(",", "").Split('.');
            List<(long Num, string Unit)> pairs = [];
            for (int i = 0; i < Math.Min(parts.Length, 2); i++)
            {
                long n = 0;
                if (parts[i].Length != 0 && !long.TryParse(parts[i], NumberStyles.None, CultureInfo.InvariantCulture, out n)) return null;
                pairs.Add((n, i == 0 ? unit : fraction));
            }
            if (pairs.Count > 1)
            {
                if (pairs[1].Num == 0) pairs.RemoveAt(1);
                else if (pairs[0].Num == 0) pairs.RemoveAt(0);
            }
            for (int i = 0; i < pairs.Count; i++)
            {
                if (i > 0) result.Add(Lookup("and", null, null));
                ExtendWords(EnglishNumberWords.Cardinal(pairs[i].Num));
                result.Add(Math.Abs(pairs[i].Num) != 1 && pairs[i].Unit != "pence"
                    ? StemS(pairs[i].Unit + "s", null, null) : Lookup(pairs[i].Unit, null, null));
            }
        }
        else
        {
            string clean = word.Replace(",", "");
            if (clean.Length == 0) return null;
            if (!clean.Contains('.'))
            {
                if (!long.TryParse(clean, NumberStyles.None, CultureInfo.InvariantCulture, out long n)) ExtendNum(clean);
                else ExtendWords(suffix is not null && Ordinals.Contains(suffix) ? EnglishNumberWords.Ordinal(n) : EnglishNumberWords.Cardinal(n));
            }
            else if (clean[0] == '.')
            {
                ExtendWords("point " + string.Join(' ', clean[1..].Select(static d => EnglishNumberWords.Cardinal(d - '0'))));
            }
            else
            {
                ExtendWords(EnglishNumberWords.Decimal(clean));
            }
        }
        if (result.Count == 0 || result.Any(static r => r is null)) return null;
        string joined = string.Join(' ', result);
        return suffix switch
        {
            "s" or "'s" => S(joined),
            "ed" or "'d" => Ed(joined),
            "ing" => Ing(joined),
            _ => joined,
        };
    }

    private string AppendCurrency(string ps, char? currency)
    {
        if (currency is not char c || !Currencies.TryGetValue(c, out (string Unit, string Fraction) names)) return ps;
        string? unit = StemS(names.Unit + "s", null, null);
        return unit is null ? ps : ps + " " + unit;
    }

    private static Dictionary<string, Entry> Parse(Stream json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        Dictionary<string, Entry> dict = new(StringComparer.Ordinal);
        foreach (JsonProperty p in doc.RootElement.EnumerateObject())
        {
            if (p.Value.ValueKind == JsonValueKind.String)
            {
                dict[p.Name] = new Entry(p.Value.GetString(), null);
                continue;
            }
            if (p.Value.ValueKind != JsonValueKind.Object) continue;
            Dictionary<string, string?> byTag = new(StringComparer.Ordinal);
            foreach (JsonProperty t in p.Value.EnumerateObject())
                byTag[t.Name] = t.Value.ValueKind == JsonValueKind.String ? t.Value.GetString() : null;
            if (!byTag.TryGetValue("DEFAULT", out string? fallback))
                throw new InvalidDataException($"misaki lexicon entry '{p.Name}' has no DEFAULT reading.");
            dict[p.Name] = new Entry(fallback, byTag);
        }
        return dict;
    }

    /// <summary>misaki <c>grow_dictionary</c>: a lowercase key also answers for its Capitalized form and vice versa,
    /// without overriding an explicit entry.</summary>
    private static Dictionary<string, Entry> Grow(Dictionary<string, Entry> d)
    {
        Dictionary<string, Entry> grown = new(d, StringComparer.Ordinal);
        foreach ((string k, Entry v) in d)
        {
            if (k.Length < 2) continue;
            string lower = k.ToLowerInvariant();
            string capital = char.ToUpperInvariant(lower[0]) + lower[1..];
            if (k == lower)
            {
                if (k != capital) grown.TryAdd(capital, v);
            }
            else if (k == capital)
            {
                grown.TryAdd(lower, v);
            }
        }
        return grown;
    }

    [GeneratedRegex(@"^(?i:vs)\.?$")]
    private static partial Regex VersusRegex();

    [GeneratedRegex(@"([bcdgklmnprstvxz])\1ing$|cking$")]
    private static partial Regex DoubledIngRegex();

    [GeneratedRegex(@"[a-z']+$")]
    private static partial Regex NumberSuffixRegex();

    [GeneratedRegex(@"[^a-z]+")]
    private static partial Regex NonLetterRegex();

    /// <summary>One dictionary value: a reading, or per-tag readings with a <c>DEFAULT</c>.</summary>
    private sealed record Entry(string? Default, IReadOnlyDictionary<string, string?>? ByTag)
    {
        /// <summary>The untagged reading; an entry with a <c>None</c> variant uses it at the end of a phrase, as
        /// misaki does when nothing with a vowel or consonant follows.</summary>
        public string? Read(bool atPhraseEnd) =>
            atPhraseEnd && ByTag is not null && ByTag.TryGetValue("None", out string? end) ? end : Default;
    }
}
