using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Port of upstream IndexTTS <c>TextNormalizer.normalize</c> (<c>indextts/utils/front.py</c>), shared by
/// IndexTTS-1.5, 2.0 and 2.5: pronunciation-annotation protection, the English/Chinese routing rule, contraction
/// expansion, the optional term glossary, tech-term / pinyin / name protection around the injected FST normalizer,
/// and the character replacement maps. The FST itself (wetext upstream) is supplied through
/// <see cref="IndexTtsFstNormalizers"/>.</summary>
/// <remarks>Upstream deduplicates protected pinyin and names through a Python <c>set</c>, so its placeholder order
/// follows string-hash order and changes between runs; this port replaces the longest match first (ties in order of
/// appearance), the order under which a name nested in a longer match is never split.</remarks>
public sealed partial class IndexTtsFrontNormalizer
{
    /// <summary>Upstream <c>char_rep_map</c>, in its declaration (alternation) order.</summary>
    private static readonly (string From, string To)[] CharRepMap =
    [
        ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", " "),
        ("·", "-"), ("、", ","), ("...", "…"), (",,,", "…"), ("，，，", "…"), ("……", "…"),
        ("“", "'"), ("”", "'"), ("\"", "'"), ("‘", "'"), ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"),
        ("《", "'"), ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", "-"), ("～", "-"), ("~", "-"),
        ("「", "'"), ("」", "'"), (":", ","),
    ];

    private static readonly Dictionary<string, string> CharRep = BuildCharRep(includeDollar: false);
    private static readonly Dictionary<string, string> ZhCharRep = BuildCharRep(includeDollar: true);
    private static readonly Regex CharRepPattern = BuildCharRepPattern(includeDollar: false);
    private static readonly Regex ZhCharRepPattern = BuildCharRepPattern(includeDollar: true);

    private readonly IndexTtsFstNormalizers _fst;
    private readonly bool _enableGlossary;
    private readonly object _glossaryLock = new();
    private KeyValuePair<string, IndexTtsGlossaryTerm>[] _glossary = [];

    /// <param name="fst">The English/Chinese FST normalizers upstream takes from wetext.</param>
    /// <param name="enableGlossary">Upstream <c>TextNormalizer(enable_glossary=...)</c>: off for 1.5, on for 2.0/2.5.</param>
    public IndexTtsFrontNormalizer(IndexTtsFstNormalizers fst, bool enableGlossary)
    {
        ArgumentNullException.ThrowIfNull(fst);
        _fst = fst;
        _enableGlossary = enableGlossary;
    }

    /// <summary>Adds or replaces glossary terms (upstream <c>load_glossary</c>); used only when the glossary is enabled.
    /// Safe to call while other threads normalize: they see the old or the new term set, never a partial one.</summary>
    public void LoadGlossary(IEnumerable<KeyValuePair<string, IndexTtsGlossaryTerm>> terms)
    {
        ArgumentNullException.ThrowIfNull(terms);
        lock (_glossaryLock)
        {
            List<KeyValuePair<string, IndexTtsGlossaryTerm>> merged = [.. Volatile.Read(ref _glossary)];
            foreach (KeyValuePair<string, IndexTtsGlossaryTerm> term in terms)
            {
                int existing = merged.FindIndex(kv => kv.Key == term.Key);
                if (existing >= 0) merged[existing] = term;
                else merged.Add(term);
            }
            Volatile.Write(ref _glossary, [.. merged]);
        }
    }

    /// <summary>Upstream <c>TextNormalizer.normalize</c>.</summary>
    public string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        List<KeyValuePair<string, string>> pronunciations = ProtectPronunciationAnnotations(ref text);
        string result;
        if (UseChinese(text))
        {
            text = ContractionPattern().Replace(text, "$1 is");
            if (_enableGlossary) text = ApplyGlossary(text, chinese: true);
            string replaced = SaveTerms(text.TrimEnd(), TechTermPattern(), out List<string>? techTerms);
            replaced = SavePlaceholders(replaced, PinyinTonePattern(), "pinyin", out List<string>? pinyin);
            replaced = SavePlaceholders(replaced, NamePattern(), "n", out List<string>? names);
            result = RunFst(_fst.Chinese, replaced, fallback: string.Empty)!;
            result = RestorePlaceholders(result, "n", names, static n => n);
            result = RestorePlaceholders(result, "pinyin", pinyin, CorrectPinyin);
            result = RestoreTechTerms(result, techTerms);
            result = ReplaceChars(result, chinese: true);
        }
        else
        {
            text = ContractionPattern().Replace(text, "$1 is");
            if (_enableGlossary) text = ApplyGlossary(text, chinese: false);
            string replaced = SaveTerms(text, TechTermPattern(), out List<string>? techTerms);
            string? normalized = RunFst(_fst.English, replaced, fallback: null);
            result = normalized is null ? text : RestoreTechTerms(normalized, techTerms);
            result = ReplaceChars(result, chinese: false);
        }
        foreach (KeyValuePair<string, string> p in pronunciations) result = result.Replace(p.Key, p.Value, StringComparison.Ordinal);
        return result;
    }

    /// <summary>Upstream <c>clean_pattern.sub(char_rep_map)</c>: the plain character map, applied by IndexTTS-2.5 to
    /// the raw text before <see cref="Normalize"/>.</summary>
    public static string ReplaceChars(string text, bool chinese)
    {
        ArgumentNullException.ThrowIfNull(text);
        return chinese
            ? ZhCharRepPattern.Replace(text, static m => ZhCharRep[m.Value])
            : CharRepPattern.Replace(text, static m => CharRep[m.Value]);
    }

    /// <summary>Upstream <c>use_chinese</c>: Chinese characters, no ASCII letter at all, an email, or pinyin with a tone digit.</summary>
    public static bool UseChinese(string s)
    {
        ArgumentNullException.ThrowIfNull(s);
        if (ChineseChar().IsMatch(s) || !AsciiLetter().IsMatch(s) || EmailPattern().IsMatch(s)) return true;
        return PinyinTonePattern().IsMatch(s);
    }

    // Upstream catches every FST failure: English falls back to the unprotected text, Chinese to an empty string.
    private static string? RunFst(Func<string, string> fst, string text, string? fallback)
    {
        try
        {
            return fst(text);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
        {
            Logs.Warning($"[IndexTTS] text normalizer failed, using the upstream fallback: {ex.Message}");
            return fallback;
        }
    }

    private string ApplyGlossary(string text, bool chinese)
    {
        KeyValuePair<string, IndexTtsGlossaryTerm>[] sorted = Volatile.Read(ref _glossary);
        if (sorted.Length == 0) return text;
        // List.Sort is unstable; OrderByDescending keeps insertion order for equal lengths, like Python's sorted().
        foreach (KeyValuePair<string, IndexTtsGlossaryTerm> term in sorted.OrderByDescending(static kv => kv.Key.Length))
        {
            string replacement = (chinese ? term.Value.Chinese : term.Value.English) ?? term.Key;
            text = Regex.Replace(text, Regex.Escape(term.Key), _ => replacement, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        }
        return text;
    }

    private static List<KeyValuePair<string, string>> ProtectPronunciationAnnotations(ref string text)
    {
        List<KeyValuePair<string, string>> placeholders = [];
        text = PronunciationAnnotationPattern().Replace(text, m =>
        {
            string key = "PRONPLACEHOLDER" + IndexToAlpha(placeholders.Count) + "PRONPLACEHOLDER";
            placeholders.Add(new KeyValuePair<string, string>(key, m.Value));
            return key;
        });
        return placeholders;
    }

    private static string IndexToAlpha(int n)
    {
        StringBuilder sb = new();
        while (true)
        {
            sb.Insert(0, (char)('a' + (n % 26)));
            n = (n / 26) - 1;
            if (n < 0) break;
        }
        return sb.ToString();
    }

    private static List<string> DistinctMatches(string text, Regex pattern)
    {
        List<string> found = [];
        foreach (Match m in pattern.Matches(text))
            if (!found.Contains(m.Value)) found.Add(m.Value);
        return found;
    }

    /// <summary>Upstream <c>save_tech_terms</c>: hyphens inside tech terms become <c>&lt;H&gt;</c>, longest term first.</summary>
    private static string SaveTerms(string text, Regex pattern, out List<string>? terms)
    {
        List<string> found = DistinctMatches(text, pattern);
        terms = null;
        if (found.Count == 0) return text;
        terms = [.. found.OrderByDescending(static t => t.Length)];
        foreach (string term in terms) text = text.Replace(term, term.Replace("-", "<H>", StringComparison.Ordinal), StringComparison.Ordinal);
        return text;
    }

    private static string RestoreTechTerms(string text, List<string>? terms) =>
        terms is { Count: > 0 } ? TechPlaceholderPattern().Replace(text, "-") : text;

    /// <summary>Upstream <c>save_pinyin_tones</c> / <c>save_names</c>: each distinct match becomes <c>&lt;{tag}_a&gt;</c>, <c>&lt;{tag}_b&gt;</c>, …</summary>
    private static string SavePlaceholders(string text, Regex pattern, string tag, out List<string>? originals)
    {
        List<string> found = DistinctMatches(text, pattern);
        originals = null;
        if (found.Count == 0) return text;
        // Longest first, so a match nested in a longer one cannot split it (upstream's set order only sometimes avoids that).
        found = [.. found.OrderByDescending(static m => m.Length)];
        originals = found;
        for (int i = 0; i < found.Count; i++) text = text.Replace(found[i], Placeholder(tag, i), StringComparison.Ordinal);
        return text;
    }

    private static string RestorePlaceholders(string text, string tag, List<string>? originals, Func<string, string> restore)
    {
        if (originals is not { Count: > 0 }) return text;
        for (int i = 0; i < originals.Count; i++) text = text.Replace(Placeholder(tag, i), restore(originals[i]), StringComparison.Ordinal);
        return text;
    }

    private static string Placeholder(string tag, int index) => "<" + tag + "_" + (char)('a' + index) + ">";

    /// <summary>Upstream <c>correct_pinyin</c>: j/q/x followed by u/ü is written with v, and the syllable uppercased.</summary>
    private static string CorrectPinyin(string pinyin)
    {
        if ("jqxJQX".IndexOf(pinyin[0], StringComparison.Ordinal) < 0) return pinyin;
        return JqxUmlautPattern().Replace(pinyin, "${1}v${2}${3}").ToUpperInvariant();
    }

    private static Dictionary<string, string> BuildCharRep(bool includeDollar)
    {
        Dictionary<string, string> map = new(StringComparer.Ordinal);
        if (includeDollar) map["$"] = ".";
        foreach ((string from, string to) in CharRepMap) map[from] = to;
        return map;
    }

    private static Regex BuildCharRepPattern(bool includeDollar)
    {
        IEnumerable<string> keys = CharRepMap.Select(static m => m.From);
        if (includeDollar) keys = keys.Prepend("$");
        return new Regex(string.Join('|', keys.Select(Regex.Escape)), RegexOptions.Compiled | RegexOptions.CultureInvariant);
    }

    [GeneratedRegex(@"(what|where|who|which|how|t?here|it|s?he|that|this)'s", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ContractionPattern();

    [GeneratedRegex(@"[一-鿿]")]
    private static partial Regex ChineseChar();

    [GeneratedRegex(@"[a-zA-Z]")]
    private static partial Regex AsciiLetter();

    [GeneratedRegex(@"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$")]
    private static partial Regex EmailPattern();

    [GeneratedRegex(@"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex PinyinTonePattern();

    [GeneratedRegex(@"[一-鿿]+(?:[-·—][一-鿿]+){1,2}", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex NamePattern();

    [GeneratedRegex(@"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")]
    private static partial Regex TechTermPattern();

    [GeneratedRegex(@"\s*<H>\s*")]
    private static partial Regex TechPlaceholderPattern();

    [GeneratedRegex(@"<([^|>\n]+)\|([^>\n]+)>")]
    internal static partial Regex PronunciationAnnotationPattern();

    [GeneratedRegex(@"([jqx])[uü](n|e|an)*(\d)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex JqxUmlautPattern();
}
