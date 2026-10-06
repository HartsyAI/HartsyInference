using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Pipelines;

/// <summary>The text normalization IndexTTS-2.0 applies before tokenizing — the reference's
/// <c>TextNormalizer.normalize</c> (<c>indextts/utils/front.py</c>): English contractions (<c>it's → it is</c>),
/// the punctuation map (<c>: → ,</c>, curly quotes → <c>'</c>, <c>… </c>, …) and spoken-form expansion of English
/// numbers. The reference's spoken-form expansion runs WeTextProcessing's FST normalizer; the English subset of it
/// that matters for ordinary prompts is reproduced by <see cref="EnglishSpokenForm"/>. Chinese text keeps its digits
/// (the Chinese FST is not ported) but still gets the contraction and punctuation handling.</summary>
internal static partial class IndexTts2TextNormalizer
{
    /// <summary>Reference <c>char_rep_map</c>, in the reference's own (alternation) order.</summary>
    private static readonly (string From, string To)[] CharRepMap =
    [
        ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", " "),
        ("·", "-"), ("、", ","), ("...", "…"), (",,,", "…"), ("，，，", "…"), ("……", "…"),
        ("“", "'"), ("”", "'"), ("\"", "'"), ("‘", "'"), ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"),
        ("《", "'"), ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", "-"), ("～", "-"), ("~", "-"),
        ("「", "'"), ("」", "'"), (":", ","),
    ];

    private static readonly Regex CharRepPattern = new(string.Join('|', CharRepMap.Select(static m => Regex.Escape(m.From))), RegexOptions.Compiled);
    private static readonly Dictionary<string, string> CharRep = CharRepMap.ToDictionary(static m => m.From, static m => m.To);

    [GeneratedRegex(@"(what|where|who|which|how|t?here|it|s?he|that|this)'s", RegexOptions.IgnoreCase)]
    private static partial Regex ContractionPattern();

    [GeneratedRegex(@"[一-鿿]")]
    private static partial Regex ChineseChar();

    [GeneratedRegex(@"[a-zA-Z]")]
    private static partial Regex AlphaChar();

    [GeneratedRegex(@"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$")]
    private static partial Regex EmailPattern();

    [GeneratedRegex(@"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])", RegexOptions.IgnoreCase)]
    private static partial Regex PinyinTone();

    /// <summary>Reference <c>use_chinese</c>: Chinese characters, no letters at all, an email, or pinyin-with-tone.</summary>
    internal static bool UseChinese(string s)
    {
        if (ChineseChar().IsMatch(s) || !AlphaChar().IsMatch(s) || EmailPattern().IsMatch(s)) return true;
        return PinyinTone().IsMatch(s);
    }

    public static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        bool chinese = UseChinese(text);
        string result = ContractionPattern().Replace(text, "$1 is");
        if (!chinese) result = EnglishSpokenForm.Expand(result);
        return ReplaceChars(result, chinese);
    }

    private static string ReplaceChars(string text, bool chinese)
    {
        string mapped = CharRepPattern.Replace(text, static m => CharRep[m.Value]);
        // The Chinese branch's zh_char_rep_map is char_rep_map plus "$" -> ".".
        return chinese ? mapped.Replace("$", ".", StringComparison.Ordinal) : mapped;
    }
}
