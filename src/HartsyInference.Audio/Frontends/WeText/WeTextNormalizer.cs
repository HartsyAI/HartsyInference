using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>Port of the <c>wetext</c> 0.0.4 text normalizer (pengzhendong/wetext, Apache-2.0) that CosyVoice 2 pins:
/// text holding a decimal digit is tagged by <c>{lang}/tn/tagger.fst</c>, its tokens reordered, then verbalized by
/// <c>{lang}/tn/verbalizer.fst</c>; text without a digit is returned untouched. Output is byte-identical to the
/// Python package on the same FST files (see <c>tools/parity/wetext_parity_reference.py</c>). Only the TN operator
/// is ported. Instances are immutable and safe to share across threads.</summary>
public sealed class WeTextNormalizer
{
    private readonly FstTextRewriter? _englishTagger;
    private readonly FstTextRewriter? _englishVerbalizer;
    private readonly FstTextRewriter? _chineseTagger;
    private readonly FstTextRewriter? _chineseVerbalizer;

    private WeTextNormalizer(WeTextLanguage language, FstTextRewriter? englishTagger, FstTextRewriter? englishVerbalizer,
        FstTextRewriter? chineseTagger, FstTextRewriter? chineseVerbalizer)
    {
        Language = language;
        _englishTagger = englishTagger;
        _englishVerbalizer = englishVerbalizer;
        _chineseTagger = chineseTagger;
        _chineseVerbalizer = chineseVerbalizer;
    }

    /// <summary>The language this normalizer was built for; <see cref="WeTextLanguage.Auto"/> holds both grammars.</summary>
    public WeTextLanguage Language { get; }

    /// <summary>Loads the TN grammars under <paramref name="fstDirectory"/> (<c>en/tn/*.fst</c>, <c>zh/tn/*.fst</c>, the
    /// layout of the wetext FST repository). <see cref="WeTextLanguage.Auto"/> loads both, as wetext does.
    /// <paramref name="removeErhua"/> selects the Chinese <c>verbalizer_remove_erhua.fst</c>.</summary>
    public static WeTextNormalizer Load(string fstDirectory, WeTextLanguage language = WeTextLanguage.Auto,
        bool removeErhua = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(fstDirectory);
        if (!Enum.IsDefined(language)) throw new ArgumentOutOfRangeException(nameof(language));
        bool english = language is WeTextLanguage.Auto or WeTextLanguage.English;
        bool chinese = language is WeTextLanguage.Auto or WeTextLanguage.Chinese;
        string chineseVerbalizer = removeErhua ? "verbalizer_remove_erhua.fst" : "verbalizer.fst";
        return new WeTextNormalizer(language,
            english ? Grammar(fstDirectory, "en", "tagger.fst") : null,
            english ? Grammar(fstDirectory, "en", "verbalizer.fst") : null,
            chinese ? Grammar(fstDirectory, "zh", "tagger.fst") : null,
            chinese ? Grammar(fstDirectory, "zh", chineseVerbalizer) : null);
    }

    /// <summary>The FST files <see cref="Load"/> reads for a language, relative to the FST directory.</summary>
    public static IReadOnlyList<string> RequiredFiles(WeTextLanguage language = WeTextLanguage.Auto,
        bool removeErhua = false)
    {
        List<string> files = [];
        if (language is WeTextLanguage.Auto or WeTextLanguage.English)
            files.AddRange(["en/tn/tagger.fst", "en/tn/verbalizer.fst"]);
        if (language is WeTextLanguage.Auto or WeTextLanguage.Chinese)
            files.AddRange(["zh/tn/tagger.fst", removeErhua ? "zh/tn/verbalizer_remove_erhua.fst" : "zh/tn/verbalizer.fst"]);
        return files;
    }

    /// <summary>wetext 0.0.4 <c>Normalizer.normalize</c>. Throws <see cref="FormatException"/> where the Python
    /// package raises: the tagger accepts no path for the text (it returns nothing for some symbols, e.g. <c>+</c>
    /// and <c>#</c> in English).</summary>
    public string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!ContainsDecimalDigit(text)) return text;
        WeTextLanguage language = Resolve(text);
        return Verbalize(Tag(text, language), language);
    }

    /// <summary><see cref="Normalize"/>, returning false instead of throwing where the Python package raises (no tagger
    /// path, malformed tags, output that is not UTF-8) and for text the FSTs cannot take: a lone surrogate, which has
    /// no UTF-8 form, or a NUL character, which the byte-labelled FSTs would read as epsilon.</summary>
    public bool TryNormalize(string text, [NotNullWhen(true)] out string? normalized)
    {
        ArgumentNullException.ThrowIfNull(text);
        normalized = null;
        if (!ContainsDecimalDigit(text))
        {
            normalized = text;
            return true;
        }
        WeTextLanguage language = Resolve(text);
        try
        {
            string tagged = Tag(text, language);
            if (tagged.Length == 0) return false;
            normalized = Verbalize(tagged, language);
            return true;
        }
        catch (FormatException)
        {
            return false;
        }
        catch (ArgumentException)
        {
            // The NUL rejection, plus strict UTF-8's Encoder/DecoderFallbackException, which derive from it.
            return false;
        }
    }

    /// <summary>The tagger output for <paramref name="text"/> (wetext <c>Normalizer.tag</c>); empty when the tagger
    /// accepts no path.</summary>
    public string Tag(string text, WeTextLanguage language) => Rewriters(Concrete(language, text)).Tagger.Rewrite(text);

    /// <summary>Reorders the tagger output's token fields and runs the verbalizer (wetext
    /// <c>Normalizer.verbalize</c>).</summary>
    public string Verbalize(string tagged, WeTextLanguage language)
    {
        WeTextLanguage concrete = Concrete(language, tagged);
        return Rewriters(concrete).Verbalizer.Rewrite(WeTextTokenParser.Reorder(tagged, concrete));
    }

    /// <summary>wetext's language test: any CJK Unified Ideograph (U+4E00..U+9FFF).</summary>
    public static bool ContainsChinese(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        foreach (char c in text)
            if (c >= '一' && c <= '鿿') return true;
        return false;
    }

    // Python's re \d on str: any Unicode decimal digit (category Nd), including full-width and other scripts.
    private static bool ContainsDecimalDigit(string text)
    {
        foreach (Rune rune in text.EnumerateRunes())
            if (Rune.IsDigit(rune)) return true;
        return false;
    }

    private static FstTextRewriter Grammar(string directory, string language, string file)
    {
        string path = Path.Combine(directory, language, "tn", file);
        if (!File.Exists(path)) throw new FileNotFoundException($"WeText grammar {language}/tn/{file} not found", path);
        return FstTextRewriter.Load(path);
    }

    private WeTextLanguage Resolve(string text) => Language == WeTextLanguage.Auto
        ? ContainsChinese(text) ? WeTextLanguage.Chinese : WeTextLanguage.English
        : Language;

    private WeTextLanguage Concrete(WeTextLanguage language, string text) =>
        language == WeTextLanguage.Auto ? Resolve(text) : language;

    private (FstTextRewriter Tagger, FstTextRewriter Verbalizer) Rewriters(WeTextLanguage language)
    {
        (FstTextRewriter? tagger, FstTextRewriter? verbalizer) = language == WeTextLanguage.Chinese
            ? (_chineseTagger, _chineseVerbalizer)
            : (_englishTagger, _englishVerbalizer);
        if (tagger is null || verbalizer is null)
            throw new InvalidOperationException($"this normalizer was loaded for {Language}, not {language}");
        return (tagger, verbalizer);
    }
}
