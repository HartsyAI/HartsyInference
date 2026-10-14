using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>Port of the <c>wetext</c> text normalizer (pengzhendong/wetext, Apache-2.0): text is tagged by
/// <c>{lang}/tn/tagger.fst</c>, its tokens reordered, then verbalized by <c>{lang}/tn/verbalizer.fst</c>.
/// <see cref="Version"/> selects the release reproduced: 0.0.4 (CosyVoice 2's pin) normalizes only text holding a
/// decimal digit; 0.1.8 (IndexTTS) strips the text, always normalizes English and re-escapes field values in the
/// reorder. Output is byte-identical to the Python package on the same FST files (see
/// <c>tools/parity/wetext_parity_reference.py</c> and <c>tools/parity/indextts_frontend_dump.py</c>). Only the TN
/// operator is ported. Instances are immutable and safe to share across threads.</summary>
public sealed class WeTextNormalizer
{
    private readonly FstTextRewriter? _englishTagger;
    private readonly FstTextRewriter? _englishVerbalizer;
    private readonly FstTextRewriter? _chineseTagger;
    private readonly FstTextRewriter? _chineseVerbalizer;

    private WeTextNormalizer(WeTextLanguage language, WeTextVersion version, FstTextRewriter? englishTagger,
        FstTextRewriter? englishVerbalizer, FstTextRewriter? chineseTagger, FstTextRewriter? chineseVerbalizer)
    {
        Language = language;
        Version = version;
        _englishTagger = englishTagger;
        _englishVerbalizer = englishVerbalizer;
        _chineseTagger = chineseTagger;
        _chineseVerbalizer = chineseVerbalizer;
    }

    /// <summary>The language this normalizer was built for; <see cref="WeTextLanguage.Auto"/> holds both grammars.</summary>
    public WeTextLanguage Language { get; }

    /// <summary>The wetext release this normalizer reproduces.</summary>
    public WeTextVersion Version { get; }

    /// <summary>Loads the TN grammars under <paramref name="fstDirectory"/> (<c>en/tn/*.fst</c>, <c>zh/tn/*.fst</c>, the
    /// layout of the wetext FST repository). <see cref="WeTextLanguage.Auto"/> loads both, as wetext does.
    /// <paramref name="removeErhua"/> selects the Chinese <c>verbalizer_remove_erhua.fst</c>.</summary>
    public static WeTextNormalizer Load(string fstDirectory, WeTextLanguage language = WeTextLanguage.Auto,
        bool removeErhua = false, WeTextVersion version = WeTextVersion.V0_0_4)
    {
        ArgumentException.ThrowIfNullOrEmpty(fstDirectory);
        if (!Enum.IsDefined(language)) throw new ArgumentOutOfRangeException(nameof(language));
        if (!Enum.IsDefined(version)) throw new ArgumentOutOfRangeException(nameof(version));
        bool english = language is WeTextLanguage.Auto or WeTextLanguage.English;
        bool chinese = language is WeTextLanguage.Auto or WeTextLanguage.Chinese;
        string chineseVerbalizer = removeErhua ? "verbalizer_remove_erhua.fst" : "verbalizer.fst";
        return new WeTextNormalizer(language, version,
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

    /// <summary>wetext <c>Normalizer.normalize</c> of <see cref="Version"/>. Throws <see cref="FormatException"/> where
    /// the Python package raises: the tagger accepts no path for the text (it returns nothing for some symbols, e.g.
    /// <c>+</c> and <c>#</c> in English).</summary>
    public string Normalize(string text) => Normalize(text, Language);

    /// <summary><see cref="Normalize(string)"/> with the grammar forced to <paramref name="language"/>, as a wetext
    /// <c>Normalizer(lang=...)</c> built for one language does; <see cref="WeTextLanguage.Auto"/> detects it.</summary>
    public string Normalize(string text, WeTextLanguage language)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (Version == WeTextVersion.V0_0_4)
        {
            if (!ContainsDecimalDigit(text)) return text;
            WeTextLanguage concrete = Concrete(language, text);
            return Verbalize(Tag(text, concrete), concrete);
        }
        string prepared = WeTextTokenParser.PythonStrip(text);
        WeTextLanguage lang = language != WeTextLanguage.Auto ? language
            : ContainsChinese(prepared) || IsAllDigits(prepared) ? WeTextLanguage.Chinese : WeTextLanguage.English;
        bool normalize = lang == WeTextLanguage.English ? prepared.Length > 0 : ContainsDecimalDigit(prepared);
        if (!normalize) return prepared;
        string tagged = WeTextTokenParser.PythonStrip(Tag(prepared, lang));
        return WeTextTokenParser.PythonStrip(Verbalize(tagged, lang));
    }

    /// <summary><see cref="Normalize"/>, returning false instead of throwing where the Python package raises (no tagger
    /// path, malformed tags, output that is not UTF-8) and for text the FSTs cannot take: a lone surrogate, which has
    /// no UTF-8 form, or a NUL character, which the byte-labelled FSTs would read as epsilon.</summary>
    public bool TryNormalize(string text, [NotNullWhen(true)] out string? normalized)
    {
        ArgumentNullException.ThrowIfNull(text);
        normalized = null;
        try
        {
            normalized = Normalize(text);
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
        // Normalize passes the language it tagged with; Auto here only serves direct callers holding tagged text.
        WeTextLanguage concrete = Concrete(language, tagged);
        return Rewriters(concrete).Verbalizer.Rewrite(WeTextTokenParser.Reorder(tagged, concrete, Version));
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

    // Python str.isdigit() on a non-empty string, approximated by Unicode decimal digits.
    private static bool IsAllDigits(string text)
    {
        if (text.Length == 0) return false;
        foreach (Rune rune in text.EnumerateRunes())
            if (!Rune.IsDigit(rune)) return false;
        return true;
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
