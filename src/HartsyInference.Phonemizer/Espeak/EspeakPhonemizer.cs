using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Phonemizer.Espeak;

/// <summary>Pure-C# espeak-ng phonemizer: turns text into the IPA the espeak-trained TTS models (Piper, MeloTTS,
/// Zonos, StyleTTS2) consume. It composes the ported pieces, splitting text into words, looking each up in the
/// dictionary (falling back to the letter-to-sound rules), placing stress, and converting the resulting phonemes to
/// IPA. This is the default <see cref="IPhonemizer"/> backend; it can be swapped for a native espeak binding behind the
/// same interface. Full clause/number normalization is layered on later; this handles word-by-word phonemization.</summary>
public sealed class EspeakPhonemizer : IPhonemizer
{
    private readonly EspeakWordLookup _lookup;
    private readonly EspeakTranslator _rules;
    private readonly EspeakStress _stress;
    private readonly EspeakPhonemeTable _phon;
    private readonly EspeakIpaMap _ipa;

    private EspeakPhonemizer(EspeakWordLookup lookup, EspeakTranslator rules, EspeakStress stress, EspeakPhonemeTable phon, EspeakIpaMap ipa)
    {
        _lookup = lookup;
        _rules = rules;
        _stress = stress;
        _phon = phon;
        _ipa = ipa;
    }

    /// <summary>Resolves the <c>espeak-ng-data</c> directory from the <c>ESPEAK_DATA_DIR</c> environment variable, then
    /// the shared model cache (<c>HARTSYINFERENCE_MODEL_CACHE</c> or <c>~/.cache/hartsyinference/models</c>), and
    /// builds a phonemizer. Throws with guidance if the data cannot be found.</summary>
    public static EspeakPhonemizer FromCache(string language = "en")
    {
        foreach (string dir in CandidateDataDirs())
        {
            if (Directory.Exists(dir) && File.Exists(Path.Combine(dir, "phontab")))
                return FromDataDirectory(dir, language);
        }
        throw new HartsyInferenceException(
            "espeak-ng-data not found. Set ESPEAK_DATA_DIR to an espeak-ng-data directory (e.g. " +
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data), or place the data under the model cache.");
    }

    private static IEnumerable<string> CandidateDataDirs()
    {
        string? env = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (!string.IsNullOrEmpty(env)) yield return env;

        string? cacheRoot = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODEL_CACHE");
        if (string.IsNullOrEmpty(cacheRoot))
            cacheRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "hartsyinference", "models");
        yield return Path.Combine(cacheRoot, "Hartsy--espeak-ng-data", "espeak-ng-data");
        yield return Path.Combine(cacheRoot, "Hartsy--espeak-ng-data");
    }

    /// <summary>Builds a phonemizer for English from an <c>espeak-ng-data</c> directory containing <c>en_dict</c> and
    /// <c>phontab</c>. Multi-language support loads the matching <c>&lt;lang&gt;_dict</c> and phoneme table.</summary>
    public static EspeakPhonemizer FromDataDirectory(string dataDir, string language = "en")
    {
        try
        {
            string dictName = LanguageToDict(language);
            EspeakDictFile dict = EspeakDictFile.Load(Path.Combine(dataDir, dictName));
            EspeakPhonemeTable phon = EspeakPhonemeTable.Load(Path.Combine(dataDir, "phontab"), PhonemeTableName(language));
            return new EspeakPhonemizer(
                new EspeakWordLookup(dict),
                new EspeakTranslator(dict, phon, EspeakLetters.Latin()),
                new EspeakStress(phon),
                phon,
                EspeakIpaMap.Load());
        }
        catch (Exception ex) when (ex is not HartsyInferenceException)
        {
            Logs.Error($"Failed to build espeak phonemizer from '{dataDir}': {ex.Message}");
            throw new HartsyInferenceException($"Failed to build espeak phonemizer from '{dataDir}'.", ex);
        }
    }

    /// <inheritdoc/>
    public string PhonemizeToIpa(string text, string language)
    {
        List<string> wordIpa = new();
        foreach (string word in SplitWords(text))
        {
            List<byte> codes = PhonemizeWord(word);
            wordIpa.Add(_ipa.ToIpa(codes, _phon));
        }
        return string.Join(" ", wordIpa);
    }

    /// <inheritdoc/>
    public int[] PhonemizeToIds(string text, string language, PhonemeIdMap idMap)
        => idMap.Encode(PhonemizeToIpa(text, language));

    /// <inheritdoc/>
    public IReadOnlyList<string> PhonemizeToMnemonics(string text, string language)
    {
        List<string> result = new();
        foreach (string word in SplitWords(text))
            result.Add(_phon.Decode(PhonemizeWord(word)));
        return result;
    }

    private List<byte> PhonemizeWord(string word)
    {
        string lower = word.ToLowerInvariant();
        List<byte> codes;
        uint dflags = 0;

        if (_lookup.Lookup(lower, out EspeakLookupResult r))
        {
            codes = r.Phonemes;
            dflags = r.Flags;
        }
        else
        {
            byte[] buf = BuildBuffer(lower);
            codes = _rules.TranslateRules(buf, WordStart, out int endType, out List<byte> endPhonemes);
            if (endType != 0)
            {
                // A suffix was matched: strip it, re-translate the (possibly respelled) stem, then append the suffix.
                _rules.RemoveEnding(buf, WordStart, endType);
                string stem = ReadWord(buf, WordStart);
                List<byte> stemCodes;
                if (_lookup.Lookup(stem, out EspeakLookupResult rs))
                {
                    stemCodes = rs.Phonemes;
                    dflags = rs.Flags;
                }
                else
                {
                    stemCodes = _rules.TranslateRules(buf, WordStart);
                }
                stemCodes.AddRange(endPhonemes);
                codes = stemCodes;
            }
        }
        return _stress.SetWordStress(codes, dflags, tonic: -1, control: 0);
    }

    private const int WordStart = 2;

    private static byte[] BuildBuffer(string lower)
    {
        byte[] buf = new byte[200];
        Array.Fill(buf, (byte)' ');
        for (int i = 0; i < lower.Length && WordStart + i < buf.Length - 1; i++)
            buf[WordStart + i] = (byte)lower[i];
        buf[^1] = 0;
        return buf;
    }

    private static string ReadWord(byte[] buf, int start)
    {
        int end = start;
        while (end < buf.Length && buf[end] != (byte)' ' && buf[end] != 0)
            end++;
        System.Text.StringBuilder sb = new(end - start);
        for (int i = start; i < end; i++)
            sb.Append((char)buf[i]);
        return sb.ToString();
    }

    // Word split for the word-by-word path: runs of letters/apostrophes are words; everything else is a separator.
    private static IEnumerable<string> SplitWords(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            if (char.IsLetter(text[i]) || text[i] == '\'')
            {
                int start = i;
                while (i < text.Length && (char.IsLetter(text[i]) || text[i] == '\''))
                    i++;
                yield return text[start..i];
            }
            else
                i++;
        }
    }

    private static string LanguageToDict(string language) => $"{PhonemeTableName(language)}_dict";

    private static string PhonemeTableName(string language)
    {
        // espeak voice names map to dictionary/table names; English dialects share the 'en' data.
        string lang = language.ToLowerInvariant();
        if (lang.StartsWith("en", StringComparison.Ordinal)) return "en";
        int dash = lang.IndexOf('-');
        return dash > 0 ? lang[..dash] : lang;
    }
}
