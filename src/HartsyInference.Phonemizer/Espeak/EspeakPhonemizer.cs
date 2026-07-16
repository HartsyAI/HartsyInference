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
    private readonly EspeakVoiceVariant _variant;
    private readonly EspeakPhonemeList? _plist;
    private readonly EspeakPhonemeRenderer? _renderer;

    private EspeakPhonemizer(EspeakWordLookup lookup, EspeakTranslator rules, EspeakStress stress, EspeakPhonemeTable phon, EspeakIpaMap ipa, EspeakVoiceVariant variant, EspeakPhonemeIndex? index)
    {
        _lookup = lookup;
        _rules = rules;
        _stress = stress;
        _phon = phon;
        _ipa = ipa;
        _variant = variant;
        if (index is not null)
        {
            // Exact allophone + IPA path: run espeak's phoneme-program VM over the clause phoneme list.
            EspeakPhonemeInterpreter interp = new(phon, index);
            _plist = new EspeakPhonemeList(phon, interp, stress.StressFlags);
            _renderer = new EspeakPhonemeRenderer(interp);
        }
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

        // Standard Linux install locations for the espeak-ng package.
        yield return "/usr/lib/x86_64-linux-gnu/espeak-ng-data";
        yield return "/usr/share/espeak-ng-data";
    }

    /// <summary>Builds a phonemizer for English from an <c>espeak-ng-data</c> directory containing <c>en_dict</c> and
    /// <c>phontab</c>. Multi-language support loads the matching <c>&lt;lang&gt;_dict</c> and phoneme table.</summary>
    public static EspeakPhonemizer FromDataDirectory(string dataDir, string language = "en")
    {
        try
        {
            EspeakVoiceVariant variant = EspeakVoiceVariant.Resolve(dataDir, language);
            EspeakDictFile dict = EspeakDictFile.Load(Path.Combine(dataDir, $"{variant.DictName}_dict"));
            EspeakPhonemeTable phon = EspeakPhonemeTable.Load(Path.Combine(dataDir, "phontab"), variant.PhonemeTable);
            string phonindex = Path.Combine(dataDir, "phonindex");
            EspeakPhonemeIndex? index = File.Exists(phonindex) ? EspeakPhonemeIndex.Load(phonindex) : null;
            return new EspeakPhonemizer(
                new EspeakWordLookup(dict, variant.DictCondition),
                new EspeakTranslator(dict, phon, EspeakLetters.Latin(), variant.DictCondition),
                // English language options (tr_languages.c L('e','n')): first-syllable stress, reduce consecutive
                // unstressed vowels in unstressed words to diminished (stress_flags 0x08).
                new EspeakStress(phon, stressRule: 0, stressFlags: 0x08),
                phon,
                EspeakIpaMap.Load(variant.PhonemeTable),
                variant,
                index);
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
        List<IReadOnlyList<byte>> words = new();
        foreach (string word in SplitWords(text))
            words.Add(PhonemizeWord(word));

        if (_plist is not null && _renderer is not null)
        {
            // Exact path: build the clause phoneme list (allophones + reduction) and render via the phoneme programs.
            List<EspeakPhonemeListEntry> list = _plist.Build(words);
            return _renderer.Render(list);
        }

        // Fallback: per-word mnemonic -> IPA map (used only when phonindex is unavailable).
        return string.Join(" ", words.Select(c => _ipa.ToIpa((List<byte>)c, _phon)));
    }

    /// <summary>Sentence/clause punctuation that carries prosody (pauses, phrasing). Preserved verbatim in the
    /// phoneme stream when <c>preservePunctuation</c> is set — TTS front-ends like StyleTTS2 encode these as their
    /// own tokens and rely on them for phrasing. Mirrors espeak-ng's <c>preserve_punctuation</c> behaviour.</summary>
    private const string ProsodyPunctuation = ".,!?;:…";

    /// <summary>As <see cref="PhonemizeToIpa(string,string)"/>, but when <paramref name="preservePunctuation"/> is
    /// set, sentence/clause punctuation (<see cref="ProsodyPunctuation"/>) is kept as standalone, space-delimited
    /// tokens between clauses. Each maximal run of words up to a punctuation mark is phonemized as one clause (so
    /// intra-clause stress/reduction is unchanged), then the punctuation is emitted verbatim. This matches the
    /// reference <c>espeak(preserve_punctuation=True)</c> → word-tokenize pipeline that speech models were trained
    /// on; without it the duration/prosody predictor sees a run-on utterance and slurs across phrase boundaries.</summary>
    public string PhonemizeToIpa(string text, string language, bool preservePunctuation)
    {
        if (!preservePunctuation)
            return PhonemizeToIpa(text, language);

        System.Text.StringBuilder sb = new();
        int clauseStart = 0;
        void FlushClause(int end)
        {
            if (end <= clauseStart)
                return;
            string ipa = PhonemizeToIpa(text[clauseStart..end], language);
            if (ipa.Length == 0)
                return;
            if (sb.Length > 0)
                sb.Append(' ');
            sb.Append(ipa);
        }
        for (int i = 0; i < text.Length; i++)
        {
            if (ProsodyPunctuation.IndexOf(text[i]) < 0)
                continue;
            FlushClause(i);
            if (sb.Length > 0)
                sb.Append(' ');
            sb.Append(text[i]);
            clauseStart = i + 1;
        }
        FlushClause(text.Length);
        return sb.ToString();
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

        bool found = _lookup.Lookup(lower, out EspeakLookupResult r);
        if (found && r.Phonemes.Count > 0)
        {
            codes = r.Phonemes;
            dflags = r.Flags;
        }
        else
        {
            if (found) dflags = r.Flags; // flag-only dictionary entry: keep its flags, pronounce via the rules
            byte[] buf = BuildBuffer(lower);
            codes = TranslateWithEndings(buf, WordStart, ref dflags, 0);
        }
        // Flapping (t/d -> ɾ), voicing assimilation, and vowel reduction are applied data-driven by the phoneme-program
        // VM during rendering (EspeakPhonemeList), so no separate flap-t post-process is needed.
        return _stress.SetWordStress(codes, dflags, tonic: -1, control: 0);
    }

    // Translates a word buffer by rules, handling a matched prefix or suffix by stripping it and re-translating the
    // stem. The stem is re-translated recursively so stacked affixes resolve (e.g. "killings" = kill+ing+s,
    // "nonpayment" = non+payment), matching espeak's repeated TranslateWord on the stem. Stripping the prefix and
    // re-translating the stem at a word boundary is what lets the stem keep its own (primary) stress.
    private List<byte> TranslateWithEndings(byte[] buf, int start, ref uint dflags, int depth)
    {
        List<byte> codes = _rules.TranslateRules(buf, start, out int endType, out List<byte> endPhonemes);
        if (endType == 0 || depth >= 8)
            return codes;

        if ((endType & EspeakRuleCodes.SufxP) != 0)
        {
            if (PrefixIsLexicalized(buf, start))
            {
                // The word minus a standard suffix is a known word (e.g. "outfit" + s), so it is lexicalized: don't
                // strip the prefix. Re-translate suppressing prefix detection and handle the suffix normally below.
                codes = _rules.TranslateRules(buf, start, out endType, out endPhonemes, EspeakTranslator.FlagNoPrefix);
                if (endType == 0)
                    return codes;
            }
            else
            {
                // Standard prefix (non-, dis-, re-, un-, ...): the prefix rule matched after consuming its letters, so
                // the stem begins at start + length. Put a space before the stem so it re-translates at a word boundary
                // (its word-start stress rules fire), then prepend the prefix phonemes (lighter stress).
                int nChars = endType & 0xf;
                int stemStart = start + nChars;
                byte saved = buf[stemStart - 1];
                buf[stemStart - 1] = (byte)' ';
                uint stemFlags = 0;
                List<byte> stem = TranslateWithEndings(buf, stemStart, ref stemFlags, depth + 1);
                buf[stemStart - 1] = saved;
                // Lock the stem's own stress (its primary may be positional) before prepending the lighter-stressed
                // prefix, so the final pass doesn't move the primary onto the prefix.
                stem = _stress.SetWordStress(stem, stemFlags, tonic: -1, control: 0);
                endPhonemes.AddRange(stem);
                return endPhonemes;
            }
        }

        // Standard suffix: strip it, re-translate the (possibly respelled) stem, then append the suffix phonemes.
        _rules.RemoveEnding(buf, start, endType);
        string stemWord = ReadWord(buf, start);
        List<byte> stemCodes;
        bool stemFound = _lookup.Lookup(stemWord, out EspeakLookupResult rs);
        if (stemFound && rs.Phonemes.Count > 0)
        {
            stemCodes = rs.Phonemes;
            dflags = rs.Flags;
        }
        else
        {
            // A flag-only stem entry still carries the stem's stress flags (e.g. "outfit" pins syllable 1); keep
            // them so adding "-s" doesn't lose the stress pin, then pronounce the stem via the rules.
            if (stemFound) dflags = rs.Flags;
            stemCodes = TranslateWithEndings(buf, start, ref dflags, depth + 1);
        }
        stemCodes.AddRange(endPhonemes);
        return stemCodes;
    }

    // espeak confirm_prefix: returns true if removing a standard suffix from the word at <paramref name="start"/>
    // leaves a word that is in the dictionary (so the word is lexicalized and the prefix should not be stripped, e.g.
    // "outfit" + s). Suppresses prefix detection during the check and restores the buffer afterwards.
    private bool PrefixIsLexicalized(byte[] buf, int start)
    {
        _rules.TranslateRules(buf, start, out int sufType, out _, EspeakTranslator.FlagNoPrefix);
        if (sufType == 0 || (sufType & EspeakRuleCodes.SufxP) != 0)
            return false;
        byte[] save = (byte[])buf.Clone();
        _rules.RemoveEnding(buf, start, sufType);
        string stem = ReadWord(buf, start);
        Array.Copy(save, buf, buf.Length);
        return _lookup.Lookup(stem, out _);
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
}
