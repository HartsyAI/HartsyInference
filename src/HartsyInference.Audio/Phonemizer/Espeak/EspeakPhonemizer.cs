using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Pure-C# espeak-ng phonemizer: turns text into the IPA the espeak-trained TTS models (Piper, MeloTTS, Zonos, StyleTTS2) consume by composing the ported pieces — splitting text into words, dictionary lookup with letter-to-sound rule fallback, stress placement, then IPA conversion. The default <see cref="IPhonemizer"/> backend; handles word-by-word phonemization only, with full clause/number normalization layered on later.</summary>
public sealed class EspeakPhonemizer : IPhonemizer
{
    private readonly EspeakWordLookup _lookup;
    private readonly EspeakTranslator _rules;
    private readonly EspeakStress _stress;
    private readonly EspeakLanguageOptions _options;
    private readonly IReadOnlyList<(string From, string To)> _replacements;
    private readonly EspeakPhonemeTable _phon;
    private readonly EspeakIpaMap _ipa;
    private readonly EspeakVoiceVariant _variant;
    private readonly EspeakPhonemeList? _plist;
    private readonly EspeakPhonemeRenderer? _renderer;

    private EspeakPhonemizer(EspeakWordLookup lookup, EspeakTranslator rules, EspeakStress stress, EspeakLanguageOptions options, IReadOnlyList<(string From, string To)> replacements, EspeakPhonemeTable phon, EspeakIpaMap ipa, EspeakVoiceVariant variant, EspeakPhonemeIndex? index)
    {
        _lookup = lookup;
        _rules = rules;
        _stress = stress;
        _options = options;
        _replacements = replacements;
        _phon = phon;
        _ipa = ipa;
        _variant = variant;
        if (index is not null)
        {
            // Exact allophone + IPA path: run espeak's phoneme-program VM over the clause phoneme list.
            EspeakPhonemeInterpreter interp = new(phon, index) { Reduce = options.Reduce };
            _plist = new EspeakPhonemeList(phon, interp, stress.StressFlags);
            _renderer = new EspeakPhonemeRenderer(interp);
        }
    }

    /// <summary>Resolves the <c>espeak-ng-data</c> directory from <c>ESPEAK_DATA_DIR</c>, then the shared model cache (<c>HARTSYINFERENCE_MODEL_CACHE</c> or <c>~/.cache/hartsyinference/models</c>), and builds a phonemizer; throws with guidance if the data cannot be found.</summary>
    public static EspeakPhonemizer FromCache(string language = "en")
    {
        if (TryFindDataDirectory(out string? dir))
            return FromDataDirectory(dir, language);
        throw new HartsyInferenceException(
            "espeak-ng-data not found. Set ESPEAK_DATA_DIR to an espeak-ng-data directory (e.g. " +
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data), or place the data under the model cache.");
    }

    /// <summary>Finds an <c>espeak-ng-data</c> directory where <see cref="FromCache"/> looks, if there is one.</summary>
    public static bool TryFindDataDirectory([System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? dataDir)
    {
        foreach (string dir in CandidateDataDirs())
        {
            if (Directory.Exists(dir) && File.Exists(Path.Combine(dir, "phontab")))
            {
                dataDir = dir;
                return true;
            }
        }
        dataDir = null;
        return false;
    }

    /// <summary>The model-cache directory <see cref="FromCache"/> reads <c>espeak-ng-data</c> from when neither
    /// <c>ESPEAK_DATA_DIR</c> nor a system install provides it; where the engine installs its copy.</summary>
    public static string CacheDataDirectory
    {
        get
        {
            string? cacheRoot = EngineKnobs.ModelCacheRoot.Value;
            if (string.IsNullOrEmpty(cacheRoot))
                cacheRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "hartsyinference", "models");
            return Path.Combine(cacheRoot, "Hartsy--espeak-ng-data", "espeak-ng-data");
        }
    }

    private static IEnumerable<string> CandidateDataDirs()
    {
        string? env = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (!string.IsNullOrEmpty(env)) yield return env;

        yield return CacheDataDirectory;
        yield return Path.GetDirectoryName(CacheDataDirectory)!;

        // Standard Linux install locations for the espeak-ng package.
        yield return "/usr/lib/x86_64-linux-gnu/espeak-ng-data";
        yield return "/usr/share/espeak-ng-data";
    }

    /// <summary>Builds a phonemizer for English from an <c>espeak-ng-data</c> directory containing <c>en_dict</c> and <c>phontab</c>; multi-language support loads the matching <c>&lt;lang&gt;_dict</c> and phoneme table.</summary>
    public static EspeakPhonemizer FromDataDirectory(string dataDir, string language = "en")
    {
        try
        {
            EspeakVoiceVariant variant = EspeakVoiceVariant.Resolve(dataDir, language);
            EspeakDictFile dict = EspeakDictFile.Load(Path.Combine(dataDir, $"{variant.DictName}_dict"));
            EspeakPhonemeTable phon = EspeakPhonemeTable.Load(Path.Combine(dataDir, "phontab"), variant.PhonemeTable);
            string phonindex = Path.Combine(dataDir, "phonindex");
            EspeakPhonemeIndex? index = File.Exists(phonindex) ? EspeakPhonemeIndex.Load(phonindex) : null;
            // The translator options of tr_languages.c for the language (stress rule and flags, letter groups).
            EspeakLanguageOptions options = EspeakLanguageOptions.For(variant.DictName);
            return new EspeakPhonemizer(
                new EspeakWordLookup(dict, variant.DictCondition),
                new EspeakTranslator(dict, phon, EspeakLetters.For(variant.DictName), variant.DictCondition),
                new EspeakStress(phon, options),
                options,
                dict.Replacements,
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
    public string PhonemizeToIpa(string text, string language) => PhonemizeClause(text, tie: null);

    /// <summary>Phonemizes one clause with <paramref name="tie"/> written between the letters of each multi-letter
    /// phoneme, as espeak-ng does with <c>espeakPHONEMES_TIE</c> (what phonemizer's <c>tie</c> option asks for).</summary>
    public string PhonemizeTied(string text, char tie) => PhonemizeClause(text, tie);

    private string PhonemizeClause(string text, char? tie)
    {
        List<IReadOnlyList<byte>> words = PhonemizeWords(text, out HashSet<int> hyphenated, out HashSet<int> fromDictionary);

        if (_plist is not null && _renderer is not null)
        {
            // Exact path: build the clause phoneme list (allophones + reduction) and render via the phoneme programs.
            List<EspeakPhonemeListEntry> list = _plist.Build(words, hyphenated, fromDictionary);
            return _renderer.Render(list, tie);
        }

        // Fallback: per-word mnemonic -> IPA map (used only when phonindex is unavailable).
        return string.Join(" ", words.Select(c => _ipa.ToIpa((List<byte>)c, _phon)));
    }

    /// <summary>Sentence/clause punctuation that carries prosody (pauses, phrasing), preserved verbatim in the phoneme stream when <c>preservePunctuation</c> is set — TTS front-ends like StyleTTS2 encode these as their own tokens and rely on them for phrasing. Mirrors espeak-ng's <c>preserve_punctuation</c> behaviour.</summary>
    private const string ProsodyPunctuation = ".,!?;:…";

    /// <summary>As <see cref="PhonemizeToIpa(string,string)"/>, but when <paramref name="preservePunctuation"/> is set, sentence/clause punctuation (<see cref="ProsodyPunctuation"/>) is kept as standalone, space-delimited tokens between clauses, each clause phonemized separately so intra-clause stress/reduction is unchanged; matches the reference <c>espeak(preserve_punctuation=True)</c> pipeline speech models were trained on — without it the duration/prosody predictor sees a run-on utterance and slurs across phrase boundaries.</summary>
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
        return PhonemizeWords(text, out _, out _).Select(_phon.Decode).ToList();
    }

    // The words of one clause, each phonemized where it stands: first/last word, case, and what the previous words
    // lead espeak to expect all pick between conditional dictionary entries (TranslateClause + TranslateWord3).
    private List<IReadOnlyList<byte>> PhonemizeWords(string text, out HashSet<int> hyphenated, out HashSet<int> fromDictionary)
    {
        fromDictionary = [];
        List<string> split = SplitNumbers(ReplaceTextWords(SplitWords(ReplaceChars(text))), out HashSet<int> digitByDigit);
        // A hyphen between letters joins two words (FLAG_HYPHEN): each is read on its own, written without a space.
        hyphenated = [];
        HashSet<int> afterHyphen = [];
        HashSet<int> digits = [];
        List<string> kept = new(split.Count);
        for (int i = 0; i < split.Count; i++)
        {
            if (split[i] == Hyphen)
            {
                if (i + 1 < split.Count) afterHyphen.Add(kept.Count); // the next word, once the marker is gone
                continue;
            }
            if (digitByDigit.Contains(i)) digits.Add(kept.Count);
            kept.Add(split[i]);
        }
        split = kept;
        digitByDigit = digits;
        List<IReadOnlyList<byte>> words = new(split.Count);
        ClauseState state = new() { Numbers = new EspeakNumbers(_lookup, _phon, _options) };
        string[] lower = split.Select(w => w.ToLowerInvariant()).ToArray();
        for (int i = 0; i < split.Count; i++)
        {
            bool joined = afterHyphen.Contains(i) && words.Count > 0;
            if (joined) hyphenated.Add(words.Count);
            int first = i, skip = 0;
            state.LastFlags = 0;
            List<byte> codes;
            if (IsSymbol(split[i][0]))
                codes = PhonemizeSymbol(split[i], state);
            else if (IsDigit09(split[i][0]))
                codes = PhonemizeNumber(lower, i, digitByDigit.Contains(i), state, out skip);
            else
                codes = PhonemizeWord(split[i], lower, i, state, out skip);
            i += skip;
            bool last = i == split.Count - 1;

            // TranslateWord2's pre-pause: $brk before a word (not the last), $pause before one inside the clause
            // (and then not again for the next two words), as pause phonemes ahead of the word.
            int prePause = 0;
            if (!joined)
            {
                if ((state.LastFlags & FlagPause1) != 0 && !last) prePause = 1;
                if ((state.LastFlags & FlagPrePause) != 0 && first > 1 && !last && state.PrePauseTimeout == 0)
                {
                    prePause = 4;
                    state.PrePauseTimeout = 3;
                }
            }
            if (prePause > 0)
            {
                List<byte> paused = [];
                for (; prePause > 0; prePause -= prePause > 1 ? 2 : 1)
                    paused.Add(prePause > 1 ? PhonPause : PhonPauseNoLink);
                paused.AddRange(codes);
                codes = paused;
            }
            if (state.PrePauseTimeout > 0) state.PrePauseTimeout--;
            // SFLAG_DICTIONARY: the dictionary spelled this word
            if ((state.LastFlags & FlagFound) != 0 && (state.LastFlags & FlagTextMode) == 0) fromDictionary.Add(words.Count);
            words.Add(codes);
        }
        return words;
    }

    private static bool IsDigit09(char c) => c is >= '0' and <= '9';

    private const string Hyphen = "-";

    // A symbol the clause speaks as a word (% ° & + = @ $ € ...): its dictionary entry, else its _-prefixed name.
    private static bool IsSymbol(char c) => !IsWordChar(c) && !char.IsDigit(c) && !char.IsWhiteSpace(c)
        && (char.IsSymbol(c) || c is '%' or '&' or '@' or '#' or '*' or '/' or '§');

    private List<byte> PhonemizeSymbol(string symbol, ClauseState state)
    {
        state.Advance();
        uint flags;
        List<byte> codes;
        if (_lookup.Lookup(symbol, out EspeakLookupResult r) && r.Phonemes.Count > 0) { codes = r.Phonemes; flags = r.Flags; }
        else if (_lookup.Lookup("_" + symbol, out r) && r.Phonemes.Count > 0) { codes = r.Phonemes; flags = r.Flags; }
        else return [];
        state.LastFlags = flags;
        return _stress.SetWordStress(codes, flags, tonic: -1, control: 0);
    }

    // A $text entry spells the word as other words (French "aujourd'hui" -> "aujourdui"), which are read in its
    // place (LookupDictList's FLAG_TEXTMODE replacement).
    private IEnumerable<string> ReplaceTextWords(IEnumerable<string> words)
    {
        foreach (string w in words)
        {
            if (w != Hyphen && !IsDigit09(w[0]) && !IsSymbol(w[0]) && _lookup.Lookup(w.ToLowerInvariant(), out EspeakLookupResult r)
                && (r.Flags & FlagTextMode) != 0 && r.Phonemes.Count > 0)
            {
                foreach (string part in SplitWords(System.Text.Encoding.UTF8.GetString(r.Phonemes.ToArray())))
                    yield return part;
            }
            else
                yield return w;
        }
    }

    private const uint FlagTextMode = 0x20000000;

    // The dictionary's .replace table (SubstituteChar): a character, or a character with the ones after it, becomes
    // its replacement (Hindi joins a consonant and its nukta into the one precomposed letter its rules expect).
    private string ReplaceChars(string text)
    {
        IReadOnlyList<(string From, string To)> table = _replacements;
        if (table.Count == 0) return text;
        System.Text.StringBuilder sb = new(text.Length);
        string lower = text.ToLowerInvariant();
        int i = 0;
        while (i < text.Length)
        {
            string? to = null;
            int used = 1;
            foreach ((string from, string rep) in table)
            {
                if (string.CompareOrdinal(lower, i, from, 0, from.Length) != 0)
                    continue;
                to = rep;
                used = from.Length;
                break;
            }
            if (to is null) sb.Append(text[i]);
            else sb.Append(char.IsUpper(text[i]) && to.Length > 0 ? char.ToUpperInvariant(to[0]) + to[1..] : to);
            i += used;
        }
        return sb.ToString();
    }

    // TranslateClause's number layout: a run of more than four digits is broken into groups (of three, or the
    // language's lakh pattern), each but the last followed by the thousands separator, so "1234567" reads as the
    // words "1." "234." "567" (Spanish). A run longer than max_digits, or with a leading zero, is read digit by digit.
    private List<string> SplitNumbers(IEnumerable<string> words, out HashSet<int> digitByDigit)
    {
        List<string> result = [];
        digitByDigit = [];
        char sep = _options.ThousandsSep;
        uint breaks = _options.BreakNumbers;
        foreach (string w in words)
        {
            int n = w.Length;
            if (!IsDigit09(w[0]) || n <= 4 || n > 32)
            {
                if (IsDigit09(w[0]) && (n > _options.MaxDigits || (n > 1 && w[0] == '0'))) digitByDigit.Add(result.Count);
                result.Add(w);
                continue;
            }
            bool individual = n > _options.MaxDigits || w[0] == '0';
            System.Text.StringBuilder group = new();
            int nx = n;
            foreach (char c in w)
            {
                group.Append(c);
                nx--;
                if (nx > 0 && (breaks & (1U << nx)) != 0)
                {
                    if (sep != ' ' && sep != '\0') group.Append(sep);
                    if (individual) digitByDigit.Add(result.Count);
                    result.Add(group.ToString());
                    group.Clear();
                    if (!individual)
                    {
                        if ((breaks & (1U << (nx - 1))) != 0) group.Append("00"); // a one-digit group, made three
                        if ((breaks & (1U << (nx - 2))) != 0) group.Append('0'); // a two-digit group (lakh)
                    }
                }
            }
            if (individual) digitByDigit.Add(result.Count);
            result.Add(group.ToString());
        }
        return result;
    }

    // A number word: TranslateNumber, else each digit by its _N entry (TranslateRules' digit path).
    private List<byte> PhonemizeNumber(string[] clause, int index, bool digitByDigit, ClauseState state, out int skip)
    {
        byte[] buf = BuildBuffer(clause, index, out int start);
        List<byte>? codes = state.Numbers!.Translate(buf, start, digitByDigit, out skip);
        if (codes is null)
        {
            codes = [];
            int count = 0;
            foreach (char c in clause[index])
            {
                if (!IsDigit09(c)) continue;
                if (_lookup.Lookup($"_{c}", out EspeakLookupResult r)) codes.AddRange(r.Phonemes);
                if (++count >= 2) { codes.Add(PhonPauseNoLink); count = 0; }
            }
        }
        List<byte> stressed = _stress.SetWordStress(codes, FlagFound, tonic: -1, control: 0);
        state.LastFlags = FlagFound;
        state.Advance();
        return stressed;
    }

    private const uint FlagFound = 0x80000000;
    private const uint FlagPause1 = 0x10000000;
    private const uint FlagPrePause = 0x100;
    private const byte PhonPause = 9;
    private const byte PhonPauseNoLink = 11;

    // expect_* countdowns carried from word to word through a clause (TranslateWord3's tail).
    private sealed class ClauseState
    {
        public int ExpectVerb, ExpectVerbS, ExpectNoun, ExpectPast;
        public EspeakNumbers? Numbers;
        public uint LastFlags;
        public int PrePauseTimeout;
        public uint Flags2;
        public int EndType;

        public EspeakLookupContext Context(EspeakLookupContext word, int endFlags) => word with
        {
            ExpectVerb = ExpectVerb > 0, ExpectVerbS = ExpectVerbS > 0, ExpectNoun = ExpectNoun > 0,
            ExpectPast = ExpectPast > 0, EndFlags = endFlags,
        };

        public void Advance()
        {
            if ((EndType & SufxF) != 0) { ExpectVerb = 2; ExpectVerbS = 2; }
            if ((Flags2 & FlagPastF) != 0) { ExpectPast = 3; ExpectVerb = 0; ExpectNoun = 0; }
            else if ((Flags2 & FlagVerbF) != 0) { ExpectVerb = 2; ExpectVerbS = 0; ExpectNoun = 0; }
            else if ((Flags2 & FlagVerbSF) != 0) { ExpectVerb = 0; ExpectVerbS = 2; ExpectPast = 0; ExpectNoun = 0; }
            else if ((Flags2 & FlagNounF) != 0) { ExpectNoun = 2; ExpectVerb = 0; ExpectVerbS = 0; ExpectPast = 0; }
            if ((Flags2 & FlagVerbExt) == 0)
            {
                if (ExpectVerb > 0) ExpectVerb--;
                if (ExpectVerbS > 0) ExpectVerbS--;
                if (ExpectNoun > 0) ExpectNoun--;
                if (ExpectPast > 0) ExpectPast--;
            }
            Flags2 = 0;
            EndType = 0;
        }

        private const uint FlagVerbF = 0x1, FlagVerbSF = 0x2, FlagNounF = 0x4, FlagPastF = 0x8, FlagVerbExt = 0x100;
        private const int SufxF = 0x2000;
    }

    private List<byte> PhonemizeWord(string word, string[] clause, int index, ClauseState state, out int skip)
    {
        skip = 0;
        string lower = clause[index];
        bool first = index == 0, last = index == clause.Length - 1;
        bool anyLower = word.Any(char.IsLower);
        EspeakLookupContext where = new()
        {
            AtStart = first, AtEnd = last, Sentence = true,
            FirstUpper = char.IsUpper(word[0]), AllUpper = !anyLower && word.Any(char.IsUpper),
            NextWords = index + 1 < clause.Length ? string.Join(' ', clause[(index + 1)..]) + " " : null,
        };
        uint dflags = 0;
        List<byte> stressed;
        bool found = _lookup.Lookup(lower, state.Context(where, 0), out EspeakLookupResult r);
        if (found && r.Phonemes.Count > 0)
        {
            dflags = r.Flags;
            state.Flags2 = r.Flags2;
            skip = r.SkipWords; // a multi-word entry speaks the next words too
            last = index + skip == clause.Length - 1;
            stressed = _stress.SetWordStress(r.Phonemes, dflags, tonic: -1, control: 0);
        }
        else
        {
            if (found) { dflags = r.Flags; state.Flags2 = r.Flags2; } // flag-only entry: keep its flags, use the rules
            byte[] buf = BuildBuffer(clause, index, out int start);
            stressed = TranslateByRules(buf, start, ref dflags, where, state);
        }
        // Flapping (t/d -> ɾ), voicing assimilation, and vowel reduction are applied data-driven by the phoneme-program
        // VM during rendering (EspeakPhonemeList), so no separate flap-t post-process is needed.
        if (last && (dflags & (FlagStressEnd | FlagStressEnd2)) != 0)
            stressed = _stress.ChangeWordStress(stressed, 4); // stressed at the end of a clause
        ApplySpecialAttribute2(stressed, dflags);
        state.LastFlags = dflags;
        state.Advance();
        return stressed;
    }

    // TranslateWord3's rules path for a word the dictionary does not spell: standard prefixes are stripped and the
    // rest looked up again (each confirmed by first removing any suffix), then a standard suffix is stripped, the stem
    // looked up or re-translated (SUFX_Q keeps the first reading, SUFX_M allows another suffix), and the stress set
    // on the stem, or on the whole word when the prefix carries stress or the stem had its own dictionary flags.
    // SUFX_T suffixes are added after the stress. Returns the stressed phonemes.
    private List<byte> TranslateByRules(byte[] buf, int start, ref uint dflags, EspeakLookupContext where, ClauseState state)
    {
        int wflags = 0;
        List<byte> phonemes = _rules.TranslateRules(buf, start, out int endType, out List<byte> endPhonemes, wflags);
        state.EndType = endType;
        List<byte> prefixPhonemes = [];
        bool prefixFlags = false;
        bool confirmPrefix = true;

        for (int loop = 0; loop < 50 && (endType & EspeakRuleCodes.SufxP) != 0; loop++)
        {
            if (confirmPrefix && (endType & SufxB) == 0)
            {
                // remove any standard suffix and confirm that the prefix is still recognised
                List<byte> phonemes2 = _rules.TranslateRules(buf, start, out int end2, out List<byte> endPhonemes2, wflags | EspeakTranslator.FlagNoPrefix);
                if (end2 != 0)
                {
                    byte[] copy = (byte[])buf.Clone();
                    _rules.RemoveEnding(buf, start, end2);
                    phonemes = _rules.TranslateRules(buf, start, out endType, out endPhonemes, wflags);
                    Array.Copy(copy, buf, buf.Length);
                    if ((endType & EspeakRuleCodes.SufxP) == 0)
                    {
                        // without the suffix the prefix is no longer recognised: keep the suffix, not the prefix
                        endType = end2;
                        phonemes = phonemes2;
                        endPhonemes = endPhonemes2;
                    }
                    confirmPrefix = false;
                    continue;
                }
            }

            int nChars = endType & 0xf;
            for (int c = 0; c < nChars && buf[start] != (byte)' '; c++)
                start += EspeakUtf8.Read(buf, start, out _);
            buf[start - 1] = (byte)' ';
            confirmPrefix = true;
            wflags |= FlagPrefixRemoved;
            prefixPhonemes.AddRange(endPhonemes);
            endPhonemes = [];
            endType = 0;

            bool found = _lookup.Lookup(ReadWord(buf, start), state.Context(where, EspeakRuleCodes.SufxP), out EspeakLookupResult rp);
            if (rp.Flags != 0 || rp.Flags2 != 0)
            {
                if (dflags == 0) { dflags = rp.Flags; state.Flags2 = rp.Flags2; }
                else prefixFlags = true;
            }
            if (found && rp.Phonemes.Count > 0)
                phonemes = new List<byte>(rp.Phonemes);
            else
                phonemes = _rules.TranslateRules(buf, start, out endType, out endPhonemes, wflags & FlagPrefixRemoved);
        }

        if (endType != 0 && (endType & EspeakRuleCodes.SufxP) == 0)
        {
            int endType1 = endType;
            List<byte> phonemes2 = phonemes;
            int endFlags = _rules.RemoveEnding(buf, start, endType);
            bool more = true;
            while (more)
            {
                more = false;
                bool found = _lookup.Lookup(ReadWord(buf, start), state.Context(where, endFlags), out EspeakLookupResult rs);
                if (dflags == 0 && (rs.Flags != 0 || rs.Flags2 != 0)) { dflags = rs.Flags; state.Flags2 = rs.Flags2; }
                if (found && rs.Phonemes.Count > 0)
                {
                    phonemes = new List<byte>(rs.Phonemes);
                    continue;
                }
                if ((endType & SufxQ) != 0)
                {
                    phonemes = phonemes2; // don't re-translate: keep the first reading
                    continue;
                }
                if ((endFlags & FlagSufx) != 0) wflags |= EspeakTranslator.FlagSuffixRemoved;
                if ((endType & SufxA) != 0) wflags |= EspeakTranslator.FlagSuffixVowel;
                if ((endType & SufxM) != 0)
                {
                    // allow more suffixes before this one
                    phonemes = _rules.TranslateRules(buf, start, out endType, out List<byte> more2, wflags);
                    more2.AddRange(endPhonemes);
                    endPhonemes = more2;
                    if (endType != 0 && (endType & EspeakRuleCodes.SufxP) == 0)
                    {
                        endFlags = _rules.RemoveEnding(buf, start, endType);
                        more = true;
                    }
                }
                else
                {
                    phonemes = _rules.TranslateRules(buf, start, wflags | EspeakTranslator.FlagKeepEndings);
                    endType = 0;
                }
            }

            if ((endType1 & SufxT) == 0)
            {
                // the default: add the suffix, then find the word's stress
                phonemes = new List<byte>(phonemes);
                phonemes.AddRange(endPhonemes);
                endPhonemes = [];
            }
        }

        bool prefixStress = prefixPhonemes.Contains(PhonStressP) || prefixPhonemes.Contains(PhonStressP2);
        List<byte> word;
        if (prefixFlags || prefixStress)
        {
            // the stress position covers the whole word, prefix included
            word = new List<byte>(prefixPhonemes);
            word.AddRange(phonemes);
            word = _stress.SetWordStress(word, dflags, tonic: -1, control: 0);
        }
        else
        {
            word = new List<byte>(prefixPhonemes);
            word.AddRange(_stress.SetWordStress(phonemes, dflags, tonic: -1, control: endPhonemes.Count > 0 ? 2 : 0));
        }
        word.AddRange(endPhonemes); // a SUFX_T suffix, after the stress is set
        return word;
    }

    private const int SufxQ = 0x4000;
    private const int SufxT = 0x10000;
    private const int SufxB = 0x20000;
    private const int SufxA = 0x40000;
    private const int SufxM = 0x80000;
    private const int FlagSufx = 0x04;
    private const int FlagPrefixRemoved = 0x800000;
    private const byte PhonStressP = 6;
    private const byte PhonStressP2 = 7;

    // $alt opens and $alt2 closes the e or o of the stressed syllable (Italian, Portuguese: LOPT_ALT & 2).
    private void ApplySpecialAttribute2(List<byte> phonemes, uint dflags)
    {
        if ((_options.Alt & 2) == 0 || (dflags & (FlagAltTrans | FlagAlt2Trans)) == 0) return;
        int ix = phonemes.IndexOf(PhonStressP);
        if (ix < 0 || ix + 1 >= phonemes.Count) return;
        bool close = (dflags & FlagAlt2Trans) != 0;
        int from1 = Code(close ? 'E' : 'e'), to1 = Code(close ? 'e' : 'E');
        int from2 = Code(close ? 'O' : 'o'), to2 = Code(close ? 'o' : 'O');
        if (phonemes[ix + 1] == from1) phonemes[ix + 1] = (byte)to1;
        else if (phonemes[ix + 1] == from2) phonemes[ix + 1] = (byte)to2;
    }

    private int Code(char mnemonic) => _phon.CodeForMnemonic(mnemonic);

    private const uint FlagAltTrans = 0x8000;
    private const uint FlagAlt2Trans = 0x10000;

    private const uint FlagStressEnd = 0x200;
    private const uint FlagStressEnd2 = 0x400;

    private const int WordStart = 2;

    // The clause as espeak's TranslateClause lays it out: UTF-8 words separated by single spaces, two spaces before
    // the first, so a rule's context can see into the neighbouring words; the rules were compiled against UTF-8 text,
    // and the interpreter walks multi-byte letters with EspeakUtf8. Returns the buffer and the word's start offset.
    private static byte[] BuildBuffer(string[] clause, int index, out int start)
    {
        byte[][] utf8 = clause.Select(System.Text.Encoding.UTF8.GetBytes).ToArray();
        byte[] buf = new byte[WordStart + utf8.Sum(w => w.Length + 1) + 200];
        Array.Fill(buf, (byte)' ');
        int pos = WordStart;
        start = WordStart;
        for (int i = 0; i < utf8.Length; i++)
        {
            if (i == index) start = pos;
            Array.Copy(utf8[i], 0, buf, pos, utf8[i].Length);
            pos += utf8[i].Length + 1;
        }
        buf[^1] = 0;
        return buf;
    }

    private static string ReadWord(byte[] buf, int start)
    {
        int end = start;
        while (end < buf.Length && buf[end] != (byte)' ' && buf[end] != 0)
            end++;
        return System.Text.Encoding.UTF8.GetString(buf, start, end - start);
    }

    // An apostrophe belongs to a word only between a letter or digit and a letter ("don't", "l'homme"); before or
    // after a word it is a quote mark (TranslateClause).
    private static bool IsWordCharAt(string text, int i)
    {
        char c = text[i];
        if (c is '\'' or '\u2019')
            return i > 0 && char.IsLetterOrDigit(text[i - 1]) && i + 1 < text.Length && char.IsLetter(text[i + 1]);
        return IsWordChar(c);
    }

    // Letters, apostrophes and combining marks (an Indic vowel sign or virama is a mark, not a letter, but part of
    // the word it attaches to).
    private static bool IsWordChar(char c) => char.IsLetter(c) || c == '\''
        || char.GetUnicodeCategory(c) is System.Globalization.UnicodeCategory.NonSpacingMark
            or System.Globalization.UnicodeCategory.SpacingCombiningMark;

    // Word split for the word-by-word path: runs of letters/apostrophes are words, and runs of digits are numbers
    // (any script's decimal digits, read as 0-9 as espeak does); everything else is a separator, and a number
    // touching a letter is its own word.
    private static IEnumerable<string> SplitWords(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            if (IsWordCharAt(text, i))
            {
                int start = i;
                while (i < text.Length && IsWordCharAt(text, i))
                    i++;
                yield return text[start..i].Replace('\u2019', '\'');
            }
            else if (text[i] == '-' && i > 0 && IsWordChar(text[i - 1]) && i + 1 < text.Length && char.IsLetter(text[i + 1]))
            {
                yield return Hyphen;
                i++;
            }
            else if (IsSymbol(text[i]))
            {
                yield return text[i].ToString();
                i++;
            }
            else if (char.IsDigit(text[i]))
            {
                System.Text.StringBuilder digits = new();
                while (i < text.Length && char.IsDigit(text[i]))
                    digits.Append((char)('0' + (int)char.GetNumericValue(text[i++])));
                yield return digits.ToString();
            }
            else
                i++;
        }
    }
}
