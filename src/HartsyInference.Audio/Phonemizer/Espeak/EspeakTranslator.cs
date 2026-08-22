namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Applies a language's letter-to-sound rules to a word, ported from espeak-ng's <c>TranslateRules</c> + <c>MatchRule</c> (dictionary.c): walks the word buffer left to right, picks the best-scoring rule chain for each letter (single-letter, two-letter, and offset groups), and emits the matched phoneme code bytes. This is the fallback path for words not found in the dictionary word list; the dictionary lookup and stress placement layer on top of it.</summary>
internal sealed class EspeakTranslator
{
    private readonly EspeakDictFile _dict;
    private readonly EspeakPhonemeTable _phon;
    private readonly EspeakLetters _letters;
    private readonly byte[] _data;

    // Language options exercised by the rule matcher; defaults match English (base Latin).
    private int _dictCondition;
    private bool _toneNumbers;
    private int _loptSuffix;

    // Per-word mutable state read by the rules.
    private int _wordVowelCount;
    private int _wordStressedCount;
    private bool _expectVerb;

    public EspeakTranslator(EspeakDictFile dict, EspeakPhonemeTable phonemeTable, EspeakLetters letters, int dictCondition = 0)
    {
        _dict = dict;
        _phon = phonemeTable;
        _letters = letters;
        _data = dict.Data;

        // Variant rules (e.g. en-us) enable extra dictionary conditions; default English (en-gb) sets none.
        _dictCondition = dictCondition;
        _toneNumbers = false;
        _loptSuffix = 0;
        _expectVerb = false;
    }

    /// <summary>Convenience for tests/diagnostics: lowercases <paramref name="word"/>, frames it with spaces, applies the letter-to-sound rules, and returns the decoded phoneme mnemonics (no stress placement or dictionary lookup). The production phonemizer drives <see cref="TranslateRules"/> directly with its own buffer.</summary>
    public string TranslateWordToMnemonics(string word)
    {
        byte[] buf = BuildWordBuffer(word);
        List<byte> codes = TranslateRules(buf, WordBufferStart);
        return _phon.Decode(codes);
    }

    // espeak translates within a large clause buffer where the word is surrounded by spaces; the rule context scans
    // rely on hitting a space to terminate. Mirror that: a space-filled buffer with the word at index 2.
    private const int WordBufferStart = 2;

    private static byte[] BuildWordBuffer(string word)
    {
        string lower = word.ToLowerInvariant();
        byte[] buf = new byte[EspeakRuleCodes.WordBytes];
        Array.Fill(buf, (byte)' ');
        for (int i = 0; i < lower.Length && WordBufferStart + i < buf.Length - 1; i++)
            buf[WordBufferStart + i] = (byte)lower[i];
        buf[^1] = 0;
        return buf;
    }

    /// <summary>Translates the word in <paramref name="buf"/> starting at <paramref name="start"/> (the byte at <c>start-1</c> must be a space) using the letter-to-sound rules, returning the matched phoneme code bytes; suffix/ending re-translation and dictionary-list lookup are handled by higher layers.</summary>
    public List<byte> TranslateRules(byte[] buf, int start, int wordFlags = 0)
        => TranslateRules(buf, start, out _, out _, wordFlags);

    /// <summary>As <see cref="TranslateRules(byte[],int,int)"/> but also reports a matched suffix ending: when a dictionary suffix rule fires, <paramref name="endType"/> is non-zero, <paramref name="endPhonemes"/> holds the suffix's phoneme codes, and the returned list is the stem translated so far, for the caller to strip the suffix (<see cref="RemoveEnding"/>), re-translate the stem, and append the suffix phonemes.</summary>
    public List<byte> TranslateRules(byte[] buf, int start, out int endType, out List<byte> endPhonemes, int wordFlags = 0)
    {
        endType = 0;
        endPhonemes = new List<byte>();
        List<byte> phonemes = new(64);
        _wordVowelCount = 0;
        _wordStressedCount = 0;

        int p = start;
        while (p < buf.Length && buf[p] != (byte)' ' && buf[p] != 0)
        {
            int wcBytes = EspeakUtf8.Read(buf, p, out int wc);
            byte c = buf[p];

            EspeakMatchRecord match1 = Empty();
            bool found = false;

            int gi = wc - _letters.LetterBitsOffset;
            if (gi >= 0 && gi < 128 && _dict.Groups3[gi] >= 0)
            {
                MatchRule(buf, ref p, start, wcBytes, _dict.Groups3[gi], out match1, wordFlags);
                found = true;
            }

            int n = _dict.Groups2Count[c];
            if (!found && n > 0)
            {
                byte c2 = buf[p + 1];
                int c12 = c + (c2 << 8);
                int g1 = _dict.Groups2Start[c];
                for (int g = g1; g < g1 + n; g++)
                {
                    if (_dict.Groups2Name[g] == c12)
                    {
                        found = true;
                        int p2 = p;
                        MatchRule(buf, ref p2, start, 2, _dict.Groups2[g], out EspeakMatchRecord match2, wordFlags);
                        if (match2.Points > 0)
                            match2.Points += 35; // account for 2 letters matching

                        MatchRule(buf, ref p, start, 1, _dict.Groups1[c], out match1, wordFlags);

                        if (match2.Points >= match1.Points)
                        {
                            match1 = match2;
                            p = p2;
                        }
                    }
                }
            }

            if (!found)
            {
                if (_dict.Groups1[c] >= 0)
                    MatchRule(buf, ref p, start, 1, _dict.Groups1[c], out match1, wordFlags);
                else
                    MatchRule(buf, ref p, start, 0, _dict.Groups1[0], out match1, wordFlags);

                if (match1.Points == 0)
                    p += wcBytes - 1; // unrecognised letter: advance and continue (letter spelling handled later)
            }

            if (match1.Points > 0)
            {
                int et = match1.EndType & ~EspeakRuleCodes.SufxUnpron;
                // A suffix ending, or (unless suppressed) a standard prefix, was found: return the phonemes so far plus
                // the affix phonemes/type so the caller can strip the affix and re-translate the stem.
                bool isPrefix = (et & EspeakRuleCodes.SufxP) != 0;
                if (et != 0 && match1.PhonemesOffset >= 0 && (!isPrefix || (wordFlags & FlagNoPrefix) == 0))
                {
                    endType = et;
                    endPhonemes = ReadCodes(match1.PhonemesOffset);
                    return phonemes;
                }

                if (match1.PhonemesOffset >= 0)
                {
                    if (match1.DelFwd >= 0)
                        buf[match1.DelFwd] = EspeakRuleCodes.ReplacedE;
                    AppendPhonemes(phonemes, match1.PhonemesOffset);
                }
            }
        }

        return phonemes;
    }

    private List<byte> ReadCodes(int offset)
    {
        List<byte> codes = new(16);
        int i = offset;
        while (_data[i] != 0)
            codes.Add(_data[i++]);
        return codes;
    }

    /// <summary>Strips a standard suffix from the word in <paramref name="buf"/> (suffix length and flags come from <paramref name="endType"/>), reversing the English spelling changes the suffix caused (y&lt;-i, silent-e restoration) so the stem can be re-translated. Ported from <c>RemoveEnding</c> (English path); returns the end-flags describing the removed suffix.</summary>
    public int RemoveEnding(byte[] buf, int start, int endType)
    {
        int wordEnd = start;
        while (buf[wordEnd] != (byte)' ')
        {
            if (buf[wordEnd] == EspeakRuleCodes.ReplacedE)
                buf[wordEnd] = (byte)'e';
            wordEnd++;
        }

        int lenEnding = endType & 0x3f;
        Span<char> ending = stackalloc char[16];
        int el = 0;
        int from = wordEnd - lenEnding;
        for (int i = 0; i < lenEnding && i < 16; i++)
        {
            ending[el++] = (char)buf[from + i];
            buf[from + i] = (byte)' ';
        }
        int stemLast = from - 1; // last character of the stem

        int endFlags = (endType & 0xfff0) | FlagSufx;

        if ((endType & SufxI) != 0 && buf[stemLast] == (byte)'i')
            buf[stemLast] = (byte)'y';

        if ((endType & SufxE) != 0)
        {
            bool addE = false;
            // vowel(incl y) + hard consonant -> add 'e', unless the stem ends in an "ion" exception.
            if (_letters.IsLetter(buf[stemLast - 1], EspeakRuleCodes.LetterGpVowel2) && _letters.IsLetter(buf[stemLast], 1))
            {
                if (!StemEndsWith(buf, stemLast, "ion"))
                    addE = true;
            }
            else if (StemEndsWithAny(buf, stemLast, AddEAdditions))
                addE = true;

            if (addE)
            {
                buf[stemLast + 1] = (byte)'e';
                endFlags |= FlagSufxEAdded;
            }
        }

        string endStr = new(ending[..el]);
        if (endStr == "s" || endStr == "es")
            endFlags |= FlagSufxS;
        if (el > 0 && ending[0] == '\'')
            endFlags &= ~FlagSufx;

        return endFlags;
    }

    private static readonly string[] AddEAdditions = ["c", "rs", "ir", "ur", "ath", "ns", "u", "spong", "rang", "larg"];

    private static bool StemEndsWith(byte[] buf, int stemLast, string s)
    {
        int n = s.Length;
        for (int i = 0; i < n; i++)
            if (buf[stemLast - n + 1 + i] != (byte)s[i]) return false;
        return true;
    }

    private static bool StemEndsWithAny(byte[] buf, int stemLast, string[] options)
    {
        foreach (string s in options)
            if (StemEndsWith(buf, stemLast, s)) return true;
        return false;
    }

    // Port of AppendPhonemes: copy code bytes from the dictionary data (until 0) into the output, tracking the
    // word's vowel and stressable-vowel counts which the rules consult.
    private void AppendPhonemes(List<byte> output, int offset)
    {
        bool unstressMark = false;
        int i = offset;
        while (_data[i] != 0)
        {
            byte code = _data[i++];
            output.Add(code);
            if (!_phon.TryGet(code, out EspeakPhoneme ph))
                continue;
            if (ph.Type == EspeakPhoneme.TypeStress)
            {
                if (ph.StdLength < 4)
                    unstressMark = true;
            }
            else if (ph.Type == EspeakPhoneme.TypeVowel)
            {
                if ((ph.PhFlags & EspeakPhoneme.FlagUnstressed) == 0 && !unstressMark)
                    _wordStressedCount++;
                unstressMark = false;
                _wordVowelCount++;
            }
        }
    }

    // Faithful port of MatchRule (dictionary.c). buf is the word buffer; wordPos is advanced past consumed letters.
    private void MatchRule(byte[] buf, ref int wordPos, int wordStart, int groupLength, int ruleStart, out EspeakMatchRecord matchOut, int wordFlags)
    {
        matchOut = Empty();
        if (ruleStart < 0)
        {
            matchOut.Points = 0;
            wordPos++;
            return;
        }

        int totalConsumed = 0;
        int commonPhonemes = -1;
        int rule = ruleStart;

        EspeakMatchRecord best = Empty();
        best.Points = 0;

        while (_data[rule] != EspeakRuleCodes.RuleGroupEnd)
        {
            bool unpronIgnore = (wordFlags & FlagUnpronTest) != 0;
            int matchType = 0;
            int consumed = 0;
            int distanceRight = -6;
            int distanceLeft = -2;
            bool checkAtStart = false;

            EspeakMatchRecord match = Empty();
            match.Points = 1;

            int prePtr = wordPos;
            int postPtr = wordPos + groupLength;

            int failed = 0;
            while (failed == 0)
            {
                byte rb = _data[rule++];

                if (rb <= EspeakRuleCodes.RuleLineNum)
                {
                    switch (rb)
                    {
                        case 0: // no phoneme string: use previous common rule
                            if (commonPhonemes != -1)
                            {
                                int cp = commonPhonemes;
                                byte rb2;
                                while ((rb2 = _data[cp++]) != 0 && rb2 != EspeakRuleCodes.RulePhonemes)
                                {
                                    if (rb2 == EspeakRuleCodes.RuleCondition) cp++;
                                    if (rb2 == EspeakRuleCodes.RuleLineNum) cp += 2;
                                }
                                match.PhonemesOffset = cp;
                            }
                            else
                                match.PhonemesOffset = -1;
                            rule--; // still pointing at the 0
                            failed = 2;
                            break;
                        case EspeakRuleCodes.RulePreAtStart:
                            checkAtStart = true;
                            unpronIgnore = false;
                            matchType = EspeakRuleCodes.RulePre;
                            break;
                        case EspeakRuleCodes.RulePre:
                            matchType = EspeakRuleCodes.RulePre;
                            if ((wordFlags & FlagUnpronTest) != 0)
                                failed = 1;
                            break;
                        case EspeakRuleCodes.RulePost:
                            matchType = EspeakRuleCodes.RulePost;
                            break;
                        case EspeakRuleCodes.RulePhonemes:
                            match.PhonemesOffset = rule;
                            failed = 2;
                            break;
                        case EspeakRuleCodes.RulePhCommon:
                            commonPhonemes = rule;
                            break;
                        case EspeakRuleCodes.RuleCondition:
                            byte conditionNum = _data[rule++];
                            if (conditionNum >= 32)
                            {
                                if ((_dictCondition & (1 << (conditionNum - 32))) != 0)
                                    failed = 1;
                            }
                            else if ((_dictCondition & (1 << conditionNum)) == 0)
                                failed = 1;
                            if (failed == 0)
                                match.Points++;
                            break;
                        case EspeakRuleCodes.RuleLineNum:
                            rule += 2;
                            break;
                    }
                    continue;
                }

                int addPoints = 0;

                if (matchType == 0)
                {
                    // match and consume this letter
                    byte letter = buf[postPtr++];
                    if (letter == rb || (letter == EspeakRuleCodes.ReplacedE && rb == (byte)'e'))
                    {
                        if ((letter & 0xc0) != 0x80)
                            addPoints = 21;
                        consumed++;
                    }
                    else
                        failed = 1;
                }
                else if (matchType == EspeakRuleCodes.RulePost)
                {
                    distanceRight += 6;
                    if (distanceRight > 18) distanceRight = 19;
                    int letterXbytes = EspeakUtf8.Read(buf, postPtr, out int letterW) - 1;
                    byte letter = buf[postPtr++];
                    failed = MatchAfter(buf, rb, ref rule, ref postPtr, ref distanceRight, letterXbytes, letterW, letter, wordPos, wordStart, groupLength, consumed, ref match, ref addPoints, ref distanceLeft);
                }
                else // RULE_PRE
                {
                    distanceLeft += 2;
                    if (distanceLeft > 18) distanceLeft = 19;
                    EspeakUtf8.Read(buf, prePtr, out int lastLetterW);
                    prePtr--;
                    int letterXbytes;
                    int letterW;
                    byte letter;
                    if (prePtr < 0)
                    {
                        // Before the word's leading boundary. espeak translates within a clause buffer that is
                        // space-padded on both sides, so a pre-context that scans past the word start reads a
                        // space (RULE_SPACE) and letter-requiring rules fail there. Our per-word buffer only has
                        // WordStart leading spaces, so mirror the space padding explicitly instead of indexing
                        // out of bounds (words like "Americans" scan a letter-group further left than the pad).
                        letterW = EspeakRuleCodes.RuleSpace;
                        letterXbytes = 0;
                        letter = (byte)' ';
                    }
                    else
                    {
                        letterXbytes = EspeakUtf8.Read2(buf, prePtr, true, out letterW) - 1;
                        letter = buf[prePtr];
                    }
                    failed = MatchBefore(buf, rb, ref rule, ref prePtr, ref distanceLeft, distanceRight, letterXbytes, letterW, lastLetterW, letter, wordFlags, ref addPoints);
                }

                if (failed == 0)
                    match.Points += addPoints;
            }

            if (failed == 2 && !unpronIgnore)
            {
                if (!checkAtStart || prePtr <= 0 || buf[prePtr - 1] == (byte)' ')
                {
                    if (checkAtStart)
                        match.Points += 4;
                    if (match.Points >= best.Points)
                    {
                        best = match;
                        totalConsumed = consumed;
                    }
                }
            }

            // skip phoneme string to reach the next template
            while (_data[rule] != 0) rule++;
            rule++;
        }

        totalConsumed += groupLength;
        if (totalConsumed == 0)
            totalConsumed = 1;
        wordPos += totalConsumed;

        if (best.Points == 0)
            best.PhonemesOffset = -1;
        matchOut = best;
    }

    // The match_type==RULE_POST branch of MatchRule (forward context). Returns the new 'failed' value.
    private int MatchAfter(byte[] buf, byte rb, ref int rule, ref int postPtr, ref int distanceRight, int letterXbytes, int letterW, byte letter, int wordPos, int wordStart, int groupLength, int consumed, ref EspeakMatchRecord match, ref int addPoints, ref int distanceLeft)
    {
        switch (rb)
        {
            case EspeakRuleCodes.RuleLetterGp:
                {
                    int group = LetterGroupNo(rule++);
                    if (_letters.IsLetter(letterW, group))
                    {
                        int lgPts = group == 2 ? 19 : 20;
                        addPoints = lgPts - distanceRight;
                        postPtr += letterXbytes;
                    }
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleLetterGp2:
                {
                    int group = LetterGroupNo(rule++);
                    int nBytes = IsLetterGroup(buf, postPtr - 1, group, false);
                    if (nBytes >= 0)
                    {
                        addPoints = 20 - distanceRight;
                        postPtr += nBytes - 1;
                    }
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleNotVowel:
                if (_letters.IsLetter(letterW, 0) || (letterW == ' ' && false))
                    return 1;
                addPoints = 20 - distanceRight;
                postPtr += letterXbytes;
                break;
            case EspeakRuleCodes.RuleDigit:
                if (IsDigit(letterW))
                {
                    addPoints = 20 - distanceRight;
                    postPtr += letterXbytes;
                }
                else if (_toneNumbers)
                {
                    addPoints = 20 - distanceRight;
                    postPtr--;
                }
                else return 1;
                break;
            case EspeakRuleCodes.RuleNonAlpha:
                if (!IsAlpha(letterW))
                {
                    addPoints = 21 - distanceRight;
                    postPtr += letterXbytes;
                }
                else return 1;
                break;
            case EspeakRuleCodes.RuleDouble:
                // RULE_DOUBLE compares against the previous letter; handled via last_letter in pre path. Forward use is rare.
                return 1;
            case (byte)'-':
                if (letter == (byte)'-')
                    addPoints = 22 - distanceRight;
                else return 1;
                break;
            case EspeakRuleCodes.RuleSyllable:
                {
                    int p = postPtr + letterXbytes;
                    int vowelCount = 0;
                    int syllableCount = 1;
                    while (_data[rule] == EspeakRuleCodes.RuleSyllable) { rule++; syllableCount++; }
                    int lw = letterW;
                    bool vowel = false;
                    while (lw != EspeakRuleCodes.RuleSpace)
                    {
                        if (!vowel && _letters.IsLetter(lw, EspeakRuleCodes.LetterGpVowel2)) vowelCount++;
                        vowel = _letters.IsLetter(lw, EspeakRuleCodes.LetterGpVowel2);
                        p += EspeakUtf8.Read(buf, p, out lw);
                    }
                    if (syllableCount <= vowelCount)
                        addPoints = 18 + syllableCount - distanceRight;
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleNoVowels:
                {
                    int p = postPtr + letterXbytes;
                    int lw = letterW;
                    while (lw != EspeakRuleCodes.RuleSpace)
                    {
                        if (_letters.IsLetter(lw, EspeakRuleCodes.LetterGpVowel2)) return 1;
                        p += EspeakUtf8.Read(buf, p, out lw);
                    }
                    addPoints = 19 - distanceRight;
                    break;
                }
            case EspeakRuleCodes.RuleIncScore:
                addPoints = 20;
                break;
            case EspeakRuleCodes.RuleDecScore:
                addPoints = -20;
                break;
            case EspeakRuleCodes.RuleDelFwd:
                for (int p = wordPos + groupLength; p < postPtr; p++)
                {
                    if (buf[p] == (byte)'e') { match.DelFwd = p; break; }
                }
                break;
            case EspeakRuleCodes.RuleEnding:
                {
                    int endType = (_data[rule] << 16) + ((_data[rule + 1] & 0x7f) << 8) + (_data[rule + 2] & 0x7f);
                    if (_wordVowelCount == 0 && (endType & EspeakRuleCodes.SufxP) == 0 && (_loptSuffix & 1) != 0)
                        return 1;
                    match.EndType = endType;
                    rule += 3;
                    break;
                }
            case EspeakRuleCodes.RuleNoSuffix:
                addPoints = 1;
                break;
            default:
                if (letter == rb)
                {
                    if ((letter & 0xc0) != 0x80)
                        addPoints = 21 - distanceRight;
                }
                else return 1;
                break;
        }
        _ = distanceLeft;
        _ = wordStart;
        _ = consumed;
        return 0;
    }

    // The match_type==RULE_PRE branch of MatchRule (backward context). Returns the new 'failed' value.
    private int MatchBefore(byte[] buf, byte rb, ref int rule, ref int prePtr, ref int distanceLeft, int distanceRight, int letterXbytes, int letterW, int lastLetterW, byte letter, int wordFlags, ref int addPoints)
    {
        switch (rb)
        {
            case EspeakRuleCodes.RuleLetterGp:
                {
                    int group = LetterGroupNo(rule++);
                    if (_letters.IsLetter(letterW, group))
                    {
                        int lgPts = group == 2 ? 19 : 20;
                        addPoints = lgPts - distanceLeft;
                        prePtr -= letterXbytes;
                    }
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleLetterGp2:
                {
                    int group = LetterGroupNo(rule++);
                    int nBytes = IsLetterGroup(buf, prePtr, group, true);
                    if (nBytes >= 0)
                    {
                        addPoints = 20 - distanceRight;
                        prePtr -= nBytes - 1;
                    }
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleNotVowel:
                if (!_letters.IsLetter(letterW, 0))
                {
                    addPoints = 20 - distanceLeft;
                    prePtr -= letterXbytes;
                }
                else return 1;
                break;
            case EspeakRuleCodes.RuleDouble:
                if (letterW == lastLetterW)
                    addPoints = 21 - distanceLeft;
                else return 1;
                break;
            case EspeakRuleCodes.RuleDigit:
                if (IsDigit(letterW))
                {
                    addPoints = 21 - distanceLeft;
                    prePtr -= letterXbytes;
                }
                else return 1;
                break;
            case EspeakRuleCodes.RuleNonAlpha:
                if (!IsAlpha(letterW))
                {
                    addPoints = 21 - distanceRight;
                    prePtr -= letterXbytes;
                }
                else return 1;
                break;
            case EspeakRuleCodes.RuleSyllable:
                {
                    int syllableCount = 1;
                    while (_data[rule] == EspeakRuleCodes.RuleSyllable) { rule++; syllableCount++; }
                    if (syllableCount <= _wordVowelCount)
                        addPoints = 18 + syllableCount - distanceLeft;
                    else return 1;
                    break;
                }
            case EspeakRuleCodes.RuleStressed:
                if (_wordStressedCount > 0)
                    addPoints = 19;
                else return 1;
                break;
            case EspeakRuleCodes.RuleNoVowels:
                {
                    int p = prePtr - letterXbytes - 1;
                    int lw = letterW;
                    while (lw != EspeakRuleCodes.RuleSpace)
                    {
                        if (_letters.IsLetter(lw, EspeakRuleCodes.LetterGpVowel2)) return 1;
                        p -= EspeakUtf8.Read2(buf, p, true, out lw);
                    }
                    addPoints = 3;
                    break;
                }
            case EspeakRuleCodes.RuleIfVerb:
                if (_expectVerb)
                    addPoints = 1;
                else return 1;
                break;
            case EspeakRuleCodes.RuleCapital:
                if ((wordFlags & FlagFirstUpper) != 0)
                    addPoints = 1;
                else return 1;
                break;
            case (byte)'.':
                {
                    int p = prePtr;
                    while (buf[p] != (byte)' ')
                    {
                        if (buf[p] == (byte)'.') { addPoints = 50; break; }
                        p--;
                    }
                    if (buf[p] == (byte)' ') return 1;
                    break;
                }
            case (byte)'-':
                if (letter == (byte)'-')
                    addPoints = 22 - distanceRight;
                else return 1;
                break;
            default:
                if (letter == rb)
                {
                    if (letter == EspeakRuleCodes.RuleSpace)
                        addPoints = 4;
                    else if ((letter & 0xc0) != 0x80)
                        addPoints = 21 - distanceLeft;
                }
                else return 1;
                break;
        }
        return 0;
    }

    // Port of IsLetterGroup: match buf at wordIdx against the utf-8 string list for the group. Returns matched length
    // or -1. For pre-rules (backwards) the group strings are matched ending at wordIdx.
    private int IsLetterGroup(byte[] buf, int wordIdx, int group, bool pre)
    {
        if (group < 0 || group >= _dict.LetterGroups.Length || _dict.LetterGroups[group] < 0)
            return -1;
        int p = _dict.LetterGroups[group];
        while (_data[p] != EspeakRuleCodes.RuleGroupEnd)
        {
            int len = StrLen(p);
            int w = pre ? wordIdx - len + 1 : wordIdx;
            if (_data[p] == (byte)'~')
                return 0;
            int pp = p;
            while (_data[pp] == buf[w] && buf[w] != 0) { w++; pp++; }
            if (_data[pp] == 0)
                return pre ? len : w - wordIdx;
            while (_data[pp++] != 0) { }
            p = pp;
        }
        return -1;
    }

    private int LetterGroupNo(int rule)
    {
        int groupNo = _data[rule] - 'A';
        if (groupNo < 0) groupNo += 256;
        return groupNo;
    }

    private int StrLen(int p)
    {
        int s = p;
        while (_data[p] != 0) p++;
        return p - s;
    }

    private static bool IsAlpha(int c)
    {
        if (c >= 0x300 && c <= 0x36f) return true;
        if (c < 0 || c > 0x10ffff) return false;
        return System.Text.Rune.IsValid(c) && System.Text.Rune.IsLetter(new System.Text.Rune(c));
    }

    private static bool IsDigit(int c) => c >= '0' && c <= '9';

    private static EspeakMatchRecord Empty() => new() { Points = 1, PhonemesOffset = -1, EndType = 0, DelFwd = -1 };

    // word_flags bits used by the matcher.
    private const int FlagUnpronTest = unchecked((int)0x80000000);
    public const int FlagNoPrefix = 0x40000000; // suppress returning a prefix match (used while re-translating a stem)
    private const int FlagFirstUpper = 0x2;

    // Suffix flags (translate.h). SufxE/SufxI are spelling-change markers; the FlagSufx* are end-flags.
    private const int SufxE = 0x0100;
    private const int SufxI = 0x0200;
    private const int FlagSufx = 0x04;
    private const int FlagSufxS = 0x08;
    private const int FlagSufxEAdded = 0x10;
}
