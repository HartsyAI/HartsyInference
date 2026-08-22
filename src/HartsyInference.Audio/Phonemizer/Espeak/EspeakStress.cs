namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Places lexical stress on a word's phoneme sequence, ported from <c>GetVowelStress</c> + <c>SetWordStress</c> in espeak-ng dictionary.c: strips any existing stress markers, decides each vowel's stress level from the language stress rule and dictionary flag bits (which may pin the stressed syllable), and re-emits the sequence with stress-marker phonemes inserted. English relies mostly on the explicit dictionary stress position with a penultimate fallback.</summary>
internal sealed class EspeakStress
{
    private const int MaxPhonemes = 200; // N_WORD_PHONEMES
    private const int MaxVowels = MaxPhonemes / 2;

    // Phoneme codes (phoneme.h).
    private const byte PhonStressPrev = 8;
    private const byte PhonPause = 9;
    private const byte PhonPauseNoLink = 11;
    private const byte PhonLengthen = 12;
    private const byte PhonSchwa = 13;
    private const byte PhonEndWord = 15;
    private const byte PhonSyllabic = 20;
    private const byte PhonPauseVShort = 23;

    // Stress levels (synthesize.h STRESS_IS_*).
    private const int Diminished = 0;
    private const int Unstressed = 1;
    private const int NotStressed = 2;
    private const int Secondary = 3;
    private const int Primary = 4;
    private const int Priority = 5;

    // stress_phonemes[] indexed by stress level (dictionary.c).
    private static readonly byte[] StressPhonemes = [3, 2, 4, 5, 6, 7, 26];

    private static readonly byte[] ConsonantTypes = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];

    private readonly EspeakPhonemeTable _phon;

    // Language options (English/base defaults).
    private readonly int _stressRule;
    private readonly int _stressFlags;
    private readonly int _unstressedWd1;
    private readonly int _unstressedWd2;

    /// <summary>The language stress flags, consulted by the phoneme-list reduction pass.</summary>
    public int StressFlags => _stressFlags;

    public EspeakStress(EspeakPhonemeTable phonemeTable, int stressRule = 2, int stressFlags = 0, int unstressedWd1 = 1, int unstressedWd2 = 3)
    {
        _phon = phonemeTable;
        _stressRule = stressRule;
        _stressFlags = stressFlags;
        _unstressedWd1 = unstressedWd1;
        _unstressedWd2 = unstressedWd2;
    }

    private bool Ph(int code, out EspeakPhoneme ph) => _phon.TryGet(code, out ph);

    private int Type(int code) => Ph(code, out EspeakPhoneme ph) ? ph.Type : 0;

    private bool IsSyllabicVowel(int code)
        => Ph(code, out EspeakPhoneme ph) && ph.Type == EspeakPhoneme.TypeVowel && (ph.PhFlags & EspeakPhoneme.FlagNonSyllabic) == 0;

    private const uint PhLong = 1U << 21;

    // Port of GetVowelStress: fills vowel_stress[], strips stress markers from phonetic in place, returns max stress.
    private int GetVowelStress(byte[] phonetic, sbyte[] vowelStress, out int vowelCount, ref int stressedSyllable, int control)
    {
        int count = 1;
        int maxStress = -1;
        int stress = -1;
        int primaryPosn = 0;
        int outPos = 0;

        vowelStress[0] = Unstressed;
        int i = 0;
        while (phonetic[i] != 0 && count < MaxVowels - 1)
        {
            byte phcode = phonetic[i++];
            if (!Ph(phcode, out EspeakPhoneme ph))
                continue;

            if (ph.Type == EspeakPhoneme.TypeStress && ph.Program == 0)
            {
                if (phcode == PhonStressPrev)
                {
                    int j = count - 1;
                    while (j > 0 && stressedSyllable == 0 && vowelStress[j] < Primary)
                    {
                        if (vowelStress[j] != Diminished && vowelStress[j] != Unstressed)
                        {
                            vowelStress[j] = Primary;
                            if (maxStress < Primary) { maxStress = Primary; primaryPosn = j; }
                            for (int ix = 1; ix < j; ix++)
                                if (vowelStress[ix] == Primary) vowelStress[ix] = Secondary;
                            break;
                        }
                        j--;
                    }
                }
                else if (ph.StdLength < 4 || stressedSyllable == 0)
                {
                    stress = ph.StdLength;
                    if (stress > maxStress) maxStress = stress;
                }
                continue;
            }

            if (ph.Type == EspeakPhoneme.TypeVowel && (ph.PhFlags & EspeakPhoneme.FlagNonSyllabic) == 0)
            {
                vowelStress[count] = (sbyte)stress;
                if (stress >= Primary && stress >= maxStress) { primaryPosn = count; maxStress = stress; }
                if (stress < 0 && (control & 1) != 0 && (ph.PhFlags & EspeakPhoneme.FlagUnstressed) != 0)
                    vowelStress[count] = Unstressed;
                count++;
                stress = -1;
            }
            else if (phcode == PhonSyllabic)
            {
                vowelStress[count] = (sbyte)stress;
                if (stress == 0 && (control & 1) != 0)
                    vowelStress[count++] = Unstressed;
            }

            phonetic[outPos++] = phcode;
        }
        vowelStress[count] = Unstressed;
        phonetic[outPos] = 0;

        if (stressedSyllable > 0)
        {
            if (stressedSyllable >= count) stressedSyllable = count - 1;
            vowelStress[stressedSyllable] = Primary;
            maxStress = Primary;
            primaryPosn = stressedSyllable;
        }

        if (maxStress == Priority)
        {
            for (int ix = 1; ix < count; ix++)
            {
                if (vowelStress[ix] == Primary) vowelStress[ix] = Secondary;
                if (vowelStress[ix] == Priority) { vowelStress[ix] = Primary; primaryPosn = ix; }
            }
            maxStress = Primary;
        }

        stressedSyllable = primaryPosn;
        vowelCount = count;
        return maxStress;
    }

    /// <summary>Places stress on <paramref name="inputCodes"/> and returns the stressed phoneme code sequence. <paramref name="dflags"/> are the dictionary flags (low bits pin the stressed syllable, pass 0 for rule-translated words); <paramref name="tonic"/> &gt;= 0 replaces the top stress with that level (-1 for none).</summary>
    public List<byte> SetWordStress(List<byte> inputCodes, uint dflags, int tonic, int control)
    {
        byte[] phonetic = new byte[MaxPhonemes + 4];
        int n = 0;
        for (int k = 0; k < inputCodes.Count && n < MaxPhonemes; k++)
        {
            byte c = inputCodes[k];
            if (c > _phon.MaxCode) c = PhonSchwa;
            phonetic[n++] = c;
        }
        phonetic[n] = 0;
        if (n == 0) return new List<byte>();

        int finalPh = phonetic[n - 1];
        int finalPh2 = phonetic[n > 1 ? n - 2 : n - 1];

        sbyte[] vowelStress = new sbyte[MaxVowels];
        byte[] syllableWeight = new byte[MaxVowels];
        byte[] vowelLength = new byte[MaxVowels];

        bool unstressedWord = false;
        int stressedSyllable = (int)(dflags & 0x7);
        if ((dflags & 0x8) != 0)
        {
            stressedSyllable = (int)(dflags & 0x3);
            // The 0x8 stress-field bit marks a word "reduce to unstressed" ($u = 0x48). But $u+ (0x4c) additionally
            // sets FLAG_STRESS_END (0x200) — "reduce to unstressed, but stress at end of clause" — and in the
            // espeak-ng 1.51 the phonemizer uses, such words (e.g. "this", "that") keep primary stress in normal
            // positions, not just at clause end. Only plain $u (no FLAG_STRESS_END) actually reduces here.
            unstressedWord = (dflags & FlagStressEnd) == 0;
        }

        int maxStressInput = GetVowelStress(phonetic, vowelStress, out int vowelCount, ref stressedSyllable, 1);
        int maxStress = maxStressInput;
        if (maxStress < 0) maxStress = Diminished;

        // heavy or light syllables
        int vi = 1;
        for (int p = 0; phonetic[p] != 0; p++)
        {
            if (IsSyllabicVowel(phonetic[p]))
            {
                int weight = 0;
                bool lengthened = Ph(phonetic[p + 1], out EspeakPhoneme nextPh) && nextPh.Code == PhonLengthen;
                if (lengthened || (Ph(phonetic[p], out EspeakPhoneme vph) && (vph.PhFlags & PhLong) != 0))
                    weight++;
                vowelLength[vi] = (byte)weight;
                if (lengthened) p++;

                int t1 = Type(phonetic[p + 1]);
                bool p1Long = Ph(phonetic[p + 1], out EspeakPhoneme p1) && (p1.PhFlags & PhLong) != 0;
                if (ConsonantTypes[t1 & 0xf] != 0 && (Type(phonetic[p + 2]) != EspeakPhoneme.TypeVowel || p1Long))
                    weight++;
                syllableWeight[vi] = (byte)weight;
                vi++;
            }
        }

        ApplyStressRule(vowelStress, syllableWeight, vowelLength, vowelCount, ref stressedSyllable, ref maxStress, finalPh, finalPh2, control);

        // guess the complete stress pattern
        int stress = maxStress < Primary ? Primary : Secondary;

        if (!unstressedWord)
        {
            if ((_stressFlags & S_2SylG2) != 0 && vowelCount == 3)
            {
                if (vowelStress[1] == Primary) vowelStress[2] = Secondary;
                if (vowelStress[2] == Primary) vowelStress[1] = Secondary;
            }
            if ((_stressFlags & S_Initial2) != 0 && vowelStress[1] < Diminished)
            {
                if (vowelCount > 3 && vowelStress[2] >= Primary) vowelStress[1] = Secondary;
            }
        }

        bool done = false;
        int firstPrimary = 0;
        for (int v = 1; v < vowelCount; v++)
        {
            if (vowelStress[v] < Diminished)
            {
                if ((_stressFlags & S_FinalNo2) != 0 && stress < Primary && v == vowelCount - 1)
                {
                    // no secondary stress on final vowel
                }
                else if ((_stressFlags & 0x8000) != 0 && !done)
                {
                    vowelStress[v] = (sbyte)stress; done = true; stress = Secondary;
                }
                else if (vowelStress[v - 1] <= Unstressed && (vowelStress[v + 1] <= Unstressed || (stress == Primary && vowelStress[v + 1] <= NotStressed)))
                {
                    if (stress == Secondary && (_stressFlags & S_NoAuto2) != 0)
                        continue;
                    vowelStress[v] = (sbyte)stress; done = true; stress = Secondary;
                }
            }
            if (vowelStress[v] >= Primary)
            {
                if (firstPrimary == 0) firstPrimary = v;
                else if ((_stressFlags & S_FirstPrimary) != 0) vowelStress[v] = Secondary;
            }
        }

        if (unstressedWord && tonic < 0)
            tonic = vowelCount <= 2 ? _unstressedWd1 : _unstressedWd2;

        maxStress = Diminished;
        int maxStressPosn = 0;
        for (int v = 1; v < vowelCount; v++)
            if (vowelStress[v] >= maxStress) { maxStress = vowelStress[v]; maxStressPosn = v; }

        if (tonic >= 0)
        {
            if (tonic > maxStress || maxStress <= Primary) vowelStress[maxStressPosn] = (sbyte)tonic;
            maxStress = tonic;
        }

        return WriteOutput(phonetic, vowelStress, vowelCount, maxStress, control);
    }

    // The big stress_rule switch; only the English-relevant cases are exercised here (2 = penultimate fallback),
    // but the other cases are ported for when additional languages are wired.
    private void ApplyStressRule(sbyte[] vowelStress, byte[] syllableWeight, byte[] vowelLength, int vowelCount, ref int stressedSyllable, ref int maxStress, int finalPh, int finalPh2, int control)
    {
        switch (_stressRule)
        {
            // No case for STRESSPOSN_1L (0): English relies on the dictionary plus the "guess complete stress
            // pattern" pass below to place primary stress; espeak's switch likewise has no case 0.
            case 8:
                if (syllableWeight[1] > 0 || syllableWeight[2] == 0) break;
                goto case 1;
            case 1:
                if (stressedSyllable == 0 && vowelCount > 2)
                {
                    stressedSyllable = 2;
                    if (maxStress == Diminished) vowelStress[stressedSyllable] = Primary;
                    maxStress = Primary;
                }
                break;
            case 10:
                if (stressedSyllable == 0 && vowelCount < 4)
                {
                    vowelStress[vowelCount - 1] = Primary;
                    maxStress = Primary;
                    break;
                }
                goto case 2;
            case 2:
                if (stressedSyllable == 0)
                {
                    maxStress = Primary;
                    if (vowelCount > 2)
                    {
                        stressedSyllable = vowelCount - 2;
                        if (vowelStress[stressedSyllable] == Diminished || vowelStress[stressedSyllable] == Unstressed)
                            stressedSyllable = stressedSyllable > 1 ? stressedSyllable - 1 : stressedSyllable + 1;
                    }
                    else
                        stressedSyllable = 1;

                    if (vowelStress[stressedSyllable] < 0)
                    {
                        if (vowelStress[stressedSyllable - 1] < Primary || vowelStress[stressedSyllable + 1] < Primary)
                            vowelStress[stressedSyllable] = (sbyte)maxStress;
                    }
                }
                break;
            case 3:
                if (stressedSyllable == 0)
                {
                    stressedSyllable = vowelCount - 1;
                    while (stressedSyllable > 0)
                    {
                        if (vowelStress[stressedSyllable] < Diminished) { vowelStress[stressedSyllable] = Primary; break; }
                        stressedSyllable--;
                    }
                    maxStress = Primary;
                }
                break;
            case 4:
                if (stressedSyllable == 0)
                {
                    stressedSyllable = vowelCount - 3;
                    if (stressedSyllable < 1) stressedSyllable = 1;
                    if (maxStress == Diminished) vowelStress[stressedSyllable] = Primary;
                    maxStress = Primary;
                }
                break;
            case 9:
                for (int ix = 1; ix < vowelCount; ix++)
                    if (vowelStress[ix] < Diminished) vowelStress[ix] = Primary;
                break;
        }
        _ = vowelLength;
        _ = finalPh;
        _ = finalPh2;
        _ = control;
    }

    private List<byte> WriteOutput(byte[] phonetic, sbyte[] vowelStress, int vowelCount, int maxStress, int control)
    {
        List<byte> output = new(MaxPhonemes);
        int p = 0;
        int v = 1;

        // leading-vowel pause handling is language-specific (vowel_pause); English does not set it, so skipped here.

        byte phcode;
        while ((phcode = phonetic[p++]) != 0 && output.Count < MaxPhonemes - 3)
        {
            if (!Ph(phcode, out EspeakPhoneme ph))
                continue;

            if (ph.Type == EspeakPhoneme.TypePause)
            {
                // pause resets running stress context (not tracked here)
            }
            else if ((ph.Type == EspeakPhoneme.TypeVowel && (ph.PhFlags & EspeakPhoneme.FlagNonSyllabic) == 0) || phonetic[p] == PhonSyllabic)
            {
                int vStress = vowelStress[v];

                if (vStress <= Unstressed)
                {
                    if (v > 1 && maxStress >= 2 && (_stressFlags & S_FinalDim) != 0 && v == vowelCount - 1)
                        vStress = Diminished;
                    else if ((_stressFlags & S_NoDim) != 0 || v == 1 || v == vowelCount - 1)
                        vStress = Unstressed;
                    else if (v == vowelCount - 2 && vowelStress[vowelCount - 1] <= Unstressed)
                        vStress = Unstressed;
                    else
                    {
                        if (vowelStress[v - 1] < Diminished || (_stressFlags & S_MidDim) == 0)
                        {
                            vStress = Diminished;
                            vowelStress[v] = (sbyte)vStress;
                        }
                    }
                }

                if (vStress == Diminished || vStress > Unstressed)
                    output.Add(StressPhonemes[vStress]);

                if (vowelStress[v] > maxStress) maxStress = vowelStress[v];
                v++;
            }

            if (phcode != 1)
                output.Add(phcode);
        }

        _ = control;
        return output;
    }

    /// <summary>Dictionary flag "full stress if at end of clause" (translate.h FLAG_STRESS_END); set by the $u+ list attribute in addition to the 0x8 unstressed-field bit.</summary>
    private const uint FlagStressEnd = 0x200;

    // stress_flags (translate.h S_*).
    private const int S_NoDim = 0x02;
    private const int S_FinalDim = 0x04;
    private const int S_FinalNo2 = 0x10;
    private const int S_NoAuto2 = 0x20;
    private const int S_FirstPrimary = 0x80;
    private const int S_2SylG2 = 0x1000;
    private const int S_Initial2 = 0x2000;
    private const int S_MidDim = 0x10000;
}
