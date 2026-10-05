namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Speaks a run of digits as a number, ported from espeak-ng 1.52 numbers.c (<c>TranslateNumber_1</c>,
/// <c>LookupNum3</c>, <c>LookupNum2</c>, <c>LookupThousands</c>): the words come from the language's <c>_list</c>
/// entries (<c>_7</c>, <c>_2X</c>, <c>_0C</c>, <c>_1M1</c>, <c>_0and</c>, ...) and are assembled by the language's
/// <c>numbers</c>/<c>numbers2</c> options. The word buffer is laid out as espeak's <c>TranslateClause</c> leaves it, a
/// long number already broken into three-digit groups followed by the thousands separator and a space
/// (<c>1. 234. 567</c>), so a group can see the groups around it. Roman numerals and the Hungarian-only paths are not
/// ported. One instance serves one clause: the missing-thousands count carries from group to group.</summary>
internal sealed class EspeakNumbers(EspeakWordLookup lookup, EspeakPhonemeTable phon, EspeakLanguageOptions options)
{
    private const byte PhonStress3 = 5;
    private const byte PhonStressP = 6;
    private const byte PhonPauseShort = 10;
    private const byte PhonPauseNoLink = 11;
    private const byte PhonEndWord = 15;

    private readonly int _numbers = options.Numbers;
    private readonly int _numbers2 = options.Numbers2;

    // numbers.c file-level state.
    private int _speakMissingThousands;
    private int _numberControl;
    private int _nDigitLookup;
    private List<byte> _digitLookup = [];
    private List<byte> _phOrdinal2 = [];
    private List<byte> _phOrdinal2x = [];

    /// <summary>Port of <c>TranslateNumber</c>: the phonemes for the number starting at <paramref name="pos"/>, or
    /// null when it is to be spoken digit by digit (too long, a leading zero, or numbers off for the language).
    /// <paramref name="skipWords"/> counts following words consumed (an ordinal suffix such as <c>º</c>).</summary>
    public List<byte>? Translate(byte[] word, int pos, bool individualDigits, out int skipWords)
    {
        skipWords = 0;
        if (individualDigits || _numbers == 0) return null;
        return TranslateNumber1(word, pos, out skipWords);
    }

    private static bool IsDigit09(byte c) => c >= '0' && c <= '9';

    private byte At(byte[] w, int i) => i >= 0 && i < w.Length ? w[i] : (byte)0;

    // Lookup(): a symbol lookup (allowed at the end of a clause), true when the entry has phonemes.
    private bool Lookup(string key, out List<byte> ph)
    {
        if (lookup.Lookup(key, EspeakLookupContext.SingleWord, out EspeakLookupResult r) && r.Phonemes.Count > 0)
        {
            ph = new List<byte>(r.Phonemes);
            return true;
        }
        ph = [];
        return false;
    }

    private List<byte> Lookup(string key) => Lookup(key, out List<byte> ph) ? ph : [];

    private int Type(byte code) => phon.TryGet(code, out EspeakPhoneme ph) ? ph.Type : 0;

    private static List<byte> Cat(params List<byte>[] parts)
    {
        List<byte> r = [];
        foreach (List<byte> p in parts) r.AddRange(p);
        return r;
    }

    private string MVariant(int value)
    {
        bool teens = value % 100 > 10 && value % 100 < 20;
        switch (_numbers2 & EspeakLanguageOptions.Num2ThousandsVarBits)
        {
            case 0x40: // ru
                if (!teens)
                {
                    if (value % 10 == 1) return "1MA";
                    if (value % 10 >= 2 && value % 10 <= 4) return "0MA";
                }
                break;
            case 0x80: // cs, sk
                if (value >= 2 && value <= 4) return "0MA";
                break;
            case 0xc0: // pl
                if (!teens && value % 10 >= 2 && value % 10 <= 4) return "0MA";
                break;
            case 0x100: // lt
                if (teens || value % 10 == 0) return "0MB";
                if (value % 10 == 1) return "0MA";
                break;
            case 0x140: // bs, hr, sr
                if (!teens)
                {
                    if (value % 10 == 1) return "1M";
                    if (value % 10 >= 2 && value % 10 <= 4) return "0MA";
                }
                break;
        }
        return "0M";
    }

    // thousandsExact: bit 0 no hundreds, tens or units; bit 1 ordinal.
    private bool LookupThousands(int value, int thousandplex, int thousandsExact, out List<byte> phOut)
    {
        bool foundValue = false;
        List<byte> phOf = [];
        List<byte> phThousands = [];

        if (value > 0)
        {
            if ((thousandsExact & 1) != 0)
            {
                if ((thousandsExact & 2) != 0)
                    foundValue = Lookup($"_{value}M{thousandplex}o", out phThousands);
                if (!foundValue && (_numberControl & 1) != 0)
                    foundValue = Lookup($"_{value}M{thousandplex}e", out phThousands);
                if (!foundValue)
                    foundValue = Lookup($"_{value}M{thousandplex}x", out phThousands);
            }
            if (!foundValue)
                foundValue = Lookup($"_{value}M{thousandplex}", out phThousands);
        }

        if (!foundValue)
        {
            if (value % 100 >= 20)
                phOf = Lookup("_0of");

            bool found = false;
            if ((thousandsExact & 1) != 0)
            {
                if ((thousandsExact & 2) != 0)
                    found = Lookup($"_{MVariant(value)}{thousandplex}o", out phThousands);
                if (!found && (_numberControl & 1) != 0)
                    found = Lookup($"_{MVariant(value)}{thousandplex}e", out phThousands);
                if (!found)
                    found = Lookup($"_{MVariant(value)}{thousandplex}x", out phThousands);
            }
            if (!found && !Lookup($"_{MVariant(value)}{thousandplex}", out phThousands))
            {
                if (thousandplex > 3 && !Lookup($"_0M{thousandplex - 1}", out _))
                {
                    // say "millions" if this name is not available and neither is the next lower
                    phThousands = Lookup("_0M2");
                    _speakMissingThousands = 3;
                }
                if (phThousands.Count == 0)
                {
                    // repeat "thousand" if higher order names are not available
                    if (!(foundValue = Lookup($"_{value}M1", out phThousands)))
                        phThousands = Lookup("_0M1");
                    _speakMissingThousands = 2;
                }
            }
        }
        phOut = Cat(phOf, phThousands);

        if (value == 1 && thousandplex == 1 && (_numbers & EspeakLanguageOptions.NumOmit1Thousand) != 0)
            return true;
        return foundValue;
    }

    // control bit 0 ordinal, bit 1 final tens and units, bit 2 no higher digits, bit 3 feminine/thousands variant,
    // bit 4 speak zero tens, bit 5 ordinal variant, bit 8 followed by a decimal fraction, bit 9 #f form for both.
    private bool LookupNum2(int value, int thousandplex, int control, out List<byte> phOut)
    {
        int units = value % 10;
        int tens = value / 10;
        bool found = false;
        bool usedAnd = false;
        bool foundOrdinal = false;
        char ordType = (control & 0x20) != 0 ? 'q' : 'o';
        bool isOrdinal = (control & 1) != 0;
        List<byte> phOrdinal = [];
        List<byte> phTens = [];
        List<byte> phDigits = [];
        List<byte> phAnd = [];

        if ((control & 2) != 0 && _nDigitLookup == 2)
        {
            phOut = new List<byte>(_digitLookup); // the final two digits were already found
        }
        else
        {
            if (_digitLookup.Count == 0)
            {
                if ((control & 8) != 0)
                {
                    if (!(found = Lookup($"_{value}fx", out phDigits)))
                        found = Lookup($"_{value}f", out phDigits);
                }
                else if (isOrdinal)
                {
                    phOrdinal = new List<byte>(_phOrdinal2);
                    if ((control & 4) != 0 && (found = Lookup($"_{value}{ordType}x", out phDigits)) && _phOrdinal2x.Count > 0)
                        phOrdinal = new List<byte>(_phOrdinal2x);
                    if (!found)
                        found = Lookup($"_{value}{ordType}", out phDigits);
                    foundOrdinal = found;
                }

                if (!found)
                {
                    if ((control & 2) != 0)
                    {
                        if ((_numberControl & 1) != 0)
                            found = Lookup($"_{value}e", out phDigits);
                    }
                    else
                    {
                        string key = (_numbers2 & EspeakLanguageOptions.Num2OrdinalAndThousands) != 0 && thousandplex <= 1
                            ? $"_{value}o" : $"_{value}a";
                        found = Lookup(key, out phDigits);
                    }

                    if (!found && !(isOrdinal && (_numbers2 & EspeakLanguageOptions.Num2NoTeenOrdinals) != 0))
                        found = Lookup($"_{value}", out phDigits);
                }
            }

            if (value < 10 && (control & 0x10) != 0)
            {
                phTens = Lookup("_0"); // speak a leading zero
            }
            else if (found)
            {
                phTens = [];
            }
            else
            {
                if (isOrdinal && Lookup($"_{tens}X{ordType}", out phTens))
                {
                    foundOrdinal = true;
                    if (units != 0 && (_numbers2 & EspeakLanguageOptions.Num2MultipleOrdinal) != 0)
                        phTens.AddRange(_phOrdinal2);
                }
                if (!foundOrdinal)
                    phTens = Lookup((control & 0x200) != 0 ? $"_{tens}Xf" : $"_{tens}X");

                if (phTens.Count == 0 && (_numbers & EspeakLanguageOptions.NumVigesimal) != 0)
                {
                    // tens not found: 73 is 60 + 13
                    units = value % 20;
                    phTens = Lookup($"_{tens & 0xfe}X");
                }

                phDigits = [];
                if (units > 0)
                {
                    found = false;
                    if ((control & 2) != 0 && _digitLookup.Count > 0)
                    {
                        phDigits = new List<byte>(_digitLookup);
                        foundOrdinal = true;
                        phOrdinal = [];
                    }
                    else
                    {
                        if ((control & 8) != 0)
                            found = Lookup($"_{units}f", out phDigits);
                        if (isOrdinal && (_numbers & EspeakLanguageOptions.NumSwapTens) == 0)
                        {
                            if ((found = Lookup($"_{units}{ordType}", out phDigits)))
                                foundOrdinal = true;
                        }
                        if (!found)
                        {
                            if ((_numberControl & 1) != 0 && (control & 2) != 0)
                                found = Lookup($"_{units}e", out phDigits);
                            else if ((control & 2) == 0 || (_numbers & EspeakLanguageOptions.NumSwapTens) != 0)
                            {
                                string key = (_numbers2 & EspeakLanguageOptions.Num2OrdinalAndThousands) != 0 && thousandplex <= 1
                                    ? $"_{units}o" : $"_{units}a";
                                found = Lookup(key, out phDigits);
                            }
                        }
                        if (!found)
                            phDigits = Lookup($"_{units}");
                    }
                }
            }

            if (isOrdinal && !foundOrdinal && phOrdinal.Count == 0)
            {
                if (value >= 20 && (value % 10 == 0 || (_numbers & EspeakLanguageOptions.NumSwapTens) != 0))
                    phOrdinal = Lookup("_ord20");
                if (phOrdinal.Count == 0)
                    phOrdinal = Lookup("_ord");
            }

            if ((_numbers & (EspeakLanguageOptions.NumSwapTens | EspeakLanguageOptions.NumAndUnits)) != 0 && phTens.Count > 0 && phDigits.Count > 0)
            {
                phAnd = Lookup("_0and");
                if (isOrdinal && (_numbers2 & EspeakLanguageOptions.Num2OrdinalNoAnd) != 0)
                    phAnd = [];
                phOut = (_numbers & EspeakLanguageOptions.NumSwapTens) != 0
                    ? Cat(phDigits, phAnd, phTens, phOrdinal)
                    : Cat(phTens, phAnd, phDigits, phOrdinal);
                usedAnd = true;
            }
            else
            {
                if ((_numbers & EspeakLanguageOptions.NumSingleVowel) != 0 && phTens.Count > 0 && phDigits.Count > 0)
                {
                    // drop the vowel ending the tens when the units start with one (Italian "ventuno")
                    int nextType = Type(phDigits[0]);
                    if (nextType == EspeakPhoneme.TypeStress && phDigits.Count > 1)
                        nextType = Type(phDigits[1]);
                    if (Type(phTens[^1]) == EspeakPhoneme.TypeVowel && nextType == EspeakPhoneme.TypeVowel)
                        phTens.RemoveAt(phTens.Count - 1);
                }

                if ((_numbers2 & EspeakLanguageOptions.Num2OrdinalDropVowel) != 0 && phOrdinal.Count > 0)
                {
                    phOut = Cat(phTens, phDigits);
                    if (phOut.Count > 0 && Type(phOut[^1]) == EspeakPhoneme.TypeVowel)
                        phOut.RemoveAt(phOut.Count - 1);
                    phOut.AddRange(phOrdinal);
                }
                else
                    phOut = Cat(phTens, phDigits, phOrdinal);
            }
        }

        if ((_numbers & EspeakLanguageOptions.NumSingleStressL) != 0)
        {
            // one primary stress, on the tens
            bool seen = false;
            for (int ix = 0; ix < phOut.Count; ix++)
            {
                if (phOut[ix] != PhonStressP) continue;
                if (seen) phOut[ix] = PhonStress3;
                else seen = true;
            }
        }
        else if ((_numbers & EspeakLanguageOptions.NumSingleStress) != 0)
        {
            // one primary stress, on the last part
            bool seen = false;
            for (int ix = phOut.Count - 1; ix >= 0; ix--)
            {
                if (phOut[ix] != PhonStressP) continue;
                if (seen) phOut[ix] = PhonStress3;
                else seen = true;
            }
        }
        return usedAnd;
    }

    // control bit 0 previous thousands, bit 1 ordinal, bit 5 ordinal variant, bit 8 followed by a decimal fraction.
    private List<byte> LookupNum3(int value, bool suppressNull, int thousandplex, int control)
    {
        int ordinal = control & 0x22;
        int hundreds = value / 100;
        int tensunits = value % 100;
        List<byte> buf1 = [];
        List<byte> phThousands = [];
        List<byte> phThousandAnd = [];
        List<byte> ph100 = [];
        List<byte> phDigits;
        bool sayZeroHundred = (_numbers & EspeakLanguageOptions.NumZeroHundred) != 0 && ((control & 1) != 0 || hundreds >= 10);

        if (hundreds > 0 || sayZeroHundred)
        {
            bool found = false;
            if (ordinal != 0 && tensunits == 0)
                found = Lookup("_0Co", out ph100);
            if (!found)
            {
                if (tensunits == 0)
                    found = Lookup("_0C0", out ph100);
                if (!found)
                    ph100 = Lookup("_0C");
            }

            if ((_numbers & EspeakLanguageOptions.Num1900) != 0 && hundreds == 19)
            {
                // a year such as 1984: nineteen eighty-four
            }
            else if (hundreds >= 10)
            {
                phDigits = [];
                int exact = value % 1000 == 0 ? 1 : 0;
                int tplex = thousandplex + 1;
                if ((_numbers2 & EspeakLanguageOptions.Num2Myriads) != 0)
                    tplex = 0;

                if (!LookupThousands(hundreds / 10, tplex, exact | ordinal, out List<byte> ph10T))
                {
                    int x = (_numbers2 & (1 << tplex)) != 0 && tplex <= 3 ? 8 : 0;
                    LookupNum2(hundreds / 10, thousandplex, x, out phDigits);
                }

                phThousands = (_numbers2 & EspeakLanguageOptions.Num2SwapThousands) != 0
                    ? Cat(ph10T, [PhonEndWord], phDigits, [PhonEndWord])
                    : Cat(phDigits, [PhonEndWord], ph10T, [PhonEndWord]);

                hundreds %= 10;
                if (hundreds == 0 && !sayZeroHundred)
                    ph100 = [];
                suppressNull = true;
                control |= 1;
            }

            phDigits = [];
            if (hundreds > 0 || sayZeroHundred)
            {
                if ((_numbers & EspeakLanguageOptions.NumAndHundred) != 0 && ((control & 1) != 0 || phThousands.Count > 0))
                    phThousandAnd = Lookup("_0and");

                suppressNull = true;
                found = false;
                if (ordinal != 0 && (tensunits == 0 || (_numbers2 & EspeakLanguageOptions.Num2MultipleOrdinal) != 0))
                {
                    found = Lookup($"_{hundreds}Co", out phDigits);
                    if ((_numbers2 & EspeakLanguageOptions.Num2MultipleOrdinal) != 0 && tensunits > 0)
                        phDigits.AddRange(_phOrdinal2);
                }

                if (hundreds == 0 && sayZeroHundred)
                    phDigits = Lookup("_0");
                else
                {
                    if (hundreds == 1 && (_numbers2 & EspeakLanguageOptions.Num2Omit1HundredOnly) != 0 && (control & 1) == 0)
                    {
                        // only look for a special 100 after thousands
                    }
                    else
                    {
                        if (!found && tensunits == 0)
                            found = Lookup($"_{hundreds}C0", out phDigits);
                        if (!found)
                            found = Lookup($"_{hundreds}C", out phDigits);
                    }

                    if (found)
                        ph100 = [];
                    else if (hundreds != 1 || (_numbers & EspeakLanguageOptions.NumOmit1Hundred) == 0)
                        LookupNum2(hundreds, thousandplex, 0, out phDigits);
                }
            }

            buf1 = Cat(phThousands, phThousandAnd, phDigits, ph100);
        }

        List<byte> phHundredAnd = [];
        if (tensunits > 0)
        {
            if ((control & 2) != 0 && (_numbers2 & EspeakLanguageOptions.Num2MultipleOrdinal) != 0)
            {
                // no "and" when the ordinal applies to both hundreds and units
            }
            else
            {
                if (value > 100 || ((control & 1) != 0 && thousandplex == 0))
                {
                    if ((_numbers & EspeakLanguageOptions.NumHundredAnd) != 0
                        || ((_numbers & EspeakLanguageOptions.NumHundredAndDigit) != 0 && tensunits < 10))
                        phHundredAnd = Lookup("_0and");
                }
                if ((_numbers & EspeakLanguageOptions.NumThousandAnd) != 0 && hundreds == 0 && ((control & 1) != 0 || phThousands.Count > 0))
                    phHundredAnd = Lookup("_0and");
            }
        }

        List<byte> buf2 = [];
        if (tensunits != 0 || !suppressNull)
        {
            int x = 0;
            if (thousandplex == 0)
            {
                x = 2; // allow "eins" for 1 rather than "ein"
                if (ordinal != 0) x = 3;
                if (value < 100 && (control & 1) == 0) x |= 4;
                if ((ordinal & 0x20) != 0) x |= 0x20;
            }
            else if ((_numbers2 & (1 << thousandplex)) != 0 && thousandplex <= 3)
                x = 8; // variant (feminine) before thousands and millions

            if ((_numbers2 & EspeakLanguageOptions.Num2ZeroTens) != 0 && ((control & 1) != 0 || hundreds > 0))
                x |= 0x10;

            if (LookupNum2(tensunits, thousandplex, x | (control & 0x100), out buf2)
                && (_numbers & EspeakLanguageOptions.NumSingleAnd) != 0)
                phHundredAnd = []; // no "and" after "hundred" when there is one between tens and units
        }
        else if (_phOrdinal2.Count > 0)
        {
            if (buf1.Count > 0 && buf1[^1] == PhonPauseShort)
                buf1.RemoveAt(buf1.Count - 1);
            buf2 = new List<byte>(_phOrdinal2);
        }

        return Cat(buf1, phHundredAnd, [PhonEndWord], buf2);
    }

    // Is this a group of 3 digits that looks like a thousands group?
    private bool CheckThousandsGroup(byte[] w, int pos, int groupLen)
    {
        for (int ix = 0; ix < groupLen; ix++)
            if (!IsDigit09(At(w, pos + ix))) return false;
        return !IsDigit09(At(w, pos + groupLen)) && !IsDigit09(At(w, pos - 1));
    }

    private List<byte>? TranslateNumber1(byte[] w, int word, out int skipWords)
    {
        skipWords = 0;
        bool suppressNull = false;
        int decimalPoint = 0;
        int thousandplex = 0;
        int thousandsExact = 1;
        int thousandsInc = 0;
        int prevThousands = 0;
        int ordinal = 0;
        byte thousandsSep = (byte)options.ThousandsSep;
        byte decimalSep = (byte)options.DecimalSep;

        _nDigitLookup = 0;
        _digitLookup = [];
        _numberControl = 0;

        int nDigits = 0;
        while (IsDigit09(At(w, word + nDigits))) nDigits++;
        if (nDigits == 0 || nDigits > 9) return null;
        long parsed = long.Parse(System.Text.Encoding.ASCII.GetString(w, word, nDigits));
        if (parsed > int.MaxValue) return null; // a long number: speak the digits
        int value = (int)parsed;

        int groupLen = (_numbers2 & EspeakLanguageOptions.Num2Myriads) != 0 ? 4 : 3;

        // is there a previous thousands part (as a previous "word")?
        if (nDigits == groupLen && At(w, word - 2) == thousandsSep && IsDigit09(At(w, word - 3)))
            prevThousands = 1;
        else if (thousandsSep == ' ' || (_numbers & EspeakLanguageOptions.NumAllowSpace) != 0)
        {
            if (nDigits == 3 && IsDigit09(At(w, word - 2)))
                prevThousands = 1;
        }
        if (prevThousands == 0)
            _speakMissingThousands = 0;

        _phOrdinal2 = [];
        _phOrdinal2x = [];
        List<byte> phZeros = [];

        // An ordinal suffix as the next word ("1 º", once the clause splits letters from digits).
        int ix = nDigits + 1;
        int sfx = word + ix;
        int sfxEnd = sfx;
        while (At(w, sfxEnd) != 0 && At(w, sfxEnd) != ' ' && sfxEnd - sfx < 29) sfxEnd++;
        if (At(w, word + nDigits) == ' ' && sfxEnd > sfx)
        {
            string suffix = System.Text.Encoding.UTF8.GetString(w, sfx, sfxEnd - sfx);
            if (options.OrdinalIndicator is string ind && suffix == ind)
                ordinal = 2;
            else if (!IsDigit09(w[sfx]) && Lookup($"_#{suffix}", out _phOrdinal2))
            {
                ordinal = 2;
                skipWords = 1;
                _phOrdinal2x = Lookup($"_x#{suffix}");
            }
        }

        if (w[word] == '0' && prevThousands == 0 && At(w, word + 1) != ' ' && At(w, word + 1) != decimalSep)
        {
            if (nDigits > 3)
            {
                skipWords = 0;
                return null; // a long digit string with a leading zero: speak the digits
            }
            for (int z = 0; w[word + z] == '0' && z < nDigits - 1; z++)
                phZeros.AddRange(Lookup("_0"));
        }

        if ((_numbers & EspeakLanguageOptions.NumAllowSpace) != 0 && At(w, word + nDigits) == ' ')
            thousandsInc = 1;
        else if (At(w, word + nDigits) == thousandsSep)
            thousandsInc = 2;

        if (thousandsInc > 0)
        {
            // count the following three-digit groups to know this group's "thousand"/"million"
            int digix = word + nDigits + thousandsInc;
            while (CheckThousandsGroup(w, digix, groupLen))
            {
                for (int g = 0; g < groupLen; g++)
                {
                    if (w[digix + g] != '0') { thousandsExact = 0; break; }
                }
                thousandplex++;
                digix += groupLen;
                if (At(w, digix) == thousandsSep || ((_numbers & EspeakLanguageOptions.NumAllowSpace) != 0 && At(w, digix) == ' '))
                    digix += thousandsInc;
                else
                    break;
            }
        }

        if (value == 0 && prevThousands != 0)
            suppressNull = true;

        List<byte> phAppend = [];
        if (At(w, word + nDigits) == decimalSep && IsDigit09(At(w, word + nDigits + 1)))
        {
            phAppend = Lookup("_dpt"); // this "word" ends with a decimal point
            decimalPoint = 0x100;
        }
        else if (!suppressNull)
        {
            if (thousandsInc > 0 && thousandplex > 0 && LookupThousands(value, thousandplex, thousandsExact, out phAppend))
            {
                // an exact match for N thousand
                value = 0;
                suppressNull = true;
            }
        }
        else if (_speakMissingThousands == 1)
        {
            // speak this thousandplex if there was no word for the previous one
            if (!Lookup($"_0M{thousandplex + 1}", out _))
                phAppend = Lookup($"_0M{thousandplex}");
        }

        if (phAppend.Count == 0 && At(w, word + nDigits) == '.' && thousandplex == 0)
            phAppend = Lookup("_.");

        if (thousandplex == 0)
        {
            // the number together with the next word: the last two digits, then the last one
            int p = word;
            while (IsDigit09(At(w, p + 1))) p++;
            EspeakLookupContext sufx = EspeakLookupContext.SingleWord with { EndFlags = FlagSufx };
            if (IsDigit09(At(w, p - 1)) && lookup.Lookup(Ascii(w, p - 1, 2), sufx, out EspeakLookupResult r2) && r2.Phonemes.Count > 0)
            {
                _digitLookup = new List<byte>(r2.Phonemes);
                _nDigitLookup = 2;
            }
            if (_digitLookup.Count == 0 && w[p] != '0'
                && lookup.Lookup(Ascii(w, p, 1), sufx, out EspeakLookupResult r1) && r1.Phonemes.Count > 0)
            {
                _digitLookup = new List<byte>(r1.Phonemes);
                _nDigitLookup = 1;
            }

            if (prevThousands == 0 && decimalPoint == 0 && ordinal == 0 && Lookup($"_{value}n", out List<byte> isolated))
                return isolated; // a special pronunciation for this number in isolation
        }

        List<byte> phBuf = LookupNum3(value, suppressNull, thousandplex, prevThousands | ordinal | decimalPoint);
        List<byte> phOut = (thousandplex > 0 && (_numbers2 & EspeakLanguageOptions.Num2SwapThousands) != 0)
            ? Cat(phZeros, phAppend, [PhonEndWord], phBuf)
            : Cat(phZeros, phBuf, [PhonEndWord], phAppend);

        while (decimalPoint != 0)
        {
            nDigits++;
            int decimalCount = 0;
            while (IsDigit09(At(w, word + nDigits + decimalCount))) decimalCount++;

            int maxDecimalCount = 2;
            int decimalMode = _numbers & EspeakLanguageOptions.NumDFractionBits;
            switch (decimalMode)
            {
                case EspeakLanguageOptions.NumDFraction4:
                case EspeakLanguageOptions.NumDFraction2:
                    if (decimalMode == EspeakLanguageOptions.NumDFraction4) maxDecimalCount = 5;
                    // French/Polish decimal fraction
                    while (At(w, word + nDigits) == '0')
                    {
                        phOut.AddRange(Lookup("_0"));
                        decimalCount--;
                        nDigits++;
                    }
                    if (decimalCount <= maxDecimalCount && IsDigit09(At(w, word + nDigits)))
                    {
                        phOut.AddRange(LookupNum3(Atoi(w, word + nDigits), false, 0, 0));
                        nDigits += decimalCount;
                    }
                    break;
                case EspeakLanguageOptions.NumDFraction1:
                case EspeakLanguageOptions.NumDFraction5:
                case EspeakLanguageOptions.NumDFraction6:
                    {
                        List<byte> frac = LookupNum3(Atoi(w, word + nDigits), false, 0, 0);
                        if (At(w, word + nDigits) == '0' || decimalMode != EspeakLanguageOptions.NumDFraction1)
                        {
                            // leading zeros: add a "hundredths"/"thousandths" suffix
                            if (!Lookup($"_0Z{decimalCount}", out List<byte> zs))
                                break; // revert to speaking single digits
                            if (decimalMode == EspeakLanguageOptions.NumDFraction6) phOut.AddRange(zs);
                            else frac.AddRange(zs);
                        }
                        phOut.AddRange(frac);
                        nDigits += decimalCount;
                        break;
                    }
                case EspeakLanguageOptions.NumDFraction3:
                    if (decimalCount <= 4 && At(w, word + nDigits) != '0')
                    {
                        phOut.AddRange(LookupNum3(Atoi(w, word + nDigits), false, 0, 0));
                        nDigits += decimalCount;
                    }
                    break;
                case EspeakLanguageOptions.NumDFraction7:
                    while (decimalCount-- > 1)
                    {
                        if (!Lookup($"_{(char)w[word + nDigits]}d", out List<byte> d)) break;
                        nDigits++;
                        phOut.AddRange(d);
                    }
                    break;
            }

            byte c;
            while (IsDigit09(c = At(w, word + nDigits)) && phOut.Count < 190)
            {
                // speak any remaining decimal digits individually
                LookupNum2(w[word + nDigits++] - '0', 0, 2, out List<byte> digit);
                phOut.Add(PhonEndWord);
                phOut.AddRange(digit);
            }

            if (Lookup("_dpt2", out List<byte> dpt2))
                phOut.AddRange(dpt2);

            if (c == decimalSep && IsDigit09(At(w, word + nDigits + 1)))
                phOut.AddRange(Lookup("_dpt"));
            else
                decimalPoint = 0;
        }

        if (phOut.Count > 0)
        {
            int next = word + nDigits + 1;
            EspeakUtf8.Read(w, Math.Min(next, w.Length - 1), out int nextChar);
            if ((_numbers & EspeakLanguageOptions.NumNoPause) != 0 && nextChar == ' ')
                EspeakUtf8.Read(w, Math.Min(next + 1, w.Length - 1), out nextChar);
            if (!char.IsLetter((char)nextChar) && thousandsExact == 0)
                phOut.Add(PhonPauseNoLink); // no pause for 100s, 6th, ...
        }

        _speakMissingThousands--;
        return phOut;
    }

    private const int FlagSufx = 0x04;

    private static string Ascii(byte[] w, int pos, int n) => System.Text.Encoding.ASCII.GetString(w, pos, n);

    private static int Atoi(byte[] w, int pos)
    {
        long v = 0;
        for (int i = pos; i < w.Length && IsDigit09(w[i]) && v < int.MaxValue; i++)
            v = v * 10 + (w[i] - '0');
        return (int)Math.Min(v, int.MaxValue);
    }
}
