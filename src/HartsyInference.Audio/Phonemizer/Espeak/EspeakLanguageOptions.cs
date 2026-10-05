namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>The per-language translator options that change phonemized output, ported from espeak-ng 1.52
/// <c>NewTranslator</c> (the defaults) and the <c>SelectTranslator</c> cases of the languages the engine phonemizes:
/// English, and the five Kokoro speaks through espeak (Spanish, French, Italian, Portuguese, Hindi).</summary>
internal sealed record EspeakLanguageOptions
{
    /// <summary><c>STRESSPOSN_*</c>: where a word's primary stress goes when the dictionary does not say.</summary>
    public int StressRule { get; init; } = StressPosn2R;

    /// <summary><c>S_*</c> stress flags.</summary>
    public int StressFlags { get; init; }

    /// <summary>Stress level of an unstressed (<c>$u</c>) one-syllable word.</summary>
    public int UnstressedWd1 { get; init; } = 1;

    /// <summary>Stress level of an unstressed multi-syllable word.</summary>
    public int UnstressedWd2 { get; init; } = 3;

    /// <summary><c>LOPT_IT_LENGTHEN</c>: drop the length mark from syllables that are not stressed enough.</summary>
    public int ItLengthen { get; init; }

    /// <summary><c>LOPT_REDUCE</c>: reduce vowels even when the dictionary spells the phonemes.</summary>
    public int Reduce { get; init; }

    /// <summary><c>LOPT_ALT</c>: bit 1 lets <c>$alt</c>/<c>$alt2</c> open or close the stressed e/o.</summary>
    public int Alt { get; init; }

    /// <summary><c>NUM_*</c> number options (0 speaks digits one by one).</summary>
    public int Numbers { get; init; } = NumDefault;

    /// <summary><c>NUM2_*</c> number options.</summary>
    public int Numbers2 { get; init; }

    /// <summary><c>BREAK_*</c>: after which digit counts a long number is cut into groups.</summary>
    public uint BreakNumbers { get; init; } = BreakThousands;

    /// <summary>Numbers with more digits than this are read digit by digit.</summary>
    public int MaxDigits { get; init; } = 14;

    /// <summary>The ordinal indicator word that follows a number, if the language has one.</summary>
    public string? OrdinalIndicator { get; init; }

    /// <summary>Thousands separator (<c>.</c> where the decimal separator is a comma).</summary>
    public char ThousandsSep => (Numbers & NumThousSpace) != 0 ? '\0' : (Numbers & NumDecimalComma) != 0 ? '.' : ',';

    /// <summary>Decimal separator.</summary>
    public char DecimalSep => (Numbers & NumDecimalComma) != 0 ? ',' : '.';

    public const int StressPosn1L = 0;
    public const int StressPosn2L = 1;
    public const int StressPosn2R = 2;
    public const int StressPosn1R = 3;
    public const int StressPosn3R = 4;
    public const int StressPosn1RH = 6;

    public const int SNoDim = 0x02;
    public const int SFinalDim = 0x04;
    public const int SFinalDimOnly = 0x06;
    public const int SFinalNo2 = 0x10;
    public const int SNoAuto2 = 0x20;
    public const int S2ToHeavy = 0x40;
    public const int SFirstPrimary = 0x80;
    public const int SFinalVowelUnstressed = 0x100;
    public const int SFinalSpanish = 0x200;
    public const int S2Syl2 = 0x1000;
    public const int SInitial2 = 0x2000;
    public const int SMidDim = 0x10000;
    public const int SPriorityStress = 0x20000;
    public const int SFinalLong = 0x80000;

    public const int NumDefault = 0x1;
    public const int NumThousSpace = 0x4;
    public const int NumDecimalComma = 0x8;
    public const int NumSwapTens = 0x10;
    public const int NumAndUnits = 0x20;
    public const int NumHundredAnd = 0x40;
    public const int NumSingleAnd = 0x80;
    public const int NumSingleStress = 0x100;
    public const int NumSingleVowel = 0x200;
    public const int NumOmit1Hundred = 0x400;
    public const int Num1900 = 0x800;
    public const int NumAllowSpace = 0x1000;
    public const int NumDFractionBits = 0xe000;
    public const int NumDFraction1 = 0x2000;
    public const int NumDFraction2 = 0x4000;
    public const int NumDFraction3 = 0x6000;
    public const int NumDFraction4 = 0x8000;
    public const int NumDFraction5 = 0xa000;
    public const int NumDFraction6 = 0xc000;
    public const int NumDFraction7 = 0xe000;
    public const int NumOrdinalDot = 0x10000;
    public const int NumNoPause = 0x20000;
    public const int NumAndHundred = 0x40000;
    public const int NumThousandAnd = 0x80000;
    public const int NumVigesimal = 0x100000;
    public const int NumOmit1Thousand = 0x200000;
    public const int NumZeroHundred = 0x400000;
    public const int NumHundredAndDigit = 0x800000;
    public const int NumRoman = 0x1000000;
    public const int NumRomanCapitals = 0x2000000;
    public const int NumRomanAfter = 0x4000000;
    public const int NumRomanOrdinal = 0x8000000;
    public const int NumSingleStressL = 0x10000000;

    public const int Num2ThousandsVarBits = 0x1c0;
    public const int Num2SwapThousands = 0x200;
    public const int Num2OrdinalNoAnd = 0x800;
    public const int Num2MultipleOrdinal = 0x1000;
    public const int Num2NoTeenOrdinals = 0x2000;
    public const int Num2Myriads = 0x4000;
    public const int Num2EnglishNumerals = 0x8000;
    public const int Num2Omit1HundredOnly = 0x20000;
    public const int Num2OrdinalAndThousands = 0x40000;
    public const int Num2OrdinalDropVowel = 0x80000;
    public const int Num2ZeroTens = 0x100000;

    public const uint BreakThousands = 0x49249248;
    public const uint BreakLakhHi = 0x00014aa8;

    /// <summary>The options for <paramref name="language"/> (<c>en-us</c>, <c>es</c>, <c>fr-fr</c>, <c>it</c>,
    /// <c>pt-br</c>, <c>hi</c>, ...), selected by its base name as espeak does.</summary>
    public static EspeakLanguageOptions For(string language) => language.Split('-')[0] switch
    {
        "en" => new()
        {
            StressRule = StressPosn1L, StressFlags = 0x08,
            Numbers = NumHundredAnd | NumRoman | Num1900, MaxDigits = 33,
        },
        "es" => new()
        {
            StressRule = StressPosn2R, StressFlags = SFinalSpanish | SFinalDimOnly | SFinalNo2,
            UnstressedWd1 = 0, UnstressedWd2 = 2,
            Numbers = NumSingleStress | NumDecimalComma | NumAndUnits | NumOmit1Hundred | NumOmit1Thousand | NumRoman
                | NumRomanAfter | NumDFraction4,
            Numbers2 = Num2MultipleOrdinal | Num2OrdinalNoAnd,
        },
        "fr" => new()
        {
            StressRule = StressPosn1R, StressFlags = SNoAuto2 | SFinalDim, ItLengthen = 1,
            Numbers = NumSingleStress | NumDecimalComma | NumAllowSpace | NumOmit1Hundred | NumNoPause | NumRoman
                | NumRomanCapitals | NumRomanAfter | NumVigesimal | NumDFraction4,
        },
        "it" => new()
        {
            StressRule = StressPosn2R, StressFlags = SNoAuto2 | SFinalDimOnly | SPriorityStress,
            UnstressedWd1 = 0, UnstressedWd2 = 2, ItLengthen = 2, Reduce = 1, Alt = 2,
            Numbers = NumSingleVowel | NumOmit1Hundred | NumDecimalComma | NumDFraction1 | NumRoman | NumRomanCapitals
                | NumRomanOrdinal,
            Numbers2 = Num2NoTeenOrdinals,
        },
        "pt" => new()
        {
            StressRule = StressPosn1R, StressFlags = SFinalDimOnly | SFinalNo2 | SInitial2 | SPriorityStress, Alt = 2,
            Numbers = NumDecimalComma | NumDFraction2 | NumHundredAnd | NumAndUnits | NumRomanCapitals,
            Numbers2 = Num2MultipleOrdinal | Num2NoTeenOrdinals | Num2OrdinalNoAnd,
        },
        "hi" => new()
        {
            StressRule = StressPosn1RH, StressFlags = SMidDim | SFinalDim,
            Numbers = NumSwapTens, BreakNumbers = BreakLakhHi,
        },
        _ => new(),
    };
}
