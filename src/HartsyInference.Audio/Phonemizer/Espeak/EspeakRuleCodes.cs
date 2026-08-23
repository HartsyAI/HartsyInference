namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Byte opcodes and flag bits used inside a compiled espeak-ng <c>*_dict</c> file, ported verbatim from espeak-ng v1.50 <c>translate.h</c>; these drive <see cref="EspeakDictFile"/> (dictionary layout) and <see cref="EspeakTranslator"/> (letter-to-sound rule matching), so the values must match espeak exactly.</summary>
internal static class EspeakRuleCodes
{
    /// <summary>Number of hash-table buckets in the compiled dictionary word list.</summary>
    public const int HashDictSize = 1024;

    /// <summary>Maximum number of named letter groups (<c>.L01</c>..<c>.L94</c>); espeak caps at 127-32.</summary>
    public const int LetterGroups = 95;

    /// <summary>Maximum bytes for the UTF-8 characters in a single word buffer.</summary>
    public const int WordBytes = 160;

    /// <summary>'e' that has been replaced by a silent-e marker during matching.</summary>
    public const byte ReplacedE = (byte)'E';

    // Rule opcodes (translate.h "codes in dictionary rules").
    public const byte RulePre = 1;
    public const byte RulePost = 2;
    public const byte RulePhonemes = 3;
    public const byte RulePhCommon = 4;
    public const byte RuleCondition = 5;
    public const byte RuleGroupStart = 6;
    public const byte RuleGroupEnd = 7;
    public const byte RulePreAtStart = 8;
    public const byte RuleLineNum = 9;
    public const byte RuleStressed = 10;
    public const byte RuleDouble = 11;
    public const byte RuleIncScore = 12;
    public const byte RuleDelFwd = 13;
    public const byte RuleEnding = 14;
    public const byte RuleDigit = 15;
    public const byte RuleNonAlpha = 16;
    public const byte RuleLetterGp = 17;
    public const byte RuleLetterGp2 = 18;
    public const byte RuleCapital = 19;
    public const byte RuleReplacements = 20;
    public const byte RuleSyllable = 21;
    public const byte RuleSkipChars = 23;
    public const byte RuleNoSuffix = 24;
    public const byte RuleNotVowel = 25;
    public const byte RuleIfVerb = 26;
    public const byte RuleDollar = 28;
    public const byte RuleNoVowels = 29;
    public const byte RuleSpelling = 31;
    public const byte RuleLastRule = 31;
    public const byte RuleSpace = 32;
    public const byte RuleDecScore = 60;

    // $ command sub-codes following RuleDollar.
    public const byte DollarUnpr = 0x01;
    public const byte DollarNoPrefix = 0x02;
    public const byte DollarList = 0x03;

    // Letter-group indices (translate.h LETTERGP_*).
    public const int LetterGpA = 0;
    public const int LetterGpB = 1;
    public const int LetterGpC = 2;
    public const int LetterGpH = 3;
    public const int LetterGpF = 4;
    public const int LetterGpG = 5;
    public const int LetterGpY = 6;
    public const int LetterGpVowel2 = 7;

    /// <summary>Bit number of <c>FLAG_ALT_TRANS - 1</c>, used to decode <c>$w_alt</c>/<c>$p_alt</c> commands.</summary>
    public const int BitNumFlagAlt = 14;

    // Suffix flags (translate.h SUFX_*), returned in MatchRecord.end_type.
    public const int SufxUnpron = 0x8000;
    public const int SufxP = 0x0400;
}
