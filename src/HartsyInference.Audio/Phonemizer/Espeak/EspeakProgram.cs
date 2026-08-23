namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Instruction opcodes and parameter ids for espeak-ng phoneme bytecode programs, ported from <c>synthesize.h</c>; the phonemizer interprets the control-flow + allophone + IPA instructions, while audio-synthesis instructions (FMT/WAV/formants/envelopes) are parsed for control flow but their data is ignored.</summary>
internal static class EspeakProgram
{
    public const int NumParam = 16; // N_PHONEME_DATA_PARAM

    // Instructions with no operand (group 0, instn2 == 0).
    public const int InstnReturn = 0x0001;
    public const int InstnContinue = 0x0002;

    // pd_param ids (group-0 instructions store `data` into Param[instn2]).
    public const int ParamChangePhoneme = 0x01;       // i_CHANGE_PHONEME
    public const int ParamReplaceNextPhoneme = 0x02;  // i_REPLACE_NEXT_PHONEME
    public const int ParamInsertPhoneme = 0x03;       // i_INSERT_PHONEME
    public const int ParamAppendPhoneme = 0x04;       // i_APPEND_PHONEME
    public const int ParamAppendIfNextVowel = 0x05;   // i_APPEND_IFNEXTVOWEL
    public const int ParamLengthMod = 0x09;           // i_LENGTH_MOD
    public const int ParamSetLength = 0x0a;           // i_SET_LENGTH
    public const int ParamAddLength = 0x0c;           // i_ADD_LENGTH
    public const int ParamIpaName = 0x0d;             // i_IPA_NAME

    // Condition / control instruction families (top nibble of the 16-bit word).
    public const int TopCondition = 0x2;   // 0x2xxx / 0x3xxx conditions
    public const int InstnConditionMask = 0xe000;
    public const int InstnConditionTag = 0x2000;
    public const int InstnOr = 0x1000;
    public const int InstnNot = 0x0003;
    public const int InstnJumpFalse = 0x6800;
    public const int InstnJumpFalseMask = 0xf800;
    public const int InstnElseMask = 0xfe00;
    public const int InstnElseTag = 0x6000;

    // pd_control bits.
    public const int ControlChangePhonemes = 0x100;

    // Condition data-class (instn & 0xe0) and the "other" sub-conditions (synthesize.h).
    public const int ConditionIsPhonemeType = 0x00;
    public const int ConditionIsPlace = 0x20;
    public const int ConditionIsPhflagSet = 0x40;
    public const int ConditionIsOther = 0x80;
    public const int IsAfterStress = 9;
    public const int IsNotVowel = 10;
    public const int IsFinalVowel = 11;
    public const int IsVoiced = 12;
    public const int IsFirstVowel = 13;
    public const int IsSecondVowel = 14;
    public const int IsTranslationGiven = 16;
    public const int IsBreak = 17;
    public const int IsWordStart = 18;
    public const int IsWordEnd = 19;

    // Stress levels (synthesize.h STRESS_IS_*).
    public const int StressDiminished = 0;
    public const int StressUnstressed = 1;
    public const int StressNotStressed = 2;
    public const int StressSecondary = 3;
    public const int StressPrimary = 4;

    // SFLAG bits used by conditions / rendering.
    public const int SflagSyllable = 0x04;
    public const int SflagDictionary = 0x10;
    public const int SflagNextPause = 0x2000;
}
