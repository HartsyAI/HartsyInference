namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Spells a word letter by letter when the dictionary gives it no pronunciation and the letters cannot be read
/// as a word: each letter is pronounced by its name, and the names are stressed as espeak does (every name secondary
/// except the last, which keeps the primary stress). Names are espeak phoneme mnemonics, resolved to codes through the
/// loaded phoneme table.</summary>
internal static class EspeakLetterNames
{
    // Letter names a..z as espeak mnemonics (the American table the en-us voice uses).
    private static readonly string[] Names =
    [
        "eI", "b i:", "s i:", "d i:", "i:", "E f", "dZ i:", "eI tS", "aI", "dZ eI", "k eI", "E l", "E m", "E n", "oU",
        "p i:", "k j u:", "A: r-", "E s", "t i:", "j u:", "v i:", "d V b @ l j u:", "E k s", "w aI", "z i:",
    ];

    private const uint FlagFound = 0x80000000;
    // Dictionary flag bit 13: set on the flag-only entries of letter strings espeak spells (ok, ai, abc).
    private const uint FlagSpellWord = 0x2000;
    // Dictionary flag bit 24: an abbreviation entry (lt, st) the rules read unless they leave it without a vowel.
    private const uint FlagAbbreviation = 0x01000000;
    private const int SecondaryStress = 3;

    // A short all-letter word with no vowel letter cannot be read as a word, so espeak spells it out. Longer words go
    // to the letter-to-sound rules.
    private const int MaxSpelledLength = 4;
    private const string VowelLetters = "aeiouyAEIOUY";

    /// <summary>True when espeak spells a letter string outright, without consulting the rules: a flag-only entry with the
    /// spell-word bit ("ok", "ai"), or a short letter string with no dictionary entry and no vowel letter ("pc", "hr").
    /// <paramref name="dictFlags"/> are 0 when the word has no entry.</summary>
    public static bool IsSpelledByFlag(string word, uint dictFlags)
    {
        if (!IsLetterString(word)) return false;
        if ((dictFlags & FlagSpellWord) != 0) return true;
        return (dictFlags & FlagAbbreviation) == 0 && word.Length <= MaxSpelledLength && !word.Any(c => VowelLetters.Contains(c));
    }

    /// <summary>True for an abbreviation entry ("lt", "st") the rules read unless they give it no vowel. espeak spells
    /// "lt" but reads "st" as sənt, so the caller spells the word only when the rules leave it without a vowel.</summary>
    public static bool IsAbbreviationCandidate(string word, uint dictFlags) =>
        IsLetterString(word) && word.Length <= MaxSpelledLength && (dictFlags & FlagAbbreviation) != 0;

    private static bool IsLetterString(string word) => word.Length >= 2 && word.All(char.IsAsciiLetter);

    /// <summary>Returns the stressed codes for the spelled letters of <paramref name="word"/>, or null when a letter name
    /// is missing from the phoneme table (the caller then falls back to the rules).</summary>
    public static List<byte>? Spell(string word, EspeakPhonemeTable phon, EspeakStress stress)
    {
        List<byte> output = [];
        for (int i = 0; i < word.Length; i++)
        {
            List<byte>? codes = NameCodes(char.ToLowerInvariant(word[i]), phon);
            if (codes is null) return null;
            List<byte> named = stress.SetWordStress(codes, FlagFound, tonic: -1, control: 0);
            if (i < word.Length - 1) named = stress.ChangeWordStress(named, SecondaryStress);
            output.AddRange(named);
        }
        return output;
    }

    private static List<byte>? NameCodes(char letter, EspeakPhonemeTable phon)
    {
        List<byte> codes = [];
        foreach (string mnemonic in Names[letter - 'a'].Split(' '))
        {
            int code = phon.CodeForMnemonic(Pack(mnemonic));
            if (code == 0) return null;
            codes.Add((byte)code);
        }
        return codes;
    }

    // espeak packs a mnemonic into a word, first character in the low byte.
    private static uint Pack(string mnemonic)
    {
        uint value = 0;
        for (int i = 0; i < mnemonic.Length; i++)
            value |= (uint)mnemonic[i] << (8 * i);
        return value;
    }
}
