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
    // Dictionary flag bit 24: an abbreviation entry the rules still read (st is read as sənt).
    private const uint FlagAbbreviation = 0x01000000;
    private const int SecondaryStress = 3;

    // A short all-letter word with no vowel letter cannot be read as a word, so espeak spells it out. Longer words go
    // to the letter-to-sound rules.
    private const int MaxVowelessLength = 4;
    private const string VowelLetters = "aeiouyAEIOUY";

    /// <summary>True when espeak spells <paramref name="word"/> out rather than reading it, given that the dictionary
    /// has no pronunciation for it. <paramref name="dictFlags"/> are the flags of a flag-only dictionary entry (0 when
    /// the word has no entry): an entry flagged for spelling is spelled, an abbreviation entry is read by the rules, and
    /// otherwise a short word with no vowel letter is spelled.</summary>
    public static bool IsSpelled(string word, uint dictFlags)
    {
        if (word.Length < 2 || !word.All(char.IsAsciiLetter)) return false;
        if ((dictFlags & FlagAbbreviation) != 0) return false;
        if ((dictFlags & FlagSpellWord) != 0) return true;
        return word.Length <= MaxVowelessLength && !word.Any(c => VowelLetters.Contains(c));
    }

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
