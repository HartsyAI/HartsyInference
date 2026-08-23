namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Per-language letter classification, ported from the default Latin setup in espeak-ng <c>NewTranslator</c> (tr_languages.c); the rule interpreter asks "is this letter a vowel / hard consonant / front vowel?" through <see cref="IsLetter"/>, answered from a 256-entry bit table whose groups 0-7 correspond to the <c>A B C H F G Y</c> rule classes plus the include-y vowel set. English uses these defaults unchanged.</summary>
internal sealed class EspeakLetters
{
    private const int RemoveAccentBase = 0xc0;

    private readonly byte[] _letterBits = new byte[256];

    /// <summary>Offset applied to letters before indexing <see cref="_letterBits"/> for non-Latin alphabets; 0 for Latin/English so codepoints index the table directly.</summary>
    public int LetterBitsOffset { get; }

    private EspeakLetters()
    {
        LetterBitsOffset = 0;
        // 0-7 sets matched by A B C H F G Y in pronunciation rules (default Latin, NewTranslator).
        SetLetterBits(0, "aeiou");                 // A  vowels, except y
        SetLetterBits(1, "bcdfgjklmnpqstvxz");     // B  hard consonants, excluding h,r,w
        SetLetterBits(2, "bcdfghjklmnpqrstvwxz");  // C  all consonants
        SetLetterBits(3, "hlmnr");                 // H  soft consonants
        SetLetterBits(4, "cfhkpqstx");             // F  voiceless consonants
        SetLetterBits(5, "bdgjlmnrvwyz");          // G  voiced
        SetLetterBits(6, "eiy");                   // Y  front vowels
        SetLetterBits(7, "aeiouy");                // vowels, including y (LETTERGP_VOWEL2)
    }

    /// <summary>The default Latin classification used by English and other Latin-script languages.</summary>
    public static EspeakLetters Latin() => new();

    /// <summary>Port of <c>IsLetter</c>: returns true when <paramref name="letter"/> belongs to rule class <paramref name="group"/> (0-7); accented Latin letters fold to their base letter first.</summary>
    public bool IsLetter(int letter, int group)
    {
        if (group > 7)
            return false;

        if (LetterBitsOffset > 0)
        {
            int letter2 = letter - LetterBitsOffset;
            if (letter2 > 0 && letter2 < 0x100)
                letter = letter2;
            else
                return false;
        }
        else if (letter >= RemoveAccentBase && letter < EspeakAccents.RemoveAccentCount)
        {
            int baseLetter = EspeakAccents.RemoveAccent[letter - RemoveAccentBase];
            return baseLetter != 0 && (_letterBits[baseLetter] & (1 << group)) != 0;
        }

        if (letter >= 0 && letter < 0x100)
            return (_letterBits[letter] & (1 << group)) != 0;

        return false;
    }

    /// <summary>Port of <c>IsVowel</c>: a letter is a vowel if it is in group <see cref="EspeakRuleCodes.LetterGpVowel2"/>.</summary>
    public bool IsVowel(int letter) => IsLetter(letter, EspeakRuleCodes.LetterGpVowel2);

    private void SetLetterBits(int group, string letters)
    {
        int bits = 1 << group;
        foreach (char c in letters)
            _letterBits[c] |= (byte)bits;
    }
}
