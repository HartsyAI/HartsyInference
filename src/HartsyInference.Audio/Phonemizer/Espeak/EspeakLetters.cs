namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Per-language letter classification, ported from the default Latin setup in espeak-ng <c>NewTranslator</c> (tr_languages.c); the rule interpreter asks "is this letter a vowel / hard consonant / front vowel?" through <see cref="IsLetter"/>, answered from a 256-entry bit table whose groups 0-7 correspond to the <c>A B C H F G Y</c> rule classes plus the include-y vowel set. English uses these defaults unchanged.</summary>
internal sealed class EspeakLetters
{
    private const int RemoveAccentBase = 0xc0;

    private readonly byte[] _letterBits = new byte[256];

    /// <summary>Offset applied to letters before indexing <see cref="_letterBits"/> for non-Latin alphabets; 0 for Latin/English so codepoints index the table directly.</summary>
    public int LetterBitsOffset { get; private set; }

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

    /// <summary>The classification <c>SelectTranslator</c> sets up for <paramref name="language"/>'s base name
    /// (<c>es</c>, <c>fr</c>, <c>it</c>, <c>pt</c>, <c>hi</c>, ...); languages it does not customize keep the Latin
    /// default.</summary>
    public static EspeakLetters For(string language)
    {
        EspeakLetters letters = new();
        switch (language.Split('-')[0])
        {
            case "fr" or "it":
                letters.SetLetterVowel('y');
                break;
            case "pt":
                letters.SetLetterVowel('y');
                letters.ResetLetterBits(0x2);
                letters.SetLetterBits(1, "bcdfgjkmnpqstvxz"); // B hard consonants, excluding h,l,r,w,y
                break;
            case "hi":
                letters.SetIndicLetters(OffsetDevanagari);
                break;
        }
        return letters;
    }

    private const int OffsetDevanagari = 0x900;

    // SetLetterVowel: a vowel in groups A and vowel2, keeping its front-vowel (Y) bit.
    private void SetLetterVowel(char c) => _letterBits[c] = (byte)((_letterBits[c] & 0x40) | 0x81);

    private void ResetLetterBits(int groups)
    {
        for (int i = 0; i < _letterBits.Length; i++) _letterBits[i] &= (byte)~groups;
    }

    private void SetLetterBitsRange(int group, int first, int last)
    {
        for (int i = first; i <= last; i++) _letterBits[i] |= (byte)(1 << group);
    }

    // SetIndicLetters (tr_languages.c): Devanagari classes indexed by codepoint minus the script offset.
    private void SetIndicLetters(int offset)
    {
        ReadOnlySpan<byte> consonants2 = [0x02, 0x03, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x7b, 0x7c, 0x7e, 0x7f];
        ReadOnlySpan<byte> vowels2 = [0x60, 0x61, 0x55, 0x56, 0x57, 0x62, 0x63];
        LetterBitsOffset = offset;
        Array.Clear(_letterBits);
        SetLetterBitsRange(0, 0x04, 0x14);
        SetLetterBitsRange(0, 0x3e, 0x4d);
        foreach (byte c in vowels2) _letterBits[c] |= 1 << 0;
        SetLetterBitsRange(1, 0x3e, 0x4d);
        foreach (byte c in vowels2) _letterBits[c] |= 1 << 1;
        SetLetterBitsRange(2, 0x15, 0x39);
        foreach (byte c in consonants2) _letterBits[c] |= 1 << 2;
        SetLetterBitsRange(6, 0x04, 0x14);
        SetLetterBitsRange(6, 0x3e, 0x4c);
        foreach (byte c in vowels2) _letterBits[c] |= 1 << 6;
    }

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
