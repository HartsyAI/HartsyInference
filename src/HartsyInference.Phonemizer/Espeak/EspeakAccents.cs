namespace HartsyInference.Phonemizer.Espeak;

/// <summary>The <c>remove_accent</c> table from espeak-ng v1.50 <c>dictionary.c</c>, mapping accented Latin
/// codepoints (starting at U+00C0) to their base ASCII letter, or 0 when there is no base letter. Used by
/// <see cref="EspeakLetters.IsLetter"/> so accented letters classify the same as their base (e.g. <c>e</c> is a
/// vowel). Generated verbatim from the source table (414 entries); a 0 entry means "no base letter".</summary>
internal static class EspeakAccents
{
    /// <summary>Upper bound (exclusive) on codepoints handled by the table; espeak's <c>N_REMOVE_ACCENT</c>.</summary>
    public const int RemoveAccentCount = 0x25e;

    private static readonly byte[] _table = System.Text.Encoding.Latin1.GetBytes(
        "aaaaaaaceeeeiiiidnooooo\0ouuuuyts" +
        "aaaaaaaceeeeiiiidnooooo\0ouuuuyty" +
        "aaaaaaccccccccddddeeeeeeeeeegggg" +
        "gggghhhhiiiiiiiiiiiijjkkklllllll" +
        "lllnnnnnnnnnoooooooorrrrrrssssss" +
        "ssttttttuuuuuuuuuuuuwwyyyzzzzzzs" +
        "bbbb\0\0occdddddeeeffgghiikkllmnno" +
        "ooooppy\0\0ssttttuuuvyyzzzzzzz\0\0\0w" +
        "tttkdddlllnnnaaiioouuuuuuuuuueaa" +
        "aaaaggggkkoooozzjdddggwwnnaaaaoo" +
        "aaaaeeeeiiiioooorrrruuuussttyyhh" +
        "ndoozzaaeeooooooooyylntjdqacclts" +
        "z\0\0buveejjqqrryyaaabocddeeeeee");

    /// <summary>Accent-folding table indexed by <c>codepoint - 0xC0</c>.</summary>
    public static System.ReadOnlySpan<byte> RemoveAccent => _table;
}
