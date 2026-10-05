namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>One code point's entry in MeCab's <c>char.bin</c>: a 32-bit field packing the category bit set (18 bits),
/// the default category id (8), the unknown-word length (4), and the group and invoke flags (1 each), low bits first.</summary>
internal readonly record struct MeCabCharInfo(uint Bits)
{
    /// <summary>Bit set of every category the character belongs to (bit <c>i</c> = category id <c>i</c>).</summary>
    public uint Type => Bits & 0x3FFFF;

    /// <summary>Id of the first category the character was listed under; picks the unknown-word entries.</summary>
    public int DefaultType => (int)((Bits >> 18) & 0xFF);

    /// <summary>Longest run (in characters) emitted as separate unknown-word candidates.</summary>
    public int Length => (int)((Bits >> 26) & 0xF);

    /// <summary>Whether a whole run of same-kind characters is also offered as one unknown word.</summary>
    public bool Group => ((Bits >> 30) & 1) != 0;

    /// <summary>Whether unknown-word candidates are added even when the dictionary matched.</summary>
    public bool Invoke => (Bits >> 31) != 0;

    /// <summary>MeCab's <c>isKindOf</c>: the two characters share at least one category.</summary>
    public bool IsKindOf(MeCabCharInfo other) => (Type & other.Type) != 0;
}
