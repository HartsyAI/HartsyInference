namespace HartsyInference.Phonemizer.Espeak;

/// <summary>Result of matching one dictionary rule against the input word, mirroring espeak-ng's <c>MatchRecord</c>.
/// <see cref="PhonemesOffset"/> is an index into the dictionary rules data where the matched phoneme code bytes begin
/// (read until a 0 terminator), or -1 for an empty match.</summary>
internal struct EspeakMatchRecord
{
    /// <summary>Score of this match; the highest-scoring rule in a group wins.</summary>
    public int Points;

    /// <summary>Offset into the dictionary data of the matched phoneme code-byte string, or -1 when empty.</summary>
    public int PhonemesOffset;

    /// <summary>Suffix/ending type bits discovered while matching (<c>SUFX_*</c>), or 0.</summary>
    public int EndType;

    /// <summary>Index in the word buffer of an 'e' to replace with the silent-e marker, or -1.</summary>
    public int DelFwd;
}
