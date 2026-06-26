namespace HartsyInference.Phonemizer.Espeak;

/// <summary>A successful dictionary word-list match: the stored phoneme code bytes plus the two dictionary flag sets
/// espeak tracks (<c>dictionary_flags</c> and <c>dictionary_flags2</c>), which carry the explicit stress level and
/// special-attribute bits used when placing stress on the word.</summary>
internal readonly struct EspeakLookupResult
{
    public EspeakLookupResult(List<byte> phonemes, uint flags, uint flags2)
    {
        Phonemes = phonemes;
        Flags = flags;
        Flags2 = flags2;
    }

    /// <summary>Stored phoneme code bytes (may be empty when the entry only sets flags, e.g. text replacements).</summary>
    public List<byte> Phonemes { get; }

    /// <summary>Primary dictionary flags; the low 4 bits hold the explicit stress position.</summary>
    public uint Flags { get; }

    /// <summary>Secondary dictionary flags (<c>dictionary_flags2</c>).</summary>
    public uint Flags2 { get; }
}
