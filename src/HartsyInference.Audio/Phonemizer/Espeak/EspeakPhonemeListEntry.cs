namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>One entry in the clause phoneme list (espeak <c>PHONEME_LIST</c>), the context over which the bytecode
/// interpreter runs. Holds the current phoneme, its resolved table entry, stress level, and word-boundary marker that
/// the condition tests (prevPh/nextPh/word-boundary) consult.</summary>
internal sealed class EspeakPhonemeListEntry
{
    /// <summary>Phoneme code (index into the active phoneme table).</summary>
    public int PhCode { get; set; }

    /// <summary>Resolved phoneme-table entry for <see cref="PhCode"/>.</summary>
    public EspeakPhoneme Ph { get; set; }

    /// <summary>Phoneme type (vowel/stress/stop/…) cached from <see cref="Ph"/>.</summary>
    public byte Type { get; set; }

    /// <summary>Stress level (0 = diminished … up). Set from the word's vowel stresses.</summary>
    public int StressLevel { get; set; }

    /// <summary>The word's maximum stress level (espeak <c>wordstress</c>), consulted by stress conditions.</summary>
    public int WordStress { get; set; }

    /// <summary>Non-zero at the first phoneme of a word (the source character index); used as the word boundary.</summary>
    public int SourceIx { get; set; }

    /// <summary>Synthesis flags (e.g. SFLAG_SYLLABLE) toggled when a change makes a phoneme a vowel/non-vowel.</summary>
    public int SynthFlags { get; set; }

    public EspeakPhonemeListEntry(int phCode, EspeakPhoneme ph)
    {
        PhCode = phCode;
        Ph = ph;
        Type = ph.Type;
    }

    public EspeakPhonemeListEntry Clone() => new(PhCode, Ph)
    {
        Type = Type,
        StressLevel = StressLevel,
        SourceIx = SourceIx,
        SynthFlags = SynthFlags,
    };
}
