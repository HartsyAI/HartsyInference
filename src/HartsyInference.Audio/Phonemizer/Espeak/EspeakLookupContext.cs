namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Where a word stands when it is looked up, which decides the conditional dictionary entries espeak-ng's
/// <c>LookupDict2</c> may use: its place in the clause (<c>FLAG_FIRST_WORD</c>/<c>FLAG_LAST_WORD</c>), its case, the
/// suffix removed from it (<c>end_flags</c>), and what the previous words lead the translator to expect
/// (<c>expect_noun</c>, <c>expect_verb</c>, <c>expect_past</c>).</summary>
internal readonly record struct EspeakLookupContext
{
    public bool AtStart { get; init; }
    public bool AtEnd { get; init; }
    public bool FirstUpper { get; init; }
    public bool AllUpper { get; init; }
    /// <summary>The clause ends a sentence (every clause the phonemizer sees ends at the end of its text).</summary>
    public bool Sentence { get; init; }
    public bool ExpectVerb { get; init; }
    public bool ExpectVerbS { get; init; }
    public bool ExpectNoun { get; init; }
    public bool ExpectPast { get; init; }
    /// <summary><c>end_flags</c>: <c>FLAG_SUFX</c>/<c>FLAG_SUFX_S</c> once a suffix is removed, <c>SUFX_P</c> for a prefix.</summary>
    public int EndFlags { get; init; }

    /// <summary>The rest of the clause after the word, its words separated by single spaces with one after the last,
    /// for multi-word entries.</summary>
    public string? NextWords { get; init; }

    /// <summary>A word spoken on its own: the whole clause.</summary>
    public static EspeakLookupContext SingleWord { get; } = new() { AtStart = true, AtEnd = true, Sentence = true };
}
