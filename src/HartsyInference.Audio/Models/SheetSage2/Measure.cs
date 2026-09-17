namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>One bar, as a span of decoded beats plus the meter it will be notated under.
///
/// <para>The span is what the downbeats actually say; the notated meter is what gets written. They differ when a
/// bar is short — a pickup, a truncated final bar, or a span the model gave a beat ID pattern it then did not
/// keep to — and the difference is serialized as rest padding rather than as a wrong bar length.</para></summary>
/// <param name="Index">Position in the score, from 0.</param>
/// <param name="StartBeat">First beat of the span.</param>
/// <param name="EndBeat">One past the last beat of the span.</param>
/// <param name="Numerator">Beats the span really holds.</param>
/// <param name="Denominator">Which note value gets the beat.</param>
/// <param name="Pickup">Whether the span precedes the first downbeat.</param>
/// <param name="Partial">Whether the span is the truncated tail after the last downbeat.</param>
/// <param name="Inferred">Whether the span's meter was reconstructed rather than taken from agreeing beats.</param>
/// <param name="NotatedNumerator">Beats to notate when that differs from the span; null keeps
/// <paramref name="Numerator"/>.</param>
/// <param name="NotatedDenominator">Beat value to notate; null keeps <paramref name="Denominator"/>.</param>
/// <param name="PadBefore">Whether the padding rest goes before the bar's notes instead of after.</param>
public readonly record struct Measure(int Index, int StartBeat, int EndBeat, int Numerator, int Denominator,
    bool Pickup, bool Partial, bool Inferred, int? NotatedNumerator, int? NotatedDenominator, bool PadBefore)
{
    /// <summary>Beats the span holds.</summary>
    public int BeatCount => EndBeat - StartBeat;

    /// <summary>First subbeat of the span.</summary>
    public int StartT => StartBeat * AbcSerializer.SubbeatDivision;

    /// <summary>One past the last subbeat of the span.</summary>
    public int EndT => EndBeat * AbcSerializer.SubbeatDivision;

    /// <summary>Beats per bar as written in the ABC.</summary>
    public int AbcNumerator => NotatedNumerator ?? Numerator;

    /// <summary>Beat value as written in the ABC.</summary>
    public int AbcDenominator => NotatedDenominator ?? Denominator;
}
