namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>A transcription resolved onto a subbeat grid, one step before it is written out as ABC.
///
/// <para>Every per-subbeat array is the same length — <c>SubbeatDivision</c> steps per beat interval plus a final
/// step for the last beat — so a subbeat index reads the key, the chord and both voices at once. Voices hold a
/// sustain code rather than a pitch: 0 is silence, <c>pitch * 2 + 2</c> continues a note and <c>pitch * 2 + 3</c>
/// starts one, which is what lets a tie be told from a repeat.</para></summary>
public sealed class RebuiltAbcScore
{
    /// <summary>The decoded beat grid, extended to the end of the clip.</summary>
    public required IReadOnlyList<BeatEvent> Beats { get; init; }

    /// <summary>Bars laid out over <see cref="Beats"/>.</summary>
    public required IReadOnlyList<Measure> Measures { get; init; }

    /// <summary>Seconds into the clip for each subbeat.</summary>
    public required double[] SubbeatTimes { get; init; }

    /// <summary>Quarter notes elapsed at each subbeat, which is what the tempo estimate divides.</summary>
    public required double[] SubbeatQuarters { get; init; }

    /// <summary>Beat value in force at each subbeat, taken from the bar it falls in.</summary>
    public required int[] SubbeatDenominators { get; init; }

    /// <summary>ABC key signature in force at each subbeat.</summary>
    public required string[] KeyArr { get; init; }

    /// <summary>Chord label in force at each subbeat; "N" is no chord.</summary>
    public required string[] ChordArr { get; init; }

    /// <summary>Section labels, each at the subbeat it starts on.</summary>
    public required IReadOnlyList<(int Subbeat, string Label)> StructureEvents { get; init; }

    /// <summary>Sustain codes per subbeat, keyed by voice id.</summary>
    public required IReadOnlyDictionary<string, int[]> VoiceArrs { get; init; }

    /// <summary>What <see cref="AbcSerializer.InferMeasures"/> had to reconstruct, in the order it did so.</summary>
    public required IReadOnlyList<string> Diagnostics { get; init; }

    /// <summary>Subbeats per beat.</summary>
    public int SubbeatDiv { get; init; } = AbcSerializer.SubbeatDivision;
}
