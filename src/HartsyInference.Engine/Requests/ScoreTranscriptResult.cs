namespace HartsyInference.Engine.Requests;

/// <summary>The result of transcribing music into a symbolic score: the same transcription written out as ABC
/// twice, once with chord symbols and once without.
///
/// <para>Both renderings come from one decode. The decode is the whole cost — the encoder attends over a fixed
/// 300-second window and the token loop is autoregressive — while serializing the decoded events is free, so
/// asking for one mode and re-running for the other would pay for the model twice to obtain something already
/// in hand.</para>
///
/// <para><b>The two are not a substitution apart.</b> Stripping the quoted chords out of <see cref="FullAbc"/>
/// does not yield <see cref="MelodyAbc"/>: a bar carrying a chord symbol cannot fold into a multi-bar rest, a
/// chord change inside a held note splits it into tied parts, and the two spell rests within a bar differently.
/// Both are valid, and which one to render under depends on the planning mode the score is going to be fed to,
/// so both are returned rather than one being derived from the other.</para></summary>
public sealed record ScoreTranscriptResult
{
    /// <summary>The score with its chord symbols, for rendering under chord-annotated planning.</summary>
    public required string FullAbc { get; init; }

    /// <summary>The score without chord symbols, for rendering under melody-only planning (covers).</summary>
    public required string MelodyAbc { get; init; }

    /// <summary>Seconds of audio the score covers.</summary>
    public required double Duration { get; init; }

    /// <summary>How many sliding windows the clip was read in; more than one means the result was stitched.</summary>
    public required int WindowCount { get; init; }

    /// <summary>True when a window hit the decoder's token limit, so the score stops short of the audio.</summary>
    public bool Truncated { get; init; }
}
