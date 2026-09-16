namespace HartsyInference.Engine.Requests;

/// <summary>The symbolic plan a music model writes before it renders any audio, plus what the context left for
/// the audio itself.
///
/// <para>YuE2 composes in two passes: an autoregressive model writes an ABC score, then a second pass turns that
/// score into sound. The score is the only editable artifact the model exposes, and planning it costs seconds
/// against minutes for the full render — so it is worth asking for on its own, editing, and feeding back through
/// <see cref="MusicRequest.Yue2Abc"/>.</para></summary>
public sealed record ScorePlanResult
{
    /// <summary>The score, as ABC notation. Empty when the model was asked not to plan one.</summary>
    public required string Abc { get; init; }

    /// <summary>True when planning stopped at its token ceiling rather than finishing the score.</summary>
    public bool Truncated { get; init; }

    /// <summary>Tokens the score itself occupies.</summary>
    public int ScoreTokens { get; init; }

    /// <summary>Tokens the whole conditioning prefix occupies — instruction, style, lyrics and the score.</summary>
    public int PrefixTokens { get; init; }

    /// <summary>Tokens left for audio behind that prefix.</summary>
    public int BudgetTokens { get; init; }

    /// <summary>Seconds of audio <see cref="BudgetTokens"/> buys. Less than the duration asked for means the
    /// prompt and score ate the context, which is otherwise indistinguishable from a model ending early.</summary>
    public double BudgetSeconds { get; init; }
}
