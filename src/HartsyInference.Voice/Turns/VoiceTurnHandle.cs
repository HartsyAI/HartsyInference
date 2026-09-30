namespace HartsyInference.Voice.Turns;

/// <summary>A running turn's id paired with its cancellation, published to the audio thread so a barge-in cancels
/// exactly the turn it heard and never the one that follows it.</summary>
internal sealed class VoiceTurnHandle(int turnId, CancellationTokenSource cancellation)
{
    /// <summary>The turn.</summary>
    public int TurnId { get; } = turnId;

    /// <summary>Its cancellation source; never disposed while the handle is published.</summary>
    public CancellationTokenSource Cancellation { get; } = cancellation;
}
