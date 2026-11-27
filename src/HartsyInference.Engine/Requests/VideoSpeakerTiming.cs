namespace HartsyInference.Engine.Requests;

/// <summary>How two speakers' audio is laid on the clip's timeline.</summary>
public enum VideoSpeakerTiming
{
    /// <summary>Both speak from the start, and the soundtrack is the two recordings added together. The speech ends with the shorter
    /// recording.</summary>
    Parallel,

    /// <summary>The second speaker starts when the first one's recording ends (the first is cut to the clip plus one second).</summary>
    Sequential,
}
