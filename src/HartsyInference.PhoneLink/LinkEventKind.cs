namespace HartsyInference.PhoneLink;

/// <summary>Discriminator of <see cref="LinkEventMessage"/>; serialized by name.</summary>
public enum LinkEventKind
{
    /// <summary>The session moved to <see cref="LinkEventMessage.State"/>.</summary>
    State,
    /// <summary>An in-progress transcript of the caller's current utterance in <see cref="LinkEventMessage.Text"/>.</summary>
    TranscriptPartial,
    /// <summary>The final transcript of an utterance in <see cref="LinkEventMessage.Text"/>.</summary>
    TranscriptFinal,
    /// <summary>Per-turn latency breakdown in <see cref="LinkEventMessage.Latency"/>.</summary>
    TurnLatency,
}
