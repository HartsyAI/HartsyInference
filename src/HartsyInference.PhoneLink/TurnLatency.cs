namespace HartsyInference.PhoneLink;

/// <summary>Milliseconds spent in each stage of one turn, carried by a <see cref="LinkEventKind.TurnLatency"/> event. Stages
/// that did not run are null.</summary>
public sealed record TurnLatency
{
    /// <summary>End of speech to transcript.</summary>
    public int? SttMs { get; init; }

    /// <summary>Transcript to first LLM token.</summary>
    public int? LlmFirstTokenMs { get; init; }

    /// <summary>Transcript to first complete sentence.</summary>
    public int? LlmFirstSentenceMs { get; init; }

    /// <summary>First sentence to first synthesized audio chunk.</summary>
    public int? TtsFirstChunkMs { get; init; }

    /// <summary>Resample, link and gateway prebuffer.</summary>
    public int? TransportMs { get; init; }

    /// <summary>End of speech to the first outbound frame on the wire.</summary>
    public required int TotalMs { get; init; }
}
