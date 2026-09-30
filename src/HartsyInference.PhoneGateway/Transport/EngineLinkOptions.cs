namespace HartsyInference.PhoneGateway.Transport;

/// <summary>Connection, liveness and queue settings for <see cref="EngineLink"/>. Defaults are the protocol's constants.</summary>
public sealed record EngineLinkOptions
{
    /// <summary>Unix domain socket the voice host listens on.</summary>
    public required string SocketPath { get; init; }

    /// <summary>Shared token carried in <c>Hello</c>; empty when the host does not require one.</summary>
    public string Token { get; init; } = "";

    public int PingIntervalMs { get; init; } = 5000;

    /// <summary>A connection with no frame at all for this long is closed and redialled.</summary>
    public int LivenessTimeoutMs { get; init; } = 20000;

    public int ReconnectBaseMs { get; init; } = 250;

    public int ReconnectCapMs { get; init; } = 30000;

    /// <summary>Inbound audio frames queued for the writer; the oldest is dropped when the lane is full.</summary>
    public int AudioLaneDepth { get; init; } = 10;

    /// <summary>Control frames queued for the writer; a full lane makes the sender wait, never drops.</summary>
    public int ControlLaneDepth { get; init; } = 64;

    /// <summary>How long a control sender waits for room before the link is declared wedged.</summary>
    public int ControlEnqueueTimeoutMs { get; init; } = 5000;
}
