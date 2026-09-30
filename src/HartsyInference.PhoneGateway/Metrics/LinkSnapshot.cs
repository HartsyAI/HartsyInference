namespace HartsyInference.PhoneGateway.Metrics;

/// <summary>The voice-host link's counters at one moment.</summary>
public readonly record struct LinkSnapshot(
    bool Connected, uint OutboundRate, long Reconnects, double RttMs,
    long AudioLaneDropped, long InboundDroppedWhileDown, long StaleOutboundDropped, long FramesSent, long FramesReceived);
