namespace HartsyInference.PhoneGateway.Transport;

/// <summary>Receives one <c>OutboundAudio</c> frame on the link reader thread. The samples are borrowed from the reader's
/// scratch buffer and are dead when the handler returns.</summary>
public delegate void OutboundAudioHandler(uint callId, uint turnId, ReadOnlySpan<short> pcm);
