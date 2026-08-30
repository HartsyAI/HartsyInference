namespace HartsyInference.API.Endpoints;

/// <summary>Payload of the optional <c>audio</c> event from <c>/v1/native/video/stream</c>.</summary>
public sealed class NativeVideoAudioEvent
{
    /// <summary>PCM sample rate.</summary>
    public required int SampleRate { get; init; }

    /// <summary>Interleaved channel count.</summary>
    public required int Channels { get; init; }

    /// <summary>Base64-encoded WAV audio.</summary>
    public required string Wav { get; init; }
}
