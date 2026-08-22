namespace HartsyInference.Engine.Requests;

/// <summary>Engine-native audio payload: encoded container bytes (WAV/MP3/FLAC/…) plus an optional format hint. Used for request inputs (voice references, source audio, continuation/cover clips) and mux tracks; the audio services decode it to the sample rate their pipeline needs.</summary>
public sealed record AudioClip
{
    /// <summary>Encoded audio container bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container/codec hint (e.g. "wav", "mp3"); null lets the decoder sniff it.</summary>
    public string? Format { get; init; }
}
