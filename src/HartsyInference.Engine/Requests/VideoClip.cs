namespace HartsyInference.Engine.Requests;

/// <summary>Engine-native video payload: encoded container bytes (MP4/AVI/MKV/…) plus an optional format hint — the video-input mirror of <see cref="AudioClip"/>. Services decode via the ffmpeg child-process decoder (subprocess, not a native-library dependency).</summary>
public sealed record VideoClip
{
    /// <summary>Encoded video container bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container hint (e.g. "mp4", "avi"); null lets ffmpeg sniff it.</summary>
    public string? Format { get; init; }
}
