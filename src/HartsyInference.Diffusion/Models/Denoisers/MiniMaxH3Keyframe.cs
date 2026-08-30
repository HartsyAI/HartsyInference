namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A visual and/or audio condition anchored to a resolved target-frame position.</summary>
public sealed record MiniMaxH3Keyframe
{
    /// <summary>Zero-based target-frame position after negative indices have been resolved by the caller.</summary>
    public required int ResolvedFrameIndex { get; init; }

    /// <summary>Conditioning video tokens carried by this anchor; zero represents an audio-only guide.</summary>
    public int VideoLatentFrames { get; init; } = 1;

    /// <summary>Conditioning audio rows per stereo channel; zero represents a visual-only guide.</summary>
    public int AudioLatentFrames { get; init; }
}
