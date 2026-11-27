namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>A rendered SANA-WM clip and how it was made. Frames are RGB24, row-major, frame 0 first; writing a container is
/// left to the caller.</summary>
public sealed record SanaWmClip
{
    /// <summary>One RGB24 buffer per frame.</summary>
    public required IReadOnlyList<byte[]> Frames { get; init; }

    /// <summary>Frame width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Frame height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Frame rate the camera path was integrated at; the clip plays back at this rate.</summary>
    public required int Fps { get; init; }

    /// <summary>Latent frames sampled.</summary>
    public required int LatentFrames { get; init; }

    /// <summary>Denoising steps run.</summary>
    public required int Steps { get; init; }

    /// <summary>Guidance scale used.</summary>
    public required double Guidance { get; init; }

    /// <summary>Schedule shift used.</summary>
    public required double FlowShift { get; init; }

    /// <summary>Seed of the initial noise; meaningless when the initial latent was injected.</summary>
    public required long Seed { get; init; }

    /// <summary>Whether the initial latent was injected rather than drawn and encoded.</summary>
    public required bool InitialLatentInjected { get; init; }
}
