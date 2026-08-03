namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Geometry and sampling knobs for one MiniMax-H3 generation. Pixel dimensions must already be snapped to
/// the VAE's 16x spatial compression; latent frame counts are supplied directly because the audio and video streams
/// advance on different rates.</summary>
public sealed record MiniMaxH3GenerationRequest
{
    public required int Width { get; init; }

    public required int Height { get; init; }

    /// <summary>Video latent frames (pixel frames / 4, the VAE's temporal compression).</summary>
    public required int LatentFrames { get; init; }

    /// <summary>Audio latent frames at 40 Hz — 800 samples each at 32 kHz.</summary>
    public required int AudioLatentFrames { get; init; }

    public int Steps { get; init; } = 30;

    public int Seed { get; init; }

    public float SigmaShiftVideo { get; init; } = MiniMaxH3Schedule.DefaultShiftVideo;

    public float SigmaShiftAudio { get; init; } = MiniMaxH3Schedule.DefaultShiftAudio;

    /// <summary>Audio latent frames covering <paramref name="pixelFrames"/> at <paramref name="fps"/>.</summary>
    public static int AudioFramesFor(int pixelFrames, double fps, int audioLatentRate = 40) =>
        Math.Max(1, (int)Math.Round(pixelFrames / fps * audioLatentRate));
}
