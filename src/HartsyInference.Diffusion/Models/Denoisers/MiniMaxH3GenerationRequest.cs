using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Geometry and sampling knobs for one MiniMax-H3 generation. Pixel dimensions must already be snapped to
/// the VAE's 16x spatial compression; latent frame counts are supplied directly because the audio and video streams
/// advance on different rates.</summary>
public sealed record MiniMaxH3GenerationRequest
{
    public required int Width { get; init; }

    public required int Height { get; init; }

    /// <summary>Video latent frames — derive with <see cref="MiniMaxH3Geometry.VideoLatentFrames"/>; it is NOT
    /// pixel frames / 4.</summary>
    public required int LatentFrames { get; init; }

    /// <summary>Audio latent frames at 40 Hz — 800 samples each at 32 kHz.</summary>
    public required int AudioLatentFrames { get; init; }

    public int Steps { get; init; } = 30;

    public int Seed { get; init; }

    public float SigmaShiftVideo { get; init; } = MiniMaxH3Schedule.DefaultShiftVideo;

    public float SigmaShiftAudio { get; init; } = MiniMaxH3Schedule.DefaultShiftAudio;

    /// <summary>First/last-frame anchors for fl2va; null is plain t2va.</summary>
    public IReadOnlyList<MiniMaxH3Keyframe>? Keyframes { get; init; }

    /// <summary>Reference blocks for ref2va; null is plain t2va.</summary>
    public IReadOnlyList<MiniMaxH3RefBlock>? Refs { get; init; }

    /// <summary>Aligned pixel frame count, required only to resolve a last-frame keyframe's anchor.</summary>
    public int? FrameCount { get; init; }

    /// <summary>Patchified conditioning video rows in packed-segment order, borrowed for the generation's lifetime.
    /// Must total the layout's non-target video rows — keyframe rows first, then each reference block's.</summary>
    public Tensor? CondVideoRows { get; init; }

    /// <summary>Channel-major conditioning audio rows in packed-segment order, borrowed for the generation's lifetime.</summary>
    public Tensor? CondAudioRows { get; init; }

    /// <summary>Blend toward noise applied to visual conditioning rows; also the timestep they modulate at. Below 1
    /// the reference blends in noise to keep conditioning from being trusted as perfectly clean.</summary>
    public float VisualCondNoiseAug { get; init; } = MiniMaxH3Schedule.VisualCondTimestep;

    /// <summary>Blend toward noise applied to audio conditioning rows; 1.0 leaves them untouched.</summary>
    public float AudioCondNoiseAug { get; init; } = MiniMaxH3Schedule.AudioCondTimestep;

    /// <summary>Audio latent frames covering <paramref name="pixelFrames"/> at <paramref name="fps"/>. Pass the
    /// <see cref="MiniMaxH3Geometry.AlignFrameCount"/>ed count — sizing this from the caller's raw request generates
    /// audio past the end of the video, which then gets trimmed away.</summary>
    public static int AudioFramesFor(int pixelFrames, double fps, int audioLatentRate = 40) =>
        Math.Max(1, (int)Math.Round(pixelFrames / fps * audioLatentRate));
}
