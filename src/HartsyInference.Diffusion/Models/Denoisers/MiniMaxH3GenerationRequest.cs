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

    /// <summary>Resolved sampler name. Native PDD revalidates this at the execution boundary.</summary>
    public string Sampler { get; init; } = "euler";

    /// <summary>Resolved classifier-free guidance value. H3 acceleration profiles require one.</summary>
    public float CfgScale { get; init; } = 1f;

    /// <summary>Resolved PDD adapter strength. Ordinary dense generations leave this at one and ignore it.</summary>
    public float PddAdapterStrength { get; init; } = 1f;

    /// <summary>Whether planning selected the Hybrid FL+reference packing contract.</summary>
    public bool HybridProfile { get; init; }

    /// <summary>Target-relative visual/audio anchors; null is plain t2va or reference-only ref2va.</summary>
    public IReadOnlyList<MiniMaxH3Keyframe>? Keyframes { get; init; }

    /// <summary>Reference blocks for ref2va; null is plain t2va.</summary>
    public IReadOnlyList<MiniMaxH3RefBlock>? Refs { get; init; }

    /// <summary>Aligned target frame count used to validate already-resolved guide indices.</summary>
    public int? FrameCount { get; init; }

    /// <summary>Patchified conditioning video rows in packed-segment order, borrowed for the generation's lifetime.
    /// Must total the layout's non-target video rows — keyframe rows first, then each reference block's.</summary>
    public Tensor? CondVideoRows { get; init; }

    /// <summary>Channel-major conditioning audio rows in packed-segment order, borrowed for the generation's lifetime.</summary>
    public Tensor? CondAudioRows { get; init; }

    /// <summary>Pre-encoded Fun ControlNet streams; identical model indices share one registered branch.</summary>
    public IReadOnlyList<MiniMaxH3FunControlCondition>? Controls { get; init; }

    /// <summary>Upward-quantized target-video token mask in packed row order, where one generates and zero
    /// preserves. Each value is the spatial <c>amax</c> of one 2x2 latent patch. Null is the exact unmasked path.</summary>
    public IReadOnlyList<float>? VideoDenoiseMaskRows { get; init; }

    /// <summary>Raw continuous target-video mask flattened as <c>[row, patchArea]</c> in channel-outer packed-feature
    /// order. For H3's 1x2x2 patch this is <c>py</c>, then <c>px</c>. Null is the exact unmasked path.</summary>
    public IReadOnlyList<float>? VideoDenoiseFeatureMaskValues { get; init; }

    /// <summary>Packed source-video rows restored below one, borrowed for the generation's lifetime.</summary>
    public Tensor? VideoDenoiseSourceRows { get; init; }

    /// <summary>Upward-quantized channel-major target-audio token-mask rows. Null is the exact unmasked path.</summary>
    public IReadOnlyList<float>? AudioDenoiseMaskRows { get; init; }

    /// <summary>Raw continuous channel-major target-audio feature-mask rows. Null is the exact unmasked path.</summary>
    public IReadOnlyList<float>? AudioDenoiseFeatureMaskRows { get; init; }

    /// <summary>Channel-major source-audio rows restored below one, borrowed for the generation's lifetime.</summary>
    public Tensor? AudioDenoiseSourceRows { get; init; }

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
