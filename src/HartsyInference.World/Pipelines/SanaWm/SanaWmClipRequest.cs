using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Everything one SANA-WM stage-1 clip is rendered from. Values arrive resolved: the frame count is on the
/// <c>8n + 1</c> grid and the camera path carries exactly one pose per frame.</summary>
public sealed record SanaWmClipRequest
{
    /// <summary>The seed image as RGB24, row-major, <see cref="ImageWidth"/> by <see cref="ImageHeight"/>.</summary>
    public required byte[] ImageRgb { get; init; }

    /// <summary>Width of the seed image in pixels.</summary>
    public required int ImageWidth { get; init; }

    /// <summary>Height of the seed image in pixels.</summary>
    public required int ImageHeight { get; init; }

    /// <summary>The prompt; the CHI prefix is added by the text path.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt for the unconditional branch.</summary>
    public string NegativePrompt { get; init; } = "";

    /// <summary>One camera-to-world pose per frame (16 row-major floats, OpenCV convention), frame 0 first.</summary>
    public required IReadOnlyList<float[]> CameraToWorld { get; init; }

    /// <summary>Intrinsics in seed-image pixels: one entry for every frame, or a per-frame sequence fitted over time.</summary>
    public required IReadOnlyList<PinholeIntrinsics> Intrinsics { get; init; }

    /// <summary>Frames to render, counting the seed frame: <c>8n + 1</c>.</summary>
    public required int Frames { get; init; }

    /// <summary>Denoising steps.</summary>
    public int Steps { get; init; } = SanaWmPipeline.DefaultSteps;

    /// <summary>Classifier-free guidance scale; at or below 1 only the conditional branch runs.</summary>
    public double Guidance { get; init; } = SanaWmPipeline.DefaultGuidance;

    /// <summary>Static schedule shift.</summary>
    public double FlowShift { get; init; } = SanaWmPipeline.DefaultFlowShift;

    /// <summary>Seed of the initial noise.</summary>
    public long Seed { get; init; }

    /// <summary>Runs the two guidance branches as two forwards per step instead of one batch-2 forward.</summary>
    public bool SequentialCfg { get; init; }

    /// <summary>Replaces the seeded initial latent, frame 0 included, with a host F32 <c>[1, 128, T, 22, 40]</c> tensor (for
    /// parity against a recorded upstream <c>z0</c>). The VAE encoder stage is skipped. Borrowed; the caller keeps it.</summary>
    public Tensor? InitialLatent { get; init; }

    /// <summary>Observes intermediates by name (<c>first_latent</c>, <c>z0</c>, <c>final_latent</c>). The tensors are borrowed
    /// for the duration of the call.</summary>
    public Action<string, Tensor>? Tap { get; init; }
}
