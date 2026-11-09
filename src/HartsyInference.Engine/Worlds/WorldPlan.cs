using System.Globalization;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>A world request after validation: every default filled in, every value checked against the model's
/// capabilities, and the effective seed recorded. <see cref="Request"/> is the resolved request a session is opened on.</summary>
public sealed record WorldPlan
{
    /// <summary>The catalog id, or the requested token for a catalog-less model.</summary>
    public required string ModelId { get; init; }

    /// <summary>The backend kind the rollout runs on (<c>cpu</c>, <c>cuda</c>, <c>vulkan</c>).</summary>
    public required string Backend { get; init; }

    /// <summary>The precision the rollout runs at.</summary>
    public required string Precision { get; init; }

    /// <summary>The frame count the caller asked for, before normalization.</summary>
    public required int RequestedFrames { get; init; }

    /// <summary>The frames the rollout produces, counting the seed frame.</summary>
    public required int Frames { get; init; }

    /// <summary>Output frame rate.</summary>
    public required int Fps { get; init; }

    /// <summary>Denoising steps per frame; zero when the pipeline fixes its own count.</summary>
    public required int Steps { get; init; }

    /// <summary>Guidance scale as requested; null means the model default.</summary>
    public double? Guidance { get; init; }

    /// <summary>Flow shift as requested; null means the model default.</summary>
    public double? FlowShift { get; init; }

    /// <summary>Sampler as requested; null means the model's own sampler.</summary>
    public string? Sampler { get; init; }

    /// <summary>Whether the refiner stage runs.</summary>
    public bool Refiner { get; init; }

    /// <summary>Whether partial frames stream while each one denoises.</summary>
    public bool ProgressivePreview { get; init; }

    /// <summary>The effective seed the rollout uses; a request with no seed records the one drawn for it.</summary>
    public required long Seed { get; init; }

    /// <summary>Whether <see cref="Seed"/> was drawn because the request named none.</summary>
    public bool SeedDrawn { get; init; }

    /// <summary>Camera poses the rollout uses, one per frame; zero when the model takes no camera path.</summary>
    public int CameraPoses { get; init; }

    /// <summary>The pose count the camera path carried when it was longer than the frame count and was truncated.</summary>
    public int? CameraTruncatedFrom { get; init; }

    /// <summary>The intrinsics the rollout projects with, validated against the seed image.</summary>
    public WorldIntrinsics? Intrinsics { get; init; }

    /// <summary>Human-readable adjustments made while resolving (normalized frame counts, ignored steps, truncated poses).</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];

    /// <summary>The resolved request a session is opened on: the same request with every resolved value written back.</summary>
    public required WorldRequest Request { get; init; }

    /// <summary>Actions a rollout is driven with: one per frame after the seed frame.</summary>
    public int ActionCount => Frames - 1;

    /// <summary>Presentation timestamp of a frame, in milliseconds, derived from its index and the plan's frame rate.</summary>
    public static long FrameTimestampMs(int index, int fps) => (long)Math.Round(index * 1000.0 / fps, MidpointRounding.AwayFromZero);

    /// <summary>A deterministic, line-per-field rendering of the plan (without the frame pixels), used to compare plans
    /// produced by different entry points.</summary>
    public string Snapshot()
    {
        List<string> lines =
        [
            $"model={ModelId}",
            $"backend={Backend}",
            $"precision={Precision}",
            $"requestedFrames={RequestedFrames}",
            $"frames={Frames}",
            $"fps={Fps}",
            $"steps={Steps}",
            $"guidance={Format(Guidance)}",
            $"flowShift={Format(FlowShift)}",
            $"sampler={Sampler ?? "-"}",
            $"refiner={Refiner}",
            $"progressivePreview={ProgressivePreview}",
            $"seed={Seed}",
            $"seedDrawn={SeedDrawn}",
            $"cameraPoses={CameraPoses}",
            $"cameraTruncatedFrom={CameraTruncatedFrom?.ToString(CultureInfo.InvariantCulture) ?? "-"}",
            $"intrinsics={FormatIntrinsics(Intrinsics)}",
        ];
        foreach (string note in Notes)
        {
            lines.Add($"note={note}");
        }
        return string.Join('\n', lines);
    }

    private static string Format(double? value) =>
        value is double v ? v.ToString("R", CultureInfo.InvariantCulture) : "-";

    private static string FormatIntrinsics(WorldIntrinsics? intrinsics) => intrinsics is null
        ? "-"
        : string.Join(',', Format(intrinsics.Fx), Format(intrinsics.Fy), Format(intrinsics.Cx), Format(intrinsics.Cy));
}
