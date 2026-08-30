namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Distinct H3 timesteps for one denoise evaluation plus scalar segment and optional target-row indices.</summary>
public sealed record MiniMaxH3TimestepPlan
{
    /// <summary>Ascending timestep values consumed by the shared time embedder.</summary>
    public required float[] Timesteps { get; init; }

    /// <summary>Scalar timestep row for every homogeneous packed segment kind.</summary>
    public required IReadOnlyDictionary<MiniMaxH3SegmentKind, int> RowOf { get; init; }

    /// <summary>Per-target-video-row timestep indices when a non-white video mask is active; otherwise null.</summary>
    public int[]? VideoRowOf { get; init; }

    /// <summary>Per-target-audio-row timestep indices when a non-white audio mask is active; otherwise null.</summary>
    public int[]? AudioRowOf { get; init; }
}
