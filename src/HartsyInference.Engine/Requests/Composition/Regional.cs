namespace HartsyInference.Engine.Requests;

/// <summary>Regional / segment prompting: the region plan (parsed from the prompt's region/segment syntax by the
/// caller) plus the mask shaping and per-segment step/cfg overrides applied to segment refinement passes.</summary>
public sealed record Regional
{
    /// <summary>The raw region/segment plan expression (the prompt's <c>&lt;region&gt;</c>/<c>&lt;segment&gt;</c> syntax).</summary>
    public string? Plan { get; init; }

    /// <summary>Ordering strategy for resolved segments.</summary>
    public string? SortOrder { get; init; }

    /// <summary>Pixels to grow each segment mask.</summary>
    public int MaskGrow { get; init; }

    /// <summary>Blur radius for each segment mask, in pixels.</summary>
    public int MaskBlur { get; init; }

    /// <summary>Oversize padding around each segment crop, in pixels.</summary>
    public int MaskOversize { get; init; }

    /// <summary>Per-segment refinement step count; null reuses the base steps.</summary>
    public int? Steps { get; init; }

    /// <summary>Per-segment refinement CFG; null reuses the base CFG.</summary>
    public double? CfgScale { get; init; }
}
