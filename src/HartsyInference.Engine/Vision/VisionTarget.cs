namespace HartsyInference.Engine.Vision;

/// <summary>A parsed vision target: which detector to run plus the model name, text query, class filter and detection
/// index decoded from the request prompt (the SwarmUI <c>&lt;segment:...&gt;</c> target grammar).</summary>
public sealed record VisionTarget
{
    /// <summary>The detector this target routes to.</summary>
    public required VisionTargetKind Kind { get; init; }

    /// <summary>YOLO checkpoint name (without extension); empty for the other detectors.</summary>
    public string ModelName { get; init; } = "";

    /// <summary>Free-text phrase for Grounding DINO / CLIPSeg; empty for the closed-set detectors.</summary>
    public string Query { get; init; } = "";

    /// <summary>Case-insensitive label substring filter; empty means keep every class.</summary>
    public string ClassFilter { get; init; } = "";

    /// <summary>Index of the detection to select when the caller wants a single result; -1 means "all".</summary>
    public int Index { get; init; } = -1;
}
