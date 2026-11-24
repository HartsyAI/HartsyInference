namespace HartsyInference.Vision.DepthPro2;

/// <summary>One image's Depth Pro 2 maps, H×W row-major: focal-normalised inverse depth, metric depth in metres (positive infinity where infinitely far) and
/// the validity mask (1 where the inverse depth is above zero). The inverse depth is null when the caller asked for metres only.</summary>
public sealed record DepthPro2Map(float[]? InverseDepth, float[] DepthMeters, float[] ValidMask);
