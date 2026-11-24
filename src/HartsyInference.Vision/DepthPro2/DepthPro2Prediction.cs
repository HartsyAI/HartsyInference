namespace HartsyInference.Vision.DepthPro2;

/// <summary>Depth Pro 2's result for one image or several views of one scene: a map per image in input order, the focal length in pixels every map used,
/// and whether the focal estimator produced it.</summary>
public sealed record DepthPro2Prediction(IReadOnlyList<DepthPro2Map> Maps, double FocalLengthPx, bool FocalEstimated, int Width, int Height);
