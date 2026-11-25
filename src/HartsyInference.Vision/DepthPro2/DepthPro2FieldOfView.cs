namespace HartsyInference.Vision.DepthPro2;

/// <summary>A frame's predicted field of view in degrees, of the whole frame.</summary>
public readonly record struct DepthPro2FieldOfView(double HorizontalDegrees, double VerticalDegrees);
