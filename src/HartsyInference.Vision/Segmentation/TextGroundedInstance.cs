namespace HartsyInference.Vision.Segmentation;

/// <summary>One text-grounded detection: a confidence score, an xyxy box in image pixels, and a binary mask (1 = inside)
/// at image resolution, row-major.</summary>
public sealed record TextGroundedInstance(float Score, float X0, float Y0, float X1, float Y1, byte[] Mask);
