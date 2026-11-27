namespace HartsyInference.Vision.Segmentation;

/// <summary>One text-grounded query above a score floor: its confidence and its low-resolution mask logits, row-major.</summary>
public sealed record TextGroundedLogit(float Score, float[] Logits);
