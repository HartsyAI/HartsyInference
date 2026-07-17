namespace HartsyInference.Vision.FaceDetection;

/// <summary>One facial landmark point in source-image pixel coordinates. <see cref="Score"/> is the model's
/// sigmoid'd visibility for checkpoints that predict a per-point confidence (kptDims = 3); it is fixed to 1
/// for landmark-only checkpoints (kptDims = 2).</summary>
public readonly record struct FaceLandmark(float X, float Y, float Score = 1f);
