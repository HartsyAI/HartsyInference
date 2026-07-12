namespace HartsyInference.Vision.Detection;

/// <summary>A single pose keypoint in source-image pixel coordinates. <see cref="Confidence"/> is the model's
/// sigmoid'd visibility score — low values mean the point is occluded or off-frame.</summary>
public readonly record struct Keypoint(float X, float Y, float Confidence);

/// <summary>A YOLO-pose detection: the person box plus its ordered keypoints. COCO-17 order:
/// 0 nose · 1 left-eye · 2 right-eye · 3 left-ear · 4 right-ear · 5 left-shoulder · 6 right-shoulder ·
/// 7 left-elbow · 8 right-elbow · 9 left-wrist · 10 right-wrist · 11 left-hip · 12 right-hip ·
/// 13 left-knee · 14 right-knee · 15 left-ankle · 16 right-ankle. Points 0–4 bound the face region.</summary>
public sealed record PoseDetection(YoloDetection Box, IReadOnlyList<Keypoint> Keypoints)
{
    /// <summary>Detection confidence (the person-class score of the box).</summary>
    public float Confidence => Box.Confidence;
}
