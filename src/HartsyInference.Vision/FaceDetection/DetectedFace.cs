using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>A single detected face: the box (xyxy in source pixels, reusing <see cref="YoloDetection"/> for its IoU
/// helper and NMS compatibility) plus its five aligned landmarks. Produced by <see cref="FaceDetector.DetectFaces"/>.</summary>
public readonly record struct DetectedFace(YoloDetection Box, Landmark5 Landmarks)
{
    /// <summary>Face-class confidence of the box.</summary>
    public float Confidence => Box.Confidence;
}
