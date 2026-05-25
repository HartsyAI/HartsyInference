using SharpInference.Core.Pipelines;

namespace SharpInference.Vision.FaceDetection;

/// <summary>Stub for a face-detection pipeline. The intended first target is YOLOv8-Face
/// (Ultralytics' face-specific YOLO variant trained on WIDER FACE) since it reuses the existing
/// <see cref="SharpInference.Vision.Detection.YoloModel"/> with a 1-class head plus an extra 5-point landmark
/// regression branch. RetinaFace and InsightFace would be alternative targets — the API surface
/// here is intentionally generic so any of them can plug in.</summary>
public sealed class FaceDetector : IVisionPipeline
{
    /// <inheritdoc/>
    public string ModelName => "face-detector-stub";

    /// <summary>Placeholder constructor.</summary>
    public FaceDetector() => throw new NotImplementedException(
        "FaceDetector scaffolding only — pick a backbone (YOLOv8-Face / RetinaFace / InsightFace) and " +
        "implement the detection + landmark head before instantiating.");

    /// <inheritdoc/>
    public void Dispose() { }
}
