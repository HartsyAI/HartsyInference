namespace HartsyInference.Vision.FaceDetection;

/// <summary>Stub for a facial-landmark extractor. Two flavors are commonly used:
/// <list type="bullet">
///   <item><b>5-point</b> — eyes, nose, mouth corners. Almost always produced as a side output by
///         face detectors (YOLOv8-Face, RetinaFace) rather than a separate model.</item>
///   <item><b>68-point or 468-point</b> — dense landmarks for alignment, expression, AR. These
///         are separate networks (FAN, MediaPipe FaceMesh) that take a cropped face as input.</item>
/// </list>
/// <para>Not yet implemented. Pick a target model in a future slice and replace this stub.</para></summary>
public sealed class LandmarkExtractor
{
    /// <summary>Placeholder.</summary>
    public LandmarkExtractor() => throw new NotImplementedException(
        "LandmarkExtractor scaffolding only — choose between 5-point detector-side output vs " +
        "68/468-point dense landmark model.");
}
