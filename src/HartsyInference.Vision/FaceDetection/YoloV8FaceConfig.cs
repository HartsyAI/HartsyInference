using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>Presets for YOLOv8-Face — a plain YOLOv8 backbone + neck driven through <see cref="YoloConfig"/> with the
/// face head knobs set: <c>NumClasses = 1</c> (face), <c>NumKeypoints = 5</c> (the eyes/nose/mouth-corner landmarks),
/// and <c>KptDims</c> selecting whether the checkpoint's landmark branch carries a visibility channel.
/// <para>Reuses <see cref="YoloConfig"/> rather than subclassing it because the width/depth scaling formula is
/// identical to detection — only the head's class and keypoint counts differ, and those already live on the config.
/// The architecture switch (v8 backbone + face head) is <see cref="YoloV8FaceModel"/>, not this config.</para></summary>
public static class YoloV8FaceConfig
{
    /// <summary>Landmark points a face checkpoint emits (eyes, nose, mouth corners).</summary>
    public const int LandmarkCount = 5;

    /// <summary>YOLOv8n-Face — nano variant, same scaling as <see cref="YoloConfig.YoloV8n"/>. <paramref name="kptDims"/>
    /// is 3 (x, y, visibility) for Ultralytics-style face checkpoints; pass 2 for landmark-only checkpoints whose
    /// <c>cv4</c> branch has no visibility channel.</summary>
    public static YoloConfig YoloV8nFace(int kptDims = 3) => new()
    {
        Name = "yolov8n-face",
        DepthMultiple = 0.33f,
        WidthMultiple = 0.25f,
        MaxChannels = 1024,
        NumClasses = 1,
        NumKeypoints = LandmarkCount,
        KptDims = kptDims,
    };

    /// <summary>YOLOv8s-Face — small variant.</summary>
    public static YoloConfig YoloV8sFace(int kptDims = 3) => YoloV8nFace(kptDims) with { Name = "yolov8s-face", WidthMultiple = 0.50f };

    /// <summary>YOLOv8m-Face — medium variant.</summary>
    public static YoloConfig YoloV8mFace(int kptDims = 3) => YoloV8nFace(kptDims) with
    {
        Name = "yolov8m-face",
        DepthMultiple = 0.67f,
        WidthMultiple = 0.75f,
        MaxChannels = 768,
    };

    /// <summary>YOLOv8l-Face — large variant.</summary>
    public static YoloConfig YoloV8lFace(int kptDims = 3) => YoloV8nFace(kptDims) with
    {
        Name = "yolov8l-face",
        DepthMultiple = 1.00f,
        WidthMultiple = 1.00f,
        MaxChannels = 512,
    };
}
