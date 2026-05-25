namespace SharpInference.Vision.Detection;

/// <summary>A single YOLO-style detection. Boxes are <b>xyxy</b> (axis-aligned, top-left/bottom-right),
/// values are in pixel coordinates of the source image (post-<see cref="YoloPreprocessor.Transform.Invert"/>
/// unprojection). Class ID is the index into the model's label vocabulary — typically COCO 80,
/// resolvable through <see cref="CocoLabels.Get"/>.</summary>
public readonly record struct YoloDetection(
    float X1,
    float Y1,
    float X2,
    float Y2,
    float Confidence,
    int ClassId)
{
    /// <summary>Box width in source pixels.</summary>
    public float Width => X2 - X1;

    /// <summary>Box height in source pixels.</summary>
    public float Height => Y2 - Y1;

    /// <summary>Box area in source pixels (clamped to ≥ 0).</summary>
    public float Area => MathF.Max(0f, Width) * MathF.Max(0f, Height);

    /// <summary>Intersection-over-union with another detection. Axis-aligned, ignores class.</summary>
    public float Iou(YoloDetection other)
    {
        float ix1 = MathF.Max(X1, other.X1);
        float iy1 = MathF.Max(Y1, other.Y1);
        float ix2 = MathF.Min(X2, other.X2);
        float iy2 = MathF.Min(Y2, other.Y2);
        float iw = MathF.Max(0f, ix2 - ix1);
        float ih = MathF.Max(0f, iy2 - iy1);
        float inter = iw * ih;
        if (inter <= 0f)
            return 0f;
        float union = Area + other.Area - inter;
        return union > 0f ? inter / union : 0f;
    }
}
