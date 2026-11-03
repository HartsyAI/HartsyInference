namespace HartsyInference.Engine.Requests;

/// <summary>The result of a vision request. Which fields are populated depends on the request mode: an embedding for
/// Embed, detections for Detect, masks for Segment, one image for the map-producing modes, and the structured fields for
/// the geometry, pose, face, matting, intrinsic and sequence modes. Structured fields follow the conventions in
/// <c>docs/Agents/ENGINE_PATTERNS.md</c> (OpenCV camera frame, pixel centres at +0.5, metres for metric values).</summary>
public sealed record VisionResult
{
    /// <summary>Image embedding vector; null unless the mode was Embed.</summary>
    public float[]? Embedding { get; init; }

    /// <summary>Detected objects; null unless the mode was Detect.</summary>
    public IReadOnlyList<Detection>? Detections { get; init; }

    /// <summary>Segmentation masks (single-channel data reused as RGB); null unless the mode was Segment.</summary>
    public IReadOnlyList<ImageData>? Masks { get; init; }

    /// <summary>A single image: a depth, edge, line, normal or segmentation map, a background-removed or matted
    /// foreground, an upscaled image, or an interpolated frame; null otherwise.</summary>
    public ImageData? Image { get; init; }

    /// <summary>Float depth with its meaning and validity (<see cref="VisionMode.MetricDepth"/>,
    /// <see cref="VisionMode.Geometry"/>).</summary>
    public VisionDepthMap? DepthMap { get; init; }

    /// <summary>H×W×3 camera-space points in the OpenCV frame (<see cref="VisionMode.Geometry"/>).</summary>
    public VisionArray? PointMap { get; init; }

    /// <summary>H×W×3 unit normals in the OpenCV camera frame (<see cref="VisionMode.Geometry"/>).</summary>
    public VisionArray? Normals { get; init; }

    /// <summary>H×W soft foreground matte in [0, 1] (<see cref="VisionMode.Matting"/>).</summary>
    public VisionArray? Matte { get; init; }

    /// <summary>The estimated camera, when the model predicts one.</summary>
    public VisionCamera? Camera { get; init; }

    /// <summary>Detected people with keypoints (<see cref="VisionMode.Pose"/>).</summary>
    public IReadOnlyList<VisionPose>? Poses { get; init; }

    /// <summary>Detected faces (<see cref="VisionMode.Face"/>).</summary>
    public IReadOnlyList<VisionFace>? Faces { get; init; }

    /// <summary>Named intrinsic maps such as <c>albedo</c> (H×W×3) or <c>roughness</c> (H×W)
    /// (<see cref="VisionMode.Intrinsic"/>).</summary>
    public IReadOnlyDictionary<string, VisionArray>? IntrinsicMaps { get; init; }

    /// <summary>Tracked objects in one frame (<see cref="VisionMode.Track"/>).</summary>
    public IReadOnlyList<VisionTrackedObject>? Tracks { get; init; }

    /// <summary>One result per input view, in order, for a multi-view <see cref="VisionBatchRequest"/>.</summary>
    public IReadOnlyList<VisionResult>? Views { get; init; }
}
