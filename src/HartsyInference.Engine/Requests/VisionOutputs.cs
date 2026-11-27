namespace HartsyInference.Engine.Requests;

/// <summary>A dense float32 array, row-major with the last dimension fastest: H×W for scalar maps, H×W×3 for point maps
/// and normals. Pixel <c>(x, y)</c> of an H×W×C map starts at <c>(y·W + x)·C</c>. The CLI writes it as a little-endian
/// <c>.npy</c>; the API sends the same bytes base64-encoded with <see cref="Shape"/> and dtype <c>float32</c>
/// (<see cref="Services.VisionArrayCodec"/>).</summary>
public sealed record VisionArray
{
    /// <summary>Creates an array, checking that <paramref name="shape"/> covers <paramref name="data"/> exactly.</summary>
    public VisionArray(float[] data, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Length == 0 || shape.Any(d => d <= 0))
            throw new ArgumentException($"Every dimension must be positive; got [{string.Join(", ", shape)}].", nameof(shape));
        long count = shape.Aggregate(1L, (a, d) => a * d);
        if (count != data.Length)
            throw new ArgumentException($"Shape [{string.Join(", ", shape)}] holds {count} values but the data has {data.Length}.", nameof(data));
        Data = data;
        Shape = shape;
    }

    /// <summary>The values.</summary>
    public float[] Data { get; }

    /// <summary>The dimensions, outermost first.</summary>
    public int[] Shape { get; }

    /// <summary>Map height (the first dimension).</summary>
    public int Height => Shape[0];

    /// <summary>Map width (the second dimension), or 1 for a vector.</summary>
    public int Width => Shape.Length > 1 ? Shape[1] : 1;
}

/// <summary>What the values of a <see cref="VisionDepthMap"/> mean.</summary>
public enum VisionDepthKind
{
    /// <summary>Relative inverse depth (disparity-like): larger is nearer, unknown scale and shift.</summary>
    RelativeInverse,

    /// <summary>Relative depth: larger is farther, unknown scale.</summary>
    Relative,

    /// <summary>Metric depth along the camera's z axis, in metres.</summary>
    MetricMeters,
}

/// <summary>A float32 depth map with what its values mean and which pixels are valid.</summary>
public sealed record VisionDepthMap
{
    /// <summary>H×W depth values.</summary>
    public required VisionArray Values { get; init; }

    /// <summary>The meaning of <see cref="Values"/>.</summary>
    public required VisionDepthKind Kind { get; init; }

    /// <summary>H×W validity, 1 valid and 0 not (sky, out of range); null when every pixel is valid.</summary>
    public VisionArray? ValidMask { get; init; }
}

/// <summary>Pinhole camera intrinsics in pixels for an image of <see cref="Width"/> × <see cref="Height"/>, OpenCV
/// convention (x right, y down, z forward, pixel centres at +0.5), with an optional world-to-camera pose.</summary>
public sealed record VisionCamera
{
    /// <summary>Focal length along x, in pixels.</summary>
    public required double Fx { get; init; }

    /// <summary>Focal length along y, in pixels.</summary>
    public required double Fy { get; init; }

    /// <summary>Principal point x, in pixels.</summary>
    public required double Cx { get; init; }

    /// <summary>Principal point y, in pixels.</summary>
    public required double Cy { get; init; }

    /// <summary>The image width the intrinsics refer to.</summary>
    public required int Width { get; init; }

    /// <summary>The image height the intrinsics refer to.</summary>
    public required int Height { get; init; }

    /// <summary>World-to-camera <c>[R | t]</c>, 3×4 row-major (12 values, metres for metric models); null when the
    /// model predicts intrinsics only.</summary>
    public double[]? WorldToCamera { get; init; }
}

/// <summary>One detected person's pose.</summary>
public sealed record VisionPose
{
    /// <summary>The person's box in pixels.</summary>
    public required VisionBox Box { get; init; }

    /// <summary>Detection score.</summary>
    public required float Score { get; init; }

    /// <summary>Keypoints as consecutive <c>[x, y, score]</c> triples in pixels, in <see cref="Schema"/>'s order.</summary>
    public required float[] Keypoints { get; init; }

    /// <summary>The keypoint layout, for example <c>coco-wholebody-133</c> or <c>coco-17</c>.</summary>
    public required string Schema { get; init; }
}

/// <summary>One detected face.</summary>
public sealed record VisionFace
{
    /// <summary>The face box in pixels.</summary>
    public required VisionBox Box { get; init; }

    /// <summary>Detection score.</summary>
    public required float Score { get; init; }

    /// <summary>Landmarks as consecutive <c>[x, y]</c> pairs in pixels, in <see cref="Schema"/>'s order.</summary>
    public float[]? Landmarks { get; init; }

    /// <summary>The landmark layout, for example <c>mediapipe-6</c>; null when there are no landmarks.</summary>
    public string? Schema { get; init; }
}

/// <summary>One object in one frame of a <see cref="VisionMode.Track"/> stream.</summary>
public sealed record VisionTrackedObject
{
    /// <summary>The object id from the <see cref="VisionFramePrompt"/> that started it.</summary>
    public required int ObjectId { get; init; }

    /// <summary>The model's confidence that the object is present in this frame.</summary>
    public required float Score { get; init; }

    /// <summary>H×W mask probability in [0, 1].</summary>
    public required VisionArray Mask { get; init; }

    /// <summary>The mask's bounding box, or null when the object is absent from the frame.</summary>
    public VisionBox? Box { get; init; }
}

/// <summary>One frame of a streamed sequence result.</summary>
public sealed record VisionFrameResult
{
    /// <summary>The output frame's index in the stream, from 0.</summary>
    public required int Index { get; init; }

    /// <summary>The frame's presentation time in seconds.</summary>
    public required double TimestampSeconds { get; init; }

    /// <summary>For <see cref="VisionMode.Interpolate"/>, the input frame this output copies, or null for a synthesized
    /// frame. For other modes, the input frame the result is for.</summary>
    public int? SourceIndex { get; init; }

    /// <summary>The frame's result: <see cref="VisionResult.Image"/> for interpolation,
    /// <see cref="VisionResult.Tracks"/> for tracking.</summary>
    public required VisionResult Result { get; init; }
}
