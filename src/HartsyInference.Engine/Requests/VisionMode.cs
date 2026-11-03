namespace HartsyInference.Engine.Requests;

/// <summary>Which vision operation a <see cref="VisionRequest"/> performs.</summary>
public enum VisionMode
{
    /// <summary>Produce an image embedding.</summary>
    Embed,

    /// <summary>Detect objects (bounding boxes), optionally text-conditioned.</summary>
    Detect,

    /// <summary>Produce segmentation masks, optionally text-conditioned.</summary>
    Segment,

    /// <summary>Produce a relative-depth grayscale map (DepthAnything-V2).</summary>
    Depth,

    /// <summary>Produce a ControlNet-style soft-edge map (HED).</summary>
    Edge,

    /// <summary>Produce a ControlNet-style line-art map (realistic or coarse variant).</summary>
    Lineart,

    /// <summary>Produce a ControlNet-style surface-normal map (NormalBAE).</summary>
    Normal,

    /// <summary>Produce an ADE20K semantic-segmentation palette map (UperNet-ConvNeXt).</summary>
    SegMap,

    /// <summary>Produce the foreground image with the background removed (RMBG-1.4). The result carries the
    /// gray-0.5 composite in <see cref="ImageData.Rgb"/> and the foreground matte in <see cref="ImageData.Alpha"/>.</summary>
    BackgroundRemoval,

    /// <summary>Enlarge the image with a Real-ESRGAN generator (x4plus by default; x2plus / anime6b by catalog id),
    /// optionally fitted to <see cref="VisionRequest.TargetWidth"/> × <see cref="VisionRequest.TargetHeight"/>.</summary>
    Upscale,

    /// <summary>Metric depth in metres: <see cref="VisionResult.DepthMap"/> with <see cref="VisionDepthKind.MetricMeters"/>,
    /// plus the estimated <see cref="VisionResult.Camera"/> when the model predicts one. <see cref="Depth"/> keeps its
    /// grayscale-image output.</summary>
    MetricDepth,

    /// <summary>Monocular geometry: a point map, normals, a validity mask and the camera
    /// (<see cref="VisionResult.PointMap"/>, <see cref="VisionResult.Normals"/>, <see cref="VisionResult.DepthMap"/>,
    /// <see cref="VisionResult.Camera"/>), each when the model predicts it.</summary>
    Geometry,

    /// <summary>Human pose keypoints with their boxes (<see cref="VisionResult.Poses"/>).</summary>
    Pose,

    /// <summary>A soft foreground matte: the foreground in <see cref="VisionResult.Image"/> with straight alpha, and the
    /// matte itself as a float map in <see cref="VisionResult.Matte"/>.</summary>
    Matting,

    /// <summary>Face boxes with landmarks (<see cref="VisionResult.Faces"/>).</summary>
    Face,

    /// <summary>Intrinsic decomposition maps such as albedo, shading or material properties
    /// (<see cref="VisionResult.IntrinsicMaps"/>).</summary>
    Intrinsic,

    /// <summary>Prompted object tracking across a frame sequence. Takes a <see cref="VisionBatchRequest"/> with
    /// <see cref="VisionSequenceKind.Frames"/> and streams one <see cref="VisionFrameResult"/> per frame.</summary>
    Track,

    /// <summary>Frame interpolation. Takes a <see cref="VisionBatchRequest"/> with <see cref="VisionSequenceKind.Frames"/>
    /// and streams the output frames, originals included, each with its exact timestamp.</summary>
    Interpolate,
}
