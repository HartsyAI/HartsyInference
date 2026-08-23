namespace HartsyInference.Engine.Requests;

/// <summary>The result of a vision request. Which field is populated depends on the request mode: an embedding for Embed, detections for Detect, masks for Segment.</summary>
public sealed record VisionResult
{
    /// <summary>Image embedding vector; null unless the mode was Embed.</summary>
    public float[]? Embedding { get; init; }

    /// <summary>Detected objects; null unless the mode was Detect.</summary>
    public IReadOnlyList<Detection>? Detections { get; init; }

    /// <summary>Segmentation masks (single-channel data reused as RGB); null unless the mode was Segment.</summary>
    public IReadOnlyList<ImageData>? Masks { get; init; }

    /// <summary>A single generated image (depth map, edge map, line map, normal map, segmentation palette, or background-removed foreground); null unless the mode was one of the single-image-output modes.</summary>
    public ImageData? Image { get; init; }
}
