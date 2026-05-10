namespace SharpInference.Core.Pipelines;

/// <summary>A single detected object with bounding box and classification.</summary>
public sealed class DetectionResult
{
    /// <summary>Bounding box: X coordinate of top-left corner (normalized 0-1).</summary>
    public float X { get; init; }

    /// <summary>Bounding box: Y coordinate of top-left corner (normalized 0-1).</summary>
    public float Y { get; init; }

    /// <summary>Bounding box width (normalized 0-1).</summary>
    public float Width { get; init; }

    /// <summary>Bounding box height (normalized 0-1).</summary>
    public float Height { get; init; }

    /// <summary>Detection confidence score (0-1).</summary>
    public float Confidence { get; init; }

    /// <summary>Detected class label.</summary>
    public required string Label { get; init; }

    /// <summary>Class index in the model's label set.</summary>
    public int ClassIndex { get; init; }
}
