namespace HartsyInference.Engine.Requests;

/// <summary>One detected object: a label, confidence, and a pixel-space bounding box (top-left origin).</summary>
public sealed record Detection
{
    /// <summary>Class/label text.</summary>
    public required string Label { get; init; }

    /// <summary>Confidence score in 0..1.</summary>
    public required double Score { get; init; }

    /// <summary>Box left edge in pixels.</summary>
    public required int X { get; init; }

    /// <summary>Box top edge in pixels.</summary>
    public required int Y { get; init; }

    /// <summary>Box width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Box height in pixels.</summary>
    public required int Height { get; init; }
}
