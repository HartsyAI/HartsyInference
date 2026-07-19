namespace HartsyInference.Engine.Requests;

/// <summary>One decoded frame streamed from a video generation: RGB24 pixels, dimensions, and the frame index.</summary>
public sealed record VideoFrame
{
    /// <summary>Row-major, top-to-bottom RGB24 bytes; length is <see cref="Width"/> * <see cref="Height"/> * 3.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }

    /// <summary>Zero-based frame index within the sequence.</summary>
    public required int Index { get; init; }
}
