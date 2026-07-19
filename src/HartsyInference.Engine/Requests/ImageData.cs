namespace HartsyInference.Engine.Requests;

/// <summary>Engine-native raster image: raw RGB24 pixels (row-major, top-to-bottom) plus dimensions. Replaces the
/// SwarmUI <c>Image</c> type on the request/result contract so the Engine carries no host-app image dependency.</summary>
public sealed record ImageData
{
    /// <summary>Row-major, top-to-bottom RGB24 bytes; length is <see cref="Width"/> * <see cref="Height"/> * 3.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }
}
