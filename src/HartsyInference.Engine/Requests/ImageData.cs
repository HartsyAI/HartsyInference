using System.Text.Json.Serialization;

namespace HartsyInference.Engine.Requests;

/// <summary>Engine-native raster image: raw RGB24 pixels (row-major, top-to-bottom) plus dimensions. Replaces the SwarmUI <c>Image</c> type on the request/result contract so the Engine carries no host-app image dependency.</summary>
public sealed record ImageData
{
    /// <summary>Row-major, top-to-bottom RGB24 bytes; length is <see cref="Width"/> * <see cref="Height"/> * 3.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }

    /// <summary>Optional straight (non-premultiplied) 8-bit coverage plane, row-major like <see cref="Rgb"/>, length
    /// <see cref="Width"/> * <see cref="Height"/>; 255 is fully opaque. Null means the image is opaque. Produced by
    /// <see cref="VisionMode.BackgroundRemoval"/>; ignored by every consumer that only reads <see cref="Rgb"/>.</summary>
    public byte[]? Alpha { get; init; }

    /// <summary>True when an alpha plane is present and sized for this image.</summary>
    [JsonIgnore]
    public bool HasAlpha => Alpha is not null && Alpha.Length == (long)Width * Height;
}
