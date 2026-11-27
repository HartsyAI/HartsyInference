namespace HartsyInference.Engine.Requests;

/// <summary>One layer of a layered image result: RGB24 plus a straight (non-premultiplied) 8-bit alpha plane at the
/// composite's size. Layers are ordered back to front; <see cref="Index"/> 0 is the bottom.</summary>
public sealed record ImageLayer
{
    /// <summary>Position from the bottom, 0-based, equal to the layer's place in <see cref="ImageResult.Layers"/>.</summary>
    public required int Index { get; init; }

    /// <summary>Row-major RGB24 bytes, <see cref="Width"/> × <see cref="Height"/> × 3.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Straight alpha, <see cref="Width"/> × <see cref="Height"/>; 255 is opaque.</summary>
    public required byte[] Alpha { get; init; }

    /// <summary>Pixel width, the composite's.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height, the composite's.</summary>
    public required int Height { get; init; }

    /// <summary>A label the model gives the layer, such as <c>background</c>; null when it gives none.</summary>
    public string? Name { get; init; }
}
