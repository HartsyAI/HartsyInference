namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>The conditioning image of a TRELLIS.2 request: RGB24 pixels and an optional straight 8-bit alpha plane.</summary>
public sealed record Trellis2InputImage
{
    /// <summary>Interleaved RGB24, row-major, length <c>Width · Height · 3</c>.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Straight coverage, row-major, length <c>Width · Height</c>; null when the image is opaque.</summary>
    public byte[]? Alpha { get; init; }

    public required int Width { get; init; }

    public required int Height { get; init; }

    /// <summary>Throws <see cref="ArgumentException"/> when the buffers do not fit the dimensions.</summary>
    public void Validate()
    {
        if (Width <= 0 || Height <= 0)
            throw new ArgumentException($"Image dimensions must be positive; got {Width}x{Height}.");
        long pixels = (long)Width * Height;
        if (Rgb is null || Rgb.Length != pixels * 3)
            throw new ArgumentException($"RGB buffer must hold {pixels * 3} bytes for a {Width}x{Height} image.");
        if (Alpha is not null && Alpha.Length != pixels)
            throw new ArgumentException($"Alpha plane must hold {pixels} bytes for a {Width}x{Height} image.");
    }

    /// <summary>True when an alpha plane exists and is not fully opaque, the condition upstream uses to skip background removal.</summary>
    public bool HasForegroundAlpha
    {
        get
        {
            if (Alpha is null)
                return false;
            foreach (byte a in Alpha)
            {
                if (a != 255)
                    return true;
            }

            return false;
        }
    }
}
