namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>A decoded character image for HairDeck: RGB24 and an optional straight 8-bit alpha plane, both row-major and top-down.</summary>
public sealed record HairDeckSourceImage
{
    /// <summary>Row-major RGB24, length <c>Width * Height * 3</c>.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Optional alpha plane, length <c>Width * Height</c>; null for an image without transparency information.</summary>
    public byte[]? Alpha { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }

    /// <summary>Validates the buffers against the size and returns the image.</summary>
    public static HairDeckSourceImage Create(byte[] rgb, byte[]? alpha, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (width <= 0 || height <= 0)
        {
            throw new HairDeckInputException(HairDeckInputFailure.InvalidImage, $"Image size must be positive; got {width}x{height}.");
        }

        if (rgb.Length != (long)width * height * 3)
        {
            throw new HairDeckInputException(HairDeckInputFailure.InvalidImage,
                $"RGB buffer holds {rgb.Length} bytes; a {width}x{height} image needs {(long)width * height * 3}.");
        }

        if (alpha is not null && alpha.Length != (long)width * height)
        {
            throw new HairDeckInputException(HairDeckInputFailure.InvalidImage,
                $"Alpha plane holds {alpha.Length} bytes; a {width}x{height} image needs {(long)width * height}.");
        }

        return new HairDeckSourceImage { Rgb = rgb, Alpha = alpha, Width = width, Height = height };
    }
}
