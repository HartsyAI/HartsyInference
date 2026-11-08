namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The image model's input for one HairDeck request: the 448 px head crop composited on white, and the crop's person matte.</summary>
/// <remarks>The reference feeds the crop as uint8 (B, 3, 448, 448) and the DINOv3 encoder applies the ImageNet normalisation itself,
/// so the crop is not normalised here. The colour model reads the same crop and the matte as a boolean mask.</remarks>
public sealed record HairDeckInput
{
    /// <summary>Side of the square crop, the image model's resolution.</summary>
    public const int Resolution = 448;

    /// <summary>Uint8 RGB in channel-major order: 3 planes of 448 * 448 bytes, index <c>(c * 448 + y) * 448 + x</c>.</summary>
    public required byte[] Rgb448 { get; init; }

    /// <summary>Person matte of the crop, row-major 448 * 448 bytes, each 0 or 255. All 255 when the image has no matte.</summary>
    public required byte[] Matte448 { get; init; }

    /// <summary>Validates the two planes and returns the input.</summary>
    public static HairDeckInput Create(byte[] rgb448, byte[] matte448)
    {
        ArgumentNullException.ThrowIfNull(rgb448);
        ArgumentNullException.ThrowIfNull(matte448);
        const int plane = Resolution * Resolution;
        if (rgb448.Length != 3 * plane)
        {
            throw new ArgumentException($"HairDeck model input RGB must hold {3 * plane} bytes; got {rgb448.Length}.", nameof(rgb448));
        }

        if (matte448.Length != plane)
        {
            throw new ArgumentException($"HairDeck model input matte must hold {plane} bytes; got {matte448.Length}.", nameof(matte448));
        }

        return new HairDeckInput { Rgb448 = rgb448, Matte448 = matte448 };
    }
}
