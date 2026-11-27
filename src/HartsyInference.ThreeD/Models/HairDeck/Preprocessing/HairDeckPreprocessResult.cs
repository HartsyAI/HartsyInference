namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>The preprocessing of one character image: the model input, and the full-resolution masks and head box it was cut from.</summary>
public sealed record HairDeckPreprocessResult
{
    /// <summary>The image model's input: the 448 px crop on white and its matte.</summary>
    public required HairDeckInput Input { get; init; }

    /// <summary>The head box in source pixels; the crop of <see cref="Input"/> is this box.</summary>
    public required HairDeckBox Box { get; init; }

    /// <summary>Outcome of the mask stage.</summary>
    public required HairDeckPrepStatus Status { get; init; }

    /// <summary>Explicit warnings of this image (for example <c>no_person</c>, <c>matte_uncertain</c>, <c>empty_matte</c>); empty when none apply.</summary>
    public required IReadOnlyList<string> Notes { get; init; }

    /// <summary>Soft person matte at source size, row-major, 0 to 255.</summary>
    public required byte[] Person { get; init; }

    /// <summary>Hair mask at source size, row-major.</summary>
    public required bool[] Hair { get; init; }

    /// <summary>Face mask at source size, row-major.</summary>
    public required bool[] Face { get; init; }

    /// <summary>Source width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Source height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Median colour of the darker hair pixels, sRGB in [0, 1]; the hair colour when the colour model is off.</summary>
    public required double[] HairColor { get; init; }

    /// <summary>Agreement of the final matte with the matting model (intersection over union), or null when the matte was not computed.</summary>
    public double? MatteAgreement { get; init; }
}
