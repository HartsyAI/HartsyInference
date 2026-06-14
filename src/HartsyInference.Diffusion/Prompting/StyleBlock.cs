namespace HartsyInference.Diffusion.Prompting;

/// <summary>Visual-style block for a <see cref="StructuredPrompt"/> (Ideogram's <c>style_description</c>). Exactly one of <see cref="Photo"/> / <see cref="ArtStyle"/> should be set: <see cref="Photo"/> for photographs, <see cref="ArtStyle"/> for illustrations / paintings / 3D renders.</summary>
public sealed record StyleBlock
{
    /// <summary>Overall aesthetic descriptor.</summary>
    public string? Aesthetics { get; init; }

    /// <summary>Lighting description.</summary>
    public string? Lighting { get; init; }

    /// <summary>Medium description.</summary>
    public string? Medium { get; init; }

    /// <summary>Photographic descriptor. Mutually exclusive with <see cref="ArtStyle"/>.</summary>
    public string? Photo { get; init; }

    /// <summary>Illustration/painting/3D descriptor. Mutually exclusive with <see cref="Photo"/>.</summary>
    public string? ArtStyle { get; init; }

    /// <summary>Up to 16 style-level hex colors (<c>#RRGGBB</c>, uppercase).</summary>
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}
