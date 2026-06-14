namespace HartsyInference.Diffusion.Prompting;

/// <summary>Base for a spatially-placed element in a <see cref="StructuredPrompt"/>. Concrete types are <see cref="ObjectElement"/> and <see cref="TextElement"/>.</summary>
public abstract record PromptElement
{
    /// <summary>Optional normalized 0–1000 placement.</summary>
    public BoundingBox? Bbox { get; init; }

    /// <summary>Detailed description of the element.</summary>
    public string Description { get; init; } = "";

    /// <summary>Up to 5 per-element hex colors (<c>#RRGGBB</c>, uppercase).</summary>
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}
