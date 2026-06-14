namespace HartsyInference.Diffusion.Prompting;

/// <summary>Model-agnostic structured prompt: a scene summary, a style block, a background, and spatially-placed elements with optional bounding boxes and color palettes. One <see cref="StructuredPrompt"/> is authored once and serialized per target model by an <see cref="IPromptDialect"/> (Ideogram-4 JSON, plain prose, regional masks, …). See <c>docs/Research/STRUCTURED_PROMPT_BUILDER.md</c>.</summary>
public sealed record StructuredPrompt
{
    /// <summary>One- or two-sentence summary of the whole image (Ideogram <c>high_level_description</c>).</summary>
    public string? Summary { get; init; }

    /// <summary>Visual style / lighting / medium / palette.</summary>
    public StyleBlock? Style { get; init; }

    /// <summary>Background / environment description (required by Ideogram's <c>compositional_deconstruction</c>).</summary>
    public string? Background { get; init; }

    /// <summary>Ordered objects and text elements.</summary>
    public IReadOnlyList<PromptElement> Elements { get; init; } = [];

    /// <summary>Up to 16 overall-palette hex colors (<c>#RRGGBB</c>, uppercase).</summary>
    public IReadOnlyList<string> ColorPalette { get; init; } = [];
}
