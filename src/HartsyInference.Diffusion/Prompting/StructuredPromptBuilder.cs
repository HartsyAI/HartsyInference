namespace HartsyInference.Diffusion.Prompting;

/// <summary>Fluent builder for a <see cref="StructuredPrompt"/>. Convenience over the records — useful for UIs (e.g. the SwarmUI extension) and tests.</summary>
public sealed class StructuredPromptBuilder
{
    private string? _summary;
    private StyleBlock? _style;
    private string? _background;
    private readonly List<PromptElement> _elements = [];
    private readonly List<string> _palette = [];

    /// <summary>Sets the high-level scene summary.</summary>
    public StructuredPromptBuilder Summary(string summary)
    {
        _summary = summary;
        return this;
    }

    /// <summary>Configures the style block via a nested builder lambda.</summary>
    public StructuredPromptBuilder Style(Action<StyleBlockBuilder> configure)
    {
        StyleBlockBuilder b = new();
        configure(b);
        _style = b.Build();
        return this;
    }

    /// <summary>Sets the background/environment description.</summary>
    public StructuredPromptBuilder Background(string background)
    {
        _background = background;
        return this;
    }

    /// <summary>Adds an object element.</summary>
    public StructuredPromptBuilder AddObject(string desc, BoundingBox? bbox = null, IReadOnlyList<string>? palette = null)
    {
        _elements.Add(new ObjectElement { Description = desc, Bbox = bbox, ColorPalette = palette ?? [] });
        return this;
    }

    /// <summary>Adds a rendered-text element.</summary>
    public StructuredPromptBuilder AddText(string text, BoundingBox? bbox = null, string desc = "", IReadOnlyList<string>? palette = null)
    {
        _elements.Add(new TextElement { Text = text, Description = desc, Bbox = bbox, ColorPalette = palette ?? [] });
        return this;
    }

    /// <summary>Sets the overall color palette.</summary>
    public StructuredPromptBuilder Palette(params string[] colors)
    {
        _palette.Clear();
        _palette.AddRange(colors);
        return this;
    }

    /// <summary>Builds the immutable <see cref="StructuredPrompt"/>.</summary>
    public StructuredPrompt Build() => new()
    {
        Summary = _summary,
        Style = _style,
        Background = _background,
        Elements = _elements.ToArray(),
        ColorPalette = _palette.ToArray(),
    };

    /// <summary>Nested builder for <see cref="StyleBlock"/>.</summary>
    public sealed class StyleBlockBuilder
    {
        private string? _aesthetics;
        private string? _lighting;
        private string? _medium;
        private string? _photo;
        private string? _artStyle;
        private readonly List<string> _palette = [];

        /// <summary>Sets the aesthetics descriptor.</summary>
        public StyleBlockBuilder Aesthetics(string v) { _aesthetics = v; return this; }

        /// <summary>Sets the lighting descriptor.</summary>
        public StyleBlockBuilder Lighting(string v) { _lighting = v; return this; }

        /// <summary>Sets the medium descriptor.</summary>
        public StyleBlockBuilder Medium(string v) { _medium = v; return this; }

        /// <summary>Sets the photographic descriptor (clears any art-style).</summary>
        public StyleBlockBuilder Photo(string v) { _photo = v; _artStyle = null; return this; }

        /// <summary>Sets the art-style descriptor (clears any photo).</summary>
        public StyleBlockBuilder ArtStyle(string v) { _artStyle = v; _photo = null; return this; }

        /// <summary>Sets the style-level palette.</summary>
        public StyleBlockBuilder Palette(params string[] colors) { _palette.Clear(); _palette.AddRange(colors); return this; }

        internal StyleBlock Build() => new()
        {
            Aesthetics = _aesthetics,
            Lighting = _lighting,
            Medium = _medium,
            Photo = _photo,
            ArtStyle = _artStyle,
            ColorPalette = _palette.ToArray(),
        };
    }
}
