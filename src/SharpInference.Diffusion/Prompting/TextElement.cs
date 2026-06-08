namespace SharpInference.Diffusion.Prompting;

/// <summary>A rendered-text element (Ideogram element <c>type:"text"</c>). <see cref="Text"/> is the literal string to draw in the image.</summary>
public sealed record TextElement : PromptElement
{
    /// <summary>Literal text to render.</summary>
    public required string Text { get; init; }
}
