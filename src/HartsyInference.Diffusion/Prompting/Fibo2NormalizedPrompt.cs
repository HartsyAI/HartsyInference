namespace HartsyInference.Diffusion.Prompting;

/// <summary>A Fibo 2.0 prompt after ComfyUI's caption handling: the text that goes into the user turn, and the text of every glyph
/// entry in caption order. With glyph images on, <see cref="Text"/> holds one vision marker per entry.</summary>
public sealed class Fibo2NormalizedPrompt
{
    /// <summary>Creates a normalised prompt.</summary>
    public Fibo2NormalizedPrompt(string text, IReadOnlyList<string> glyphTexts)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(glyphTexts);
        Text = text;
        GlyphTexts = glyphTexts;
    }

    /// <summary>The user-turn text.</summary>
    public string Text { get; }

    /// <summary>The <c>text</c> of each <c>text_render</c> entry that gets a glyph image, in order.</summary>
    public IReadOnlyList<string> GlyphTexts { get; }
}
