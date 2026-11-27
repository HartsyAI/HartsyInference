namespace HartsyInference.LLM.ChatTemplates;

/// <summary>An image inside a structured message body; <paramref name="ImageIndex"/> indexes <see cref="EncodeOptions.Images"/>.</summary>
public sealed record ImageBlock(int ImageIndex) : ContentBlock
{
    /// <summary>Pixel width of the image, for templates that write its size into the prompt; null when the template does not need it.</summary>
    public int? Width { get; init; }

    /// <summary>Pixel height of the image, for templates that write its size into the prompt; null when the template does not need it.</summary>
    public int? Height { get; init; }
}
