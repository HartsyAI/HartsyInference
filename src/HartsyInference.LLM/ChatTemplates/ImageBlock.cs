namespace HartsyInference.LLM.ChatTemplates;

/// <summary>An image inside a structured message body; <paramref name="ImageIndex"/> indexes <see cref="EncodeOptions.Images"/>.</summary>
public sealed record ImageBlock(int ImageIndex) : ContentBlock;
